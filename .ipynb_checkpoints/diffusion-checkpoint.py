import itertools
from dataclasses import dataclass
from collections import OrderedDict

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from omegaconf import ListConfig, OmegaConf

import dataloader # Assuming dataloader.py is present
import metrics   # Assuming metrics.py is present
import models    # Assuming models/__init__.py properly imports dit, ema, dimamba, etc.
import noise_schedule # Assuming noise_schedule.py is present
import utils     # Assuming utils.py is present

from models.meta_controller import MetaController

# For feature extraction
from transformers import AutoModel, AutoTokenizer
import spacy
import benepar # Needs nltk.download('punkt') and benepar.download('benepar_en3')
from nltk.tree import Tree # For parsing benepar's output string


def _sample_categorical(categorical_probs):
  # Detach probabilities before sampling if they might have grads
  categorical_probs_no_grad = categorical_probs.detach()
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs_no_grad) + 1e-10).log())
  samples = (categorical_probs_no_grad / gumbel_norm).argmax(dim=-1)
  return samples

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size
        self.sampler = self.config.algo.sampler
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.cross_attn = self.config.algo.cross_attn # Note: Generation impact needs careful model design
        self.ignore_bos = self.config.algo.ignore_bos
        self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale
        if (not hasattr(self.tokenizer, 'mask_token')
            or self.tokenizer.mask_token is None):
          self.mask_index = self.vocab_size
          self.vocab_size += 1
        else:
          self.mask_index = self.tokenizer.mask_token_id

        if hasattr(self.config, 'algo'):
          self.parameterization = self.config.algo.parameterization
        else:
          self.parameterization = self.config.parameterization
        if hasattr(self.config, 'block_size'):
          self.block_size = self.config.block_size
        else:
          self.block_size = self.config.model.length
        if self.parameterization == 'ar':
          self.block_size = 1
        
        if self.config.algo.backbone == 'dit':
          self.backbone = models.dit.DIT(
            self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'dimamba':
          if not hasattr(models, 'dimamba') or not hasattr(models.dimamba, 'DiMamba'):
              raise ImportError("models.dimamba.DiMamba not found.")
          self.backbone = models.dimamba.DiMamba(
            self.config,
            vocab_size=self.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id)
        elif self.config.algo.backbone == 'hf_dit':
          self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
            config.eval.checkpoint_path, trust_remote_code=True)
          if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
            self.config.model.attn_backend == 'sdpa':
            self.backbone.config.attn_backend = 'sdpa'
            if hasattr(self.backbone, 'backbone') and hasattr(self.backbone.backbone, 'blocks'):
                for i_block_internal in self.backbone.backbone.blocks: 
                    i_block_internal.attn_backend = 'sdpa'
                if hasattr(self.backbone.backbone, 'gen_mask'):
                    self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
            else:
                print("Warning: hf_dit backbone structure for attn_backend patching not as expected.")
        else:
          raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

        self.T = self.config.algo.T
        self.num_tokens = self.config.model.length
        self.noise = noise_schedule.get_noise(self.config) # This is the global, fixed noise schedule
        self.metrics = metrics.Metrics(config)

        self.var_min = self.config.algo.var_min
        if self.var_min:
          self.register_buffer('sampling_eps_min', torch.tensor(
            self.config.training.sampling_eps_min))
          self.register_buffer('sampling_eps_max', torch.tensor(
            self.config.training.sampling_eps_max))

        self.time_conditioning = self.config.algo.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        
        # --- Feature Extractor: DistilBERT ---
        self.distilbert_model_name = getattr(config.algo, "feature_extractor_model_name", "distilbert-base-uncased")
        try:
            self.distilbert_tokenizer = AutoTokenizer.from_pretrained(self.distilbert_model_name)
            self.distilbert_model = AutoModel.from_pretrained(self.distilbert_model_name)
            self.distilbert_model.eval()
            for param in self.distilbert_model.parameters():
                param.requires_grad = False
            self.distilbert_hidden_size = self.distilbert_model.config.hidden_size
            print(f"Successfully loaded DistilBERT model '{self.distilbert_model_name}' (Hidden: {self.distilbert_hidden_size}).")
        except Exception as e:
            print(f"Warning: Could not load DistilBERT '{self.distilbert_model_name}'. Semantic features will be zero/limited. Error: {e}")
            self.distilbert_tokenizer = None
            self.distilbert_model = None
            self.distilbert_hidden_size = 768 

        # --- Feature Extractor: SpaCy + Benepar for Syntactic Depth ---
        self.spacy_model_name = getattr(config.algo, "spacy_model_name", "en_core_web_sm")
        self.benepar_model_name = getattr(config.algo, "benepar_model_name", "benepar_en3")
        try:
            self.nlp = spacy.load(self.spacy_model_name, exclude=["ner", "lemmatizer"])
            if not self.nlp.has_pipe("benepar"):
                if spacy.__version__.startswith("2"):
                    self.nlp.add_pipe(benepar.BeneparComponent(self.benepar_model_name))
                else: 
                    self.nlp.add_pipe("benepar", config={"model": self.benepar_model_name})
            print(f"Successfully loaded spaCy model '{self.spacy_model_name}' and benepar model '{self.benepar_model_name}'.")
        except Exception as e:
            print(f"Warning: Could not load spaCy/benepar. Syntactic depth feature will be zero/limited. Error: {e}")
            self.nlp = None
            
        self.feature_dim_parts = { # This should reflect features available at GENERATION
            "pos_ratio": 1,
            # Semantic feature of the *context* or a global signal
            "semantic_context": self.distilbert_hidden_size if self.distilbert_model is not None else 0,
            # Proxies for current block features (e.g., average from context or zero)
            "block_entropy_proxy": 1, 
            "token_variance_proxy": 1,
            "syntactic_depth_context": 1 if self.nlp is not None else 0, # Syntactic depth of context
            "context_entropy": 1, # Entropy of accumulated context
        }
        # This calculated_feature_dim_generation is what _extract_features_for_block_generation should produce.
        # The MetaController's feature_dim MUST match this.
        calculated_feature_dim_generation = sum(self.feature_dim_parts.values())
        
        original_mc_config_dict = OmegaConf.to_container(config.algo.meta_controller, resolve=True)
        current_meta_controller_config_obj = OmegaConf.create(original_mc_config_dict)

        if current_meta_controller_config_obj.feature_dim != calculated_feature_dim_generation:
            print(f"ADJUSTING MetaController 'feature_dim' for generation: Config had {current_meta_controller_config_obj.feature_dim}, "
                  f"but calculated {calculated_feature_dim_generation} based on generation-available features. Using calculated value.")
            print(f"Generation feature breakdown: {self.feature_dim_parts}")
            # This assumes the MetaController was trained understanding this feature structure.
            current_meta_controller_config_obj.feature_dim = calculated_feature_dim_generation
        
        self.meta_controller_config = current_meta_controller_config_obj
        
        temp_config_for_mc_init = OmegaConf.create({'algo': {'meta_controller': self.meta_controller_config}})
        self.meta_controller = MetaController(temp_config_for_mc_init)

        # Base noise schedule for warping
        self.base_noise_schedule = noise_schedule.get_noise(config, config.algo.base_noise_type)
        
        _dummy_zero_t = torch.tensor(0.0, dtype=torch.float32) 
        clamped_base_alpha_at_0 = self.base_noise_schedule.get_alpha_bar(_dummy_zero_t).clamp(
            config.algo.schedule_clamp_epsilon, 1.0 - config.algo.schedule_clamp_epsilon
        )

        self.register_buffer("target_alpha_at_0", torch.tensor(1.0 - config.algo.schedule_clamp_epsilon, dtype=torch.float32))
        self.register_buffer("base_log_alpha_bar_at_0", torch.logit(clamped_base_alpha_at_0))

        self.min_alpha_1_target = self.config.algo.min_alpha_1_target
        self.lambda_min_alpha_1_penalty = self.config.algo.lambda_min_alpha_1_penalty
        self.alpha_1_clamp_min = self.config.algo.alpha_1_clamp_min
        self.alpha_1_clamp_max = self.config.algo.alpha_1_clamp_max
        
        self.lambda_steps_penalty = self.config.algo.lambda_steps_penalty
        self.lambda_s_b_l2_penalty = getattr(self.config.algo, 'lambda_s_b_l2_penalty', 0.0)

        # For Dynamic NFE remapping
        self.dynamic_nfe_grid_size = getattr(self.config.algo, "dynamic_nfe_grid_size", 1000)
            
        if self.config.training.ema > 0:
          self.ema = models.ema.ExponentialMovingAverage(
            self._get_parameters(), # Initial parameters might change with stage, so EMA re-init in on_train_start
            decay=self.config.training.ema)
        else:
          self.ema = None

        self._validate_configuration()

    def _get_parameters(self):
        params_to_optimize = []
        # Ensure config.training.stage is accessible, default if not
        training_stage = getattr(self.config.training, "stage", "joint") 

        if training_stage == "controller_only":
            print("Optimizer: MetaController parameters only.")
            params_to_optimize.append(self.meta_controller.parameters())
        elif training_stage == "joint":
            print("Optimizer: Jointly training LM and MetaController.")
            params_to_optimize.append(self.backbone.parameters())
            # Assuming self.noise (base noise schedule) is not learnable by default
            # If it were learnable:
            # if hasattr(self.noise, 'parameters') and any(p.requires_grad for p in self.noise.parameters()):
            #      params_to_optimize.append(self.noise.parameters())
            params_to_optimize.append(self.meta_controller.parameters())
        elif training_stage == "lm_only": # Standard LM training or finetuning
            print("Optimizer: LM parameters only (MetaController frozen).")
            params_to_optimize.append(self.backbone.parameters())
            # if hasattr(self.noise, 'parameters') and any(p.requires_grad for p in self.noise.parameters()):
            #      params_to_optimize.append(self.noise.parameters())
        else:
            raise ValueError(f"Unknown training stage: {training_stage}")

        return itertools.chain(*params_to_optimize)

    def _get_parse_tree_height(self, parse_string: str) -> int:
        if not parse_string or not parse_string.startswith("("):
            return 0
        try:
            tree = Tree.fromstring(parse_string)
            return tree.height()
        except ValueError: 
            return 0

    def _get_block_features(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # This is the TRAINING-time feature extractor, operating on x0.
        # It should produce features that the MetaController is trained on.
        # Some of these features might not be available at generation time for the *current block being generated*.
        # The MetaController should be robust to this, or generation-time features should be proxies.
        B, N_Tokens = tokens.shape
        target_feature_dim = self.meta_controller_config.feature_dim # From __init__, potentially adjusted

        if self.block_size == 0 or N_Tokens == 0 :
            return torch.empty(B, 0, target_feature_dim, device=tokens.device, dtype=torch.float32)

        num_blocks = N_Tokens // self.block_size
        if num_blocks == 0:
             return torch.empty(B, 0, target_feature_dim, device=tokens.device, dtype=torch.float32)

        if N_Tokens % self.block_size != 0:
            N_Tokens_for_feat = num_blocks * self.block_size
            # print(f"Warning (_get_block_features): N_Tokens ({N_Tokens}) not multiple of block_size ({self.block_size}). Using first {N_Tokens_for_feat} tokens.")
            tokens_for_feat = tokens[:, :N_Tokens_for_feat]
        else:
            tokens_for_feat = tokens
        
        block_tokens_view = tokens_for_feat.reshape(B, num_blocks, self.block_size)

        # Initialize based on self.feature_dim_parts, which should align with MetaController's expectation
        # (even if some are proxied/zeroed during generation)
        # Training features:
        _feature_dim_parts_training = {
            "pos_ratio": 1,
            "semantic": self.distilbert_hidden_size if self.distilbert_model is not None else 0, # Semantic of current block
            "block_entropy": 1,          # Entropy of current block
            "token_variance": 1,         # Variance of current block tokens
            "syntactic_depth": 1 if self.nlp is not None else 0,      # Syntactic depth of current block
            "context_entropy": 1,        # Entropy of preceding context
        }
        # Ensure `target_feature_dim` (from MetaController config) matches sum of these training feature parts
        if sum(_feature_dim_parts_training.values()) != target_feature_dim:
            # This can happen if generation features are different, and MC config was set to generation one.
            # For training, we MUST use the features it was intended to be trained on.
            # This indicates a mismatch in config setup if target_feature_dim was set to generation one.
            # Assuming target_feature_dim IS for training here.
            pass # Allow mismatch if target_feature_dim is the "generation" one, and we are robust. Best if they match.


        pos_ratio_feat = torch.zeros(B, num_blocks, _feature_dim_parts_training["pos_ratio"], device=tokens.device, dtype=torch.float32)
        semantic_feat = torch.zeros(B, num_blocks, _feature_dim_parts_training["semantic"], device=tokens.device, dtype=torch.float32)
        block_entropy_feat = torch.zeros(B, num_blocks, _feature_dim_parts_training["block_entropy"], device=tokens.device, dtype=torch.float32)
        token_variance_feat = torch.zeros(B, num_blocks, _feature_dim_parts_training["token_variance"], device=tokens.device, dtype=torch.float32)
        syntactic_depth_feat = torch.zeros(B, num_blocks, _feature_dim_parts_training["syntactic_depth"], device=tokens.device, dtype=torch.float32)
        context_entropy_feat = torch.zeros(B, num_blocks, _feature_dim_parts_training["context_entropy"], device=tokens.device, dtype=torch.float32)

        if _feature_dim_parts_training["pos_ratio"] > 0:
            block_indices_vals = torch.arange(num_blocks, device=tokens.device, dtype=torch.float32)
            pos_ratio_unexpanded = block_indices_vals / (num_blocks - 1 + 1e-9) if num_blocks > 1 else torch.zeros_like(block_indices_vals)
            pos_ratio_feat.copy_(pos_ratio_unexpanded.view(1, num_blocks, 1).expand(B, -1, -1))

        if self.distilbert_model is not None and self.distilbert_tokenizer is not None and _feature_dim_parts_training["semantic"] > 0:
            flat_block_token_ids = block_tokens_view.reshape(B * num_blocks, self.block_size)
            # Handle empty blocks for tokenizer, provide pad token if block is all padding
            block_texts_flat = [
                self.tokenizer.decode(ids.cpu().tolist(), skip_special_tokens=True).strip() or self.distilbert_tokenizer.pad_token
                for ids in flat_block_token_ids
            ]
            distilbert_inputs = self.distilbert_tokenizer(
                block_texts_flat, padding='longest', truncation=True,
                max_length=self.distilbert_tokenizer.model_max_length, return_tensors="pt"
            ).to(self.distilbert_model.device) # Ensure model is on the correct device
            with torch.no_grad():
                distilbert_outputs = self.distilbert_model(**distilbert_inputs)
            cls_embeddings_flat = distilbert_outputs.last_hidden_state[:, 0, :] # CLS token embedding
            semantic_feat.copy_(cls_embeddings_flat.reshape(B, num_blocks, self.distilbert_hidden_size).to(tokens.device))

        for b_idx in range(B):
            accumulated_context_tokens_list = []
            for blk_idx in range(num_blocks):
                current_block_token_ids = block_tokens_view[b_idx, blk_idx]

                if _feature_dim_parts_training["block_entropy"] > 0 and current_block_token_ids.numel() > 0:
                    _, counts = torch.unique(current_block_token_ids, return_counts=True)
                    if counts.sum() > 0: probs = counts.float() / counts.sum(); block_entropy_feat[b_idx, blk_idx, 0] = torch.special.entr(probs).sum()

                if _feature_dim_parts_training["token_variance"] > 0 and current_block_token_ids.numel() > 0:
                    token_variance_feat[b_idx, blk_idx, 0] = current_block_token_ids.float().var(unbiased=False)
                
                if self.nlp is not None and _feature_dim_parts_training["syntactic_depth"] > 0:
                    block_text = self.tokenizer.decode(current_block_token_ids.cpu().tolist(), skip_special_tokens=True).strip()
                    avg_depth = 0.0
                    if block_text:
                        doc = self.nlp(block_text)
                        sentence_depths = [self._get_parse_tree_height(sent._.parse_string) for sent in doc.sents if hasattr(sent._, 'parse_string') and sent._.parse_string]
                        sentence_depths = [d for d in sentence_depths if d > 0]
                        if sentence_depths: avg_depth = sum(sentence_depths) / len(sentence_depths)
                    syntactic_depth_feat[b_idx, blk_idx, 0] = avg_depth
                
                if _feature_dim_parts_training["context_entropy"] > 0:
                    if blk_idx > 0 and accumulated_context_tokens_list:
                        context_tensor = torch.tensor(accumulated_context_tokens_list, device=tokens.device, dtype=torch.long)
                        if context_tensor.numel() > 0:
                            _, counts = torch.unique(context_tensor, return_counts=True)
                            if counts.sum() > 0: probs = counts.float() / counts.sum(); context_entropy_feat[b_idx, blk_idx, 0] = torch.special.entr(probs).sum()
                
                if _feature_dim_parts_training["context_entropy"] > 0: # Accumulate for next block's context
                    accumulated_context_tokens_list.extend(current_block_token_ids.cpu().tolist())
        
        all_features_collected = []
        if _feature_dim_parts_training["pos_ratio"] > 0: all_features_collected.append(pos_ratio_feat)
        if _feature_dim_parts_training["semantic"] > 0: all_features_collected.append(semantic_feat) # This is current block semantic
        if _feature_dim_parts_training["block_entropy"] > 0: all_features_collected.append(block_entropy_feat)
        if _feature_dim_parts_training["token_variance"] > 0: all_features_collected.append(token_variance_feat)
        if _feature_dim_parts_training["syntactic_depth"] > 0: all_features_collected.append(syntactic_depth_feat) # This is current block syntax
        if _feature_dim_parts_training["context_entropy"] > 0: all_features_collected.append(context_entropy_feat)

        if not all_features_collected:
            return torch.zeros(B, num_blocks, target_feature_dim, device=tokens.device, dtype=torch.float32)

        final_features_cat = torch.cat(all_features_collected, dim=-1)
        
        if final_features_cat.shape[-1] != target_feature_dim:
             # This can be an issue if target_feature_dim was set to the generation-time one in __init__
             # And the training features sum to something different.
             # The MetaController expects `target_feature_dim`.
             # Solution: Ensure target_feature_dim used for initializing MetaController is for training.
             # Or, ensure _get_block_features only computes `target_feature_dim` features.
             # For now, assume `target_feature_dim` is for training.
             print(f"Warning (_get_block_features): Feature dimension mismatch. Training features sum to {final_features_cat.shape[-1]}, "
                   f"MC expects {target_feature_dim}. Ensure MC config matches training features.")
             # If it's critical, pad or truncate, but ideally fix the config/feature list.
             if final_features_cat.shape[-1] > target_feature_dim:
                 final_features_cat = final_features_cat[..., :target_feature_dim]
             elif final_features_cat.shape[-1] < target_feature_dim:
                 padding = torch.zeros(B, num_blocks, target_feature_dim - final_features_cat.shape[-1], device=tokens.device, dtype=torch.float32)
                 final_features_cat = torch.cat([final_features_cat, padding], dim=-1)

        return final_features_cat

    def _extract_features_for_block_generation(
        self, 
        x_accum_context: torch.Tensor, # (B, L_context)
        current_block_idx_to_generate: int,
        total_num_blocks: int
        ) -> torch.Tensor:
        """
        Extracts features for the Meta-Controller for the *next block to be generated*.
        This version needs to be carefully aligned with how the MetaController was trained.
        Features that depend on the *current block's actual tokens* (e.g. its entropy, its specific semantic embedding)
        must be handled (e.g., zeroed out, proxied by context, or the MC trained to not rely on them).
        """
        B = x_accum_context.shape[0] if x_accum_context is not None and x_accum_context.numel() > 0 else 1 # Assume B=1 if no context
        device = x_accum_context.device if x_accum_context is not None and x_accum_context.numel() > 0 else self.device
        
        # Use self.feature_dim_parts which was defined in __init__ for generation-available features
        # and should match self.meta_controller_config.feature_dim
        target_gen_feature_dim = self.meta_controller_config.feature_dim

        # Initialize feature tensors (for a single block, shape (B, 1, FeatPartDim))
        pos_ratio_feat = torch.zeros(B, 1, self.feature_dim_parts["pos_ratio"], device=device, dtype=torch.float32)
        semantic_context_feat = torch.zeros(B, 1, self.feature_dim_parts["semantic_context"], device=device, dtype=torch.float32)
        block_entropy_proxy_feat = torch.zeros(B, 1, self.feature_dim_parts["block_entropy_proxy"], device=device, dtype=torch.float32)
        token_variance_proxy_feat = torch.zeros(B, 1, self.feature_dim_parts["token_variance_proxy"], device=device, dtype=torch.float32)
        syntactic_depth_context_feat = torch.zeros(B, 1, self.feature_dim_parts["syntactic_depth_context"], device=device, dtype=torch.float32)
        context_entropy_feat = torch.zeros(B, 1, self.feature_dim_parts["context_entropy"], device=device, dtype=torch.float32)

        # 1. Position Ratio
        if self.feature_dim_parts["pos_ratio"] > 0:
            if total_num_blocks > 1:
                pos_ratio = current_block_idx_to_generate / (total_num_blocks - 1 + 1e-9)
            else:
                pos_ratio = 0.0
            pos_ratio_feat.fill_(pos_ratio)

        # Context available?
        has_context = x_accum_context is not None and x_accum_context.numel() > 0

        # 2. Semantic Context Feature (e.g., CLS of last N tokens of context)
        if has_context and self.distilbert_model is not None and self.distilbert_tokenizer is not None and self.feature_dim_parts["semantic_context"] > 0:
            # Take last N tokens, or all if shorter. Max length for distilbert is 512.
            context_for_sem = x_accum_context[:, -self.distilbert_tokenizer.model_max_length // 2:] # Heuristic for length
            
            context_texts_flat = [
                self.tokenizer.decode(ids.cpu().tolist(), skip_special_tokens=True).strip() or self.distilbert_tokenizer.pad_token
                for ids in context_for_sem
            ]
            if not any(context_texts_flat): # All empty strings
                 pass # semantic_context_feat remains zero
            else:
                distilbert_inputs = self.distilbert_tokenizer(
                    context_texts_flat, padding='longest', truncation=True,
                    max_length=self.distilbert_tokenizer.model_max_length, return_tensors="pt"
                ).to(self.distilbert_model.device)
                with torch.no_grad():
                    distilbert_outputs = self.distilbert_model(**distilbert_inputs)
                cls_embeddings_flat = distilbert_outputs.last_hidden_state[:, 0, :] # (B, HiddenSize)
                semantic_context_feat = cls_embeddings_flat.unsqueeze(1).to(device) # (B, 1, HiddenSize)
        
        # 3. Block Entropy Proxy (e.g., set to 0 or avg entropy of context blocks)
        # For simplicity, setting to 0. A better proxy could be developed.
        if self.feature_dim_parts["block_entropy_proxy"] > 0:
            block_entropy_proxy_feat.fill_(0.0) 

        # 4. Token Variance Proxy (e.g., set to 0)
        if self.feature_dim_parts["token_variance_proxy"] > 0:
            token_variance_proxy_feat.fill_(0.0)

        # 5. Syntactic Depth of Context
        if has_context and self.nlp is not None and self.feature_dim_parts["syntactic_depth_context"] > 0:
            avg_depth_context_batch = []
            for b_idx in range(B):
                context_text_item = self.tokenizer.decode(x_accum_context[b_idx].cpu().tolist(), skip_special_tokens=True).strip()
                avg_depth_item = 0.0
                if context_text_item:
                    doc = self.nlp(context_text_item)
                    sentence_depths = [self._get_parse_tree_height(sent._.parse_string) for sent in doc.sents if hasattr(sent._, 'parse_string') and sent._.parse_string]
                    sentence_depths = [d for d in sentence_depths if d > 0]
                    if sentence_depths: avg_depth_item = sum(sentence_depths) / len(sentence_depths)
                avg_depth_context_batch.append(avg_depth_item)
            syntactic_depth_context_feat = torch.tensor(avg_depth_context_batch, device=device, dtype=torch.float32).view(B, 1, 1)
        
        # 6. Context Entropy (Entropy of x_accum_context)
        if has_context and self.feature_dim_parts["context_entropy"] > 0:
            context_entropy_batch = []
            for b_idx in range(B):
                context_item_tokens = x_accum_context[b_idx]
                entropy_val = 0.0
                if context_item_tokens.numel() > 0:
                    _, counts = torch.unique(context_item_tokens, return_counts=True)
                    if counts.sum() > 0: probs = counts.float() / counts.sum(); entropy_val = torch.special.entr(probs).sum().item()
                context_entropy_batch.append(entropy_val)
            context_entropy_feat = torch.tensor(context_entropy_batch, device=device, dtype=torch.float32).view(B, 1, 1)

        # Concatenate features based on self.feature_dim_parts (order defined in __init__)
        all_gen_features_collected = []
        if self.feature_dim_parts["pos_ratio"] > 0: all_gen_features_collected.append(pos_ratio_feat)
        if self.feature_dim_parts["semantic_context"] > 0: all_gen_features_collected.append(semantic_context_feat)
        if self.feature_dim_parts["block_entropy_proxy"] > 0: all_gen_features_collected.append(block_entropy_proxy_feat)
        if self.feature_dim_parts["token_variance_proxy"] > 0: all_gen_features_collected.append(token_variance_proxy_feat)
        if self.feature_dim_parts["syntactic_depth_context"] > 0: all_gen_features_collected.append(syntactic_depth_context_feat)
        if self.feature_dim_parts["context_entropy"] > 0: all_gen_features_collected.append(context_entropy_feat)
        
        if not all_gen_features_collected:
            return torch.zeros(B, 1, target_gen_feature_dim, device=device, dtype=torch.float32)

        final_gen_features_cat = torch.cat(all_gen_features_collected, dim=-1) # Shape (B, 1, FeatDim)

        if final_gen_features_cat.shape[-1] != target_gen_feature_dim:
            raise ValueError(
                f"CRITICAL Dimension Mismatch for GENERATION MetaController features: "
                f"Concatenated feature dimension is {final_gen_features_cat.shape[-1]}, "
                f"but MetaController expects {target_gen_feature_dim} (from config, should match self.feature_dim_parts sum). "
                f"Generation feature_dim_parts: {self.feature_dim_parts} (sum: {sum(self.feature_dim_parts.values())})."
            )
        return final_gen_features_cat


    def _get_warped_noise_outputs_for_block_batch(
        self,
        x0_block_features: torch.Tensor, # (B, N_Blk, FeatDim)
        u_values_per_block: torch.Tensor # (B, N_Blk) - u are the t-values for training
    ):
        # This is used in TRAINING (_loss method)
        log_s_tilde_b = self.meta_controller(x0_block_features) # (B, N_Blk, 1)
        s_b = self.meta_controller.get_s_b(log_s_tilde_b)       # (B, N_Blk, 1)
        s_b_squeezed = s_b.squeeze(-1) # (B, N_Blk) for get_warped_schedule_outputs

        # u_values_per_block is (B, N_Blk). Expand for schedule funcs if they expect (B,N_Blk,N_t) or (N_t,)
        # Here, N_t = 1 for each block's u value.
        base_alpha_bar_u = self.base_noise_schedule.get_alpha_bar(u_values_per_block).clamp(
             self.config.algo.schedule_clamp_epsilon, 1.0 - self.config.algo.schedule_clamp_epsilon
        ) # (B, N_Blk)
        base_log_alpha_bar_base_derivative_u = self.base_noise_schedule.get_log_alpha_bar_base_derivative_t(u_values_per_block) # (B, N_Blk)

        target_alpha_at_0_device = self.target_alpha_at_0.to(u_values_per_block.device)

        # get_warped_schedule_outputs expects:
        # base_alpha_bar_t: (B, N_Blk, N_t) or (N_t,). Here (B, N_Blk) implies N_t=1 effectively per block
        # s_b: (B, N_Blk)
        alpha_b_u_all_blocks, loss_scale_b_u_all_blocks, p_b_u_all_blocks = \
            noise_schedule.get_warped_schedule_outputs(
                base_alpha_bar_t=base_alpha_bar_u, # (B, N_Blk)
                base_log_alpha_bar_base_derivative_t=base_log_alpha_bar_base_derivative_u, # (B, N_Blk)
                base_log_alpha_bar_at_0=self.base_log_alpha_bar_at_0, # Scalar
                s_b=s_b_squeezed, # (B, N_Blk)
                target_log_alpha_at_0=torch.logit(target_alpha_at_0_device) # Scalar
            )
        # Outputs will be (B, N_Blk)
        return alpha_b_u_all_blocks, loss_scale_b_u_all_blocks, p_b_u_all_blocks, s_b, log_s_tilde_b

    # --- Phase 4: New helper methods for inference with adaptive schedules ---
    @torch.no_grad()
    def _get_remapped_timesteps_for_block(self, 
                                          s_b_current_block: torch.Tensor, # Shape: (B, 1, 1)
                                          global_step_k: int, # Current global step index (1 to total_global_steps)
                                          total_global_steps: int
                                         ) -> tuple[torch.Tensor, torch.Tensor]:
        B = s_b_current_block.shape[0]
        device = s_b_current_block.device

        tau_grid = torch.linspace(0.0, 1.0, self.dynamic_nfe_grid_size, device=device, dtype=torch.float32) # (grid_size,)
        
        base_alpha_bar_tau_unexpanded = self.base_noise_schedule.get_alpha_bar(tau_grid).clamp(
            self.config.algo.schedule_clamp_epsilon, 1.0 - self.config.algo.schedule_clamp_epsilon
        ) # (grid_size,)
        base_log_alpha_bar_base_derivative_tau_unexpanded = self.base_noise_schedule.get_log_alpha_bar_base_derivative_t(tau_grid) # (grid_size,)

        # Expand for batch operation: (B, 1, grid_size)
        base_alpha_bar_tau = base_alpha_bar_tau_unexpanded.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        base_log_alpha_bar_base_derivative_tau = base_log_alpha_bar_base_derivative_tau_unexpanded.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

        alpha_b_j_tau_grid, beta_b_j_tau_grid, _ = noise_schedule.get_warped_schedule_outputs(
            base_alpha_bar_t=base_alpha_bar_tau, # (B, 1, grid_size)
            base_log_alpha_bar_base_derivative_t=base_log_alpha_bar_base_derivative_tau, # (B, 1, grid_size)
            base_log_alpha_bar_at_0=self.base_log_alpha_bar_at_0, # Scalar
            s_b=s_b_current_block.squeeze(-1), # (B, 1) from (B,1,1) for get_warped_schedule_outputs
            target_log_alpha_at_0=torch.logit(self.target_alpha_at_0)
        )
        # Outputs are (B, 1, grid_size), squeeze to (B, grid_size)
        alpha_b_j_tau_grid = alpha_b_j_tau_grid.squeeze(1) 
        beta_b_j_tau_grid = beta_b_j_tau_grid.squeeze(1)   

        alpha_b_at_1_clamped = torch.clamp(alpha_b_j_tau_grid[:, -1], min=self.alpha_1_clamp_min, max=self.alpha_1_clamp_max)
        T_b_j = -torch.log(alpha_b_at_1_clamped).unsqueeze(-1) # (B, 1)

        C_k_target = (global_step_k / total_global_steps) * T_b_j
        C_k_minus_1_target = ((global_step_k - 1) / total_global_steps) * T_b_j if global_step_k > 0 else torch.zeros_like(T_b_j)
        
        dt_tau = 1.0 / (self.dynamic_nfe_grid_size -1) if self.dynamic_nfe_grid_size > 1 else 1.0 # step size in tau_grid
        #Integral approx: sum(beta_values * dt_tau)
        cumulative_beta_integral = torch.cumsum(beta_b_j_tau_grid, dim=-1) * dt_tau # (B, grid_size)
        
        max_integral_val = cumulative_beta_integral[:, -1].unsqueeze(-1).clamp(min=1e-9) # Ensure positive for clamping
        C_k_target_clamped = torch.min(C_k_target, max_integral_val)
        C_k_minus_1_target_clamped = torch.min(C_k_minus_1_target, max_integral_val)

        # searchsorted needs C_k_target_clamped to be 1D per batch item
        idx_k_list = [torch.searchsorted(cumulative_beta_integral[b], C_k_target_clamped[b,0], right=True) for b in range(B)]
        idx_k_minus_1_list = [torch.searchsorted(cumulative_beta_integral[b], C_k_minus_1_target_clamped[b,0], right=True) for b in range(B)]
        
        idx_k = torch.stack(idx_k_list).clamp(max=self.dynamic_nfe_grid_size - 1) # (B, 1)
        idx_k_minus_1 = torch.stack(idx_k_minus_1_list).clamp(max=self.dynamic_nfe_grid_size - 1) # (B, 1)

        t_prime_k_for_block = torch.gather(tau_grid.expand(B, -1), 1, idx_k) 
        t_prime_k_minus_1_for_block = torch.gather(tau_grid.expand(B, -1), 1, idx_k_minus_1)
        
        return t_prime_k_for_block, t_prime_k_minus_1_for_block

    @torch.no_grad()
    def _get_warped_outputs_at_t(self, 
                                 s_b_current_block: torch.Tensor, # Shape: (B, 1, 1)
                                 t_values: torch.Tensor          # Shape: (B, 1) or (B, N_t)
                                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = s_b_current_block.shape[0]
        if t_values.ndim == 1: t_values = t_values.unsqueeze(-1) # (B,1)
        # t_values is (B, N_t), s_b is (B,1,1). Expand t_values for broadcasting with base schedule outputs.
        # Base schedule expects (N_t) or (B, N_Blk, N_t). Here N_Blk=1.
        t_values_for_base = t_values.view(B, 1, -1) # (B, 1, N_t)

        base_alpha_bar_t = self.base_noise_schedule.get_alpha_bar(t_values_for_base).clamp(
            self.config.algo.schedule_clamp_epsilon, 1.0 - self.config.algo.schedule_clamp_epsilon
        ) # (B, 1, N_t)
        base_log_alpha_bar_base_derivative_t = self.base_noise_schedule.get_log_alpha_bar_base_derivative_t(t_values_for_base) # (B, 1, N_t)

        alpha_b_t, beta_b_t, p_b_t = noise_schedule.get_warped_schedule_outputs(
            base_alpha_bar_t=base_alpha_bar_t, # (B, 1, N_t)
            base_log_alpha_bar_base_derivative_t=base_log_alpha_bar_base_derivative_t, # (B, 1, N_t)
            base_log_alpha_bar_at_0=self.base_log_alpha_bar_at_0, # Scalar
            s_b=s_b_current_block.squeeze(-1), # (B, 1) for broadcasting with N_t dim
            target_log_alpha_at_0=torch.logit(self.target_alpha_at_0)
        )
        # Output shapes will be (B, 1, N_t). If N_t=1, effectively (B,1,1)
        return alpha_b_t, beta_b_t, p_b_t
    # --- End of Phase 4 New helper methods ---


    def on_validation_model_zero_grad(self) -> None:
        # Prevent PL from zeroing grads during accumulating_grad_batches > 1
        super().on_validation_model_zero_grad()
        # Skip initial validation when restarting from checkpoint
        if hasattr(self,'trainer') and self.trainer is not None and self.trainer.ckpt_path is not None and \
           getattr(self, '_restarting_skip_val_flag', True):
            # print("Sanity checking, skipping initial validation.")
            self.trainer.sanity_checking = True # Tells PL it's a sanity check run
            self._restarting_skip_val_flag = False # Reset flag

    def _validate_configuration(self):
        if self.config.mode == 'sample_eval' and \
            self.config.sampling.first_hitting:
          assert self.config.loader.eval_batch_size == 1
        assert self.config.algo.backbone in {
          'dit', 'ar', 'hf_dit', 'dimamba'} # Added dimamba
        if self.config.algo.parameterization == 'ar':
          assert not self.config.algo.time_conditioning
        if self.config.sampling.kv_cache:
          assert self.config.algo.name in {'ar', 'bd3lm'} # KV cache for BD3LM

        if self.parameterization in {'subs', 'sedd'}: # Removed 'bd3lm' as it handles time differently
          assert self.time_conditioning
        
        if self.config.algo.name == 'bd3lm' and self.parameterization != 'bd3lm':
            print(f"Warning: algo.name is 'bd3lm' but parameterization is '{self.parameterization}'. Ensure this is intentional.")

        if self.config.mode == 'sample_eval':
          assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
        if self.config.model.attn_backend == 'flex':
          assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
          if hasattr(self.backbone.block_diff_mask, 'to'): # Make sure it's a tensor or module
            self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
        elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
          # FlexAttention mask might be managed differently or not be a direct tensor attribute
          if hasattr(self.backbone.block_diff_mask, 'to'):
              self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device) # Flex attention mask usually stays on device
        if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min): # Check if tensor before calling .to()
          self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
          self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
        
        # Move feature extractors and MetaController if they exist
        if hasattr(self, 'distilbert_model') and self.distilbert_model is not None:
            self.distilbert_model = self.distilbert_model.to(*args, **kwargs)
        # SpaCy model (self.nlp) typically runs on CPU and manages its own device placement.

        if hasattr(self, 'meta_controller'):
            self.meta_controller = self.meta_controller.to(*args, **kwargs)
        if hasattr(self, 'target_alpha_at_0'): # Buffer
            self.target_alpha_at_0 = self.target_alpha_at_0.to(*args, **kwargs)
        if hasattr(self, 'base_log_alpha_bar_at_0'): # Buffer
            self.base_log_alpha_bar_at_0 = self.base_log_alpha_bar_at_0.to(*args, **kwargs)
        if hasattr(self, 'base_noise_schedule'): # Module
            self.base_noise_schedule = self.base_noise_schedule.to(*args, **kwargs)
        return self

    def _replace_ckpt_keys(self, checkpoint):
        # Model was possibly saved with DDP, remove prefix
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
          new_state_dict[k.replace('_orig_mod.', '')] = v # For FSDP/DDP compiled
        checkpoint['state_dict'] = new_state_dict
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        print('Loading checkpoint at', checkpoint['global_step'])
        self._restarting_skip_val_flag = True # For skipping val on restart

        # Handle checkpoints saved with FSDP/DDP _orig_mod prefix
        if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
          checkpoint = self._replace_ckpt_keys(checkpoint)

        if self.ema and 'ema' in checkpoint: # Check if 'ema' key exists
          self.ema.load_state_dict(checkpoint['ema'])
            
        if 'sampling_eps_min' in checkpoint.keys():
          self.sampling_eps_min = checkpoint['sampling_eps_min']
          self.sampling_eps_max = checkpoint['sampling_eps_max']
        # Fast forward dataloader state.
        if 'loops' in checkpoint and 'fit_loop' in checkpoint['loops']: # Check if loops exist
            self.fast_forward_epochs = checkpoint['loops'][
              'fit_loop']['epoch_progress']['current']['completed']
            self.fast_forward_batches = checkpoint['loops'][
              'fit_loop']['epoch_loop.batch_progress'][
                'current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
          checkpoint['ema'] = self.ema.state_dict()
        if hasattr(self, 'sampling_eps_min'): # Check if attribute exists before saving
          checkpoint['sampling_eps_min'] = self.sampling_eps_min
          checkpoint['sampling_eps_max'] = self.sampling_eps_max
        
        # Need to save batch_sampler state for fault tolerance.
        # Simplified version, actual saving might be more complex depending on sampler type
        if 'loops' not in checkpoint: checkpoint['loops'] = {}
        if 'fit_loop' not in checkpoint['loops']: checkpoint['loops']['fit_loop'] = {}
        if 'epoch_loop.batch_progress' not in checkpoint['loops']['fit_loop']:
            checkpoint['loops']['fit_loop']['epoch_loop.batch_progress'] = {'total': {'completed':0}, 'current':{'completed':0}}
        if 'epoch_loop.automatic_optimization.optim_progress' not in checkpoint['loops']['fit_loop']: # Path for automatic opt
            checkpoint['loops']['fit_loop']['epoch_loop.automatic_optimization.optim_progress'] = {'optimizer': {'step': {'total':{'completed':0}, 'current':{'completed':0}}}}
        if 'epoch_loop.state_dict' not in checkpoint['loops']['fit_loop']: # Fallback path
            checkpoint['loops']['fit_loop']['epoch_loop.state_dict'] = {'_batches_that_stepped': 0}


        # Update completed batches based on optimizer steps and grad accumulation
        checkpoint['loops']['fit_loop'][
          'epoch_loop.batch_progress']['total'][
            'completed'] = checkpoint['loops']['fit_loop'][
              'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total'][
                  'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
          'epoch_loop.batch_progress']['current'][
            'completed'] = checkpoint['loops']['fit_loop'][
              'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['current'][
                  'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
          'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
              'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total']['completed']
        # Save dataloader sampler state if applicable
        if 'sampler' not in checkpoint.keys():
          checkpoint['sampler'] = {}
        
        train_dataloader_obj = self.trainer.train_dataloader
        actual_sampler = None
        if hasattr(train_dataloader_obj, 'sampler') and hasattr(train_dataloader_obj.sampler, 'state_dict'): # Direct attribute
            actual_sampler = train_dataloader_obj.sampler
        elif hasattr(train_dataloader_obj, 'loader') and hasattr(train_dataloader_obj.loader, 'sampler') and \
             hasattr(train_dataloader_obj.loader.sampler, 'state_dict'): # Wrapped in CombinedLoader
            actual_sampler = train_dataloader_obj.loader.sampler
        
        if actual_sampler:
            sampler_state_dict = actual_sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None) # specific to RandomFaultTolerantSampler
        else:
            checkpoint['sampler']['random_state'] = None


    def on_train_start(self):
        # This hook is called before training begins.
        # Set requires_grad based on the current training stage.
        training_stage = getattr(self.config.training, "stage", "joint")
        print(f"on_train_start: Setting requires_grad for stage: {training_stage}")

        if training_stage == "controller_only":
            for param in self.backbone.parameters():
                param.requires_grad = False
            if hasattr(self.noise, 'parameters'): # If base noise schedule were learnable
                 for param in self.noise.parameters(): param.requires_grad = False
            for param in self.meta_controller.parameters():
                param.requires_grad = True
        elif training_stage == "lm_only":
            for param in self.backbone.parameters():
                param.requires_grad = True
            if hasattr(self.noise, 'parameters'):
                 for param in self.noise.parameters(): param.requires_grad = True # Or False if fixed base schedule
            for param in self.meta_controller.parameters():
                param.requires_grad = False
        elif training_stage == "joint":
            for param in self.backbone.parameters():
                param.requires_grad = True
            if hasattr(self.noise, 'parameters'):
                 for param in self.noise.parameters(): param.requires_grad = True # Or False
            for param in self.meta_controller.parameters():
                param.requires_grad = True
        
        # Initialize or re-initialize EMA here, now that _get_parameters will work correctly for the current stage.
        if self.config.training.ema > 0: # EMA is configured
            current_params_for_ema = list(self._get_parameters()) # Get currently trainable params
            if not any(p.requires_grad for p in current_params_for_ema): # Handles controller_only where backbone is frozen
                # If only controller is trained, EMA might apply to controller only or be disabled.
                # For simplicity, if _get_parameters() returns only controller params, EMA applies to them.
                # If _get_parameters() returned an empty list (e.g. if all frozen mistakenly), EMA would fail.
                # The current _get_parameters should always return something if a stage is active.
                pass

            if not self.ema or (hasattr(self.ema, '_original_params') and self.ema._original_params != current_params_for_ema):
                print(f"Re-initializing EMA for stage {training_stage}.")
                self.ema = models.ema.ExponentialMovingAverage(
                    current_params_for_ema, 
                    decay=self.config.training.ema
                )
            if self.ema: # Ensure shadow params are on the correct device
                 self.ema.move_shadow_params_to_device(self.device)
        
        # Dataloader sampler setup for fault tolerance
        if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, '_accelerator_connector'):
            # This part assumes a Pytorch Lightning Trainer context
            distributed = (
              self.trainer._accelerator_connector.use_distributed_sampler
              and self.trainer._accelerator_connector.is_distributed)
            if distributed:
              sampler_cls = dataloader.FaultTolerantDistributedSampler
            else:
              sampler_cls = dataloader.RandomFaultTolerantSampler
            # This logic reinitializes dataloaders, might be complex with CombinedLoader
            updated_dls = []
            
            # Accessing flattened dataloaders if using CombinedLoader (common in PL for multiple val_dataloaders)
            # For train_dataloader, it's usually simpler.
            if hasattr(self.trainer.fit_loop,'_combined_loader') and self.trainer.fit_loop._combined_loader is not None:
                # This applies if multiple train_dataloaders are used, uncommon.
                # More likely, self.trainer.train_dataloader is the one.
                # For simplicity, assuming single train dataloader. If multiple, this loop is needed.
                for dl in self.trainer.fit_loop._combined_loader.flattened: # Iterate through all dataloaders managed by fit_loop
                  # Check if this DL is a train_dataloader, might need more specific check if val_dls are also here
                  if hasattr(dl.sampler, 'shuffle'): # Check if original sampler had shuffle
                    dl_sampler = sampler_cls(
                      dl.dataset, shuffle=dl.sampler.shuffle)
                  else: # Default to no shuffle if not specified, or handle based on sampler_cls default
                    dl_sampler = sampler_cls(dl.dataset) # shuffle defaults to True in RandomFaultTolerantSampler
                  
                  # Restore sampler state if restarting
                  if (distributed # Or not distributed if RandomFaultTolerantSampler also supports load_state_dict
                      and self.fast_forward_epochs is not None
                      and self.fast_forward_batches is not None):
                    # This state dict loading is specific to how FaultTolerantDistributedSampler saves state
                    dl_sampler.load_state_dict({
                      'epoch': self.fast_forward_epochs,
                      'counter': (self.fast_forward_batches # This should be #samples processed, not #batches
                                  * self.config.loader.batch_size)}) # Assuming batch_size means per GPU in DDP
                  updated_dls.append(
                    torch.utils.data.DataLoader(
                      dl.dataset,
                      batch_size=self.config.loader.batch_size, # Use global batch size from config
                      num_workers=self.config.loader.num_workers,
                      pin_memory=self.config.loader.pin_memory,
                      sampler=dl_sampler,
                      shuffle=False, # Sampler handles shuffling
                      persistent_workers=True if self.config.loader.num_workers > 0 else False))
                self.trainer.fit_loop._combined_loader.flattened = updated_dls # Replace Dataloaders in CombinedLoader


    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
          self.ema.update(self._get_parameters()) # Pass currently trainable parameters

    def _subs_parameterization(self, logits, xt):
        # Parameterization for SUBS model from "Structured Denoising Diffusion Models in Discrete State-Spaces"
        logits[:, :, self.mask_index] += self.neg_infinity # Prevent sampling mask token
        logits = logits - torch.logsumexp(logits, dim=-1,
                                          keepdim=True) # Normalize
        # Set probability of non-masked tokens to 0, and prob of true token to 1
        # This ensures p(x_t | x_0, x_t) = 1 if x_t is not mask, and p(mask | x_0, x_t) = 0
        unmasked_indices = (xt != self.mask_index)
        if xt[unmasked_indices].numel() > 0 : # Check if there are any unmasked tokens
             logits[unmasked_indices] = self.neg_infinity # Zero out all probs for unmasked
             logits[unmasked_indices, xt[unmasked_indices]] = 0.0 # Set true token prob to 1 (log(1)=0)
        return logits

    def _sedd_parameterization(self, logits, xt, sigma_batch):
        # Parameterization for SEDD model from "Scalable Diffusion Models with Transformers"
        # Assuming sigma_batch is (B,) or broadcastable with logits (B, N, V)
        # This needs sigma_batch to be per-sample, not per-token if logits are (B,N,V)
        # Ensure sigma_batch is correctly shaped, e.g., (B,1,1) for broadcasting
        if sigma_batch.ndim == 1: sigma_batch = sigma_batch.view(-1,1,1)

        esigm1_log = torch.where( # Numerically stable expm1
          sigma_batch < 0.5, # Heuristic threshold
          torch.expm1(sigma_batch), # exp(sigma) - 1 for small sigma
          sigma_batch.exp() - 1.0 # Direct computation for larger sigma
          ).log().to(logits.dtype) # log(exp(sigma) - 1)
        
        # Equation (5) from SEDD paper: log p_theta(x_0 | x_t)
        # = log q_phi(x_t | x_0) - log(exp(sigma_t) - 1) - log(V-1)
        # Here, logits are model's direct output for q_phi (up to a softmax)
        # So, logits_final = logits_raw - log(exp(sigma_t)-1) - log(V-1)
        # And then scatter value for x_0 = x_t to be 0 (as per their objective, if p_theta(x_t|x_t)=1 implicitly)
        # This part is tricky: SEDD's loss is on score s_theta(x_0, x_t).
        # If `logits` here are for p_theta(x_0|x_t), then the adjustment is as above.
        
        # The provided code seems to implement the parameterization for the *output distribution*
        # log p(x_0 | x_t) = s_theta(x_0, x_t) - log Z_t(x_t)
        # where s_theta(x_0, x_t) = { M(x_0, x_t) if x_0 != x_t; 0 if x_0 = x_t }
        # and M(x_0,x_t) is the model output.
        # The normalization log Z_t(x_t) involves sum over all x'_0.
        # This is closer to the final log_softmax stage.
        
        # The provided code:
        logits = logits - esigm1_log - np.log(logits.shape[-1] - 1.0) # Adjusts raw model output
        # Scatter 0 for x_0 = x_t. This means model does not predict x_t as x_0.
        # This aligns with s_theta(x_t, x_t) = 0.
        logits = torch.scatter(logits, -1, xt.unsqueeze(-1), # xt is (B,N), need (B,N,1)
                               torch.zeros_like(logits[..., :1])) # Scatter 0
        return logits


    def _process_sigma(self, sigma): # Process sigma for backbone input
        if sigma is None and self.parameterization == 'ar': # AR models don't use sigma
            return None
            
        current_sigma = sigma # Can be (B, N_Tokens) or (B, N_Blk) or (B,1)
        # Backbone might expect a single sigma per batch item, e.g., mean sigma if per-token varies.
        # Or, if backbone is DiT-like, it can take per-token sigma if time_conditioning is on token level.

        # This logic prepares sigma for backbones that expect (B,) or scalar sigma.
        # If BD3LM's backbone takes per-token sigma, this needs adjustment.
        # Assuming BD3LM backbone (like DiT) takes (B,) timesteps/sigmas.
        
        if current_sigma.ndim == 3 and current_sigma.shape[-1] == 1: # (B, N, 1) -> (B, N)
            current_sigma = current_sigma.squeeze(-1) 

        if self.parameterization == 'ar': # Should have been caught by sigma is None
            return None 

        if current_sigma.ndim == 2: # (B, N_Tokens) or (B, N_Blk)
            # If backbone expects (B,), take mean. This is a simplification if per-token/block sigma is important.
            sigma_for_backbone = current_sigma.mean(dim=-1) 
        elif current_sigma.ndim == 1: # (B,) - already in desired shape
            sigma_for_backbone = current_sigma
        else:
            raise ValueError(f"Unexpected sigma shape after initial processing: {current_sigma.shape}")

        if not self.time_conditioning: # If no time conditioning, backbone sees fixed (zero) sigma
            sigma_for_backbone = torch.zeros_like(sigma_for_backbone)

        assert sigma_for_backbone.ndim == 1, sigma_for_backbone.shape
        return sigma_for_backbone

    def forward(self, x, sigma, sample_mode=False, store_kv=False): # x is token_ids
        # Process sigma for the backbone.
        # For BD3LM with adaptive schedules, sigma can be (B, N_Tokens_in_block) if processing one block.
        # Or (B, N_Total_Tokens) if processing full sequence (training).
        # _process_sigma will average it to (B,) if backbone expects that.
        # If backbone handles per-token sigma, _process_sigma needs to be by-passed or adapted.
        # For now, assume _process_sigma is okay and backbone gets (B,).
        
        # If parameterization is 'bd3lm', sigma might be per-token from warped schedule.
        # The self.backbone for bd3lm (e.g. DiT) expects `timesteps` which is (B,).
        # So, if `sigma` is (B, N), it needs to be averaged for such backbones.
        
        sigma_processed_for_backbone = self._process_sigma(sigma) # Averages if sigma is (B,N)
        
        # Determine input argument name based on backbone type
        is_hf_bd3lm_backbone = self.config.algo.name == 'bd3lm' and self.config.algo.backbone == 'hf_dit'
        is_custom_bd3lm_backbone = self.config.algo.name == 'bd3lm' and self.config.algo.backbone != 'hf_dit'
        
        backbone_input_arg_name = 'input_ids' if self.config.algo.backbone == 'hf_dit' else 'indices'
        backbone_input = {backbone_input_arg_name: x}

        # Autocast for mixed precision
        with torch.amp.autocast('cuda', dtype=torch.bfloat16): # Assuming 'cuda' and 'bfloat16'
          if is_hf_bd3lm_backbone:
            # HF DiT-like model for BD3LM
            output = self.backbone(**backbone_input, timesteps=sigma_processed_for_backbone, # Pass (B,) sigma
                                   store_kv=store_kv,
                                   sample_mode=sample_mode)
            logits = output.logits if hasattr(output, 'logits') else output # Handle HF model output
          elif is_custom_bd3lm_backbone: # Custom DIT/DiMamba for BD3LM
            logits = self.backbone(**backbone_input, sigma=sigma_processed_for_backbone, # Pass (B,) sigma
                                   store_kv=store_kv,
                                   sample_mode=sample_mode)
          elif self.config.algo.name == 'ar': # Autoregressive model
            if self.config.algo.backbone == 'hf_dit': # Using an HF model (e.g. GPT2) as AR backbone
              output = self.backbone(**backbone_input, timesteps=None) # No time conditioning for AR
              logits = output.logits if hasattr(output, 'logits') else output
            else: # Custom AR backbone
              logits = self.backbone(x, None, sample_mode=sample_mode, store_kv=store_kv) # x is indices
            if self.mask_index < logits.shape[-1]:
                logits[:, :, self.mask_index] = self.neg_infinity # Prevent generating mask
            logits = logits.log_softmax(-1) # AR models usually output log_probs
          else: # Other diffusion models (MDLM, SEDD type)
            # These typically use a DiT-like backbone that takes (B,) sigma
            logits = self.backbone(**backbone_input, sigma=sigma_processed_for_backbone) 

        # Handle cross-attention for parameterization (SUBS, SEDD) if x was concatenated
        # This part is subtle. If self.cross_attn is True, 'x' was cat(xt, x0_context_for_attn).
        # The parameterization (_subs_parameterization, _sedd_parameterization) needs 'xt'.
        x_orig_for_param = x # By default, parameterization uses the full input 'x' as 'xt'
        
        if self.cross_attn :
            # This assumes 'x' was torch.cat((xt_actual, context_tokens), dim=1)
            # And backbone processed this, but logits correspond to xt_actual part.
            # This logic is for models where backbone output matches only the `xt` part of a concatenated input.
            # Not standard for DiT backbones unless they internally segment.
            if not (self.config.algo.backbone == 'hf_dit'): # Original condition
                x_orig_len = x.shape[1] // 2 # Assumes xt and context are same length
                x_orig_for_param = x[:, :x_orig_len] # This is the `xt` part for parameterization
                if logits.shape[1] == x.shape[1] : # If logits are for full cat(xt,ctx)
                     logits = logits[:, :x_orig_len, :] # Take logits for `xt` part only


        # Apply parameterization based on model type
        if self.parameterization == 'subs':
          return self._subs_parameterization(logits=logits, xt=x_orig_for_param)
        elif self.parameterization == 'sedd':
          # SEDD parameterization needs sigma corresponding to x_orig_for_param
          # If sigma was (B, N_total) and x_orig_for_param is first half, then sigma needs slicing too.
          # But _process_sigma already made it (B,). So this sigma_processed_for_backbone is fine for SEDD.
          return self._sedd_parameterization(logits=logits, xt=x_orig_for_param, sigma_batch=sigma_processed_for_backbone) 
        
        # For 'bd3lm' or 'ar', logits are returned directly (or after log_softmax for 'ar')
        return logits 


    def on_train_epoch_start(self):
        self.backbone.train()
        self.meta_controller.train() # Ensure controller is in train mode
        self.noise.train() # Base noise schedule (if it had params)
        self.metrics.reset()
        # Check metric state
        main_metric_collection = self.metrics.train_nlls
        if hasattr(main_metric_collection, 'nll') and hasattr(main_metric_collection.nll, 'mean_value'):
             assert main_metric_collection.nll.mean_value == 0
             assert main_metric_collection.nll.weight == 0
        elif hasattr(main_metric_collection, '_forward_cache') and main_metric_collection._forward_cache is not None:
             main_metric_collection.reset() # Reset if using torchmetrics cache


    def training_step(self, batch, batch_idx):
        del batch_idx # Unused
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'])
        self.metrics.train_nlls.update(losses.nlls.detach(), losses.token_mask.detach())
        self.log(name='trainer/loss', # Main loss for optimizer
                 value=losses.loss.item(), # .item() for scalar tensor
                 on_step=True,
                 on_epoch=False, # Typically log step loss
                 sync_dist=True)
        return losses.loss

    def on_validation_epoch_start(self):
        self.metrics.reset()
        if self.ema: # If EMA is enabled
          self.ema.store(self._get_parameters()) # Store original weights
          self.ema.copy_to(self._get_parameters()) # Copy EMA weights to model
        self.eval() # Set LightningModule to eval mode (affects dropout, batchnorm etc. in self and submodules)
        # Explicitly set submodules to eval if not covered by self.eval() or if behavior differs
        self.backbone.eval()
        self.meta_controller.eval() 
        self.noise.eval() # Base noise schedule
        
        main_val_metric_collection = self.metrics.valid_nlls
        if hasattr(main_val_metric_collection, 'nll') and hasattr(main_val_metric_collection.nll, 'mean_value'):
            assert main_val_metric_collection.nll.mean_value == 0
            assert main_val_metric_collection.nll.weight == 0
        elif hasattr(main_val_metric_collection, '_forward_cache') and main_val_metric_collection._forward_cache is not None:
            main_val_metric_collection.reset()

        # Set sampling_eps for validation based on config (can be fixed or a range for var_min search)
        if isinstance(self.config.training.sampling_eps, (tuple, ListConfig)): # Range like [0.001, 1.0]
            self.sampling_eps_min_val_current = torch.tensor(self.config.training.sampling_eps[0], device=self.device, dtype=torch.float32)
            self.sampling_eps_max_val_current = torch.tensor(self.config.training.sampling_eps[1], device=self.device, dtype=torch.float32)
        else: # Scalar, implies sampling_eps_min, max defaults to 1.0
            self.sampling_eps_min_val_current = torch.tensor(self.config.training.sampling_eps, device=self.device, dtype=torch.float32)
            self.sampling_eps_max_val_current = torch.tensor(1.0, device=self.device, dtype=torch.float32)


    def on_validation_epoch_end(self):
        # Log all computed validation metrics
        for k, v_metric in self.metrics.valid_nlls.items(): # Assuming valid_nlls is a dict of Metric objects
          if hasattr(v_metric, 'compute'): # Check if it's a TorchMetric object
              computed_val = v_metric.compute()
              if torch.is_tensor(computed_val) and computed_val.numel() == 1: # Scalar tensor
                self.log(name=k, value=computed_val.item(), on_step=False,
                         on_epoch=True, sync_dist=True)
              elif isinstance(computed_val, float): # Python float
                 self.log(name=k, value=computed_val, on_step=False,
                         on_epoch=True, sync_dist=True)
              # Else: might be a non-scalar tensor or other type, handle as needed or skip logging
          elif isinstance(v_metric, torch.Tensor) and v_metric.numel() == 1: # Already a scalar tensor
               self.log(name=k, value=v_metric.item(), on_step=False,
                       on_epoch=True, sync_dist=True)
               # Else: might be a non-scalar tensor, handle or skip

        if self.ema: # If EMA was used
          self.ema.restore(self._get_parameters()) # Restore original weights
        if self.var_min and hasattr(self, 'trainer') and self.trainer is not None and not self.trainer.sanity_checking:
          self._clipped_schedule_search() # Perform search if var_min is active
          self.log('sampling_eps_min', # Log the chosen min/max eps
                   self.sampling_eps_min, # This should be a scalar tensor or float
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
          self.log('sampling_eps_max',
                   self.sampling_eps_max, # This should be a scalar tensor or float
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)

    def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
        # Helper to check if current (eps_min, eps_max) matches ELBO or NLL eval conditions
        s_eps_min_f = sampling_eps_min.item() if isinstance(sampling_eps_min, torch.Tensor) else sampling_eps_min
        s_eps_max_f = sampling_eps_max.item() if isinstance(sampling_eps_max, torch.Tensor) else sampling_eps_max
            
        is_elbo_range = abs(s_eps_min_f - 1e-3) < 1e-6 and abs(s_eps_max_f - 1.0) < 1e-6 # Standard ELBO
        
        eval_nll_configured = getattr(self.config.training, 'eval_nll', False)

        is_nll_bs1_eval_config = eval_nll_configured # General flag for NLL eval
        # Specific condition for block_size=1 NLL (t chosen near 1.0)
        is_nll_bs1_case = self.block_size == 1 and is_nll_bs1_eval_config

        if is_elbo_range and not is_nll_bs1_case: # If it's ELBO range AND not the special bs1 NLL case
          return True # This interval should be logged as the main 'val/nll' or 'val/elbo'
        
        # Condition for NLL evaluation when block_size is 1 (t -> 1)
        is_nll_bs1_condition_strict = self.block_size == 1 and s_eps_min_f >= (1.0 - 1e-6) # Max set to 1.0 or slightly less
        if is_nll_bs1_condition_strict and is_nll_bs1_eval_config: # If bs1 NLL eval is active
            return True # This interval should be logged as main 'val/nll'
            
        return False # Not the main interval for standard logging under current config


    def validation_step(self, batch, batch_idx):
        # Determine sampling_eps for this validation step
        current_sampling_eps_min_for_val = self.sampling_eps_min_val_current
        current_sampling_eps_max_for_val = self.sampling_eps_max_val_current
        loss_to_return = None # Placeholder for the loss value PL expects

        if self.var_min: # If variance minimization / schedule search is active
          main_interval_processed = False
          # Iterate through predefined (eps_min, eps_max) intervals for variance calculation
          for noise_clip_start_tuple, collected_nlls_for_var in self.metrics.valid_vars.items(): 
            sampling_eps_min_tensor = torch.tensor(noise_clip_start_tuple[0], device=self.device, dtype=torch.float32)
            sampling_eps_max_tensor = torch.tensor(noise_clip_start_tuple[1], device=self.device, dtype=torch.float32)

            # Check if this interval is the one for standard val loss logging (e.g. ELBO or NLL)
            is_main_interval_for_logging = self._check_val_sampling_intvl(sampling_eps_min_tensor, sampling_eps_max_tensor)
            needs_collection_for_var = len(collected_nlls_for_var) < 100 # Collect up to 100 samples for variance

            if is_main_interval_for_logging or needs_collection_for_var:
                losses_clip = self._loss(batch['input_ids'],
                                  batch['attention_mask'],
                                  sampling_eps_min=sampling_eps_min_tensor,
                                  sampling_eps_max=sampling_eps_max_tensor)
                if is_main_interval_for_logging: # If this is the primary interval for val_loss
                    self.metrics.valid_nlls.update(losses_clip.nlls.detach(), losses_clip.token_mask.detach())
                    loss_to_return = losses_clip.loss # This loss will be logged by PL as 'val_loss'
                    main_interval_processed = True
                
                if needs_collection_for_var: # Collect NLLs for variance calculation
                    nlls_for_var_calc = losses_clip.nlls # (B, N)
                    # Average NLL over block if block_size > 0, else use per-token
                    if nlls_for_var_calc.numel() > 0 and self.block_size > 0: 
                        # Reshape to (B, NumBlocks, BlockSize) then mean over BlockSize
                        nlls_reshaped_for_var = nlls_for_var_calc.view(nlls_for_var_calc.shape[0], -1, self.block_size).mean(-1)
                        collected_nlls_for_var.append(nlls_reshaped_for_var.detach().cpu()) # Store on CPU
                    elif nlls_for_var_calc.numel() > 0: # block_size is 0 or not applicable
                        collected_nlls_for_var.append(nlls_for_var_calc.detach().cpu())

            
          # If the main logging interval wasn't processed in the loop (e.g., not in valid_vars keys)
          if not main_interval_processed and loss_to_return is None:
              losses_main = self._loss(batch['input_ids'], batch['attention_mask'],
                                          sampling_eps_min=current_sampling_eps_min_for_val, # Default val eps
                                          sampling_eps_max=current_sampling_eps_max_for_val)
              self.metrics.valid_nlls.update(losses_main.nlls.detach(), losses_main.token_mask.detach())
              loss_to_return = losses_main.loss


        elif self.block_size == 1 and getattr(self.config.training, 'eval_nll', False) : # NLL eval for AR-like models
          losses = self._loss(batch['input_ids'],
                              batch['attention_mask'],
                              sampling_eps_min=torch.tensor(1.0, device=self.device, dtype=torch.float32), # t_eval near 1
                              sampling_eps_max=torch.tensor(1.0, device=self.device, dtype=torch.float32))
          self.metrics.valid_nlls.update(losses.nlls.detach(), losses.token_mask.detach())
          loss_to_return = losses.loss
        else: # Standard validation (e.g., ELBO)
          losses = self._loss(batch['input_ids'],
                              batch['attention_mask'],
                              sampling_eps_min=current_sampling_eps_min_for_val,
                              sampling_eps_max=current_sampling_eps_max_for_val)
          self.metrics.valid_nlls.update(losses.nlls.detach(), losses.token_mask.detach())
          loss_to_return = losses.loss
        
        # Fallback if loss_to_return is still None (should not happen with above logic)
        if loss_to_return is None: 
            losses_fallback = self._loss(batch['input_ids'], batch['attention_mask'],
                                     sampling_eps_min=current_sampling_eps_min_for_val,
                                     sampling_eps_max=current_sampling_eps_max_for_val)
            self.metrics.valid_nlls.update(losses_fallback.nlls.detach(), losses_fallback.token_mask.detach())
            loss_to_return = losses_fallback.loss


        return loss_to_return # PL will log this as 'val_loss'

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
          self._get_parameters(), # Gets parameters based on current training stage
          lr=self.config.optim.lr,
          betas=(self.config.optim.beta1,
                 self.config.optim.beta2),
          eps=self.config.optim.eps,
          weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate( # Uses hydra to build scheduler from config
          self.config.lr_scheduler, optimizer=optimizer)
        
        # Metric to monitor for ReduceLROnPlateau or similar schedulers
        monitor_metric = 'val/loss' # Default monitor
        if self.parameterization == 'bd3lm': # Specific metric for BD3LM
            monitor_metric = 'val/nll' # Assuming 'val/nll' is primary NLL/ELBO metric
        elif self.block_size == 1 and getattr(self.config.training, 'eval_nll', False): # AR NLL case
            monitor_metric = 'val/nll'


        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step', # Scheduler steps every training step
                          'monitor': monitor_metric, 
                          'name': 'trainer/lr'} # Name for logging LR

        return [optimizer], [scheduler_dict]

    def _resample_q_xt(self, x_blocks, xt_blocks_initial,
                         block_size_resample, sampling_eps_min_val, sampling_eps_max_val):
        # Resamples xt to ensure mask rate is within [sampling_eps_min, sampling_eps_max] per block
        B, N_Blk, BlkS = x_blocks.shape
        xt_blocks = xt_blocks_initial.clone() 

        max_resample_iters = 10 # Max iterations to try and fix mask rate
        for _iter in range(max_resample_iters):
            current_mask_count = (xt_blocks == self.mask_index).long().sum(dim=-1) # (B, N_Blk)
            current_mask_rate = current_mask_count.float() / BlkS

            # Target number of masks (float, to be ceiled/floored)
            target_min_masks_float = sampling_eps_min_val * BlkS
            target_max_masks_float = sampling_eps_max_val * BlkS

            # Check if min_bound is active (not effectively zero)
            active_min_bound = abs(sampling_eps_min_val - 1e-3) > 1e-6 # Check if not default near-zero min
            # Blocks with too few masks
            too_few_masks_indices = (current_mask_rate < sampling_eps_min_val - 1e-6) & active_min_bound # (B, N_Blk) bool

            if too_few_masks_indices.any():
                for b_idx in range(B):
                    for blk_jdx in range(N_Blk):
                        if too_few_masks_indices[b_idx, blk_jdx]:
                            num_to_add = torch.ceil(torch.tensor(target_min_masks_float, device=current_mask_count.device)).long() - current_mask_count[b_idx, blk_jdx]
                            if num_to_add.item() <= 0: continue

                            non_masked_indices_in_block = (xt_blocks[b_idx, blk_jdx] != self.mask_index).nonzero(as_tuple=False).squeeze(-1)
                            num_available_to_mask = non_masked_indices_in_block.numel()
                            
                            actual_num_to_add = min(num_to_add.item(), num_available_to_mask)
                            if actual_num_to_add > 0:
                                indices_to_mask = non_masked_indices_in_block[torch.randperm(num_available_to_mask, device=self.device)[:actual_num_to_add]]
                                xt_blocks[b_idx, blk_jdx, indices_to_mask] = self.mask_index
            
            # Re-calculate mask count/rate after adding masks
            current_mask_count = (xt_blocks == self.mask_index).long().sum(dim=-1)
            current_mask_rate = current_mask_count.float() / BlkS

            # Check if max_bound is active (not effectively one)
            active_max_bound = abs(sampling_eps_max_val - 1.0) > 1e-6 # Check if not default near-one max
            # Blocks with too many masks
            too_many_masks_indices = (current_mask_rate > sampling_eps_max_val + 1e-6) & active_max_bound # (B, N_Blk) bool

            if too_many_masks_indices.any():
                for b_idx in range(B):
                    for blk_jdx in range(N_Blk):
                        if too_many_masks_indices[b_idx, blk_jdx]:
                            num_to_remove = current_mask_count[b_idx, blk_jdx] - torch.floor(torch.tensor(target_max_masks_float, device=current_mask_count.device)).long()
                            if num_to_remove.item() <= 0: continue
                            
                            masked_indices_in_block = (xt_blocks[b_idx, blk_jdx] == self.mask_index).nonzero(as_tuple=False).squeeze(-1)
                            num_available_to_unmask = masked_indices_in_block.numel()

                            actual_num_to_remove = min(num_to_remove.item(), num_available_to_unmask)
                            if actual_num_to_remove > 0:
                                indices_to_unmask = masked_indices_in_block[torch.randperm(num_available_to_unmask, device=self.device)[:actual_num_to_remove]]
                                xt_blocks[b_idx, blk_jdx, indices_to_unmask] = x_blocks[b_idx, blk_jdx, indices_to_unmask] # Restore original token
            
            # Final check if all blocks satisfy constraints
            final_mask_rate = (xt_blocks == self.mask_index).float().sum(dim=-1) / BlkS
            
            violations = torch.zeros_like(final_mask_rate, dtype=torch.bool)
            if active_min_bound:
                violations |= (final_mask_rate < sampling_eps_min_val - 1e-6)
            if active_max_bound:
                violations |= (final_mask_rate > sampling_eps_max_val + 1e-6)
                
            if not violations.any(): # If no blocks violate, resample successful
                break 
            if _iter == max_resample_iters - 1 and violations.any():
                # print(f"Warning: Resample q_xt failed to meet mask rate after {_iter+1} iterations.")
                pass # Continue with current xt_blocks
        return xt_blocks


    def q_xt(self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
        # Sample x_t from x_0 using probability p (move_chance)
        # p can be (B,1) or (B,N_Tokens) for per-token noising
        block_size_resample = self.block_size if block_size is None else block_size
        if block_size_resample == 0 : # Handle cases where block_size might be 0 (e.g. AR mode)
            block_size_resample = x.shape[1] if x.shape[1] > 0 else 1 # Use full seq len or 1

        # Ensure p is broadcastable to x's shape (B, N_Tokens)
        if p.ndim == 3 and p.shape[-1] == 1: # (B, N_Blk_or_N_Tok, 1) -> (B, N_Blk_or_N_Tok)
            p_broadcastable = p.squeeze(-1) 
        elif p.ndim == 2 and p.shape[1] == x.shape[1]: # (B, N_Tokens) - already fine
            p_broadcastable = p
        elif p.ndim == 2 and p.shape[1] == 1: # (B, 1) -> expand to (B, N_Tokens)
            p_broadcastable = p.expand_as(x) 
        else: # Should not happen if p comes from noise schedule or warped schedule correctly
            raise ValueError(f"Unsupported shape for p: {p.shape}, x shape: {x.shape}")

        move_indices = torch.rand_like(x, dtype=torch.float32) <= p_broadcastable
        xt = torch.where(move_indices, self.mask_index, x)

        # Get current sampling_eps values (scalar floats)
        current_sampling_eps_min_tensor = sampling_eps_min if sampling_eps_min is not None else torch.tensor(1e-3, device=x.device)
        current_sampling_eps_max_tensor = sampling_eps_max if sampling_eps_max is not None else torch.tensor(1.0, device=x.device)
        
        sampling_eps_min_val = current_sampling_eps_min_tensor.item() if isinstance(current_sampling_eps_min_tensor, torch.Tensor) else current_sampling_eps_min_tensor
        sampling_eps_max_val = current_sampling_eps_max_tensor.item() if isinstance(current_sampling_eps_max_tensor, torch.Tensor) else current_sampling_eps_max_tensor

        # If target mask rate is full (NLL case for AR-like), return all masks
        if sampling_eps_min_val >= (1.0 - 1e-6): # Effectively, min_mask_rate = 1.0
            return torch.full_like(x, self.mask_index)

        # Check if it's the standard ELBO range where resampling might not be needed or is disabled
        is_standard_elbo_range = abs(sampling_eps_min_val - 1e-3) < 1e-6 and abs(sampling_eps_max_val - 1.0) < 1e-6
        
        if self.config.training.resample and not is_standard_elbo_range:
            # Resample only if resample is enabled and not in standard ELBO range (where it might be vacuous)
            if x.shape[1] % block_size_resample != 0 or x.shape[1]==0: # Cannot reshape into blocks
                 # print(f"Warning: Cannot resample q_xt, N_Tokens {x.shape[1]} not divisible by block_size_resample {block_size_resample}")
                 pass # Skip resampling
            else:
                # Reshape to (B, NumBlocks, BlockSize)
                xt_blocks_view = xt.view(xt.shape[0], -1, block_size_resample)
                x_blocks_view = x.view_as(xt_blocks_view) # Original tokens in block view
                
                xt_blocks_resampled = self._resample_q_xt( 
                    x_blocks_view, xt_blocks_view, 
                    block_size_resample,
                    sampling_eps_min_val, # Pass scalar floats
                    sampling_eps_max_val
                )
                xt = xt_blocks_resampled.reshape(xt.shape[0], -1) # Reshape back to (B, N_Tokens)
        return xt
    
    @torch.no_grad()
    def _sample_prior(self, *batch_dims): # e.g. (B, N_Tokens)
        # Returns a tensor of mask_index tokens
        return self.mask_index * torch.ones(
          *batch_dims, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def _nucleus_sample(self, p_x0_probs): # p_x0_probs is (B, N, V)
        p_nucleus = self.config.sampling.nucleus_p
        if abs(p_nucleus - 1.0) < 1e-6: # If p_nucleus is 1, no change
          return p_x0_probs
        
        # For BD3LM, nucleus sampling applies to the current block being generated
        is_full_sequence_sampling_context = p_x0_probs.shape[1] > self.block_size and self.block_size > 0
        
        if is_full_sequence_sampling_context: # If sampling full sequence, target last block
            p_x0_target_block_probs = p_x0_probs[:, -self.block_size:].clone()
        else: # If sampling just one block (or shorter than block_size)
            p_x0_target_block_probs = p_x0_probs.clone()

        sorted_probs, sorted_indices = p_x0_target_block_probs.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        
        # Create mask for tokens outside the nucleus
        remove_mask = cum_probs > p_nucleus
        # Shift mask to include the first token that exceeded p_nucleus, but keep at least one token
        remove_mask_shifted = torch.zeros_like(remove_mask)
        if remove_mask.shape[-1] > 1: # Ensure there's a dimension to shift into
            remove_mask_shifted[..., 1:] = remove_mask[..., :-1]
        
        final_keep_mask = ~remove_mask_shifted # Tokens to keep
        final_keep_mask[..., 0] = True # Always keep the most probable token

        # Apply mask and renormalize
        probs_in_nucleus_sorted_order = sorted_probs * final_keep_mask
        
        probs_after_nucleus_sorted = probs_in_nucleus_sorted_order / probs_in_nucleus_sorted_order.sum(-1, keepdim=True).clamp(min=1e-9)
        
        # Scatter back to original order
        probs_after_nucleus = torch.zeros_like(p_x0_target_block_probs)
        probs_after_nucleus.scatter_(-1, sorted_indices, probs_after_nucleus_sorted)
        
        if is_full_sequence_sampling_context: # If modified only last block, put it back
            p_x0_probs_out = p_x0_probs.clone()
            p_x0_probs_out[:, -self.block_size:] = probs_after_nucleus
            return p_x0_probs_out
        else: # If operated on the whole input (e.g. single block sampling)
            return probs_after_nucleus

    @torch.no_grad()
    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        # This is the ORIGINAL DDPM update based on GLOBAL noise schedule.
        # For ADAPTIVE schedule, see the logic integrated into _semi_ar_sampler.
        # This method is kept for non-adaptive samplers or reference.
        
        # t, dt are based on global schedule time
        _, move_chance_t = self.noise(t) # Global schedule: p(t)
        _, move_chance_s = self.noise(t - dt) # Global schedule: p(t-dt)
        sigma_t_global = self._sigma_from_p(move_chance_t) # Global sigma(t)
        
        # Ensure shapes are (B,1) for broadcasting with (B,N,V)
        move_chance_t_expanded = move_chance_t.view(-1,1) if move_chance_t.ndim == 1 else move_chance_t
        move_chance_s_expanded = move_chance_s.view(-1,1) if move_chance_s.ndim == 1 else move_chance_s
        
        mask_prob = move_chance_s_expanded / move_chance_t_expanded.clamp(min=1e-9) # p(t-dt)/p(t)

        current_block_len = x.shape[1] 

        if p_x0 is None: # If p_theta(x0|xt) is not cached
          # Sigma for forward pass, needs to match backbone's expectation (e.g. (B,) or (B,N))
          # If backbone takes (B,), use sigma_t_global. If (B,N), expand.
          sigma_for_fwd = sigma_t_global.unsqueeze(-1).expand(-1, current_block_len) # (B, N)
          
          model_input_for_fwd = x 
          # Handle cross_attn if applicable for this model type during sampling
          # This part is complex: if cross_attn needs x0 of current block, it's not available.
          # Assuming for sampling, if cross_attn=True, it means attending to x_accum (context)
          # which would need to be passed and handled by self.forward.
          # For simplicity here, assume self.forward takes xt and handles context internally if needed.
          
          raw_logits_or_logprobs = self.forward(model_input_for_fwd, 
                                                sigma=sigma_for_fwd, 
                                                sample_mode=True).to(torch.float64) # Use high precision
          
          # Convert model output to p(x0|xt) based on parameterization
          if self.parameterization == 'subs' or self.parameterization == 'ar': # Model outputs log_probs
              p_x0 = raw_logits_or_logprobs.exp()
          else: # Model outputs direct logits (SEDD, BD3LM)
              p_x0 = F.softmax(raw_logits_or_logprobs, dim=-1)

          p_x0 = self._nucleus_sample(p_x0) # Apply nucleus sampling

        # Standard DDPM reverse step (Equation 12 from original DDPM paper, adapted for discrete)
        # q(x_{t-1} | x_t, x_0) = Cat(x0; p_alpha(x0|xt) * (1 - p(t-dt)/p(t)))
        #                          + Cat(mask; p_alpha(mask|xt) * p(t-dt)/p(t) )
        # where p_alpha(mask|xt) is implicitly 1 if all other x0 have sum < 1 for p_alpha(x0|xt).
        # More simply: sample x0 ~ p_theta(x0|xt), then sample xt-1 ~ q(xt-1|xt, x0_sampled)
        
        # Simplified approach used in many discrete diff models:
        # Sample x_hat_0 ~ p_theta(x0|xt). Then effectively use Bayes rule for q(x_{t-1}|x_t, x_hat_0)
        # Or, directly sample from the model's predicted distribution for x_{t-1}
        # The code implements the latter with a "masking" probability trick:
        q_xs = p_x0 * (1.0 - mask_prob.unsqueeze(-1)) # Prob to keep x0_sampled
        if self.mask_index < q_xs.shape[-1]: # Vocab has mask token
            q_xs[..., self.mask_index] += mask_prob.unsqueeze(-1) # Prob to re-mask (revert to prior for x_{t-1})
        x_block_new = _sample_categorical(q_xs) 

        # First Hitting Logic (Optional, complex to adapt for warped schedules)
        if self.config.sampling.first_hitting:
          # This logic changes only one masked token per step, typically used for specific types of generation.
          # It requires careful indexing.
          x_block_sampled_for_fh = _sample_categorical(p_x0) # Sample a full x0 prediction
          
          one_change_mask = torch.zeros_like(x_block_sampled_for_fh, dtype=torch.bool)
          for b_idx in range(x.shape[0]): # Iterate over batch
              masked_positions = (x[b_idx] == self.mask_index).nonzero(as_tuple=False).squeeze(-1)
              num_masked_in_this_sample_block = masked_positions.shape[0]
              if num_masked_in_this_sample_block > 0:
                  # Choose one random masked position to update
                  chosen_idx_in_masked_list = torch.randint(0, num_masked_in_this_sample_block, (1,), device=self.device).item()
                  actual_idx_in_block = masked_positions[chosen_idx_in_masked_list]
                  one_change_mask[b_idx, actual_idx_in_block] = True
          
          x_block_new = torch.where(one_change_mask, x_block_sampled_for_fh, x) # Update only chosen pos

        # Ensure already known tokens (non-masked in x_t) are not changed
        copy_flag = (x != self.mask_index) 
        x_block_final = torch.where(copy_flag, x, x_block_new)
        
        # Update KV cache if block is fully denoised
        if self.config.sampling.kv_cache and self.mask_index not in x_block_final: 
          # Sigma for KV cache should match what backbone expects
          sigma_for_kv_cache = sigma_t_global.unsqueeze(-1).expand(-1, current_block_len)
          _ = self.forward(x_block_final, sigma_for_kv_cache, sample_mode=True, store_kv=True)

        # If x_block_final changed from x, invalidate p_x0 cache for next step
        if not torch.allclose(x_block_final, x): 
          return None, x_block_final # p_x0 becomes None, new x_t
        else: # No change, p_x0 can be reused
          return p_x0, x_block_final 

    @torch.no_grad()
    def _ar_sampler(self, bsz, context_len=1024):
        # Standard Autoregressive sampler (not using diffusion or adaptive schedules)
        if self.config.sampling.kv_cache:
          self.backbone.reset_kv_cache(eval_batch_size=bsz) 

        with torch.amp.autocast('cuda', dtype=torch.float32): # Adjust dtype if needed
          num_pred_tokens = self.num_tokens - 1 # Predict up to num_tokens total (BOS + num_pred)
          if num_pred_tokens < 0: num_pred_tokens = 0 

          x = torch.zeros( # (B, TotalLength)
            (bsz, num_pred_tokens + 1), # +1 for BOS
            dtype=torch.long,
            device=self.device)
          if x.numel() > 0 : x[:, 0] = self.tokenizer.bos_token_id # Start with BOS
          
          stop_flags = torch.zeros(bsz, dtype=torch.bool, device=self.device) # For variable length
          final_lengths = torch.full((bsz,), num_pred_tokens + 1, dtype=torch.long, device=self.device)


          for i in tqdm(range(num_pred_tokens), desc="AR Sampling"):
            if stop_flags.all(): break # All sequences finished

            active_indices = (~stop_flags).nonzero(as_tuple=True)[0] # Indices of sequences still generating
            if len(active_indices) == 0: break

            # Current context for active sequences
            x_active_context = x[active_indices, :i + 1]
            if context_len > 0 and x_active_context.shape[1] > context_len: # Truncate context if too long
                x_active_context = x_active_context[:, -context_len:]
            
            if x_active_context.shape[1] == 0: # Should not happen if BOS is there
                # Handle empty context if BOS wasn't added or seq starts empty
                # This case might need specific model handling (e.g. unconditional first token)
                # For now, assume BOS is always present.
                if x.shape[1] > 0: # If x has sequence dim
                    x_active_context = x[active_indices, :1] # Use BOS if available
                else: # Cannot form context
                    continue 
            
            # Gumbel noise for sampling (alternative to top-k/nucleus then multinomial)
            noise_active = (torch.distributions.Gumbel(0, 1)
                    .sample((len(active_indices), self.vocab_size)) # (B_active, VocabSize)
                    .to(self.device))
            
            # Get next token logits from AR model
            next_logits_active = self.forward(
              x_active_context, 
              None, # No sigma for AR
              store_kv=self.config.sampling.kv_cache,
              sample_mode=True)[:, -1:].to(torch.float64) # (B_active, 1, VocabSize), take last token's logits

            next_probs_active = next_logits_active.exp() # Convert log_probs to probs
            next_probs_active_nuclear = self._nucleus_sample(next_probs_active) # Apply nucleus
            
            # Sample next token using Gumbel-Max trick (log_probs + Gumbel noise, then argmax)
            y_active = (next_probs_active_nuclear.log().clamp(min=self.neg_infinity) + noise_active).argmax(-1) # (B_active, 1)
            
            x[active_indices, i + 1] = y_active.squeeze(-1) # Store sampled token

            # Check stop conditions (EOS, low entropy) for variable length generation
            if (i + 1) > 256: # Start checking after a certain length
                 current_x_for_stop_check = x[active_indices, :i+2] # Tokens generated so far for active seqs
                 stop_now_active_flags, x_truncated_active_list = self._check_stop_conds_ar_batch(current_x_for_stop_check)
                 
                 for idx_in_active_list, original_batch_idx_tensor in enumerate(active_indices):
                     original_batch_idx = original_batch_idx_tensor.item()
                     if stop_now_active_flags[idx_in_active_list]:
                         stop_flags[original_batch_idx] = True
                         final_lengths[original_batch_idx] = x_truncated_active_list[idx_in_active_list].shape[0]
          
          if not self.config.sampling.var_length:
              return x # Return full fixed-length sequences
          else: # Return variable-length sequences
                output_samples_list_of_tensors = []
                for b_idx_final in range(bsz):
                    output_samples_list_of_tensors.append(x[b_idx_final, :final_lengths[b_idx_final]])
                return output_samples_list_of_tensors 


    def _check_stop_conds_ar_batch(self, x_batch_active): # For AR sampler variable length
        B_active = x_batch_active.shape[0]
        stop_flags_for_active = torch.zeros(B_active, dtype=torch.bool, device=x_batch_active.device)
        truncated_x_list_for_active = list(x_batch_active.unbind(0)) # List of tensors

        for i in range(B_active):
            current_x_sample = x_batch_active[i].clone() # (SeqLen_current)
            stop_this_sample_now = False
            current_truncate_idx_for_sample = current_x_sample.shape[0] # Default to full length

            # 1. Entropy check
            check_entropy_len = min(256, current_x_sample.shape[0])
            if check_entropy_len > 0:
                entropy_val = self._compute_entropy(current_x_sample[-check_entropy_len:])
                if entropy_val < 4.0: # Threshold for low entropy (repetitive)
                    stop_this_sample_now = True
                    if self.config.sampling.var_length: # Truncate before low-entropy part
                        current_truncate_idx_for_sample = max(1, current_x_sample.shape[0] - check_entropy_len)

            # 2. EOS token check
            if self.config.sampling.var_length:
                eos_indices = (current_x_sample == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    first_valid_eos_pos = -1
                    for eos_idx_val_item in eos_indices: # Find first EOS not at BOS (unless BOS==EOS)
                        eos_idx = eos_idx_val_item.item()
                        is_bos_token_val = (current_x_sample.numel()>0 and current_x_sample[0] == self.tokenizer.bos_token_id)
                        is_eos_at_bos_val = (eos_idx == 0)
                        
                        if is_eos_at_bos_val and is_bos_token_val and (self.tokenizer.bos_token_id == self.tokenizer.eos_token_id):
                            if current_x_sample.shape[0] == 1: # BOS=EOS, and it's the only token
                                first_valid_eos_pos = 1 # Keep BOS
                                break
                            elif len(eos_indices) > 1: # BOS=EOS, look for next EOS
                                first_valid_eos_pos = eos_indices[1].item() + 1 # Truncate after next EOS
                                break
                        elif not is_eos_at_bos_val : # EOS is not at BOS, valid stop
                            first_valid_eos_pos = eos_idx + 1 # Truncate after this EOS
                            break
                        elif is_eos_at_bos_val and not (is_bos_token_val and self.tokenizer.bos_token_id == self.tokenizer.eos_token_id):
                            # EOS at pos 0, but BOS is different or not present. Treat as stop.
                            first_valid_eos_pos = 1 # Keep only EOS if it's the first token
                            break
                            
                    if first_valid_eos_pos != -1: # Valid EOS found
                        stop_this_sample_now = True
                        current_truncate_idx_for_sample = min(current_truncate_idx_for_sample, first_valid_eos_pos)
            
            if stop_this_sample_now:
                stop_flags_for_active[i] = True
                final_sample_tensor = current_x_sample[:current_truncate_idx_for_sample]
                if final_sample_tensor.numel() == 0: # Ensure not empty
                    final_sample_tensor = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long, device=x_batch_active.device)
                truncated_x_list_for_active[i] = final_sample_tensor
        
        return stop_flags_for_active, truncated_x_list_for_active


    @torch.no_grad()
    def _sample(
      self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
        # Main dispatch function for sampling
        if seqlen is None:
          seqlen = self.config.model.length
        if batch_size_per_gpu is None:
          batch_size_per_gpu = self.config.loader.eval_batch_size 
        
        samples_list_of_decoded_texts = [] # To store decoded text samples
        
        if self.parameterization == 'ar': # If model is AR, use AR sampler
          for _ in range(self.config.sampling.num_sample_batches): # Number of batches to sample
            ar_output = self._ar_sampler(batch_size_per_gpu) # Returns list of tensors or single tensor
            if isinstance(ar_output, list): # Variable length outputs list of tensors
                for s_tensor in ar_output:
                    if s_tensor is not None and s_tensor.numel() > 0:
                         samples_list_of_decoded_texts.append(self.tokenizer.decode(s_tensor.cpu().tolist(), skip_special_tokens=True))
            elif ar_output is not None and ar_output.numel() > 0 : # Fixed length outputs tensor
                samples_list_of_decoded_texts.extend(self.tokenizer.batch_decode(ar_output.cpu().tolist(), skip_special_tokens=True))
            # NFE for AR is roughly sequence length
            self.metrics.gen_nfes.append(self.config.model.length) 
          return samples_list_of_decoded_texts 
        
        # For diffusion models
        if num_steps is None: num_steps = self.T # Default to T from config

        tensor_samples_list = [] # Store raw tensor samples before decoding

        if self.sampler == 'semi_ar': # BD3LM-style block-by-block generation
          for _ in range(self.config.sampling.num_sample_batches):
            sample_i_tensor, num_tries_semi_ar = None, 0
            actual_nfes_semi_ar = 0
            while sample_i_tensor is None and num_tries_semi_ar < 10: # Retry loop
              num_tries_semi_ar += 1
              if self.block_size == 0 : num_strides = 1 # Should not happen for BD3LM
              else: num_strides = seqlen // self.block_size # Number of blocks
              
              seqlen_eff = seqlen # Effective sequence length for this batch
              if seqlen % self.block_size != 0 and self.parameterization == 'bd3lm' and self.block_size > 0:
                  seqlen_eff = num_strides * self.block_size # Align to block boundary
                  if seqlen_eff == 0 : # Handle if seqlen < block_size
                      seqlen_eff = self.block_size; num_strides = 1
              
              if num_strides == 0 and seqlen_eff > 0: num_strides = 1 # Min 1 stride if seqlen > 0

              sample_i_tensor, nfes_this_try = self._semi_ar_sampler(
                n_samples=batch_size_per_gpu,
                num_strides=num_strides, 
                num_steps=num_steps, # This is total_global_steps for dynamic NFE
                seqlen=seqlen_eff) 
              
              if sample_i_tensor is not None: actual_nfes_semi_ar = nfes_this_try
              if num_tries_semi_ar >= 10 and sample_i_tensor is None: 
                print(f'Warning: {self.sampler} Sampling failed after multiple tries. Returning empty list for this batch.')
                break 
            if sample_i_tensor is not None: 
                tensor_samples_list.append(sample_i_tensor) 
                self.metrics.nfes.update(actual_nfes_semi_ar if actual_nfes_semi_ar is not None else 0)
                self.metrics.gen_nfes.append(actual_nfes_semi_ar if actual_nfes_semi_ar is not None else 0)
        
        elif self.sampler == 'analytic': # Analytic D3PM sampler
            actual_nfes_analytic = 0
            for _ in range(self.config.sampling.num_sample_batches):
                sample_i_tensor, num_tries_analytic = None, 0
                
                while sample_i_tensor is None and num_tries_analytic < 10:
                    num_tries_analytic +=1
                    # Analytic sampler also needs to be block-sequential for adaptive schedules
                    if self.block_size == 0: num_strides_analytic = 1
                    else: num_strides_analytic = seqlen // self.block_size
                    
                    seqlen_eff_analytic = seqlen
                    if seqlen % self.block_size != 0 and self.parameterization == 'bd3lm' and self.block_size > 0:
                        seqlen_eff_analytic = num_strides_analytic * self.block_size
                        if seqlen_eff_analytic == 0: seqlen_eff_analytic = self.block_size; num_strides_analytic = 1
                    if num_strides_analytic == 0 and seqlen_eff_analytic > 0: num_strides_analytic = 1
                    
                    sample_i_tensor, nfes_this_try_analytic = self._analytic_sampler(
                        n_samples=batch_size_per_gpu,
                        num_strides=num_strides_analytic,
                        num_steps=num_steps, # total_global_steps
                        seqlen=seqlen_eff_analytic,
                        eps=eps) # eps is for final timestep in original analytic sampler
                    
                    if sample_i_tensor is not None: actual_nfes_analytic = nfes_this_try_analytic
                    if num_tries_analytic >=10 and sample_i_tensor is None:
                        print('Warning: Analytic Sampling failed. Returning empty list for this batch.')
                        break
                if sample_i_tensor is not None:
                    tensor_samples_list.append(sample_i_tensor)
                    self.metrics.nfes.update(actual_nfes_analytic if actual_nfes_analytic is not None else 0)
                    self.metrics.gen_nfes.append(actual_nfes_analytic if actual_nfes_analytic is not None else 0)
        else: 
          # Fallback or other samplers (e.g. original DDPM sampler if you had one)
          # This part would need to be implemented if other samplers are supported.
          print(f"Sampler type '{self.sampler}' not fully adapted for adaptive schedules or not recognized.")
          return [] # Return empty if sampler is not supported
        
        if not tensor_samples_list: return [] # No samples generated
        
        # Decode all collected tensor samples
        for s_batch_tensor in tensor_samples_list:
            if s_batch_tensor is None or s_batch_tensor.numel() == 0: continue
            if s_batch_tensor.ndim == 1: s_batch_tensor = s_batch_tensor.unsqueeze(0) # Ensure batch dim
            # batch_decode expects list of lists of token_ids or tensor
            samples_list_of_decoded_texts.extend(self.tokenizer.batch_decode(s_batch_tensor.cpu().tolist(), skip_special_tokens=True))
            
        return samples_list_of_decoded_texts


    def _sigma_from_p(self, p_move_chance): # p_move_chance is alpha_t in some notations, 1-alpha_bar in others. Here it's "move chance".
        # sigma = -log(1 - p_move_chance) if p_move_chance is prob of masking (alpha_t in absorbing state)
        # This depends on definition of p from noise schedule.
        # If self.noise(t) returns (loss_scale, move_chance), and move_chance is like alpha_t in absorbing state models:
        # then sigma_t = -log(alpha_t) (if alpha_t = prob of staying x0, so 1-alpha_t is move to mask)
        # Or sigma_t = -log(1-alpha_t) (if alpha_t = prob of moving to mask)
        # The current self.noise(t) returns (loss_scale, move_chance=p_t)
        # And q_xt uses move_chance to decide mask vs x0. So move_chance is prob of becoming mask.
        # So, if p_t is prob of mask, then alpha_bar_t (prob of x0) is 1-p_t.
        # In SEDD, sigma_t = -log gamma_t, where gamma_t is prob of x0. So sigma_t = -log(1-p_t)
        
        p_device = p_move_chance.device
        # Clamp p_move_chance to avoid log(0). Max value clamp depends on if p is near 1.
        # If p_move_chance is probability of *masking*, 1-p_move_chance is prob of *not masking* (staying original)
        # sigma = -log(prob_of_staying_original)
        prob_staying_original = (1.0 - p_move_chance).clamp(min=1e-9) # Add epsilon for stability near p=1
        log_val = torch.log(prob_staying_original) 
        
        # Sigma_max from base noise schedule, if defined (e.g. LogLinearNoise)
        sigma_max_val = getattr(self.noise, 'sigma_max', 10.0) # Default large sigma_max
        if isinstance(sigma_max_val, torch.Tensor):
            sigma_max_tensor = sigma_max_val.to(p_device)
        else: 
            sigma_max_tensor = torch.tensor(sigma_max_val, device=p_device, dtype=p_move_chance.dtype) 
            
        return torch.min(-log_val, sigma_max_tensor) # sigma = -log(1-p_mask), capped


    def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
        # Utility to load EMA weights and sample (for eval scripts)
        if self.ema:
          self.ema.store(self._get_parameters()) # Store current model weights
          self.ema.copy_to(self._get_parameters()) # Load EMA weights into model
        self.backbone.eval()
        self.meta_controller.eval() 
        self.noise.eval() # Base noise schedule
        
        actual_seqlen = seqlen if seqlen is not None else self.config.model.length
        
        # Call main _sample dispatcher
        samples_text = self._sample( 
          seqlen=actual_seqlen,
          batch_size_per_gpu=self.config.loader.eval_batch_size, # From config
          num_steps=num_steps, # Passed in
          eps=eps) # Passed in, for analytic sampler
        
        # Optionally compute perplexity of generated samples (if a LM is available for scoring)
        if samples_text: # If samples were generated
            self.metrics.record_generative_perplexity( # This metric needs implementation
              samples_text, 
              actual_seqlen, 
              self.config.loader.eval_batch_size, 
              self.device)
        return samples_text


    def get_score(self, x_token_ids, sigma_batch_eff): 
        # Get model's prediction p_theta(x0 | xt, sigma_eff_t)
        # sigma_batch_eff is the effective sigma for the current block at remapped time t'
        # It should be (B,1) or (B,) if backbone expects that, or (B,N) if per-token.
        
        # Ensure sigma_batch_eff is correctly shaped for self.forward's expectation for `sigma`
        # If x_token_ids is (B, N_block), sigma_batch_eff might be (B,1) or (B, N_block)
        if sigma_batch_eff.ndim == 1: # (B,) -> (B,1) then expand to (B, N_block)
            sigma_for_fwd = sigma_batch_eff.unsqueeze(-1).expand(-1, x_token_ids.shape[1])
        elif sigma_batch_eff.ndim == 2 and sigma_batch_eff.shape[1] == 1: # (B,1) -> expand
            sigma_for_fwd = sigma_batch_eff.expand(-1, x_token_ids.shape[1])
        elif sigma_batch_eff.ndim == 2 and sigma_batch_eff.shape[1] == x_token_ids.shape[1]: # (B, N_block)
            sigma_for_fwd = sigma_batch_eff
        else:
            raise ValueError(f"get_score: sigma_batch_eff shape {sigma_batch_eff.shape} incompatible with x_token_ids {x_token_ids.shape}")

        # self.forward will internally call _process_sigma if backbone expects (B,) sigma
        # If backbone handles (B,N) sigma, _process_sigma should be adapted or bypassed.
        model_output_logits = self.forward(x_token_ids, sigma_for_fwd).to(torch.float64) 
        
        # Convert logits to probabilities based on parameterization
        if self.parameterization == 'ar' or self.parameterization == 'subs': # Model outputs log_probs
            model_probs = model_output_logits.exp()
        elif self.parameterization == 'sedd': # SEDD needs softmax after parameterization adjustments
             model_probs = F.softmax(model_output_logits, dim=-1) # Assuming SEDD logits are pre-softmax
        else: # BD3LM and others output direct logits for x0
             model_probs = F.softmax(model_output_logits, dim=-1)

        # Apply nucleus sampling to the probabilities
        if abs(self.config.sampling.nucleus_p - 1.0) < 1e-6: # If nucleus_p is 1, no change
          return model_probs
        
        model_probs_nuclear = self._nucleus_sample(model_probs) 
        return model_probs_nuclear 

    def _staggered_score(self, score_probs, dsigma_batch): 
        # Apply correction for staggered D3PM sampler (Analytic D3PM)
        # score_probs: p_theta(x0 | xt, sigma_t), shape (B, N, V)
        # dsigma_batch: sigma_t - sigma_s, shape (B,1) or (B,)
        
        score_probs_clone = score_probs.clone() # Avoid in-place modification
        
        # Reshape dsigma for broadcasting: (B,1,1)
        dsigma_reshaped_for_sum = dsigma_batch.view(-1,1,1) if dsigma_batch.ndim == 1 else dsigma_batch
        dsigma_reshaped_for_mult = dsigma_reshaped_for_sum # Same shape needed

        # Equation 11 from Analytic D3PM paper, discrete version for p_hat(x0 | xs)
        # p_hat(x0 | xs) propto p_theta(x0 | xt) * exp(dsigma) for x0 != mask
        # and p_hat(mask | xs) propto p_theta(mask | xt) * exp(dsigma) + (1 - exp(dsigma)) * sum_{x0'!=mask} p_theta(x0'|xt)
        # This simplifies if p_theta(mask|xt) is assumed small or zero.
        # A common simplification: score_tilde(x0) = score(x0) * exp(dsigma) if x0 != mask
        # score_tilde(mask) = score(mask) * exp(dsigma) + (1-exp(dsigma))
        # This assumes sum score(x0) = 1.
        
        sum_probs_over_vocab = score_probs_clone.sum(dim=-1, keepdim=True) # Should be ~1.0
        # Term (1 - exp(dsigma)) * sum_{x0'} p_theta(x0'|xt) to add to mask probability
        extra_const_for_mask = (1.0 - torch.exp(dsigma_reshaped_for_sum)) * sum_probs_over_vocab 
        
        # Multiply all probabilities by exp(dsigma)
        score_probs_clone = score_probs_clone * torch.exp(dsigma_reshaped_for_mult) 
        
        # Add the extra term to the mask token's probability
        if self.mask_index < score_probs_clone.shape[-1]: # If mask_index is a valid vocab index
             score_probs_clone[..., self.mask_index] += extra_const_for_mask.squeeze(-1) # Squeeze if extra_const is (B,N,1)
        
        # Renormalize (optional, but good practice if approximations were made)
        # score_probs_clone = score_probs_clone / score_probs_clone.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return score_probs_clone

    # --- Methods for Analytic Sampler with Warped Schedules ---
    @torch.no_grad()
    def _analytic_update_warped(self, x_current_block, s_b_current_block, t_prime_k, t_prime_k_minus_1):
        # Get warped schedule values at remapped times t'_k and t'_{k-1}
        _, _, p_b_j_curr = self._get_warped_outputs_at_t(s_b_current_block, t_prime_k)
        _, _, p_b_j_prev = self._get_warped_outputs_at_t(s_b_current_block, t_prime_k_minus_1)

        # Squeeze from (B,1,1) to (B,1) if N_t=1 for t_prime_k
        sigma_t_eff = self._sigma_from_p(p_b_j_curr.squeeze(-1)) # (B,1)
        sigma_s_eff = self._sigma_from_p(p_b_j_prev.squeeze(-1)) # (B,1)
        dsigma_eff = sigma_t_eff - sigma_s_eff # (B,1)
        
        # score_probs: p_theta(x0 | xt_current_block, sigma_t_eff)
        score_probs = self.get_score(x_current_block, sigma_t_eff) # sigma_t_eff is (B,1)
        
        stag_score_probs = self._staggered_score(score_probs, dsigma_eff) # dsigma_eff is (B,1)
        
        # Transition probability q(xt | x_{t-1}=i, dsigma_eff)
        # dsigma_eff needs to be (B, N_block_len) for _transp_transition's current use of sigma_per_token
        dsigma_eff_per_token = dsigma_eff.expand(-1, x_current_block.shape[1])
        trans_probs = self._transp_transition(x_current_block, dsigma_eff_per_token) 
        
        # Combine: Equation 10 from Analytic D3PM
        # p_hat(x_{s} | x_t) = sum_{x0} q(x_s | x0, x_t) p_hat(x0 | x_t)
        # Here, p_hat(x0|xt) is stag_score_probs
        # q(xs | x0, xt) is approximated by transp_transition based on x0_sampled implicitly from stag_score_probs
        # A common implementation directly samples x0_hat ~ stag_score_probs, then x_s ~ transp_transition(x0_hat, dsigma)
        # Or, uses a mixture form: probs_final = stag_score_probs * trans_probs (element-wise product, then sample)
        # The element-wise product implies p(x_s=i | x_t) = p_hat(x0=i | x_t) * q(x_s=i | x0=i, dsigma) which is an approx.
        
        # Assuming the product form common in some codebases:
        probs_final = stag_score_probs * trans_probs # (B,N,V) * (B,N,V)
        return _sample_categorical(probs_final)

    @torch.no_grad()
    def _denoiser_update_warped(self, x_current_block, s_b_current_block, t_prime_final):
        # Get warped schedule values at the final remapped time t'_final
        _, _, p_b_j_final = self._get_warped_outputs_at_t(s_b_current_block, t_prime_final)
        
        sigma_eff_final = self._sigma_from_p(p_b_j_final.squeeze(-1)) # (B,1)
        
        score_probs = self.get_score(x_current_block, sigma_eff_final)
        # For denoiser, "dsigma" in staggered_score is sigma_eff_final itself
        stag_score_probs = self._staggered_score(score_probs, sigma_eff_final) 
        
        sigma_eff_final_per_token = sigma_eff_final.expand(-1, x_current_block.shape[1])
        trans_probs = self._transp_transition(x_current_block, sigma_eff_final_per_token)
        
        probs_final = stag_score_probs * trans_probs
        if self.mask_index < probs_final.shape[-1]: # Don't sample mask at final step
            probs_final[..., self.mask_index] = 0.0 
            # Renormalize if mask prob was non-zero and large
            # probs_final = probs_final / probs_final.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        samples = _sample_categorical(probs_final)
        return samples
    # --- End of Analytic Sampler Warped Methods ---


    def _transp_transition(self, i_tokens, sigma_per_token): # q(x_s | x_0=i, effective_sigma)
        # sigma_per_token is the effective sigma (e.g. dsigma_eff or sigma_eff_final) for this transition
        # Shape (B, N_Tokens)
        sigma_expanded = sigma_per_token.unsqueeze(-1) # (B, N, 1) for broadcasting with vocab
        
        # Create one-hot for i_tokens (x0 candidates)
        one_hot_i = F.one_hot(i_tokens, num_classes=self.vocab_size).type_as(sigma_expanded) # (B,N,V)
        
        # Transition prob: if x_s == i, prob is exp(-sigma_eff). Otherwise, (1-exp(-sigma_eff))/(V-1)
        # Simplified: diagonal is exp(-sigma), off-diagonal part for mask handling
        # This is q(x_s | x_0) under absorbing state assumption, where x_s can be x_0 or MASK.
        # Edge case: if x_0 is MASK, then x_s is MASK.
        
        # prob(x_s = i | x_0 = i) = exp(-sigma_eff)
        # prob(x_s = MASK | x_0 = i) = 1 - exp(-sigma_eff)
        # prob(x_s = k | x_0 = i) = 0 for k != i, k != MASK
        
        # diagonal term: exp(-sigma_eff) * one_hot(x0)
        edge_diag = torch.exp(-sigma_expanded) * one_hot_i # (B,N,V), only non-zero at x0=i
        
        # off-diagonal term for MASK: (1 - exp(-sigma_eff)) to be placed at MASK column
        # This should only be added if x0 != MASK. If x0 == MASK, then q(xs=MASK|x0=MASK)=1.
        
        prob_to_mask = (1.0 - torch.exp(-sigma_expanded)) # (B,N,1)
        
        # Construct the transition matrix probabilities for each token position
        # Final q(x_s | x_0=i_tokens[n]) will have:
        #   exp(-sigma) at i_tokens[n] column
        #   1-exp(-sigma) at MASK column (if i_tokens[n] != MASK)
        #   0 elsewhere
        # If i_tokens[n] == MASK:
        #   1 at MASK column
        #   0 elsewhere
        
        final_probs = torch.zeros_like(edge_diag) # (B,N,V)
        
        # Populate diagonal (prob of staying original token)
        # Scatter exp(-sigma) to columns indexed by i_tokens
        final_probs.scatter_add_(-1, i_tokens.unsqueeze(-1), torch.exp(-sigma_expanded)) # Careful with multiple i_tokens being same

        # Clear and rebuild using a safer gather/scatter approach for clarity
        final_probs.zero_()
        current_token_indices = i_tokens.unsqueeze(-1) # (B,N,1)
        prob_stay_original_expanded = torch.exp(-sigma_expanded) # (B,N,1)
        final_probs.scatter_(-1, current_token_indices, prob_stay_original_expanded)


        # Add probability for transitioning to MASK (if original token was not MASK)
        if self.mask_index < self.vocab_size: # Ensure MASK is in vocab
            mask_col_probs = prob_to_mask.expand(-1, -1, self.vocab_size)[..., self.mask_index:self.mask_index+1] # (B,N,1)
            # Need to ensure this is only added if current_token_indices != self.mask_index
            
            # Create a mask for where i_tokens is NOT the MASK token
            not_mask_token_condition = (i_tokens != self.mask_index).unsqueeze(-1) # (B,N,1)
            
            # Add prob_to_mask to the MASK column, only where original token was not MASK
            final_probs[..., self.mask_index:self.mask_index+1] += (prob_to_mask * not_mask_token_condition)

            # If original token *was* MASK, prob of staying MASK is 1
            is_mask_token_condition = (i_tokens == self.mask_index) # (B,N)
            final_probs[is_mask_token_condition, :] = 0.0 # Zero out row if original was MASK
            final_probs[is_mask_token_condition, self.mask_index] = 1.0 # Set P(MASK|x0=MASK)=1

        # Renormalize just in case, though analytically it should sum to 1 (or close)
        # final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return final_probs


    def _sample_t(self, batch_dims_for_blocks, device, sampling_eps_min, sampling_eps_max, block_size=None): 
        # Samples t ~ U[eps_min, eps_max] for each block (or per sample if not block-based)
        # batch_dims_for_blocks typically (B, N_Blk)
        _eps_b = torch.rand(batch_dims_for_blocks, device=device, dtype=torch.float32) 

        if self.antithetic_sampling: # Stratified sampling for t
            num_total_block_samples = batch_dims_for_blocks[0] * batch_dims_for_blocks[1] # B * N_Blk
            if num_total_block_samples > 0: 
                # Create strata offsets
                offset_b = torch.arange(num_total_block_samples, device=device, dtype=torch.float32) / num_total_block_samples
                offset_b = offset_b.view(batch_dims_for_blocks[0], batch_dims_for_blocks[1]) # Reshape to (B, N_Blk)
                _eps_b = (_eps_b / num_total_block_samples + offset_b) % 1.0 # Add jitter within strata
        
        t_per_block_uniform_0_1 = _eps_b # Now in [0,1)

        # Convert scalar eps_min/max to tensors if they aren't already
        s_eps_min_tensor = sampling_eps_min.to(device=device, dtype=torch.float32) if isinstance(sampling_eps_min, torch.Tensor) else torch.tensor(sampling_eps_min, device=device, dtype=torch.float32)
        s_eps_max_tensor = sampling_eps_max.to(device=device, dtype=torch.float32) if isinstance(sampling_eps_max, torch.Tensor) else torch.tensor(sampling_eps_max, device=device, dtype=torch.float32)

        # Handle NLL evaluation case (t -> 1.0)
        s_eps_max_val = s_eps_max_tensor.item()
        s_eps_min_val = s_eps_min_tensor.item()
        
        # block_size here refers to context for NLL (e.g. 1 for AR)
        is_nll_case_for_block_level_t = (block_size == 1 and s_eps_max_val >= (1.0-1e-6) and s_eps_min_val >= (1.0-1e-6))
        if is_nll_case_for_block_level_t :
            return torch.ones_like(t_per_block_uniform_0_1, dtype=torch.float32) # t = 1.0

        # Scale t from [0,1) to [eps_min, eps_max)
        t_per_block_scaled = t_per_block_uniform_0_1 * (s_eps_max_tensor - s_eps_min_tensor) + s_eps_min_tensor
        return t_per_block_scaled.clamp(min=0.0, max=1.0) # Ensure t is within [0,1]


    def _maybe_sub_sample(self, x0, attention_mask):
        # Subsamples a portion of x0 if x0 is longer than model's num_tokens
        seqlen = x0.shape[1]
        if seqlen == 0 : 
            return x0, None, attention_mask # Nothing to subsample

        if seqlen > self.num_tokens and self.num_tokens > 0 : # If x0 is too long
          start_max = seqlen - self.num_tokens
          start = np.random.randint(0, start_max + 1) if start_max >=0 else 0 # Random start
          
          end = start + self.num_tokens
          # Ensure end does not exceed actual sequence length (shouldn't if start_max is correct)
          if end > seqlen: 
              end = seqlen
              start = max(0, end - self.num_tokens) # Adjust start if end was clipped


          input_tokens = x0[:, start: end]
          # For AR, output_tokens are shifted input_tokens
          output_tokens = x0[:, start + 1: end + 1] if self.parameterization == 'ar' and (end+1 <= seqlen) else None
          new_attention_mask = attention_mask[:, start: end]

          # Special token handling (BOS/EOS) if configured (more for AR models)
          if hasattr(self.config.data, 'insert_train_special') and self.config.data.insert_train_special == True:
            if input_tokens.numel() > 0 and input_tokens.shape[1]>0 : input_tokens[:, 0] = self.tokenizer.bos_token_id
            if output_tokens is not None and output_tokens.numel() > 0 and output_tokens.shape[1]>0: output_tokens[:, -1] = self.tokenizer.eos_token_id
        
        elif self.parameterization == 'ar': # If AR model and x0 is not longer (or exactly num_tokens)
          if seqlen <= 1: # Very short sequence, cannot form input/output pair well
              input_tokens = x0 
              output_tokens = None # No target if input is 1 token
              new_attention_mask = attention_mask[:, :seqlen] 
          else: # Standard AR: input is x0[:-1], target is x0[1:]
              input_tokens = x0[:, :-1]
              output_tokens = x0[:, 1:]
              new_attention_mask = attention_mask[:, :-1] # Mask matches input_tokens
        else: # Not AR, and not longer than num_tokens. Use full x0.
          input_tokens = x0
          output_tokens = None # No shifted target for non-AR
          new_attention_mask = attention_mask

        return input_tokens, output_tokens, new_attention_mask


    def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
        # Determine sampling_eps_min and sampling_eps_max for this loss calculation
        if sampling_eps_min is None: # Not passed explicitly, use training defaults
            if hasattr(self, 'sampling_eps_min') and isinstance(self.sampling_eps_min, torch.Tensor):
                # Use instance attributes if they exist (e.g., from var_min search)
                current_sampling_eps_min = self.sampling_eps_min
                current_sampling_eps_max = self.sampling_eps_max
            elif isinstance(self.config.training.sampling_eps, (tuple, ListConfig)) and len(self.config.training.sampling_eps) == 2:
                # Config specifies a range [min, max]
                current_sampling_eps_min = torch.tensor(self.config.training.sampling_eps[0], device=x0.device, dtype=torch.float32)
                current_sampling_eps_max = torch.tensor(self.config.training.sampling_eps[1], device=x0.device, dtype=torch.float32)
            elif not isinstance(self.config.training.sampling_eps, (tuple, ListConfig)): # Config specifies scalar (treat as min_eps)
                current_sampling_eps_min = torch.tensor(self.config.training.sampling_eps, device=x0.device, dtype=torch.float32)
                current_sampling_eps_max = torch.tensor(1.0, device=x0.device, dtype=torch.float32) # Max defaults to 1.0
            else: 
                raise ValueError(f"Unexpected type/format for self.config.training.sampling_eps: {self.config.training.sampling_eps}")
        else: # sampling_eps_min/max explicitly passed (e.g., during validation step for var_min)
            current_sampling_eps_min = sampling_eps_min.to(x0.device) if isinstance(sampling_eps_min, torch.Tensor) else torch.tensor(sampling_eps_min, device=x0.device, dtype=torch.float32)
            current_sampling_eps_max = sampling_eps_max.to(x0.device) if isinstance(sampling_eps_max, torch.Tensor) else torch.tensor(sampling_eps_max, device=x0.device, dtype=torch.float32)

        # Subsample if x0 is longer than model's processing length
        input_tokens, _, new_attention_mask = self._maybe_sub_sample(x0, attention_mask)
        B, N_Tokens = input_tokens.shape
        if N_Tokens == 0: # If after subsampling, sequence is empty
            return Loss(loss=torch.tensor(0.0, device=x0.device, requires_grad=True), 
                        nlls=torch.empty((B,0), device=x0.device), 
                        token_mask=torch.empty((B,0), device=x0.device))

        # --- BD3LM Parameterization (Adaptive Schedule) ---
        if self.parameterization == 'bd3lm':
            # Check if input can be blocked. If not, use a fallback (e.g. treat as single block or skip)
            if self.block_size == 0 or N_Tokens < self.block_size or N_Tokens % self.block_size != 0: 
                 # Fallback for non-block-divisible inputs or very short sequences
                 # For simplicity, compute a standard diffusion loss using global schedule (or skip if too short)
                 # This part needs careful design: how to handle non-ideal inputs for BD3LM?
                 # Here, we'll compute a simple loss with global noise, similar to MDLM/SEDD.
                 _t_fallback = self._sample_t((B,1), input_tokens.device, current_sampling_eps_min, current_sampling_eps_max, block_size=N_Tokens) # (B,1)
                 _loss_scale_fallback, _move_chance_fallback = self.noise(_t_fallback) # Global schedule
                 _sigma_fallback = self._sigma_from_p(_move_chance_fallback) # (B,1)
                 _xt_fallback = self.q_xt(input_tokens, _move_chance_fallback, block_size=N_Tokens, sampling_eps_min=current_sampling_eps_min, sampling_eps_max=current_sampling_eps_max)
                 
                 _model_input_fallback = _xt_fallback
                 # Handle cross_attn for fallback if applicable (though problematic for generation consistency)
                 # if self.cross_attn: _model_input_fallback = torch.cat((_xt_fallback, input_tokens), dim=-1)
                
                 _sigma_fwd_fallback = _sigma_fallback.unsqueeze(-1).expand(-1,N_Tokens) if N_Tokens > 0 else _sigma_fallback.unsqueeze(-1) # (B,N)
                 _logits_fallback = self.forward(_model_input_fallback, sigma=_sigma_fwd_fallback)
                 # if self.cross_attn and self.config.algo.backbone != 'hf_dit' and N_Tokens > 0 : _logits_fallback = _logits_fallback[:,:N_Tokens,:] # If output needs slicing

                 if N_Tokens > 0:
                    _log_probs_fallback = F.log_softmax(_logits_fallback, dim=-1)
                    _nll_term_fallback = -torch.gather(_log_probs_fallback, -1, input_tokens.unsqueeze(-1)).squeeze(-1)
                    if self.mdlm_loss_scale: _nll_term_fallback *= _loss_scale_fallback.squeeze(-1) # Apply loss scale
                    _masked_nll_fallback = (_nll_term_fallback * new_attention_mask)
                    _mean_nll_fallback = _masked_nll_fallback.sum() / new_attention_mask.sum().clamp(min=1.0)
                 else: # Should not happen if N_Tokens check passed
                    _mean_nll_fallback = torch.tensor(0.0, device=x0.device, requires_grad=True)
                    _nll_term_fallback = torch.empty((B,0), device=x0.device)

                 return Loss(loss=_mean_nll_fallback, nlls=_nll_term_fallback, token_mask=new_attention_mask)

            # --- Main BD3LM loss computation ---
            N_Blk = N_Tokens // self.block_size
            # Get features for each block of input_tokens (x0)
            x0_block_features = self._get_block_features(input_tokens, new_attention_mask) # (B, N_Blk, FeatDim)

            # Sample u ~ U[eps_min, eps_max] for each block. u is the 't' for schedule warping.
            u_samples_per_block = self._sample_t( # (B, N_Blk)
                (B, N_Blk), 
                input_tokens.device,
                current_sampling_eps_min,
                current_sampling_eps_max,
                block_size=1 # block_size for _sample_t refers to NLL context, not BD3LM block_size
            ) 

            # Get warped schedule outputs (alpha_b(u), beta_b(u), p_b(u)) and s_b for each block
            # alpha_b_u, loss_scale_b_u (beta_b_u), p_b_u are all (B, N_Blk)
            _, loss_scale_b_u_all_blocks, p_b_u_all_blocks, s_b_for_penalty, log_s_tilde_b = \
                self._get_warped_noise_outputs_for_block_batch(x0_block_features, u_samples_per_block)

            # Repeat per-block values to per-token: (B, N_Tokens)
            p_per_token = p_b_u_all_blocks.repeat_interleave(self.block_size, dim=1) 
            loss_scale_per_token = loss_scale_b_u_all_blocks.repeat_interleave(self.block_size, dim=1)
            
            # Sigma for backbone input, derived from per-token warped p_b(u)
            # self.forward will handle _process_sigma if backbone expects (B,)
            sigma_all_tokens_for_backbone = self._sigma_from_p(p_per_token) # (B, N_Tokens)

            # Get x_t by noising input_tokens with per-token p_b(u)
            xt = self.q_xt(input_tokens,
                           p_per_token.unsqueeze(-1), # q_xt expects p to be (B,N,1) or (B,N)
                           block_size=self.block_size, # For resampling within q_xt if active
                           sampling_eps_min=current_sampling_eps_min, 
                           sampling_eps_max=current_sampling_eps_max)

            if self.ignore_bos and N_Tokens > 0: # Don't noise BOS token
                xt[:, 0] = input_tokens[:, 0]

            # Prepare model input (xt, potentially with context if cross_attn)
            # Note: If cross_attn for BD3LM means attending to x0 of current block, this is problematic for generation.
            # Assuming here self.forward handles its inputs appropriately.
            x_model_input = xt
            # if self.cross_attn: # This specific form of cross-attn is tricky for generation
            #     x_model_input = torch.cat((xt, input_tokens), dim=-1)

            # Get model logits p_theta(x0_pred | xt, sigma_warped)
            model_output_logits = self.forward(x_model_input, sigma=sigma_all_tokens_for_backbone) # Pass (B,N) sigma
            
            # If cross_attn implied slicing logits (original code), do it.
            # if self.cross_attn and self.config.algo.backbone != 'hf_dit' and logits.shape[1] == x_model_input.shape[1]:
            #      model_output_logits = model_output_logits[:, :N_Tokens, :] # Ensure logits match input_tokens length

            log_probs_x0 = F.log_softmax(model_output_logits, dim=-1) # (B, N_Tokens, VocabSize)
            # Negative ELBO term (cross-entropy part)
            neg_log_p_theta_x0_given_xt_per_token = -torch.gather(log_probs_x0, -1, input_tokens.unsqueeze(-1)).squeeze(-1) # (B, N_Tokens)
            
            # Weighted NELBO term using warped loss scale (beta_b(u))
            nelbo_term_per_token = loss_scale_per_token * neg_log_p_theta_x0_given_xt_per_token # (B, N_Tokens)
            
            # --- Surrogate Penalty Term ---
            # s_b_for_penalty is (B, N_Blk, 1)
            s_b_for_penalty_device = s_b_for_penalty.to(self.device)
            # alpha_b_at_1_arg_t: dummy t=1 values, shape (B, N_Blk)
            alpha_b_at_1_arg_t = torch.ones_like(s_b_for_penalty_device.squeeze(-1)) 
            
            base_alpha_bar_at_1_for_penalty = self.base_noise_schedule.get_alpha_bar(alpha_b_at_1_arg_t).clamp(
                 self.config.algo.schedule_clamp_epsilon, 1.0 - self.config.algo.schedule_clamp_epsilon
            ) # (B, N_Blk)
            
            # Get warped alpha_b(t=1) using s_b_for_penalty
            _alpha_b_at_1_warped = noise_schedule.get_warped_alpha_b_t(
                base_alpha_bar_t=base_alpha_bar_at_1_for_penalty, # (B, N_Blk)
                base_log_alpha_bar_at_0=self.base_log_alpha_bar_at_0, # Scalar
                s_b=s_b_for_penalty_device.squeeze(-1),  # (B, N_Blk)
                target_log_alpha_at_0=torch.logit(self.target_alpha_at_0.to(base_alpha_bar_at_1_for_penalty.device)) # Scalar
            ) # Output should be (B, N_Blk)

            surrogate_penalty_per_block = noise_schedule.compute_surrogate_steps_penalty(
                _alpha_b_at_1_warped, # (B, N_Blk)
                self.min_alpha_1_target,
                self.lambda_min_alpha_1_penalty,
                self.alpha_1_clamp_min,
                self.alpha_1_clamp_max
            ) # (B, N_Blk)
            mean_surrogate_penalty = surrogate_penalty_per_block.mean() # Mean over all blocks in batch

            # L2 penalty on log_s_tilde_b (raw output of MetaController)
            s_b_l2_reg_term = torch.tensor(0.0, device=input_tokens.device)
            if self.lambda_s_b_l2_penalty > 0 and log_s_tilde_b is not None: # log_s_tilde_b is (B, N_Blk, 1)
                s_b_l2_reg_term = self.lambda_s_b_l2_penalty * (log_s_tilde_b**2).mean()

            # Finalize NELBO
            masked_nelbo_term = (nelbo_term_per_token * new_attention_mask) # Apply attention mask
            # Sum over tokens, then divide by num active tokens (masked sum)
            # Note: Original NELBO is negative log likelihood. Here it's positive due to -torch.gather.
            # So, we are minimizing this positive NELBO.
            mean_nelbo = masked_nelbo_term.sum() / new_attention_mask.sum().clamp(min=1.0)

            # Total loss
            total_loss = mean_nelbo + self.lambda_steps_penalty * mean_surrogate_penalty + s_b_l2_reg_term
            
            # Logging for BD3LM
            if hasattr(self, 'trainer') and self.trainer is not None and not self.trainer.sanity_checking:
                self.log('train/nelbo', mean_nelbo.item(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
                self.log('train/surrogate_penalty', mean_surrogate_penalty.item(), on_step=True, on_epoch=False, sync_dist=True)
                if self.lambda_s_b_l2_penalty > 0:
                     self.log('train/s_b_l2_penalty', s_b_l2_reg_term.item(), on_step=True, on_epoch=False, sync_dist=True)
            
            nlls_for_metric = -nelbo_term_per_token # Metric should be NLL (positive)
            current_loss_for_return = total_loss

        # --- AR Parameterization ---
        elif self.parameterization == 'ar':
            if input_tokens.shape[1] <= 1 and x0.shape[1] <=1 : # Not enough length for AR target
                 return Loss(loss=torch.tensor(0.0, device=x0.device, requires_grad=True), 
                             nlls=torch.zeros_like(input_tokens, dtype=torch.float32), 
                             token_mask=new_attention_mask)

            # AR model forward pass (no sigma)
            output_ar_logprobs = self.forward(input_tokens, None) # (B, N_Input, V) log_probs
            
            # Determine target tokens (shifted input)
            if x0.shape[1] > input_tokens.shape[1]: # If input_tokens was subsampled from x0
                 # Target needs to align with the subsampled input_tokens
                 # Example: x0 = [0,1,2,3,4], num_tokens=3. input_tokens might be [0,1,2] from x0[0:3]. Target is x0[1:4] -> [1,2,3]
                 # This depends on how _maybe_sub_sample produced input_tokens relative to x0.
                 # Assuming _maybe_sub_sample: input=[x_s:x_e], output=[x_{s+1}:x_{e+1}]
                 # So target_ar should be the `output_tokens` from `_maybe_sub_sample` if it was created,
                 # or derived correctly from x0.
                 # The _maybe_sub_sample's output_tokens handles this for AR. Let's re-fetch:
                 _, target_ar_from_subsample, _ = self._maybe_sub_sample(x0, attention_mask) # Get correctly shifted target
                 target_ar = target_ar_from_subsample

            else: # input_tokens is x0[:, :-1]
                 target_ar = x0[:, 1:]


            if target_ar is None or target_ar.shape[1] == 0: # No target tokens available
                 return Loss(loss=torch.tensor(0.0, device=x0.device, requires_grad=True), 
                             nlls=torch.zeros_like(input_tokens[:,:output_ar_logprobs.shape[1]], dtype=torch.float32), # Match output length
                             token_mask=new_attention_mask[:,:output_ar_logprobs.shape[1]])
            
            # Align lengths of output_ar_logprobs and target_ar
            min_len_ar = min(output_ar_logprobs.shape[1], target_ar.shape[1])
            output_ar_matched = output_ar_logprobs[:,:min_len_ar,:]
            target_ar_matched = target_ar[:,:min_len_ar]
            mask_ar = new_attention_mask[:, :min_len_ar] # Attention mask for targets

            # NLL for AR model (negative log likelihood)
            loss_ar_nll = - output_ar_matched.gather(-1, target_ar_matched.unsqueeze(-1)).squeeze(-1) # (B, min_len_ar)
            
            mean_loss_ar = (loss_ar_nll * mask_ar).sum() / mask_ar.sum().clamp(min=1.0)
            nlls_for_metric = loss_ar_nll 
            current_loss_for_return = mean_loss_ar
        
        # --- Other Diffusion Parameterizations (MDLM, SEDD) ---
        else: 
            # Sample t from U[eps_min, eps_max] per batch item (not per block)
            if t is None: # If t not passed (usual training case)
                # _sample_t here expects (B,1) for batch_dims if not block-based
                t_for_loss = self._sample_t((input_tokens.shape[0], 1), input_tokens.device, 
                                             current_sampling_eps_min, current_sampling_eps_max, 
                                             block_size=self.block_size) # (B,1)
            else: # t is passed (e.g. for specific t evaluation in validation)
                t_val = t.item() if isinstance(t,torch.Tensor) and t.numel()==1 else t # Get scalar t
                t_for_loss = t_val * torch.ones(input_tokens.shape[0], 1, device=input_tokens.device, dtype=torch.float32)
            
            # Get global noise schedule parameters
            loss_scale_global, move_chance_global = self.noise(t_for_loss) # (B,1), (B,1)
            sigma_for_loss_global = self._sigma_from_p(move_chance_global) # (B,1)
            
            # Get x_t by noising input_tokens
            xt = self.q_xt(input_tokens,
                           move_chance_global, # (B,1) -> broadcasts to (B,N) in q_xt
                           block_size=self.block_size, # For resampling in q_xt
                           sampling_eps_min=current_sampling_eps_min, 
                           sampling_eps_max=current_sampling_eps_max)
            
            if self.ignore_bos and N_Tokens > 0: # Don't noise BOS
                xt[:, 0] = input_tokens[:, 0]

            # Prepare model input
            x_model_input_other = xt
            # if self.cross_attn: # Problematic for generation, see BD3LM notes
            #     x_model_input_other = torch.cat((xt, input_tokens), dim=-1)
            
            # Sigma for forward pass, needs to be (B,N) if backbone expects per-token
            # self.forward will call _process_sigma to make it (B,) if needed
            sigma_for_fwd_other = sigma_for_loss_global.expand(-1, N_Tokens) if N_Tokens > 0 else sigma_for_loss_global # (B,N)
            
            model_output_logits_other = self.forward(x_model_input_other, sigma=sigma_for_fwd_other) 
            # if self.cross_attn and self.config.algo.backbone != 'hf_dit' and logits.shape[1] == x_model_input_other.shape[1]:
            #      model_output_logits_other = model_output_logits_other[:, :N_Tokens, :]

            # Calculate loss based on parameterization
            if self.parameterization == 'sedd':
                # SEDD loss is entropy-based for masked tokens, needs sigma_for_loss_global (B,1)
                loss_other = self._score_entropy(model_output_logits_other, sigma_for_loss_global, xt, input_tokens)
            else: # MDLM / SUBS (direct NLL on p_theta(x0|xt))
                log_probs_x0_other = F.log_softmax(model_output_logits_other, dim=-1) # (B,N,V)
                loss_other = -torch.gather(log_probs_x0_other, -1, input_tokens.unsqueeze(-1)).squeeze(-1) # (B,N)
            
            # Apply MDLMLossScale if configured (weights loss by 1/sigma^2 related term)
            if self.mdlm_loss_scale: # loss_scale_global is (B,1)
                loss_other *= loss_scale_global.squeeze(-1) # loss_other is (B,N), broadcast B dim
            
            masked_loss_other = (loss_other * new_attention_mask) # Apply attention mask
            mean_loss_other = masked_loss_other.sum() / new_attention_mask.sum().clamp(min=1.0)
            
            nlls_for_metric = loss_other # NLL per token (positive)
            current_loss_for_return = mean_loss_other
            
        return Loss(loss=current_loss_for_return,
                    nlls=nlls_for_metric, 
                    token_mask=new_attention_mask)


    def _clipped_schedule_search(self):
        # Implements search for optimal (eps_min, eps_max) for t sampling based on variance
        best_var = float('inf')
        # Ensure sampling_eps_min/max are initialized as scalar tensors on device
        if not hasattr(self, 'sampling_eps_min') or not isinstance(self.sampling_eps_min, torch.Tensor):
             self.sampling_eps_min = torch.tensor(self.config.training.sampling_eps_min, device=self.device, dtype=torch.float32)
        if not hasattr(self, 'sampling_eps_max') or not isinstance(self.sampling_eps_max, torch.Tensor):
             self.sampling_eps_max = torch.tensor(self.config.training.sampling_eps_max, device=self.device, dtype=torch.float32)

        sampling_eps_min_best = self.sampling_eps_min.item() # Current best (scalar float)
        sampling_eps_max_best = self.sampling_eps_max.item() # Current best (scalar float)

        # self.metrics.valid_vars is a dict: {(eps_min_f, eps_max_f): [list_of_nll_tensors]}
        for (eps_min_float, eps_max_float), var_list_cpu_tensors in self.metrics.valid_vars.items():
          if not var_list_cpu_tensors: continue # Skip if no NLLs collected for this interval
          
          # Concatenate all collected NLL tensors (which are on CPU)
          var_list_cpu_tensors_on_cpu = [t.cpu() for t in var_list_cpu_tensors if t is not None and t.numel() > 0] 
          if not var_list_cpu_tensors_on_cpu: continue 

          all_vars_tensor_cpu = torch.cat(var_list_cpu_tensors_on_cpu, dim=0) # Cat along batch dim
          if all_vars_tensor_cpu.numel() == 0: continue 

          # Move to current device for DDP gathering
          all_vars_tensor_device = all_vars_tensor_cpu.to(self.device)
          
          # Gather NLLs from all DDP processes
          gathered_vars = self.all_gather(all_vars_tensor_device) # List of tensors from each rank, or single tensor

          # Flatten gathered tensors into a single 1D tensor for variance calculation
          if isinstance(gathered_vars, list) and all(isinstance(t, torch.Tensor) for t in gathered_vars):
            if not gathered_vars : continue # Empty list from all_gather
            gathered_vars_flat = torch.cat([gv.view(-1) for gv in gathered_vars if gv.numel() > 0]) 
          elif isinstance(gathered_vars, torch.Tensor): # Non-DDP or single-rank DDP
            gathered_vars_flat = gathered_vars.view(-1)
          else: 
            print(f"Warning: Unexpected output from all_gather: {type(gathered_vars)}")
            continue
          
          if gathered_vars_flat.numel() == 0: continue 

          # Calculate variance of NLLs for this (eps_min, eps_max) interval
          if gathered_vars_flat.numel() > 1 : # Variance needs at least 2 samples
            current_total_variance = gathered_vars_flat.var()
          else: # Not enough samples for variance
            current_total_variance = torch.tensor(float('inf'), device=self.device, dtype=torch.float32) # Or handle differently
          
          if current_total_variance < best_var:
            best_var = current_total_variance
            sampling_eps_min_best = eps_min_float
            sampling_eps_max_best = eps_max_float
          
          # Log variance for this specific interval
          self.log(f'valid_var_{round(eps_min_float, 3)}_{round(eps_max_float, 3)}', 
                    current_total_variance.item() if isinstance(current_total_variance, torch.Tensor) else current_total_variance, 
                    on_epoch=True, on_step=False, sync_dist=False) # sync_dist=False as already gathered
        
        # Update self.sampling_eps_min/max if fix_clipping is False
        if self.config.algo.fix_clipping == False:
          self.sampling_eps_min.fill_(sampling_eps_min_best)
          self.sampling_eps_max.fill_(sampling_eps_max_best)

    def _score_entropy(self, log_score, sigma_batch, xt, x0): 
        # Calculates score entropy for SEDD model loss (Equation 6 in SEDD paper)
        # log_score: model output after parameterization adjustment, (B,N,V)
        # sigma_batch: (B,1) or (B,) effective sigma for these samples
        # xt: (B,N) noisy tokens
        # x0: (B,N) original tokens
        
        masked_indices = xt == self.mask_index # (B,N) bool, where xt is mask
        # Ensure sigma_batch is (B,N) for element-wise ops if needed, or (B,1) for broadcasting
        sigma_expanded_like_xt = sigma_batch.view(-1,1).expand_as(xt) if sigma_batch.ndim==1 else sigma_batch.expand_as(xt)

        expsig_minus_1 = torch.expm1(sigma_expanded_like_xt) # exp(sigma)-1, shape (B,N)
        
        entropy = torch.zeros_like(xt, dtype=torch.float32) # (B,N)

        if masked_indices.any(): # Only compute for tokens that were masked in xt
            # q_ratio = 1 / (exp(sigma_t) - 1)
            q_ratio_masked = 1.0 / expsig_minus_1[masked_indices].clamp(min=1e-9) # (NumMaskedTokens,)
            
            words_that_were_masked_gt = x0[masked_indices] # Ground truth tokens at masked positions
            
            log_score_at_masked_pos = log_score[masked_indices] # (NumMaskedTokens, V)
            
            # Term 1: - (1/(e^sig-1)) * s_theta(x0, xt_mask)
            # s_theta(x0, xt_mask) is log_score_at_masked_pos gathered at words_that_were_masked_gt
            neg_term = q_ratio_masked * torch.gather(
              log_score_at_masked_pos, # (NumMasked, V)
              -1, # Gather along vocab dim
              words_that_were_masked_gt.unsqueeze(-1) # (NumMasked, 1) indices
              ).squeeze(-1) # (NumMasked,)
            
            # Term 2: Sum_{x'_0 != x0_gt, x'_0 != MASK} exp(s_theta(x'_0, xt_mask))
            # This is sum of exp(log_score) over relevant vocab
            score_exp_masked = log_score_at_masked_pos.exp() # (NumMasked, V) probabilities

            # Sum exp(score) for all vocab EXCEPT the ground truth word (words_that_were_masked_gt)
            # AND EXCEPT the MASK token itself (model shouldn't predict MASK as x0)
            pos_term_sum_parts = []
            # This loop is inefficient, better to sum all and subtract specific terms
            # Total sum: score_exp_masked.sum(dim=-1)
            # Subtract exp(s_theta(x0_gt, xt_mask))
            # Subtract exp(s_theta(MASK, xt_mask))
            
            sum_exp_score_all_vocab = score_exp_masked.sum(dim=-1) # (NumMasked,)
            exp_score_at_gt = torch.gather(score_exp_masked, -1, words_that_were_masked_gt.unsqueeze(-1)).squeeze(-1)
            
            exp_score_at_mask_token = torch.zeros_like(exp_score_at_gt) # Default if no mask in vocab
            if self.mask_index < score_exp_masked.shape[-1]:
                exp_score_at_mask_token = score_exp_masked[..., self.mask_index]
            
            pos_term = sum_exp_score_all_vocab - exp_score_at_gt - exp_score_at_mask_token
            
            # Term 3: const = (1/(e^sig-1)) * (log(1/(e^sig-1)) - 1)
            # This seems to be related to q(x_t|x_0)'s entropy part if x_t=MASK
            # The SEDD paper's loss form is simpler: L_vlb = E_q [ sum_t lambda_t D_KL(q(x0|xt)||p_theta(x0|xt)) ]
            # And D_KL for discrete is sum q log(q/p).
            # The provided formula seems to be a specific expansion.
            # Assuming the formula structure is from a source:
            const = q_ratio_masked * (q_ratio_masked.log() - 1.0) # (NumMasked,)
            
            entropy[masked_indices] = pos_term - neg_term + const # Store for masked positions
        return entropy
    
    @torch.no_grad()
    def _analytic_sampler(self, n_samples, num_strides, num_steps, seqlen, eps=1e-5):
        # Analytic D3PM sampler, adapted for block-sequential generation with warped schedules
        if seqlen == 0:
            return torch.empty((n_samples, 0), dtype=torch.long, device=self.device), 0
        
        sampling_steps_nfe = 0 # NFE for the backbone
        x_accum = None

        if self.config.sampling.kv_cache:
            self.backbone.reset_kv_cache(eval_batch_size=n_samples)

        for stride_num in tqdm(range(num_strides), desc="Analytic Sampler Strides (Adaptive)"):
            current_block_gen_len = self.block_size
            if x_accum is not None and x_accum.shape[1] + current_block_gen_len > seqlen:
                current_block_gen_len = seqlen - x_accum.shape[1]
            if current_block_gen_len <= 0: break

            # Initialize current block (noisy)
            x_current_block = self._sample_prior(n_samples, current_block_gen_len).to(self.device)
            if stride_num == 0 and x_current_block.numel() > 0 and x_current_block.shape[1] > 0:
                x_current_block[:, 0] = self.tokenizer.bos_token_id

            # Get features and s_b for this block
            context_for_features = x_accum if x_accum is not None else torch.empty((n_samples, 0), dtype=torch.long, device=self.device)
            block_features_j = self._extract_features_for_block_generation(
                x_accum_context=context_for_features,
                current_block_idx_to_generate=stride_num,
                total_num_blocks=num_strides
            ) # (B, 1, FeatDim)
            log_s_tilde_b_j = self.meta_controller(block_features_j)
            s_b_j = self.meta_controller.get_s_b(log_s_tilde_b_j) # (B, 1, 1)

            # Inner loop for diffusion steps (global_step_k from 1 to num_steps)
            # Corresponds to iterating t from 1.0 down to eps in Analytic D3PM
            for global_step_k in range(1, num_steps + 1): # k from 1 to N
                if x_current_block.numel() == 0: break
                
                # Get remapped t'_k (current time) and t'_{k-1} (previous time for dt)
                # For Analytic D3PM, t_k goes from 1 down to eps.
                # We need t_prime_k corresponding to global_t_k = 1 - (k-1)*dt_global
                # and t_prime_{k-1} corresponding to global_t_{k-1} = 1 - k*dt_global
                # Let's use _get_remapped_timesteps_for_block by interpreting global_step_k as progress.
                # The original analytic sampler has t_current and t_next (t_current > t_next).
                # t_prime_k_eff will be for current "more noisy" time, t_prime_k_minus_1_eff for "less noisy"
                
                # global_step_k in Analytic D3PM context means step index in the DDIM-like iteration
                # We need t_current for sigma_t, and t_previous for sigma_s.
                # Let's align with DDPM: step k uses t_k and t_{k-1} from the discrete schedule.
                # Here, we remap global step indices to warped schedule times.
                
                # t_current_global is like timesteps[i] in original _analytic_sampler
                # t_next_global is like timesteps[i+1]
                # global_t_k_progress = (num_steps - (global_step_k -1)) / num_steps # Progress from 1 down to 1/N
                # global_t_k_minus_1_progress = (num_steps - global_step_k) / num_steps # Progress from (N-1)/N down to 0

                # Simpler: use global_step_k directly for remapping.
                # _get_remapped_timesteps_for_block(s_b, k, N) gives t'_k and t'_{k-1}
                # where t'_k is "more diffused" than t'_{k-1} if beta integral is increasing.
                # This matches DDIM t_i and t_{i-1} (t_i > t_{i-1}).
                t_prime_curr, t_prime_prev = self._get_remapped_timesteps_for_block(
                    s_b_j, global_step_k, num_steps
                ) # t_prime_curr > t_prime_prev (potentially)

                x_current_block = self._analytic_update_warped(
                    x_current_block, s_b_j, t_prime_curr, t_prime_prev
                )
                sampling_steps_nfe += 1 # One NFE per analytic update step

            # Final denoiser step using t_prime_final (remapped from global eps)
            # Need to find t_prime corresponding to global schedule time `eps`
            # This is tricky. Let's use t_prime_k_minus_1 from the last step as t_prime_final.
            # (This t_prime_prev was for the *target* state of the last _analytic_update_warped step)
            # Or, more robustly, get t_prime corresponding to a very small global t like `eps`.
            # For simplicity, let t_prime_final be t_prime_prev from last step.
            t_prime_final_for_denoise = t_prime_prev # From last iteration of loop

            x_current_block = self._denoiser_update_warped(
                x_current_block, s_b_j, t_prime_final_for_denoise
            )
            sampling_steps_nfe += 1 # One NFE for denoiser step

            # Accumulate
            if x_accum is None: x_accum = x_current_block
            else: x_accum = torch.cat((x_accum, x_current_block), dim=1)

            # KV cache update (if block is clean and KV enabled)
            if self.config.sampling.kv_cache and self.mask_index not in x_current_block:
                 # Need a sigma for KV cache. Use sigma derived from t_prime_final_for_denoise
                 _, _, p_b_j_final_kv = self._get_warped_outputs_at_t(s_b_j, t_prime_final_for_denoise)
                 sigma_kv = self._sigma_from_p(p_b_j_final_kv.squeeze(-1)) # (B,1)
                 sigma_kv_expanded = sigma_kv.expand(-1, x_current_block.shape[1])
                 _ = self.forward(x_current_block, sigma_kv_expanded, sample_mode=True, store_kv=True)
            
            # Stop conditions
            if x_accum.shape[1] > 256: # Or other threshold
                stop_batch_flag, x_accum_maybe_truncated = self._check_stop_conds(x_accum)
                if self.config.sampling.var_length:
                    if stop_batch_flag:
                        final_output_val = x_accum_maybe_truncated
                        if isinstance(x_accum_maybe_truncated, list): # _check_stop_conds returns list for var_len
                            if not x_accum_maybe_truncated: return None, None
                            final_output_val = x_accum_maybe_truncated[0].unsqueeze(0) if x_accum_maybe_truncated[0].ndim == 1 else x_accum_maybe_truncated[0]

                        if final_output_val is None or final_output_val.numel() == 0: return None, None
                        is_just_bos = (final_output_val.shape[0]>0 and final_output_val.shape[1] == 1 and final_output_val[0,0] == self.tokenizer.bos_token_id)
                        if is_just_bos : return None, None
                        return final_output_val, sampling_steps_nfe
                elif stop_batch_flag and x_accum_maybe_truncated is None: # Fixed length, but sample failed stop cond
                     return None, None
        
        if x_accum is None or x_accum.numel() == 0: return None, None
        return x_accum, sampling_steps_nfe


    @torch.no_grad()
    def _semi_ar_sampler(
      self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
        # num_steps is total_global_steps for adaptive schedule
        if seqlen is None:
          seqlen = self.config.model.length
        if seqlen == 0: 
            return torch.empty((n_samples, 0), dtype=torch.long, device=self.device), 0

        sampling_steps_nfe = 0 # Tracks NFE for the backbone model

        if self.config.sampling.kv_cache:
          self.backbone.reset_kv_cache(eval_batch_size=n_samples) 

        x_accum = None 
        
        for stride_num in tqdm(range(num_strides), desc="Semi-AR Strides (Adaptive Schedule)"):
          current_block_gen_len = self.block_size
          
          if x_accum is not None and x_accum.shape[1] + current_block_gen_len > seqlen :
              current_block_gen_len = seqlen - x_accum.shape[1]
          if current_block_gen_len <=0 : break

          # 1. Prepare current block to be generated (starts noisy)
          x_current_block_noisy = self._sample_prior(n_samples, current_block_gen_len).to(self.device)
          if stride_num == 0 and x_current_block_noisy.numel() > 0 and x_current_block_noisy.shape[1] > 0 :
              x_current_block_noisy[:, 0] = self.tokenizer.bos_token_id # BOS for first block
          
          # 2. Get features for the current block based on x_accum (context)
          context_for_features = x_accum if x_accum is not None else torch.empty((n_samples, 0), dtype=torch.long, device=self.device)
          
          block_features_j = self._extract_features_for_block_generation(
              x_accum_context=context_for_features,
              current_block_idx_to_generate=stride_num,
              total_num_blocks=num_strides
          ) # (B, 1, FeatDim)
          log_s_tilde_b_j = self.meta_controller(block_features_j)
          s_b_j = self.meta_controller.get_s_b(log_s_tilde_b_j) # (B, 1, 1)
          
          p_x0_cache_block = None # For DDPM caching trick

          # 3. Inner loop for DDPM steps (global_step_k from 1 to num_steps)
          for global_step_k in range(1, num_steps + 1):
            if x_current_block_noisy.numel() == 0: break

            # Get remapped t'_k (current) and t'_{k-1} (previous) for this block j
            t_prime_k, t_prime_k_minus_1 = self._get_remapped_timesteps_for_block(
                s_b_j, global_step_k, num_steps
            ) # Shapes: (B, 1)

            # Get warped schedule outputs (alpha, beta, p) at these remapped times
            # p_b_j_curr is p_b(t'_k), p_b_j_prev is p_b(t'_{k-1})
            _, _, p_b_j_curr = self._get_warped_outputs_at_t(s_b_j, t_prime_k) # (B,1,1)
            _, _, p_b_j_prev = self._get_warped_outputs_at_t(s_b_j, t_prime_k_minus_1) # (B,1,1)

            move_chance_t_eff = p_b_j_curr.squeeze(-1) # (B, 1)
            move_chance_s_eff = p_b_j_prev.squeeze(-1) # (B, 1)
            
            sigma_t_eff_for_model = self._sigma_from_p(move_chance_t_eff) # (B, 1)
            
            # Expand sigma for backbone: (B, current_block_gen_len)
            sigma_for_backbone_expanded = sigma_t_eff_for_model.expand(-1, current_block_gen_len)

            if p_x0_cache_block is None: # If p_theta(x0|xt) is not cached for this block
                model_input_for_fwd = x_current_block_noisy
                # self.forward handles input based on its architecture and self.cross_attn.
                # For BD3LM, if cross_attn implies attending to x0 of current block, this is an issue at generation.
                # Assuming self.forward(xt, sigma) is the call.
                raw_logits_or_logprobs = self.forward(
                    model_input_for_fwd, 
                    sigma=sigma_for_backbone_expanded, 
                    sample_mode=True
                ).to(torch.float64)
                sampling_steps_nfe += 1

                if self.parameterization == 'subs' or self.parameterization == 'ar':
                    p_x0_cache_block = raw_logits_or_logprobs.exp()
                else: # sedd, bd3lm (direct logits)
                    p_x0_cache_block = F.softmax(raw_logits_or_logprobs, dim=-1)
                p_x0_cache_block = self._nucleus_sample(p_x0_cache_block)

            # Effective mask probability for DDPM step: p_eff(t'_{k-1}) / p_eff(t'_k)
            mask_prob_eff = move_chance_s_eff / move_chance_t_eff.clamp(min=1e-9) # (B, 1)

            # DDPM reverse step logic (from _ddpm_caching_update)
            if self.config.sampling.first_hitting:
                x_block_sampled_hitting = _sample_categorical(p_x0_cache_block)
                one_change_mask = torch.zeros_like(x_block_sampled_hitting, dtype=torch.bool)
                for b_idx_fh in range(x_current_block_noisy.shape[0]):
                    masked_pos_fh = (x_current_block_noisy[b_idx_fh] == self.mask_index).nonzero(as_tuple=False).squeeze(-1)
                    if masked_pos_fh.numel() > 0:
                        chosen_idx_fh = torch.randint(0, masked_pos_fh.numel(), (1,), device=self.device).item()
                        actual_idx_fh = masked_pos_fh[chosen_idx_fh]
                        one_change_mask[b_idx_fh, actual_idx_fh] = True
                x_block_new_noisy = torch.where(one_change_mask, x_block_sampled_hitting, x_current_block_noisy)
            else: # Standard DDPM reverse step
                q_xs = p_x0_cache_block * (1.0 - mask_prob_eff.unsqueeze(-1)) # (B, L_blk, V)
                if self.mask_index < q_xs.shape[-1]:
                    q_xs[..., self.mask_index] += mask_prob_eff.unsqueeze(-1)
                x_block_new_noisy = _sample_categorical(q_xs)

            copy_flag = (x_current_block_noisy != self.mask_index)
            x_current_block_noisy_updated = torch.where(copy_flag, x_current_block_noisy, x_block_new_noisy)
            
            if not torch.allclose(x_current_block_noisy_updated, x_current_block_noisy):
                p_x0_cache_block = None # Invalidate cache if change occurred

            x_current_block_noisy = x_current_block_noisy_updated

            if self.mask_index not in x_current_block_noisy: # Block is fully denoised
                if self.config.sampling.kv_cache: # Update KV cache
                     sigma_final_kv = self._sigma_from_p(move_chance_t_eff).expand(-1,current_block_gen_len)
                     _ = self.forward(x_current_block_noisy, sigma_final_kv, sample_mode=True, store_kv=True)
                break # Exit inner DDPM loop for this block
          
          # End of DDPM steps for current block
          # Accumulate the generated/denoised block
          if x_accum is None: x_accum = x_current_block_noisy
          else: x_accum = torch.cat((x_accum, x_current_block_noisy), dim=1)
          
          # Stop conditions check
          if x_accum.shape[1] > 256 : # Or other suitable length for checking
            stop_batch_flag, x_accum_maybe_truncated = self._check_stop_conds(x_accum)
            if self.config.sampling.var_length:
                if stop_batch_flag : 
                    final_output_val = x_accum_maybe_truncated
                    if isinstance(x_accum_maybe_truncated, list):
                        if not x_accum_maybe_truncated: return None, None
                        final_output_val = x_accum_maybe_truncated[0].unsqueeze(0) if x_accum_maybe_truncated[0].ndim==1 else x_accum_maybe_truncated[0]
                    
                    if final_output_val is None or final_output_val.numel() == 0: return None, None
                    is_just_bos = (final_output_val.shape[0]>0 and final_output_val.shape[1] == 1 and final_output_val[0,0] == self.tokenizer.bos_token_id)
                    if is_just_bos : return None, None
                    return final_output_val, sampling_steps_nfe
            elif stop_batch_flag and x_accum_maybe_truncated is None: 
                 return None, None
        
        if x_accum is None or x_accum.numel() == 0 : return None, None
        
        # Final denoise for non-BD3LM if param is not bd3lm (original code had this)
        # For BD3LM, the loop above should suffice. This part is more for MDLM/SEDD if they use semi-AR.
        # If parameterization is 'bd3lm', x_accum should be clean.
        # If not 'bd3lm' and a final denoise is needed:
        # if self.parameterization != 'bd3lm' and x_current_block_noisy.numel() > 0 :
        #     # This requires a final t and s_b or global schedule
        #     # Simplified: assume x_accum is the result for BD3LM.
        #     pass

        return x_accum, sampling_steps_nfe


    def _compute_entropy(self, x_sample): # x_sample is a 1D tensor of token_ids
        if x_sample.numel() == 0: return torch.tensor(0.0, device=x_sample.device, dtype=torch.float32)
        _, counts = torch.unique(x_sample, return_counts=True, sorted=False)
        if counts.sum() == 0: return torch.tensor(0.0, device=x_sample.device, dtype=torch.float32)
        probs = counts.float() / counts.sum()
        entropy = torch.special.entr(probs).sum() # sum(-p log p)
        return entropy


    def _check_stop_conds(self, x_batch): # For diffusion samplers variable length
        # x_batch is (B, SeqLen_current_total)
        B, SeqLen = x_batch.shape
        stop_flags_batch = torch.zeros(B, dtype=torch.bool, device=x_batch.device)
        final_x_batch_list_of_tensors = list(x_batch.clone().unbind(0)) # List of (SeqLen,) tensors

        for b_idx in range(B):
            current_x_sample = x_batch[b_idx].clone() # (SeqLen,)
            stop_this_sample = False
            current_truncate_idx = current_x_sample.shape[0] # Default to full current length

            # 1. Entropy check on last N tokens (e.g. 256)
            check_entropy_len = min(256, current_truncate_idx)
            if check_entropy_len > 0:
                entropy_val = self._compute_entropy(current_x_sample[-check_entropy_len:])
                if entropy_val < 4.0: # Low entropy threshold
                    stop_this_sample = True
                    if self.config.sampling.var_length: # Truncate before low-entropy part
                        current_truncate_idx = max(1, current_truncate_idx - check_entropy_len)

            # 2. EOS token check
            if self.config.sampling.var_length:
                eos_indices = (current_x_sample == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    first_valid_eos_pos = -1
                    for eos_idx_val_item in eos_indices:
                        eos_idx = eos_idx_val_item.item()
                        is_bos_present = current_x_sample.numel() > 0 and current_x_sample[0] == self.tokenizer.bos_token_id
                        is_eos_at_bos = (eos_idx == 0)
                        
                        if is_eos_at_bos and is_bos_present and (self.tokenizer.bos_token_id == self.tokenizer.eos_token_id):
                            if current_x_sample.shape[0] == 1: first_valid_eos_pos = 1; break
                            elif len(eos_indices) > 1: first_valid_eos_pos = eos_indices[1].item() + 1; break
                        elif not is_eos_at_bos: first_valid_eos_pos = eos_idx + 1; break
                        elif is_eos_at_bos and not (is_bos_present and self.tokenizer.bos_token_id == self.tokenizer.eos_token_id):
                            first_valid_eos_pos = 1; break # EOS is first token, stop after it
                            
                    if first_valid_eos_pos != -1:
                        stop_this_sample = True
                        current_truncate_idx = min(current_truncate_idx, first_valid_eos_pos)
            
            if stop_this_sample:
                stop_flags_batch[b_idx] = True
                final_tensor_for_list = current_x_sample[:current_truncate_idx]
                if final_tensor_for_list.numel() == 0: # Ensure not empty after truncation
                    final_tensor_for_list = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long, device=x_batch.device)
                final_x_batch_list_of_tensors[b_idx] = final_tensor_for_list


        # Return value handling based on var_length config
        if not self.config.sampling.var_length: # Fixed length generation
            if stop_flags_batch.any(): # If any sample met early stop criteria (e.g. low entropy but fixed len)
                return True, None # Signal failure or issue for fixed length case
            return False, x_batch # Return original full batch
        else: # Variable length generation
            # Check if all samples became effectively empty (e.g. just BOS after truncation)
            all_effectively_empty_or_just_bos = True
            if B > 0 : 
                for s_tensor in final_x_batch_list_of_tensors:
                    is_just_bos = (s_tensor.shape[0] == 1 and s_tensor.numel()>0 and s_tensor[0] == self.tokenizer.bos_token_id)
                    if s_tensor.numel() > 0 and not is_just_bos : # If any sample is non-empty and not just BOS
                        all_effectively_empty_or_just_bos = False
                        break
                if all_effectively_empty_or_just_bos : # All samples are bad
                    return True, None # Signal all samples failed or are trivial
            elif B == 0: # No samples in batch
                 return False, [] # Return empty list

            # If var_length, return list of (potentially truncated) tensors.
            # stop_flags_batch.all() indicates if ALL met some stop criterion.
            # Individual tensors in list are already truncated if their specific flag was true.
            return stop_flags_batch.all(), final_x_batch_list_of_tensors