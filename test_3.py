import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig, ListConfig
import math
import os
import sys
from collections import OrderedDict
from unittest.mock import MagicMock, patch, ANY 
from nltk.tree import Tree 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from diffusion import Diffusion, Loss 
import noise_schedule 
import dataloader # dataloader.Text8Tokenizer is used by MockTokenizer
from models.meta_controller import MetaController
from transformers import AutoModel, AutoTokenizer, AutoConfig # Correctly imported for direct use
import spacy # For spacy.__version__ and for patching spacy.load
import benepar # Ensure benepar is importable for the Diffusion class if used

# --- Mocks ---
class MockTokenizer(dataloader.Text8Tokenizer):
    def __init__(self, vocab_size_override=50, bos_token_id=0, eos_token_id=1, pad_token_id=2, mask_token_id=3, unk_token_id=4):
        self._vocab_size_override = vocab_size_override
        _special_tokens_map = {
            '[BOS]': bos_token_id, '[EOS]': eos_token_id, '[PAD]': pad_token_id,
            '[MASK]': mask_token_id, '[UNK]': unk_token_id,
        }
        self._vocab_str_to_int_override = {}
        self._vocab_int_to_str_override = {}
        current_ids_assigned = set()

        for token, token_id in _special_tokens_map.items():
            if token_id < self._vocab_size_override and token_id not in current_ids_assigned:
                self._vocab_str_to_int_override[token] = token_id
                self._vocab_int_to_str_override[token_id] = token
                current_ids_assigned.add(token_id)
        
        next_available_id = 0
        for i in range(self._vocab_size_override - len(current_ids_assigned)):
            while next_available_id in current_ids_assigned or next_available_id >= self._vocab_size_override :
                next_available_id +=1
            if next_available_id >= self._vocab_size_override: break
            token_str = f'token{i+len(_special_tokens_map)}' 
            if token_str not in self._vocab_str_to_int_override:
                 self._vocab_str_to_int_override[token_str] = next_available_id
                 self._vocab_int_to_str_override[next_available_id] = token_str
                 current_ids_assigned.add(next_available_id)
        
        super_kwargs_init = {
            'bos_token': '[BOS]', 'eos_token': '[EOS]', 'sep_token': '[SEP]',
            'cls_token': '[CLS]', 'pad_token': '[PAD]', 'mask_token': '[MASK]',
            'unk_token': '[UNK]'
        }
        for token_key_super, token_val_super in _special_tokens_map.items():
             super_kwargs_init[token_key_super.lower().strip('[]') + "_token_id"] = token_val_super
        
        super().__init__(**super_kwargs_init)

        final_vocab_str_to_int = {}
        final_vocab_int_to_str = {}
        
        for token, token_id in _special_tokens_map.items():
            if token_id < self._vocab_size_override:
                final_vocab_str_to_int[token] = token_id
                final_vocab_int_to_str[token_id] = token
        
        for token, token_id in self._vocab_str_to_int_override.items():
            if len(final_vocab_str_to_int) >= self._vocab_size_override: break
            if token not in final_vocab_str_to_int and token_id not in final_vocab_int_to_str and token_id < self._vocab_size_override:
                final_vocab_str_to_int[token] = token_id
                final_vocab_int_to_str[token_id] = token
        
        if hasattr(super(), '_vocab_str_to_int'): # Check if superclass has _vocab_str_to_int
            for token, token_id in super()._vocab_str_to_int.items():
                if len(final_vocab_str_to_int) >= self._vocab_size_override: break
                if token not in final_vocab_str_to_int and token_id not in final_vocab_int_to_str and token_id < self._vocab_size_override:
                    final_vocab_str_to_int[token] = token_id
                    final_vocab_int_to_str[token_id] = token

        self._vocab_str_to_int = final_vocab_str_to_int
        self._vocab_int_to_str = final_vocab_int_to_str
        
        self.bos_token_id = self._vocab_str_to_int.get('[BOS]', bos_token_id)
        self.eos_token_id = self._vocab_str_to_int.get('[EOS]', eos_token_id)
        self.pad_token_id = self._vocab_str_to_int.get('[PAD]', pad_token_id)
        self.mask_token_id = self._vocab_str_to_int.get('[MASK]', mask_token_id)
        self.unk_token_id = self._vocab_str_to_int.get('[UNK]', unk_token_id)

    @property
    def vocab_size(self) -> int: return self._vocab_size_override
    def get_vocab(self) -> dict: return self._vocab_str_to_int.copy()
    def _convert_token_to_id(self, token: str) -> int: return self._vocab_str_to_int.get(token, self.unk_token_id)
    def _convert_id_to_token(self, index: int) -> str: return self._vocab_int_to_str.get(index, self.unk_token)
    def _tokenize(self, text, **kwargs): return list(text)

class MockOutput:
    def __init__(self, logits, last_hidden_state=None):
        self.logits = logits
        self.last_hidden_state = last_hidden_state if last_hidden_state is not None else logits

class MockBackbone(nn.Module):
    def __init__(self, vocab_size, hidden_dim, length, is_hf_model=False):
        super().__init__()
        self.vocab_size = vocab_size; self.hidden_dim = hidden_dim; self.length = length; self.is_hf_model = is_hf_model
        self.dummy_embed = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        if is_hf_model: self.config = OmegaConf.create({'hidden_size': hidden_dim, 'attn_backend': 'sdpa', 'cross_attn': False})
    def forward(self, x=None, input_ids=None, indices=None, sigma=None, timesteps=None, **kwargs):
        if self.is_hf_model: inp = input_ids if input_ids is not None else x
        else: inp = indices if indices is not None else x
        if inp is None: raise ValueError("MockBackbone needs input")
        embeds = self.dummy_embed(inp) if inp.dtype == torch.long else inp
        logits = self.fc(embeds)
        return MockOutput(logits=logits, last_hidden_state=embeds) if self.is_hf_model else logits
    def get_input_embeddings(self): return self.dummy_embed
    def reset_kv_cache(self, eval_batch_size=1): pass

class MockLightningTrainer:
    def __init__(self, accumulate_grad_batches=1):
        class MockAcceleratorConnector:
            def __init__(self): self.use_distributed_sampler = False; self.is_distributed = False
        class MockFitLoop:
            def __init__(self):
                class MockCombinedLoader:
                    def __init__(self): self.flattened = []
                self._combined_loader = MockCombinedLoader()
        self._accelerator_connector = MockAcceleratorConnector(); self.fit_loop = MockFitLoop()
        self.accumulate_grad_batches = accumulate_grad_batches; self.sanity_checking = False
        self.ckpt_path = None; self.barebones = False

class MockSpaCySpan:
    def __init__(self, text, parse_string):
        self.text = text; self._ = MagicMock(); self._.parse_string = parse_string
class MockSpaCyDoc:
    def __init__(self, sents_data): self.sents = [MockSpaCySpan(text, ps) for text, ps in sents_data]
class MockSpaCyNLP:
    def __init__(self, sentence_parse_map=None): self.sentence_parse_map = sentence_parse_map or {}
    def __call__(self, text): return MockSpaCyDoc([(text.strip(), self.sentence_parse_map.get(text.strip(), ""))])
    def has_pipe(self, name): return True if name == "benepar" else False # Simple mock for has_pipe
    def add_pipe(self, name, model=None, config=None, **kwargs): pass # More complete mock


class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.mock_main_tokenizer = MockTokenizer(vocab_size_override=30,
                                                bos_token_id=0, eos_token_id=1,
                                                pad_token_id=5, mask_token_id=4, unk_token_id=6)
        cls.real_distilbert_model_name = "distilbert-base-uncased"
        try:
            distilbert_config = AutoConfig.from_pretrained(cls.real_distilbert_model_name)
            cls.distilbert_hidden_size = distilbert_config.hidden_size
        except OSError:
            print(f"Warning: Could not load REAL config for {cls.real_distilbert_model_name} during setUpClass. Defaulting hidden_size to 768 for tests.")
            cls.distilbert_hidden_size = 768

        cls.feature_dim_parts_config_template = {
            "pos_ratio": 1, "semantic": cls.distilbert_hidden_size, "block_entropy": 1,
            "token_variance": 1, "syntactic_depth": 1, "context_entropy": 1,
        }
        cls.total_feature_dim_all_on = sum(cls.feature_dim_parts_config_template.values())

        cls.sampling_config_dict = { 
            'first_hitting': False, 'kv_cache': False, 'nucleus_p': 1.0, 
            'var_length': False, 'logdir': './sample_logs/test_samples.csv', 
            'num_sample_batches': 1, 
        }

        cls.base_config_dict = {
            'mode': 'train',
            'algo': {
                'name': 'bd3lm', 'backbone': 'dit', 'parameterization': 'bd3lm',
                'base_noise_type': 'cosine', 'schedule_clamp_epsilon': 1.0e-6,
                'lambda_steps_penalty': 0.01, 'lambda_s_b_l2_penalty': 0.001,
                'min_alpha_1_target': 0.005, 'lambda_min_alpha_1_penalty': 1.0,
                'alpha_1_clamp_min': 1.0e-5, 'alpha_1_clamp_max': 0.99999,
                'T': 1000, 'ignore_bos': False, 'time_conditioning': True,
                'var_min': False, 'cross_attn': False, 'sampler': 'semi_ar',
                'mdlm_loss_scale': False,
                'feature_extractor_model_name': cls.real_distilbert_model_name,
                'spacy_model_name': "mock-spacy-for-config", 
                'benepar_model_name': "mock-benepar-for-config",
                'clip_search_delta': 0.1, 'clip_search_widths': [0.5], 'fix_clipping': False,
                'meta_controller': {
                    'feature_dim': cls.total_feature_dim_all_on, # This will be adjusted in Diffusion.__init__ if necessary
                    'hidden_dim': 32, 's_tilde_squash_factor': 4.0, 's_min_epsilon': 0.01,
                }
            },
            'model': {
                'length': 64, 'hidden_size': 16, 'attn_backend': 'sdpa',
                'cond_dim': 128, 'n_heads': 4, 'n_blocks': 2, 'dropout': 0.1,
                'tie_word_embeddings': False,
            },
            'training': {
                'antithetic_sampling': False, 'resample': True, 'ema': 0.0, 'stage':'joint',
                'sampling_eps_min': 1e-3, 'sampling_eps_max': 1.0,
                'sampling_eps': [1e-3, 1.0], 'eval_nll': False,
            },
            'sampling': cls.sampling_config_dict,
            'block_size': 16,
            'vocab_size': cls.mock_main_tokenizer.vocab_size, # Will be updated in _create_diffusion_model
            'data': {'tokenizer_name_or_path': 'mock-main-tokenizer'},
            'loader': {'eval_batch_size': 1, 'batch_size':2, 'num_workers': 0, 'pin_memory': False},
            'noise': {'type': 'cosine', 'eps': 1e-3 },
            'eval': {
                'checkpoint_path': "dummy-checkpoint-path",
                'perplexity_batch_size': 1,
                'gen_ppl_eval_model_name_or_path': 'gpt2'
            }
        }
        cls.base_config = OmegaConf.create(cls.base_config_dict)

    def _create_diffusion_model_with_patches(self, config_overrides=None, 
                                          mock_distilbert_model_obj=None, 
                                          mock_distilbert_tokenizer_obj=None, 
                                          mock_spacy_nlp_obj=None):
        current_config = self.base_config.copy()
        if config_overrides:
            current_config = OmegaConf.merge(current_config, OmegaConf.create(config_overrides))
        
        current_config.vocab_size = self.mock_main_tokenizer.vocab_size

        original_automodel_from_pretrained_local = AutoModel.from_pretrained
        original_autotokenizer_from_pretrained_local = AutoTokenizer.from_pretrained
        original_spacy_load_local = spacy.load if 'spacy' in sys.modules else MagicMock(side_effect=OSError("Spacy module not available for original call"))


        def side_effect_automodel(name_or_path, **kwargs):
            if name_or_path == current_config.algo.feature_extractor_model_name:
                if mock_distilbert_model_obj is False: raise OSError(f"Intentionally failed to load model: {name_or_path}")
                if mock_distilbert_model_obj is None: return original_automodel_from_pretrained_local(name_or_path, **kwargs) 
                return mock_distilbert_model_obj
            return original_automodel_from_pretrained_local(name_or_path, **kwargs)

        def side_effect_autotokenizer(name_or_path, **kwargs):
            if name_or_path == current_config.algo.feature_extractor_model_name:
                if mock_distilbert_tokenizer_obj is False: raise OSError(f"Intentionally failed to load tokenizer: {name_or_path}")
                if mock_distilbert_tokenizer_obj is None: return original_autotokenizer_from_pretrained_local(name_or_path, **kwargs)
                return mock_distilbert_tokenizer_obj
            return original_autotokenizer_from_pretrained_local(name_or_path, **kwargs)

        def side_effect_spacy_load(name, **kwargs):
            if name == current_config.algo.spacy_model_name:
                if mock_spacy_nlp_obj is False: raise OSError(f"Intentionally failed to load spacy model: {name}")
                if mock_spacy_nlp_obj is None: return original_spacy_load_local(name, **kwargs)
                return mock_spacy_nlp_obj
            return original_spacy_load_local(name, **kwargs)
            
        with patch('diffusion.AutoModel.from_pretrained', side_effect=side_effect_automodel), \
             patch('diffusion.AutoTokenizer.from_pretrained', side_effect=side_effect_autotokenizer), \
             patch('diffusion.spacy.load', side_effect=side_effect_spacy_load):
            
            model = Diffusion(current_config, tokenizer=self.mock_main_tokenizer)

        model.backbone = MockBackbone(
            vocab_size=model.vocab_size,
            hidden_dim=current_config.model.hidden_size,
            length=current_config.model.length,
            is_hf_model=(current_config.algo.backbone == 'hf_dit')
        ).to(self.device)
        
        if mock_spacy_nlp_obj is False : model.nlp = None
        elif mock_spacy_nlp_obj is not None and model.nlp != mock_spacy_nlp_obj : model.nlp = mock_spacy_nlp_obj
        
        if mock_distilbert_model_obj is False : model.distilbert_model = None
        elif mock_distilbert_model_obj is not None and model.distilbert_model != mock_distilbert_model_obj: model.distilbert_model = mock_distilbert_model_obj
            
        if mock_distilbert_tokenizer_obj is False : model.distilbert_tokenizer = None
        elif mock_distilbert_tokenizer_obj is not None and model.distilbert_tokenizer != mock_distilbert_tokenizer_obj: model.distilbert_tokenizer = mock_distilbert_tokenizer_obj


        model = model.to(self.device)
        model.trainer = MockLightningTrainer()
        return model, current_config

    def test_get_parse_tree_height(self):
        print("\nTesting _get_parse_tree_height...")
        model, _ = self._create_diffusion_model_with_patches() 
        self.assertEqual(model._get_parse_tree_height("(S (A B))"), 3)
        self.assertEqual(model._get_parse_tree_height("(S (C (D E)))"), 4)
        simple_parse = "(S (NP (DT The) (NN cat)) (VP (VBD sat)))"
        self.assertEqual(model._get_parse_tree_height(simple_parse), 4)
        complex_parse = "(S (NP (DT The) (JJ big) (NN cat)) (VP (VBD sat) (PP (IN on) (NP (DT the) (NN mat)))))"
        self.assertEqual(model._get_parse_tree_height(complex_parse), 6)
        self.assertEqual(model._get_parse_tree_height(""), 0)
        self.assertEqual(model._get_parse_tree_height("(S (NP cat"), 0) # Malformed
        print("_get_parse_tree_height tests passed.")

    def test_syntactic_depth_feature(self):
        print("\nTesting syntactic_depth_feature integration...")
        B, N_Tokens, Blk_Size = 1, 32, 16; num_blocks = N_Tokens // Blk_Size
        valid_token_ids = [i + 7 for i in range(N_Tokens)] 
        tokens = torch.tensor([valid_token_ids], device=self.device)
        attn_mask = torch.ones_like(tokens)
        text_block0 = self.mock_main_tokenizer.decode(tokens[0, 0:Blk_Size].cpu().tolist(), skip_special_tokens=True).strip()
        text_block1 = self.mock_main_tokenizer.decode(tokens[0, Blk_Size:2*Blk_Size].cpu().tolist(), skip_special_tokens=True).strip()

        print("  Case 1: nlp is None (syntactic_depth feature should have zero contribution)")
        expected_dim_no_nlp = self.total_feature_dim_all_on - self.feature_dim_parts_config_template["syntactic_depth"]
        cfg_no_nlp_override = {'algo': {'meta_controller': {'feature_dim': expected_dim_no_nlp}}}
        model_no_nlp, cfg_no_nlp = self._create_diffusion_model_with_patches(
            config_overrides=cfg_no_nlp_override, mock_spacy_nlp_obj=None 
        )
        self.assertIsNone(model_no_nlp.nlp)
        self.assertEqual(model_no_nlp.feature_dim_parts["syntactic_depth"], 0)
        self.assertEqual(model_no_nlp.meta_controller_config.feature_dim, expected_dim_no_nlp)
        features_no_nlp = model_no_nlp._get_block_features(tokens, attn_mask)
        self.assertEqual(features_no_nlp.shape[-1], expected_dim_no_nlp)

        print("  Case 2: nlp active, successful parse")
        mock_parse_db = {text_block0: "(S (A B))", text_block1: "(S (C (D E)))"} # Heights: 3, 4
        active_nlp = MockSpaCyNLP(sentence_parse_map=mock_parse_db)
        cfg_with_nlp_override = {'algo': {'meta_controller': {'feature_dim': self.total_feature_dim_all_on}}}
        model_with_nlp, _ = self._create_diffusion_model_with_patches(
            config_overrides=cfg_with_nlp_override, mock_spacy_nlp_obj=active_nlp
        )
        self.assertIsNotNone(model_with_nlp.nlp)
        self.assertEqual(model_with_nlp.feature_dim_parts["syntactic_depth"], 1)
        self.assertEqual(model_with_nlp.meta_controller_config.feature_dim, self.total_feature_dim_all_on)
        
        ordered_keys = ["pos_ratio", "semantic", "block_entropy", "token_variance", "syntactic_depth", "context_entropy"]
        syntactic_depth_idx_offset = 0
        for key_idx, k_feat in enumerate(ordered_keys):
            if k_feat == "syntactic_depth":
                break
            syntactic_depth_idx_offset += model_with_nlp.feature_dim_parts[k_feat]
        
        features_with_nlp = model_with_nlp._get_block_features(tokens, attn_mask)
        self.assertEqual(features_with_nlp.shape, (B, num_blocks, self.total_feature_dim_all_on))
        self.assertAlmostEqual(features_with_nlp[0, 0, syntactic_depth_idx_offset].item(), 3.0, places=5)
        self.assertAlmostEqual(features_with_nlp[0, 1, syntactic_depth_idx_offset].item(), 4.0, places=5)
        print("Syntactic depth feature tests passed.")

    def test_context_entropy_feature(self):
        print("\nTesting context_entropy_feature integration...")
        B, N_Tokens, Blk_Size = 1, 48, 16; num_blocks = N_Tokens // Blk_Size
        model, _ = self._create_diffusion_model_with_patches() 

        tok7_id = self.mock_main_tokenizer._convert_token_to_id('token7')
        tok8_id = self.mock_main_tokenizer._convert_token_to_id('token8')
        block0_toks = [tok7_id] * Blk_Size; block1_toks = [tok8_id] * Blk_Size
        block2_toks = [tok7_id] * (Blk_Size // 2) + [tok8_id] * (Blk_Size // 2)
        tokens = torch.tensor([block0_toks + block1_toks + block2_toks], device=self.device)
        attn_mask = torch.ones_like(tokens)
        features = model._get_block_features(tokens, attn_mask)
        
        ordered_keys = ["pos_ratio", "semantic", "block_entropy", "token_variance", "syntactic_depth", "context_entropy"]
        context_entropy_idx_offset = 0
        for key_idx, k_feat in enumerate(ordered_keys):
            if k_feat == "context_entropy":
                break
            context_entropy_idx_offset += model.feature_dim_parts[k_feat]

        self.assertEqual(features.shape, (B, num_blocks, model.meta_controller_config.feature_dim))
        self.assertAlmostEqual(features[0, 0, context_entropy_idx_offset].item(), 0.0, places=5, msg="Block 0 context entropy")
        self.assertAlmostEqual(features[0, 1, context_entropy_idx_offset].item(), 0.0, places=5, msg="Block 1 context entropy")
        expected_ent_blk2_ctx = -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
        self.assertAlmostEqual(features[0, 2, context_entropy_idx_offset].item(), expected_ent_blk2_ctx, places=5, msg="Block 2 context entropy")
        print("Context entropy feature tests passed.")

    def test_feature_dim_consistency(self):
        print("\nTesting overall feature dimension consistency...")
        B, N_Tokens, Blk_Size = 1, 32, 16
        tokens = torch.randint(7, 15, (B, N_Tokens), device=self.device) 
        attn_mask = torch.ones_like(tokens)

        mock_distil_model_obj = MagicMock(spec=AutoModel)
        mock_distil_model_obj.config = MagicMock(hidden_size=self.distilbert_hidden_size)
        mock_distil_model_obj.device = self.device 
        mock_distil_model_obj.parameters = MagicMock(return_value=iter([])) 
        mock_distil_model_obj.eval = MagicMock()
        mock_distil_model_obj.to = MagicMock(return_value=mock_distil_model_obj) # Fix AttributeError

        dummy_logits_for_mock_output = torch.randn(B * (N_Tokens // Blk_Size), 5, self.mock_main_tokenizer.vocab_size, device=self.device)
        mock_distil_model_obj.return_value = MockOutput(
            logits=dummy_logits_for_mock_output,
            last_hidden_state=torch.randn(B * (N_Tokens // Blk_Size), 5, self.distilbert_hidden_size, device=self.device)
        )
        mock_distil_tokenizer_obj = MagicMock(spec=AutoTokenizer)
        mock_distil_tokenizer_obj.model_max_length = 128
        mock_distil_tokenizer_obj.pad_token = "[PAD]"
        
        class MockBatchEncoding(dict):
            def to(self, device):
                for key, value in self.items():
                    if isinstance(value, torch.Tensor):
                        self[key] = value.to(device)
                return self

        def tokenizer_side_effect(texts, padding, truncation, max_length, return_tensors):
            num_texts = len(texts)
            data = {
                'input_ids': torch.randint(0, 100, (num_texts, 10), device='cpu'), # Create on CPU
                'attention_mask': torch.ones((num_texts, 10), device='cpu')
            }
            return MockBatchEncoding(data) # Return our mock object

        mock_distil_tokenizer_obj.side_effect = tokenizer_side_effect
        mock_distil_tokenizer_obj.__call__ = mock_distil_tokenizer_obj.side_effect
        
        mock_nlp_active = MockSpaCyNLP()

        print("  Case 1: All features on")
        expected_dim_case1 = self.total_feature_dim_all_on
        cfg_override_case1 = {'algo': {'meta_controller': {'feature_dim': expected_dim_case1}}}
        model_all_on, _ = self._create_diffusion_model_with_patches(
            config_overrides=cfg_override_case1,
            mock_distilbert_model_obj=mock_distil_model_obj,
            mock_distilbert_tokenizer_obj=mock_distil_tokenizer_obj,
            mock_spacy_nlp_obj=mock_nlp_active
        )
        self.assertIsNotNone(model_all_on.distilbert_model)
        self.assertIsNotNone(model_all_on.nlp)
        self.assertEqual(model_all_on.meta_controller_config.feature_dim, expected_dim_case1)
        features_all_on = model_all_on._get_block_features(tokens, attn_mask)
        self.assertEqual(features_all_on.shape[-1], expected_dim_case1)

        print("  Case 2: Syntactic depth off (simulating nlp=None)")
        expected_dim_case2 = self.total_feature_dim_all_on - self.feature_dim_parts_config_template["syntactic_depth"]
        cfg_override_case2 = {'algo': {'meta_controller': {'feature_dim': expected_dim_case2}}}
        model_nlp_off, _ = self._create_diffusion_model_with_patches(
            config_overrides=cfg_override_case2,
            mock_distilbert_model_obj=mock_distil_model_obj,
            mock_distilbert_tokenizer_obj=mock_distil_tokenizer_obj,
            mock_spacy_nlp_obj=None 
        )
        self.assertIsNone(model_nlp_off.nlp) 
        self.assertEqual(model_nlp_off.meta_controller_config.feature_dim, expected_dim_case2)
        features_nlp_off = model_nlp_off._get_block_features(tokens, attn_mask)
        self.assertEqual(features_nlp_off.shape[-1], expected_dim_case2)

        print("  Case 3: Semantic feature off (simulating distilbert load failure)")
        expected_dim_case3 = self.total_feature_dim_all_on - self.feature_dim_parts_config_template["semantic"]
        cfg_override_case3 = {
            'algo': {
                'meta_controller': {'feature_dim': expected_dim_case3},
                'feature_extractor_model_name': 'distilbert-will-fail-to-load' 
            }
        }
        model_sem_off, _ = self._create_diffusion_model_with_patches(
            config_overrides=cfg_override_case3,
            mock_distilbert_model_obj=False, 
            mock_distilbert_tokenizer_obj=False, 
            mock_spacy_nlp_obj=mock_nlp_active 
        )
        self.assertIsNone(model_sem_off.distilbert_model)
        self.assertIsNone(model_sem_off.distilbert_tokenizer)
        self.assertEqual(model_sem_off.meta_controller_config.feature_dim, expected_dim_case3)
        features_sem_off = model_sem_off._get_block_features(tokens, attn_mask)
        self.assertEqual(features_sem_off.shape[-1], expected_dim_case3)

        print("Feature dimension consistency tests passed.")


if __name__ == '__main__':
    print("Running tests for new features (syntactic depth, context entropy)...")
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(TestFeatureEngineering))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)