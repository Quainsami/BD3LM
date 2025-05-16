import itertools
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from collections import OrderedDict

import dataloader
import metrics
import models
import noise_schedule
import utils
import numpy as np
import itertools

from models.meta_controller import MetaController

def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
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
        self.cross_attn = self.config.algo.cross_attn
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
          self.backbone = models.dimamba.DiMamba(
            self.config,
            vocab_size=self.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id)
        elif self.config.algo.backbone == 'hf_dit':
          self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
            config.eval.checkpoint_path, trust_remote_code=True)
          #  egenerate mask if pretrained model uses flex attention mask
          # and current model uses sdpa mask
          if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
            self.config.model.attn_backend == 'sdpa':
            self.backbone.config.attn_backend = 'sdpa'
            for i in self.backbone.backbone.blocks:
              i.attn_backend = 'sdpa'
            self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
        else:
          raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

        self.T = self.config.algo.T
        self.num_tokens = self.config.model.length

        self.noise = noise_schedule.get_noise(self.config)
        self.metrics = metrics.Metrics(config)

        if self.config.training.ema > 0:
          self.ema = models.ema.ExponentialMovingAverage(
            self._get_parameters(),
            decay=self.config.training.ema)
        else:
          self.ema = None

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
        self._validate_configuration()
            # Meta-Controller and related schedule warping parameters
        self.meta_controller_config = config.algo.meta_controller
        self.meta_controller = MetaController(config) # Pass the full config

        self.base_noise_schedule = noise_schedule.get_noise(config, config.algo.base_noise_type)
        # Ensure base_noise_schedule has schedule_clamp_epsilon
        self.base_noise_schedule.schedule_clamp_epsilon = config.algo.schedule_clamp_epsilon

        # Pre-calculate scalar constants on the correct device
        # These will be properly moved to device by PyTorch Lightning
        _dummy_zero_t = torch.tensor(0.0, dtype=torch.float32) # device will be set by to()
        clamped_base_alpha_at_0 = self.base_noise_schedule.get_alpha_bar(_dummy_zero_t)

        self.register_buffer("target_alpha_at_0", torch.tensor(1.0 - config.algo.schedule_clamp_epsilon, dtype=torch.float32))
        self.register_buffer("base_log_alpha_bar_at_0", torch.logit(clamped_base_alpha_at_0))

        # For penalty calculation
        self.min_alpha_1_target = self.meta_controller_config.min_alpha_1_target
        self.lambda_min_alpha_1_penalty = self.meta_controller_config.lambda_min_alpha_1_penalty
        self.alpha_1_clamp_min = self.meta_controller_config.alpha_1_clamp_min
        self.alpha_1_clamp_max = self.meta_controller_config.alpha_1_clamp_max
        self.lambda_steps_penalty = self.config.algo.lambda_steps_penalty # Renamed from lambda_steps

        if hasattr(self.config.algo, 'lambda_s_b_l2_penalty'):
            self.lambda_s_b_l2_penalty = self.config.algo.lambda_s_b_l2_penalty
        else:
            self.lambda_s_b_l2_penalty = 0.0


    def _get_parameters(self): # Modified to support training stages
        params_to_optimize = []
        training_stage = getattr(self.config.training, "stage", "joint") # Default to joint

        if training_stage == "controller_only":
            print("Optimizer: MetaController parameters only.")
            params_to_optimize.append(self.meta_controller.parameters())
        elif training_stage == "joint":
            print("Optimizer: Jointly training LM and MetaController.")
            params_to_optimize.append(self.backbone.parameters())
            # self.noise parameters are usually not learnable for fixed schedules like CosineNoise
            # If self.noise (global schedule) becomes learnable, add it here.
            # For now, assume self.base_noise_schedule is fixed.
            if hasattr(self.noise, 'parameters') and any(p.requires_grad for p in self.noise.parameters()):
                 params_to_optimize.append(self.noise.parameters())
            params_to_optimize.append(self.meta_controller.parameters())
        elif training_stage == "lm_only": # For initial LM pretraining
            print("Optimizer: LM parameters only.")
            params_to_optimize.append(self.backbone.parameters())
            if hasattr(self.noise, 'parameters') and any(p.requires_grad for p in self.noise.parameters()):
                 params_to_optimize.append(self.noise.parameters())
        else:
            raise ValueError(f"Unknown training stage: {training_stage}")

        return itertools.chain(*params_to_optimize)
    def _get_block_features(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for extracting block features.
        Args:
            tokens: (B, N_Tokens)
            attention_mask: (B, N_Tokens)
        Returns:
            block_features: (B, N_Blk, FeatDim)
        """
        B, N_Tokens = tokens.shape
        num_blocks = N_Tokens // self.block_size

        # For now, a simple feature: normalized block index
        block_indices_feat = torch.arange(num_blocks, device=tokens.device, dtype=torch.float32)
        if num_blocks > 1:
            block_indices_feat = block_indices_feat / (num_blocks - 1 + 1e-9) # Normalize to [0,1]
        else: # Avoid division by zero if only one block
            block_indices_feat = torch.zeros_like(block_indices_feat)

        block_indices_feat = block_indices_feat.view(1, num_blocks, 1).expand(B, -1, -1)

        # TODO: Add more features:
        # - block length (always self.block_size here if no padding issues)
        # - rarity score (mean unigram log-freq) - requires vocab freq map
        # - syntactic complexity (avg. parse depth) - requires external parser
        # - running entropy of preceding context - requires model fwd pass or approximation

        # Example: Add mean token ID in block (as a dummy feature)
        # reshaped_tokens = tokens.view(B, num_blocks, self.block_size).float()
        # mean_token_id_feat = reshaped_tokens.mean(dim=2, keepdim=True) / self.vocab_size # Normalized
        # block_features = torch.cat([block_indices_feat, mean_token_id_feat], dim=-1)

        return block_indices_feat # Current feature_dim = 1
    def _get_warped_noise_outputs_for_block_batch(
        self,
        x0_block_features: torch.Tensor, # (B, N_Blk, FeatDim)
        t_values_per_block: torch.Tensor # (B, N_Blk) - these are the "u_j" values
    ):
        """
        Vectorized calculation of warped schedule outputs for all blocks in a batch.
        Args:
            x0_block_features: (B, N_Blk, FeatDim)
            t_values_per_block: (B, N_Blk) - abstract time `u` for each block
        Returns:
            alpha_b_t_all_blocks: (B, N_Blk)
            loss_scale_b_t_all_blocks: (B, N_Blk)
            p_b_t_all_blocks: (B, N_Blk)
            s_b_all_blocks: (B, N_Blk, 1) - scale factor for penalty
        """
        log_s_tilde_b = self.meta_controller(x0_block_features) # (B, N_Blk, 1)
        s_b = self.meta_controller.get_s_b(log_s_tilde_b)       # (B, N_Blk, 1)

        # Expand t_values for base schedule methods if they expect (B, N_Blk, N_t_points)
        # Here t_values_per_block is already (B, N_Blk), representing one t point per block
        # So base_alpha_bar_t and base_log_alpha_bar_base_derivative_t will be (B, N_Blk)

        # .unsqueeze(-1) to make it (B, N_Blk, 1) like s_b if base methods handle it
        # Or, base methods might handle (B, N_Blk) directly. Let's assume they do.
        base_alpha_bar_t = self.base_noise_schedule.get_alpha_bar(t_values_per_block)
        base_log_alpha_bar_base_derivative_t = self.base_noise_schedule.get_log_alpha_bar_base_derivative_t(t_values_per_block)

        # Unsqueeze s_b to match dimensions if base_alpha_bar_t is (B, N_Blk) rather than (B, N_Blk, 1)
        # get_warped_schedule_outputs expects s_b as (B, N_Blk, 1)
        # and base_alpha_bar_t etc with compatible trailing dimensions.
        # If base_alpha_bar_t is (B,N_Blk), then for get_warped_schedule_outputs:
        #   base_alpha_bar_t -> (B, N_Blk)
        #   base_log_alpha_bar_base_derivative_t -> (B, N_Blk)
        #   s_b -> (B, N_Blk, 1)
        # This means we need to unsqueeze base_alpha_bar_t and its derivative before passing or adjust inside.
        # Let's assume get_warped_schedule_outputs handles broadcasting from (B,N_Blk,1) for s_b and (B,N_Blk) for others.
        # The implementation of get_warped_alpha_b_t was:
        # A_b_t = s_b * (L_base_t - self.base_log_alpha_bar_at_0.unsqueeze(-1).unsqueeze(-1)) + \
        #         self.target_log_alpha_at_0.unsqueeze(-1).unsqueeze(-1)
        # If L_base_t is (B,N_Blk) and s_b is (B,N_Blk,1), this will fail.
        # s_b should be (B,N_Blk) or (B,N_Blk,1) and L_base_t (B,N_Blk)

        # Let's adjust s_b to be (B, N_Blk) for easier broadcasting with (B, N_Blk) time-dependent terms.
        s_b_squeezed = s_b.squeeze(-1) # (B, N_Blk)

        alpha_b_t_all_blocks, loss_scale_b_t_all_blocks, p_b_t_all_blocks = \
            noise_schedule.get_warped_schedule_outputs(
                base_alpha_bar_t, # (B, N_Blk)
                base_log_alpha_bar_base_derivative_t, # (B, N_Blk)
                self.base_log_alpha_bar_at_0, # scalar
                s_b_squeezed, # (B, N_Blk) - modified get_warped_schedule_outputs to take this
                self.target_alpha_at_0 # scalar - this should be logit(target_alpha_at_0)
            )

        # The get_warped_schedule_outputs needs logit(target_alpha_at_0)
        # We have self.target_alpha_at_0 (value) and self.base_log_alpha_bar_at_0 (logit)
        # We need logit(self.target_alpha_at_0)
        # Let's pass self.target_alpha_at_0 and do logit inside get_warped_alpha_b_t if it's not already logit

        # Re-check noise_schedule.get_warped_alpha_b_t:
        # It takes target_log_alpha_at_0. So, self.target_alpha_at_0 needs to be logit-transformed before passing.
        # self.register_buffer("target_log_alpha_at_0_val", torch.logit(self.target_alpha_at_0)) # in __init__

        alpha_b_t_all_blocks, loss_scale_b_t_all_blocks, p_b_t_all_blocks = \
            noise_schedule.get_warped_schedule_outputs(
                base_alpha_bar_t,
                base_log_alpha_bar_base_derivative_t,
                self.base_log_alpha_bar_at_0,
                s_b_squeezed, # (B, N_Blk)
                torch.logit(self.target_alpha_at_0) # Pass the logit directly
            )

        return alpha_b_t_all_blocks, loss_scale_b_t_all_blocks, p_b_t_all_blocks, s_b # Return original s_b for penalty

    def on_validation_model_zero_grad(self) -> None:
        '''
        Small hack to avoid first validation on resume.
        This will NOT work if the gradient accumulation step should be performed at this point.
        '''
        super().on_validation_model_zero_grad()
        if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
            self.trainer.sanity_checking = True
            self._restarting_skip_val_flag = False

    def _validate_configuration(self):
        if self.config.mode == 'sample_eval' and \
            self.config.sampling.first_hitting:
          assert self.config.loader.eval_batch_size == 1
        assert self.config.algo.backbone in {
          'dit', 'ar', 'hf_dit'}
        if self.config.algo.parameterization == 'ar':
          assert not self.config.algo.time_conditioning
        if self.config.sampling.kv_cache:
          assert self.config.algo.name in {'ar', 'bd3lm'}

        if self.parameterization in {'sedd'}:
          assert self.time_conditioning

        if self.config.mode == 'sample_eval':
          assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
        if self.config.model.attn_backend == 'flex':
          assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
          self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
        elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
          self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
        if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
          self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
          self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
        return self

    def _replace_ckpt_keys(self, checkpoint):
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
          new_state_dict[k.replace('_orig_mod.', '')] = v
        checkpoint['state_dict'] = new_state_dict
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        print('Loading checkpoint at', checkpoint['global_step'])
        self._restarting_skip_val_flag = True

        # for models compiled with `torch.compile`
        if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
          checkpoint = self._replace_ckpt_keys(checkpoint)

        if self.ema:
          self.ema.load_state_dict(checkpoint['ema'])
        if 'sampling_eps_min' in checkpoint.keys():
          self.sampling_eps_min = checkpoint['sampling_eps_min']
          self.sampling_eps_max = checkpoint['sampling_eps_max']
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
          'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
          'fit_loop']['epoch_loop.batch_progress'][
            'current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
          checkpoint['ema'] = self.ema.state_dict()
        if hasattr(self, 'sampling_eps_min'):
          checkpoint['sampling_eps_min'] = self.sampling_eps_min
          checkpoint['sampling_eps_max'] = self.sampling_eps_max
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
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
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
          'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
              'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total']['completed']
        if 'sampler' not in checkpoint.keys():
          checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler,
                   'state_dict'):
          sampler_state_dict = self.trainer.\
            train_dataloader.sampler.state_dict()
          checkpoint['sampler'][
            'random_state'] = sampler_state_dict.get(
              'random_state', None)
        else:
          checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
        if self.ema:
          self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
          self.trainer._accelerator_connector.use_distributed_sampler
          and self.trainer._accelerator_connector.is_distributed)
        if distributed:
          sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
          sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
          if hasattr(dl.sampler, 'shuffle'):
            dl_sampler = sampler_cls(
              dl.dataset, shuffle=dl.sampler.shuffle)
          else:
            dl_sampler = sampler_cls(dl.dataset)
          if (distributed
              and self.fast_forward_epochs is not None
              and self.fast_forward_batches is not None):
            dl_sampler.load_state_dict({
              'epoch': self.fast_forward_epochs,
              'counter': (self.fast_forward_batches
                          * self.config.loader.batch_size)})
          updated_dls.append(
            torch.utils.data.DataLoader(
              dl.dataset,
              batch_size=self.config.loader.batch_size,
              num_workers=self.config.loader.num_workers,
              pin_memory=self.config.loader.pin_memory,
              sampler=dl_sampler,
              shuffle=False,
              persistent_workers=True))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
          self.ema.update(self._get_parameters())

    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1,
                                          keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _sedd_parameterization(self, logits, xt, sigma):
        esigm1_log = torch.where(
          sigma < 0.5,
          torch.expm1(sigma),
          sigma.exp() - 1).log().to(logits.dtype)
        # logits shape
        # (batch_size, diffusion_model_input_length, vocab_size)
        logits = logits - esigm1_log[:, None, None] - np.log(
          logits.shape[-1] - 1)
        # The below scatter operation sets the log score
        # for the input word to 0.
        logits = torch.scatter(logits, -1, xt[..., None],
                               torch.zeros_like(logits[..., :1]))
        return logits

    def _process_sigma(self, sigma): # Modified to handle (B,N_T) and average if needed
        # Input sigma can be (B, N_T) or (B, N_T, 1) from _forward_pass_diffusion
        # The backbone (HF DiT) expects timesteps of shape (B)
        if sigma.ndim == 3 and sigma.shape[-1] == 1: # (B, N_T, 1)
            sigma = sigma.squeeze(-1) # (B, N_T)

        if self.parameterization == 'ar':
            return None # AR models don't use sigma for backbone

        # Average over sequence length if sigma is per-token
        if sigma.ndim == 2: # (B, N_T)
            sigma_for_backbone = sigma.mean(dim=-1) # (B)
        elif sigma.ndim == 1: # Already (B)
            sigma_for_backbone = sigma
        else:
            raise ValueError(f"Unexpected sigma shape: {sigma.shape}")

        if not self.time_conditioning:
            sigma_for_backbone = torch.zeros_like(sigma_for_backbone)

        assert sigma_for_backbone.ndim == 1, sigma_for_backbone.shape
        return sigma_for_backbone

    def forward(self, x, sigma, sample_mode=False, store_kv=False):
        """Returns log score."""
        sigma = self._process_sigma(sigma)
        with torch.amp.autocast('cuda', dtype=torch.float32):
          if self.config.algo.name == 'bd3lm':
            logits = self.backbone(x, sigma,
                                  store_kv=store_kv,
                                  sample_mode=sample_mode)
          elif self.config.algo.name == 'ar':
            if self.config.algo.backbone == 'hf_dit':
              logits = self.backbone(x, None)
            else:
              logits = self.backbone(x, sigma, sample_mode=sample_mode, store_kv=store_kv)
            logits[:, :, self.mask_index] = self.neg_infinity
            logits = logits.log_softmax(-1)
          else:
            logits = self.backbone(x, sigma)

        if self.cross_attn:
          x = x[:, :self.config.model.length]
        if self.parameterization == 'subs':
          return self._subs_parameterization(logits=logits,
                                          xt=x)
        elif self.parameterization == 'sedd':
          return self._sedd_parameterization(logits=logits,
                                            xt=x,
                                            sigma=sigma)
        return logits

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()
        self.metrics.reset()
        assert self.metrics.train_nlls.nll.mean_value == 0
        assert self.metrics.train_nlls.nll.weight == 0

    def training_step(self, batch, batch_idx):
        del batch_idx
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'])
        self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
        self.log(name='trainer/loss',
                 value=losses.loss.item(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return losses.loss

    def on_validation_epoch_start(self):
        self.metrics.reset()
        if self.ema:
          self.ema.store(itertools.chain(
            self.backbone.parameters(),
            self.noise.parameters()))
          self.ema.copy_to(itertools.chain(
            self.backbone.parameters(),
            self.noise.parameters()))
        self.eval()
        self.backbone.eval()
        self.noise.eval()
        assert self.metrics.valid_nlls.nll.mean_value == 0
        assert self.metrics.valid_nlls.nll.weight == 0
        self.sampling_eps = self.config.training.sampling_eps

    def on_validation_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
          self.log(name=k,  value=v.compute(), on_step=False,
                  on_epoch=True, sync_dist=True)
        if self.ema:
          self.ema.restore(self._get_parameters())
        if self.var_min and not self.trainer.sanity_checking:
          self._clipped_schedule_search()
          self.log('sampling_eps_min',
                   self.sampling_eps_min,
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
          self.log('sampling_eps_max',
                   self.sampling_eps_max,
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)

    def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
        """Checks if the current sampling interval is valid for reporting likelihood."""
        if (sampling_eps_min == 1e-3 \
            and sampling_eps_max == 1 \
            and not (self.block_size == 1 and self.config.training.eval_nll)):
          return True # elbo
        elif (self.block_size == 1 and sampling_eps_min >= 1):
          return True # nll (block size 1)
        return False # not a valid elbo (biased estimate)

    def validation_step(self, batch, batch_idx):
        if self.var_min:
          for noise_clip_start in self.metrics.valid_vars.keys():
            sampling_eps_min, sampling_eps_max = noise_clip_start
            if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max) == True:
              # compute and record nelbo
              losses_clip = self._loss(batch['input_ids'],
                                batch['attention_mask'],
                                sampling_eps_min=sampling_eps_min,
                                sampling_eps_max=sampling_eps_max)
              losses = Loss(
                nlls=losses_clip.nlls.clone(),
                token_mask=losses_clip.token_mask,
                loss=losses_clip.loss.clone())
            elif len(self.metrics.valid_vars[noise_clip_start]) < 100:
              # elbo from clipped schedule (biased estimate)
              losses_clip = self._loss(batch['input_ids'],
                                batch['attention_mask'],
                                sampling_eps_min=sampling_eps_min,
                                sampling_eps_max=sampling_eps_max)
            if len(self.metrics.valid_vars[noise_clip_start]) < 100:
              # only report variance over 100 batches
              nlls = losses_clip.nlls
              self.metrics.valid_vars[noise_clip_start].append(
                nlls.reshape(
                  nlls.shape[0], -1, self.block_size).mean(-1))
        elif self.block_size == 1:
          # nll
          losses = self._loss(batch['input_ids'],
                              batch['attention_mask'],
                              sampling_eps_min=1,
                              sampling_eps_max=1)
        else:
          # nelbo
          losses = self._loss(batch['input_ids'],
                              batch['attention_mask'],
                              sampling_eps_min=1e-3,
                              sampling_eps_max=1)
        self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
        return losses.loss

    def configure_optimizers(self):
        # TODO(yair): Lightning currently giving this warning when using `fp16`:
        #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        #  Not clear if this is a problem or not.
        #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
        optimizer = torch.optim.AdamW(
          self._get_parameters(),
          lr=self.config.optim.lr,
          betas=(self.config.optim.beta1,
                 self.config.optim.beta2),
          eps=self.config.optim.eps,
          weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
          self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step',
                          'monitor': 'val/loss',
                          'name': 'trainer/lr'}
        return [optimizer], [scheduler_dict]

    def _resample_q_xt(
        self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
        """Resamples x_t if the percentage of masked tokens is outside the bounds
        defined by sampling_eps_min and sampling_eps_max."""
        perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
        while (perc_masked < sampling_eps_min).any() or \
          (perc_masked > sampling_eps_max).any():
          # if a bound is epsilon, don't resample
          if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
            regen_idx = (perc_masked > sampling_eps_max)
            if regen_idx.max() == 0:
              break
          elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
            regen_idx = (perc_masked < sampling_eps_min)
            if regen_idx.max() == 0:
              break
          elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
            regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
          regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
          move_indices[regen_idx] = (torch.rand(
            * x.shape, device=x.device) < p)[regen_idx]
          xt = torch.where(move_indices, self.mask_index, x)
          xt = xt.reshape(xt.shape[0], -1, block_size)
          perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
        return xt

    def q_xt(self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None): # p can be (B,N_T,1)
        """Computes the noisy sample xt.
        Args:
            x: int torch.Tensor with shape (batch_size, N_Tokens), input.
            p: float torch.Tensor with shape (batch_size, N_Tokens, 1) or (batch_size, 1). Per-token move chance.
        """
        if block_size is None: # block_size for resampling logic, not for p application
            block_size_resample = self.block_size
        else:
            block_size_resample = block_size # if explicitly passed for other use cases

        if p.ndim == 3 and p.shape[-1] == 1:
            p_broadcastable = p.squeeze(-1) # (B, N_Tokens) for element-wise comparison with x
        elif p.ndim == 2 and p.shape[1] == x.shape[1]: # (B, N_Tokens)
            p_broadcastable = p
        elif p.ndim == 2 and p.shape[1] == 1: # (B, 1)
            p_broadcastable = p.expand_as(x) # Broadcast to (B, N_Tokens)
        else:
            raise ValueError(f"Unsupported shape for p: {p.shape}, x shape: {x.shape}")

        move_indices = torch.rand_like(x, dtype=torch.float32) <= p_broadcastable # uses float comparison
        xt = torch.where(move_indices, self.mask_index, x)

        if block_size_resample == 1 and sampling_eps_min is not None and sampling_eps_min >= 1.0:
             # This check is usually for NLL calculation where all tokens are masked.
             # The `p` passed here would be derived from t=1.
             # If p_broadcastable corresponding to t=1 is indeed 1.0 everywhere, xt will be all masks.
             # If p_broadcastable is not 1.0 (e.g. due to clamping in warped schedule), then xt won't be all masks.
             # Forcing it here ensures consistency with the NLL interpretation.
            return torch.full_like(x, self.mask_index)

        # Resampling logic needs to be carefully considered with per-token `p`.
        # The original _resample_q_xt used a single `p` for the whole sequence.
        # If resampling is block-wise, then `p` should be constant within a block for that logic.
        # Current `p_per_token` is constant within a block due to `t_samples_per_block`.
        if self.config.training.resample and \
           not (sampling_eps_min is not None and sampling_eps_min <= 1e-3 and \
                sampling_eps_max is not None and sampling_eps_max >= 1.0):
            # Need sampling_eps_min_val_for_no_resample, etc. from config or define them.
            # Let's assume these are 1e-3 and 1.0 for standard ELBO.
            # no_resample_flag = (sampling_eps_min is not None and sampling_eps_min <= 1e-3 and \
            #                     sampling_eps_max is not None and sampling_eps_max >= 1.0)

            # if not no_resample_flag:
            # Reshape xt for block-wise processing. p_broadcastable is also effectively block-wise constant.
            xt_blocks = xt.view(xt.shape[0], -1, block_size_resample)
            # Take the p value for the first token in each block for the resampling logic
            p_for_resampling_blocks = p_broadcastable[:, ::block_size_resample].view(xt.shape[0],-1,1)

            xt_blocks = self._resample_q_xt(
                x.view_as(xt_blocks), # x also needs to be block-viewed
                xt_blocks,
                move_indices.view_as(xt_blocks), # move_indices also needs to be block-viewed for initial state
                p_for_resampling_blocks, # Pass block-constant p
                block_size_resample,
                sampling_eps_min if sampling_eps_min is not None else 1e-3, # Fallback values
                sampling_eps_max if sampling_eps_max is not None else 1.0
            )
            xt = xt_blocks.reshape(xt.shape[0], -1)
        return xt



    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(
          * batch_dims, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def _nucleus_sample(self, p_x0):
        p = self.config.sampling.nucleus_p
        if p == 1.0:
          return p_x0
        p_x0_ = p_x0[:, -self.block_size:].clone()
        sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cum_probs <= p
        nucleus_mask[..., 0] = 1
        sorted_probs = sorted_probs * nucleus_mask
        p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
        p_x0_ /= p_x0_.sum(-1, keepdim=True)
        p_x0[:, -self.block_size:] = p_x0_
        return p_x0

    @torch.no_grad()
    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        _, move_chance_t = self.noise(t)
        _, move_chance_s = self.noise(t - dt)
        sigma_t = self._sigma_from_p(move_chance_t)
        move_chance_t = move_chance_t[:, None]
        move_chance_s = move_chance_s[:, None]
        mask_prob = move_chance_s / move_chance_t

        if p_x0 is None:
          if self.config.sampling.kv_cache:
            p_x0 = self.forward(x[:, -self.block_size:],
                            sigma_t,
                            sample_mode=True).to(torch.float64)
          else:
            p_x0 = self.forward(x,
                              sigma_t,
                              sample_mode=True).to(torch.float64)
            p_x0 = p_x0[:, -self.block_size:]
          p_x0 = p_x0.exp()
          p_x0 = self._nucleus_sample(p_x0)

        if self.config.sampling.first_hitting:
          x_block = _sample_categorical(p_x0)
          # randomly and uniformly select an index in the block (among masked tokens)
          num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
          ind = torch.randint(0, num_masked, (x_block.shape[0],))
          ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
          mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
          x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
        else:
          q_xs = p_x0 * (1 - mask_prob)
          q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
          x_block = _sample_categorical(q_xs)
        copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
        x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
        x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

        # compute kv cache if all tokens in a block are sampled
        if self.config.sampling.kv_cache and self.mask_index not in x_block:
          _ = self.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

        if not torch.allclose(x_new, x):
          return None, x_new
        else:
          return p_x0, x_new

    @torch.no_grad()
    def _ar_sampler(self, bsz, context_len=1024):
        # reset kvs
        if self.config.sampling.kv_cache:
          self.backbone.reset_kv_cache()

        with torch.amp.autocast('cuda', dtype=torch.float32):
          # precompute token buffer
          num_pred_tokens = self.num_tokens - 1
          x = torch.zeros(
            (bsz, num_pred_tokens + 1),
            dtype=torch.long,
            device=self.device)
          x[:, 0] = self.tokenizer.bos_token_id
          stop = False
          for i in tqdm(range(num_pred_tokens)):
            # need to sample a gumbel for each token
            # to save memory in variable-length sampling
            noise = (torch.distributions.Gumbel(0, 1)
                    .sample((bsz, self.vocab_size))
                    .to(self.device))
            next_logits = self.forward(
              x[:, :i + 1][:, -context_len:],
              None,
              store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)

            next_logits = next_logits.exp()
            next_logits = self._nucleus_sample(next_logits).log()
            y = (next_logits[:, -1] + noise).argmax(-1)
            # check if we need to resample (or stop sampling for variable-length sampling)
            if (i+1) > 256:
              stop, x_out = self._check_stop_conds(x[:, :i+1])
              if stop:
                x = x_out
            if (stop and not self.config.sampling.var_length) \
              or (stop and x.shape[-1] == 1):
              return None
            elif stop:
              break
            x[:, i + 1] = y
          return x

    @torch.no_grad()
    def _sample(
      self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
        """Generate samples from the model."""
        if seqlen is None:
          seqlen = self.config.model.length
        if batch_size_per_gpu is None:
          batch_size_per_gpu = self.config.loader.eval_batch_size
        samples = []
        if self.parameterization == 'ar':
          for _ in range(self.config.sampling.num_sample_batches):
            sample_i, num_tries = None, 0
            while sample_i is None:
              num_tries += 1
              sample_i = self._ar_sampler(batch_size_per_gpu)
              if num_tries > 10:
                raise ValueError('Sampling failed.')
            samples.append(sample_i)
            self.metrics.gen_nfes.append(self.config.model.length)
          samples = torch.cat(samples, dim=0)
          return self.tokenizer.batch_decode(samples)
        if self.sampler == 'semi_ar':
          for _ in range(self.config.sampling.num_sample_batches):
            sample_i, num_tries = None, 0
            while sample_i is None:
              num_tries += 1
              sample_i, nfes = self._semi_ar_sampler(
                n_samples=batch_size_per_gpu,
                num_strides=(seqlen // self.block_size),
                num_steps=num_steps,
                seqlen=seqlen)
              if num_tries > 10:
                raise ValueError('Sampling failed.')
            samples.append(sample_i)
            self.metrics.nfes.update(nfes)
            self.metrics.gen_nfes.append(nfes)
        else:
          nfes = num_steps
          for _ in range(self.config.sampling.num_sample_batches):
            sample_i, num_tries = None, 0
            while sample_i is None:
              sample_i = self._analytic_sampler(
                n_samples=batch_size_per_gpu,
                num_steps=num_steps,
                seqlen=seqlen,
                eps=eps)
              num_tries += 1
              if num_tries > 10 and sample_i is None:
                raise ValueError('Sampling failed.')
            samples.append(sample_i)
            self.metrics.nfes.update(nfes)
            self.metrics.gen_nfes.append(nfes)
        samples = torch.cat(samples, dim=0)
        return self.tokenizer.batch_decode(samples)

    def _sigma_from_p(self, p):
        return torch.min(- torch.log(1 - p), self.noise.sigma_max)

    def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
        """Generate samples from the model."""
        if self.ema:
          self.ema.store(self._get_parameters())
          self.ema.copy_to(self._get_parameters())
        self.backbone.eval()
        self.noise.eval()
        samples = self._sample(
          seqlen=seqlen,
          batch_size_per_gpu=self.config.loader.eval_batch_size,
          num_steps=num_steps,
          eps=eps)
        self.metrics.record_generative_perplexity(
          samples,
          self.config.model.length,
          self.config.loader.eval_batch_size,
          self.device)
        return samples

    def get_score(self, x, sigma):
        model_output = self.forward(x, sigma).to(torch.float64)
        if self.config.sampling.nucleus_p == 1.0:
          return model_output.exp()
        model_output = model_output - model_output.logsumexp(-1, keepdim=True)
        model_output = self._nucleus_sample(model_output.exp())
        return model_output

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, dt):
        sigma_t = self._sigma_from_p(self.noise(t)[1])
        sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
        dsigma = sigma_t - sigma_s
        score = self.get_score(x, sigma_t)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return _sample_categorical(probs)


    def _denoiser_update(self, x, t):
        sigma = self._sigma_from_p(self.noise(t)[1])
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = _sample_categorical(probs)
        return samples


    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(
          i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index,
                            1 - torch.exp(-sigma).squeeze(-1),
                            0)[..., None]
        return edge

    def _sample_t(self, batch_dims_for_blocks, device, sampling_eps_min, sampling_eps_max, block_size=None):
        """
        Samples t per block.
        Args:
            batch_dims_for_blocks: Tuple (batch_size, num_blocks)
            device: torch.device
            sampling_eps_min: float or tensor
            sampling_eps_max: float or tensor
            block_size: int, used by original to repeat_interleave. Here, means "size of these blocks is 1"
                       because we are sampling *for* blocks.
        Returns:
            t_per_block: Tensor of shape (batch_size, num_blocks)
        """
        # num_blocks is batch_dims_for_blocks[1]
        # batch_size is batch_dims_for_blocks[0]

        _eps_b = torch.rand(batch_dims_for_blocks, device=device) # (B, N_Blk)

        if self.antithetic_sampling:
            # This antithetic sampling is over (B * N_Blk) flattened.
            num_total_block_samples = batch_dims_for_blocks[0] * batch_dims_for_blocks[1]
            offset_b = torch.arange(num_total_block_samples, device=device, dtype=torch.float32) / num_total_block_samples
            offset_b = offset_b.view(batch_dims_for_blocks[0], batch_dims_for_blocks[1])
            _eps_b = (_eps_b / num_total_block_samples + offset_b) % 1.0

        t_per_block = _eps_b # Values in [0,1)

        # NLL case for AR-style (block_size=1 in original _sample_t context)
        # If sampling_eps_min/max indicate NLL (t=1 always)
        if isinstance(sampling_eps_max, torch.Tensor): # Convert to float for comparison
            s_eps_max_val = sampling_eps_max.item()
            s_eps_min_val = sampling_eps_min.item()
        else: # float
            s_eps_max_val = sampling_eps_max
            s_eps_min_val = sampling_eps_min

        if s_eps_max_val >= 1.0 and s_eps_min_val >= 1.0:
            return torch.ones_like(t_per_block) # t=1 for all blocks

        # Scale t to [sampling_eps_min, sampling_eps_max]
        t_per_block = t_per_block * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
        return t_per_block

    # Ensure these are defined in __init__ if used by q_xt's resample conditions
    # self.sampling_eps_min_val_for_no_resample = 1e-3
    # self.sampling_eps_max_val_for_no_resample = 1.0


    def _maybe_sub_sample(self, x0, attention_mask):
        seqlen = x0.shape[1]
        if seqlen > self.num_tokens:
          assert seqlen == 2 * self.num_tokens
          # cropping is needed for text8-crop dataset
          # try the same starting point for now
          start = np.random.choice(self.num_tokens)
          end = start + self.num_tokens
          input_tokens = x0[:, start: end]
          output_tokens = x0[:, start + 1: end + 1]
          new_attention_mask = attention_mask[:, start: end]

          # Helps with validation ppl, since the val
          # examples will all start and end with BOS/EOS
          if self.config.data.insert_train_special == True:
            input_tokens[:, 0] = self.tokenizer.bos_token_id
            output_tokens[:, -1] = self.tokenizer.eos_token_id
        elif self.parameterization == 'ar':
          input_tokens = x0[:, :-1]
          output_tokens = x0[:, 1:]
          new_attention_mask = attention_mask[:, 1:]
        else:
          input_tokens = x0
          output_tokens = None
          new_attention_mask = attention_mask

        return input_tokens, output_tokens, new_attention_mask

    def _forward_pass_diffusion(self, x0, x0_block_features, t_samples_per_block):
        B, N_Tokens = x0.shape
        num_blocks = N_Tokens // self.block_size

        alpha_b_t_all_blocks, loss_scale_b_t_all_blocks, p_b_t_all_blocks, s_b = \
            self._get_warped_noise_outputs_for_block_batch(x0_block_features, t_samples_per_block)

        # Expand per-block values to per-token
        # p_b_t_all_blocks, loss_scale_b_t_all_blocks are (B, N_Blk)
        p_per_token = p_b_t_all_blocks.repeat_interleave(self.block_size, dim=1) # (B, N_Tokens)
        loss_scale_per_token = loss_scale_b_t_all_blocks.repeat_interleave(self.block_size, dim=1) # (B, N_Tokens)

        # sigma for the backbone model's timestep conditioning
        # self._sigma_from_p expects (B,1) or (B). Let's pass (B*N_Tokens)
        # or ensure it can handle (B, N_Tokens)
        # Existing _sigma_from_p calls self.noise(t)[1] -> self.noise is global, not base!
        # This should be: sigma based on p_per_token.
        # self.noise.sigma_max should be self.base_noise_schedule.sigma_max or a global constant.

        # If self.noise is the global schedule (e.g. LogLinearNoise instance)
        # then sigma_max is defined there.
        # The argument to _sigma_from_p is move_chance 'p'.
        sigma_all_tokens = self._sigma_from_p(p_per_token) # (B, N_Tokens)

        # q_xt needs p_per_token to be (B, N_Tokens, 1) if it was expecting (B,1) and unsqueezing
        # self.q_xt(x0, p, sampling_eps_min, sampling_eps_max)
        #   p: float torch.Tensor with shape (batch_size, 1)
        #   move_indices = torch.rand(*x.shape, device=x.device) <= p
        # So, p needs to be (B, N_Tokens) for element-wise comparison if x is (B, N_Tokens)
        # The unsqueeze(-1) was for the old q_xt. The current q_xt should handle p=(B,N_Tokens) or (B,1)
        # Let's assume current q_xt takes p of shape (B, N_Tokens) or (B,1) and broadcasts.
        # The provided `q_xt` takes p of shape (B,1). This MUST be updated.
        # For now, let's pass p_per_token.unsqueeze(-1)
        xt = self.q_xt(x0,
                       p_per_token.unsqueeze(-1), # Make it (B, N_Tokens, 1) for broadcasting
                       sampling_eps_min=self.sampling_eps_min, # Global min/max for resampling logic
                       sampling_eps_max=self.sampling_eps_max)

        if self.ignore_bos: # This applies to xt after noising
            xt[:, 0] = x0[:, 0]

        x_input = xt
        if self.cross_attn:
            x_input = torch.cat((xt, x0), dim=-1)

        # self.forward (backbone) needs sigma of shape (B) or (B, N_Tokens) if per-token time cond.
        # Current self.forward calls self._process_sigma which averages sigma over seq dim.
        # This needs to be fixed if sigma_all_tokens is (B, N_Tokens).
        # For now, assume backbone takes per-token sigma, or _process_sigma handles it.
        # Let's make _process_sigma take (B, N_Tokens) and return (B) by averaging, if backbone needs (B).
        # Or, modify backbone to take (B, N_Tokens) sigma.
        # The HF DiT backbone's `timesteps` arg is (B). So average is needed.
        processed_sigma_for_backbone = self._process_sigma(sigma_all_tokens.unsqueeze(-1)) # Pass (B,N_T,1) to get (B)

        model_output = self.forward(x_input, sigma=processed_sigma_for_backbone) # self.forward is Diffusion.forward

        if self.cross_attn: # Model output corresponds to xt part
             model_output = model_output[:, :N_Tokens, :]

        # Original BD3LM loss: loss_scale * log_p_theta
        # log_p_theta is -CE. So loss is -loss_scale * CE_term
        # Here, CE_term is positive. loss_scale_per_token is currently negative.
        # So product is positive. This seems correct.

        # log_p_theta = torch.gather(F.log_softmax(model_output, dim=-1), -1, x0.unsqueeze(-1)).squeeze(-1)
        # weighted_loss_per_token = loss_scale_per_token * log_p_theta

        # Alternative: CE loss, weighted by |loss_scale_per_token|
        # Ensure loss_scale_per_token is positive for weighting a positive CE loss.
        # User's formula implies loss_scale_b_t is the weight for log_p_theta(x0|xt).
        # As derived, L_base' < 0, so user_beta_b_t = alpha_b_t * s_b * L_base' is negative.
        # log_p_theta is negative. Product is positive.

        log_probs_x0 = F.log_softmax(model_output, dim=-1)
        log_p_theta_x0_given_xt = torch.gather(log_probs_x0, -1, x0.unsqueeze(-1)).squeeze(-1) # (B, N_Tokens)

        # loss_scale_per_token is user_beta_b_t (negative)
        weighted_loss_per_token = loss_scale_per_token * log_p_theta_x0_given_xt

        return weighted_loss_per_token, s_b # s_b from _get_warped_noise_outputs_for_block_batch


    def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
        if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
            sampling_eps_min = self.sampling_eps_min
            sampling_eps_max = self.sampling_eps_max
        elif not hasattr(self, 'sampling_eps_min'): # Should always be hasattr after __init__
            sampling_eps_min = torch.tensor(1e-3, device=x0.device)
            sampling_eps_max = torch.tensor(1.0, device=x0.device)

        input_tokens, _, new_attention_mask = self._maybe_sub_sample(x0, attention_mask)
        # output_tokens is not used if not AR

        if self.parameterization == 'ar': # AR handled by original code, not modified here
            output = self.forward(input_tokens, None)
            loss = - output.gather(-1, x0[:, 1:].unsqueeze(-1))[:, :, 0] # x0[:,1:] is output_tokens
            mean_loss = (loss * new_attention_mask).sum() / new_attention_mask.sum()
            s_b_for_penalty = None # No s_b for AR
            nlls_for_metric = loss
            current_loss_for_return = mean_loss
        else:
            # (B, N_Tokens) -> (B, N_Blk, FeatDim)
            x0_block_features = self._get_block_features(input_tokens, new_attention_mask)

            # Sample t per block: (B, N_Blk)
            # _sample_t takes batch_dims of tokens, so pass (B, N_Blk) as shape for block-level t
            # No, _sample_t current shape arg is (B, N_Tokens) and it divides by block_size internally
            # This means t is already sampled per-block if block_size != model.length in _sample_t
            # If self.block_size is used in _sample_t for num_blocks, it's fine.
            # Let's make _sample_t explicitly return (B, N_Blk)
            t_samples_per_block = self._sample_t(
                (input_tokens.shape[0], input_tokens.shape[1] // self.block_size), # (B, N_Blk)
                input_tokens.device,
                sampling_eps_min,
                sampling_eps_max,
                block_size=1 # Tell _sample_t these are already "blocks" of size 1
            ) # (B, N_Blk)

            weighted_loss_per_token, s_b_all_blocks = self._forward_pass_diffusion(
                input_tokens,
                x0_block_features,
                t_samples_per_block # (B, N_Blk)
            ) # weighted_loss_per_token is (B, N_Tokens), s_b_all_blocks is (B, N_Blk, 1)

            loss = weighted_loss_per_token # This is already the NELBO term per token, needs masking and averaging

            # Surrogate Penalty Calculation
            # s_b_all_blocks is (B, N_Blk, 1), from _get_warped_noise_outputs_for_block_batch (via _forward_pass_diffusion)

            # Need alpha_b(1) using s_b_all_blocks
            # Arg t for alpha_b(1) is a tensor of ones, shape like s_b_all_blocks (or broadcastable)
            t_one_for_penalty = torch.ones_like(s_b_all_blocks.squeeze(-1), device=self.device) # (B, N_Blk)

            base_alpha_bar_at_1 = self.base_noise_schedule.get_alpha_bar(t_one_for_penalty) # (B, N_Blk)

            _alpha_b_at_1 = noise_schedule.get_warped_alpha_b_t(
                base_alpha_bar_at_1,              # (B, N_Blk)
                self.base_log_alpha_bar_at_0,     # scalar
                s_b_all_blocks.squeeze(-1),       # (B, N_Blk) - adjust get_warped_alpha_b_t if it needs (B,N_Blk,1)
                torch.logit(self.target_alpha_at_0) # scalar
            ) # (B, N_Blk)

            surrogate_penalty_per_block = noise_schedule.compute_surrogate_steps_penalty(
                _alpha_b_at_1.unsqueeze(-1), # (B, N_Blk, 1) for the function
                self.min_alpha_1_target,
                self.lambda_min_alpha_1_penalty,
                self.alpha_1_clamp_min,
                self.alpha_1_clamp_max
            ) # (B, N_Blk, 1)

            mean_surrogate_penalty = surrogate_penalty_per_block.mean()

            # L2 penalty on s_b (or log_s_tilde_b)
            # s_b_all_blocks is (B, N_Blk, 1). Let's penalize (s_b - 1)^2 to encourage s_b around 1.
            s_b_l2_penalty = 0.0
            if self.lambda_s_b_l2_penalty > 0:
                 # Recompute log_s_tilde from x0_block_features if not returned by _forward_pass_diffusion
                log_s_tilde_b_all_blocks = self.meta_controller(x0_block_features) # (B, N_Blk, 1)
                s_b_l2_penalty = self.lambda_s_b_l2_penalty * (log_s_tilde_b_all_blocks**2).mean()
                # Or, penalize (s_b - target_s_b)^2 if desired. (s_b-1)^2 encourages no scaling.
                # s_b_l2_penalty = self.lambda_s_b_l2_penalty * ((s_b_all_blocks - 1.0)**2).mean()


            # Mask and average the NELBO term
            masked_nelbo_per_token = (loss * new_attention_mask)
            mean_nelbo = masked_nelbo_per_token.sum() / new_attention_mask.sum().clamp(min=1.0)

            total_loss = mean_nelbo + self.lambda_steps_penalty * mean_surrogate_penalty + s_b_l2_penalty

            # For logging individual components
            self.log('train/nelbo', mean_nelbo, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log('train/surrogate_penalty', mean_surrogate_penalty, on_step=True, on_epoch=False, sync_dist=True)
            if self.lambda_s_b_l2_penalty > 0:
                self.log('train/s_b_l2_penalty', s_b_l2_penalty, on_step=True, on_epoch=False, sync_dist=True)

            # Return values for metrics.Metrics
            # `loss` here is per-token, already weighted by loss_scale (user_beta_b_t)
            # metrics.train_nlls.update(nlls, token_mask) where nlls is (loss * token_mask)
            # So we should pass `loss` (per token) and `new_attention_mask`.
            # The `loss` here is already the "final" loss contribution from each token to NELBO.
            nlls_for_metric = loss # These are the per-token NELBO contributions
            current_loss_for_return = total_loss

        return Loss(loss=current_loss_for_return,
                    nlls=nlls_for_metric, # Per-token NELBO values
                    token_mask=new_attention_mask)


    def _clipped_schedule_search(self):
        # collect losses per batch across devices and sum them per interval
        best_var = float('inf')
        for (eps_min, eps_max), var in self.metrics.valid_vars.items():
          all_vars = torch.tensor(0., device=self.device)
          for i in range(len(var)):
            agg_var = var[i].to(self.device)
            agg_var = self.all_gather(agg_var)
            all_vars += agg_var.var()
          if all_vars < best_var:
            best_var = all_vars
            sampling_eps_min_best = eps_min
            sampling_eps_max_best = eps_max
          self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                    all_vars / len(var),
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True)
        if self.config.algo.fix_clipping == False:
          self.sampling_eps_min.fill_(sampling_eps_min_best)
          self.sampling_eps_max.fill_(sampling_eps_max_best)

    def _score_entropy(self, log_score, sigma, xt, x0):
        """Computes the SEDD loss.

        Args:
          log_score: float torch.Tensor with shape (batch_size,
              diffusion_model_input_length, vocab_size),
              log score, output of the denoising network.
          xt: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          x0: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          sigma: float torch.Tensor with shape (batch_size, 1).

        Returns:
          loss with shape (batch_size, diffusion_model_input_length)
        """
        masked_indices = xt == self.mask_index

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
          log_score[masked_indices],
          -1,
          words_that_were_masked[..., None]).squeeze(-1)
        score = log_score[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
          pos_term = score[:, :-1].sum(dim=-1)
        else:
          pos_term = score[:, : self.mask_index].sum(
            dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(* xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return entropy

    @torch.no_grad
    def _analytic_sampler(
      self, n_samples, num_steps, seqlen, eps=1e-5):
        x = self._sample_prior(
          n_samples,
          seqlen).to(self.device)
        x[:, 0] = self.tokenizer.bos_token_id
        timesteps = torch.linspace(
          1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        for i in tqdm(range(num_steps), desc='step'):
          t = timesteps[i] * torch.ones(
            x.shape[0], 1, device=self.device)
          x = self._analytic_update(x=x, t=t, dt=dt)
        # denoising step
        t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                      device=self.device)
        x = self._denoiser_update(x=x, t=t)

        stop, x = self._check_stop_conds(x)
        if stop:
          return None
        return x

    @torch.no_grad
    def _semi_ar_sampler(
      self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
        if seqlen is None:
          seqlen = self.config.model.length
        sampling_steps = 0

        mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
        if mdlm_semi_ar:
          # sliding window of length 512 for mdlm semi-ar decoding
          num_strides = self.config.model.length // 512
          num_strides -= 1

        ones = torch.ones((n_samples,1), dtype=self.dtype,
                          device=self.device)

        # reset kvs
        if self.config.sampling.kv_cache:
          self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

        for stride_num in tqdm(range(num_strides)):
          # sample next block
          if stride_num == 0:
            x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
            x_accum[:, 0] = self.tokenizer.bos_token_id
          else:
            if mdlm_semi_ar:
              x = self._sample_prior(n_samples, 512).to(self.device)
            else:
              x = self._sample_prior(n_samples, self.block_size).to(self.device)
            x_accum = torch.cat((x_accum, x), dim=1)

          # compute logits in a sliding window (context passed to model can't exceed context_size)
          end_idx = (stride_num + 1) * self.block_size
          start_idx = max(end_idx - context_size, 0)
          fwd_idx = torch.arange(start_idx, end_idx)
          if mdlm_semi_ar and stride_num > 0: # MDLM
            fwd_idx = torch.arange(512*(stride_num), (512*(stride_num))+self.block_size)

          dt = 1 / num_steps
          p_x0_cache = None
          timesteps = torch.linspace(1, 0, num_steps, device=self.device)
          t = 1
          for i in range(num_steps):
            if self.mask_index not in x_accum:
              break

            # faster (equivalent) sampler from zheng et al (2025)
            if self.config.sampling.first_hitting:
              u = np.random.rand()
              num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
              t *= u**(1 / num_masked)

            elif not self.config.sampling.first_hitting:
              t = timesteps[i]

            p_x0_cache, x_next = self._ddpm_caching_update(
                x=x_accum[:, fwd_idx],
                t=t * ones,
                dt=dt,
                p_x0=p_x0_cache,)
            if p_x0_cache is None:
              sampling_steps += 1

            x_accum[:, fwd_idx] = x_next

          # check if we need to resample (or stop sampling for variable-length sampling)
          if x_accum.shape[1] > 256:
            stop, x_accum = self._check_stop_conds(x_accum)
            if (stop and not self.config.sampling.var_length) \
              or (stop and x.shape[-1] == 1):
              return None, None
            elif stop:
              break
        return x_accum, sampling_steps

    def _compute_entropy(self, x):
        _, counts = torch.unique(x, return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum()
        return entropy

    def _check_stop_conds(self, x):
        """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
        Entropy/likelihood evaluated on last 256 token-block.

        Args:
          x: torch.Tensor, current sample.
        Returns:
          stop: bool, whether to stop sampling.
          x: torch.Tensor, sample (potentially truncated for variable-length sampling).
        """
        stop = False # stop sampling?
        truncate_idx = None # truncate sample? (variable-length sampling only)

        # CRITERION 2: always stop sampling if entropy is low
        entropy = self._compute_entropy(x[:, -256:])
        if entropy < 4:
          stop = True

        # for variable length sampling, check if we should stop
        # sampling, and where to truncate the sample
        if self.config.sampling.var_length:
          # CRITERION 1: stop at sampled EOS token
          if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
            stop = True
            eos_idx = torch.where(x == self.tokenizer.eos_token_id)
            if len(eos_idx[0]) > 1:
              truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

          # CRITERION 2: stop if entropy/likelihood is low
          if entropy < 4:
            stop = True
            truncate_idx = x.shape[1] - 256

        # truncate sample (variable-length sampling only)
        if truncate_idx is not None:
          x = x[:, :truncate_idx]
          if x.ndim == 1:
            x = x.unsqueeze(0)

        return stop, x