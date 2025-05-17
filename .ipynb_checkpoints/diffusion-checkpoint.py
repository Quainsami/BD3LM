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
from omegaconf import ListConfig # <--- ADD THIS LINE

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
        
        self.meta_controller_config = config.algo.meta_controller 
        self.meta_controller = MetaController(config) 

        self.base_noise_schedule = noise_schedule.get_noise(config, config.algo.base_noise_type)
        
        _dummy_zero_t = torch.tensor(0.0, dtype=torch.float32) 
        clamped_base_alpha_at_0 = self.base_noise_schedule.get_alpha_bar(_dummy_zero_t)

        self.register_buffer("target_alpha_at_0", torch.tensor(1.0 - config.algo.schedule_clamp_epsilon, dtype=torch.float32))
        self.register_buffer("base_log_alpha_bar_at_0", torch.logit(clamped_base_alpha_at_0))

        self.min_alpha_1_target = self.config.algo.min_alpha_1_target
        self.lambda_min_alpha_1_penalty = self.config.algo.lambda_min_alpha_1_penalty
        self.alpha_1_clamp_min = self.config.algo.alpha_1_clamp_min
        self.alpha_1_clamp_max = self.config.algo.alpha_1_clamp_max
        
        self.lambda_steps_penalty = self.config.algo.lambda_steps_penalty
        
        if hasattr(self.config.algo, 'lambda_s_b_l2_penalty'):
            self.lambda_s_b_l2_penalty = self.config.algo.lambda_s_b_l2_penalty
        else:
            self.lambda_s_b_l2_penalty = 0.0
            
        self._validate_configuration()

    def _get_parameters(self):
        params_to_optimize = []
        training_stage = getattr(self.config.training, "stage", "joint") 

        if training_stage == "controller_only":
            print("Optimizer: MetaController parameters only.")
            params_to_optimize.append(self.meta_controller.parameters())
        elif training_stage == "joint":
            print("Optimizer: Jointly training LM and MetaController.")
            params_to_optimize.append(self.backbone.parameters())
            if hasattr(self.noise, 'parameters') and any(p.requires_grad for p in self.noise.parameters()):
                 params_to_optimize.append(self.noise.parameters())
            params_to_optimize.append(self.meta_controller.parameters())
        elif training_stage == "lm_only": 
            print("Optimizer: LM parameters only.")
            params_to_optimize.append(self.backbone.parameters())
            if hasattr(self.noise, 'parameters') and any(p.requires_grad for p in self.noise.parameters()):
                 params_to_optimize.append(self.noise.parameters())
        else:
            raise ValueError(f"Unknown training stage: {training_stage}")

        return itertools.chain(*params_to_optimize)

    def _get_block_features(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, N_Tokens = tokens.shape
        if self.block_size == 0: 
            return torch.empty(B, 0, self.meta_controller_config.feature_dim, device=tokens.device)
        num_blocks = N_Tokens // self.block_size
        if num_blocks == 0: 
             return torch.empty(B, 0, self.meta_controller_config.feature_dim, device=tokens.device)


        block_indices_feat = torch.arange(num_blocks, device=tokens.device, dtype=torch.float32)
        if num_blocks > 1:
            block_indices_feat = block_indices_feat / (num_blocks - 1 + 1e-9) 
        else: 
            block_indices_feat = torch.zeros_like(block_indices_feat)
        
        block_indices_feat = block_indices_feat.view(1, num_blocks, 1).expand(B, -1, -1)
        return block_indices_feat

    def _get_warped_noise_outputs_for_block_batch(
        self,
        x0_block_features: torch.Tensor, 
        u_values_per_block: torch.Tensor 
    ):
        log_s_tilde_b = self.meta_controller(x0_block_features) 
        s_b = self.meta_controller.get_s_b(log_s_tilde_b)       
        s_b_squeezed = s_b.squeeze(-1) 

        base_alpha_bar_u = self.base_noise_schedule.get_alpha_bar(u_values_per_block)
        base_log_alpha_bar_base_derivative_u = self.base_noise_schedule.get_log_alpha_bar_base_derivative_t(u_values_per_block)

        target_alpha_at_0_device = self.target_alpha_at_0.to(u_values_per_block.device)

        alpha_b_u_all_blocks, loss_scale_b_u_all_blocks, p_b_u_all_blocks = \
            noise_schedule.get_warped_schedule_outputs(
                base_alpha_bar_u,
                base_log_alpha_bar_base_derivative_u,
                self.base_log_alpha_bar_at_0, 
                s_b_squeezed, 
                torch.logit(target_alpha_at_0_device) 
            )
        return alpha_b_u_all_blocks, loss_scale_b_u_all_blocks, p_b_u_all_blocks, s_b, log_s_tilde_b

    def on_validation_model_zero_grad(self) -> None:
        super().on_validation_model_zero_grad()
        if hasattr(self,'trainer') and self.trainer is not None and self.trainer.ckpt_path is not None and \
           getattr(self, '_restarting_skip_val_flag', True):
            self.trainer.sanity_checking = True
            self._restarting_skip_val_flag = False

    def _validate_configuration(self):
        if self.config.mode == 'sample_eval' and \
            self.config.sampling.first_hitting:
          assert self.config.loader.eval_batch_size == 1
        assert self.config.algo.backbone in {
          'dit', 'ar', 'hf_dit', 'dimamba'} 
        if self.config.algo.parameterization == 'ar':
          assert not self.config.algo.time_conditioning
        if self.config.sampling.kv_cache:
          assert self.config.algo.name in {'ar', 'bd3lm'}

        if self.parameterization in {'sedd'}:
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
          if hasattr(self.backbone.block_diff_mask, 'to'): 
            self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
        elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
          if hasattr(self.backbone.block_diff_mask, 'to'):
              self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
        if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
          self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
          self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
        if hasattr(self, 'meta_controller'):
            self.meta_controller = self.meta_controller.to(*args, **kwargs)
        if hasattr(self, 'target_alpha_at_0'):
            self.target_alpha_at_0 = self.target_alpha_at_0.to(*args, **kwargs)
        if hasattr(self, 'base_log_alpha_bar_at_0'):
            self.base_log_alpha_bar_at_0 = self.base_log_alpha_bar_at_0.to(*args, **kwargs)
        if hasattr(self, 'base_noise_schedule'):
            self.base_noise_schedule = self.base_noise_schedule.to(*args, **kwargs)
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

        if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
          checkpoint = self._replace_ckpt_keys(checkpoint)

        if self.ema and 'ema' in checkpoint: 
          self.ema.load_state_dict(checkpoint['ema'])
            
        if 'sampling_eps_min' in checkpoint.keys():
          self.sampling_eps_min = checkpoint['sampling_eps_min']
          self.sampling_eps_max = checkpoint['sampling_eps_max']
        if 'loops' in checkpoint and 'fit_loop' in checkpoint['loops']: 
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
        
        if 'loops' not in checkpoint: checkpoint['loops'] = {}
        if 'fit_loop' not in checkpoint['loops']: checkpoint['loops']['fit_loop'] = {}
        if 'epoch_loop.batch_progress' not in checkpoint['loops']['fit_loop']:
            checkpoint['loops']['fit_loop']['epoch_loop.batch_progress'] = {'total': {'completed':0}, 'current':{'completed':0}}
        if 'epoch_loop.automatic_optimization.optim_progress' not in checkpoint['loops']['fit_loop']: 
            checkpoint['loops']['fit_loop']['epoch_loop.automatic_optimization.optim_progress'] = {'optimizer': {'step': {'total':{'completed':0}, 'current':{'completed':0}}}}
        if 'epoch_loop.state_dict' not in checkpoint['loops']['fit_loop']:
            checkpoint['loops']['fit_loop']['epoch_loop.state_dict'] = {'_batches_that_stepped': 0}


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
        if 'sampler' not in checkpoint.keys():
          checkpoint['sampler'] = {}
        
        train_dataloader_obj = self.trainer.train_dataloader
        actual_sampler = None
        if hasattr(train_dataloader_obj, 'sampler') and hasattr(train_dataloader_obj.sampler, 'state_dict'): 
            actual_sampler = train_dataloader_obj.sampler
        elif hasattr(train_dataloader_obj, 'loader') and hasattr(train_dataloader_obj.loader, 'sampler') and \
             hasattr(train_dataloader_obj.loader.sampler, 'state_dict'): 
            actual_sampler = train_dataloader_obj.loader.sampler
        
        if actual_sampler:
            sampler_state_dict = actual_sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None


    def on_train_start(self):
        if self.ema:
          self.ema.move_shadow_params_to_device(self.device)
        
        if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, '_accelerator_connector'):
            distributed = (
              self.trainer._accelerator_connector.use_distributed_sampler
              and self.trainer._accelerator_connector.is_distributed)
            if distributed:
              sampler_cls = dataloader.FaultTolerantDistributedSampler
            else:
              sampler_cls = dataloader.RandomFaultTolerantSampler
            updated_dls = []
            
            if hasattr(self.trainer.fit_loop,'_combined_loader') and self.trainer.fit_loop._combined_loader is not None:
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
                      persistent_workers=True if self.config.loader.num_workers > 0 else False))
                self.trainer.fit_loop._combined_loader.flattened = updated_dls
        
        training_stage = getattr(self.config.training, "stage", "joint")
        if training_stage == "controller_only":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.meta_controller.parameters():
                param.requires_grad = True
        elif training_stage == "lm_only":
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.meta_controller.parameters():
                param.requires_grad = False
        elif training_stage == "joint":
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.meta_controller.parameters():
                param.requires_grad = True
        
        if self.ema: 
            self.ema = models.ema.ExponentialMovingAverage(
                self._get_parameters(), decay=self.config.training.ema
            )
            self.ema.move_shadow_params_to_device(self.device)


    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
          self.ema.update(self._get_parameters())

    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1,
                                          keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        if xt[unmasked_indices].numel() > 0 : 
             logits[unmasked_indices] = self.neg_infinity
             logits[unmasked_indices, xt[unmasked_indices]] = 0.0 
        return logits

    def _sedd_parameterization(self, logits, xt, sigma_batch):
        esigm1_log = torch.where(
          sigma_batch < 0.5, 
          torch.expm1(sigma_batch),
          sigma_batch.exp() - 1.0 
          ).log().to(logits.dtype) 
        logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1.0) 
        logits = torch.scatter(logits, -1, xt.unsqueeze(-1), 
                               torch.zeros_like(logits[..., :1]))
        return logits


    def _process_sigma(self, sigma): 
        if sigma is None and self.parameterization == 'ar': 
            return None
            
        current_sigma = sigma
        if current_sigma.ndim == 3 and current_sigma.shape[-1] == 1: 
            current_sigma = current_sigma.squeeze(-1) 

        if self.parameterization == 'ar': 
            return None 

        if current_sigma.ndim == 2: 
            sigma_for_backbone = current_sigma.mean(dim=-1) 
        elif current_sigma.ndim == 1: 
            sigma_for_backbone = current_sigma
        else:
            raise ValueError(f"Unexpected sigma shape after initial processing: {current_sigma.shape}")

        if not self.time_conditioning:
            sigma_for_backbone = torch.zeros_like(sigma_for_backbone)

        assert sigma_for_backbone.ndim == 1, sigma_for_backbone.shape
        return sigma_for_backbone

    def forward(self, x, sigma, sample_mode=False, store_kv=False): 
        sigma_processed = self._process_sigma(sigma) 
        
        is_hf_bd3lm_backbone = self.config.algo.name == 'bd3lm' and self.config.algo.backbone == 'hf_dit'
        is_custom_bd3lm_backbone = self.config.algo.name == 'bd3lm' and self.config.algo.backbone != 'hf_dit'
        
        backbone_input_arg_name = 'input_ids' if self.config.algo.backbone == 'hf_dit' else 'indices'
        backbone_input = {backbone_input_arg_name: x}

        with torch.amp.autocast('cuda', dtype=torch.bfloat16): 
          if is_hf_bd3lm_backbone:
            output = self.backbone(**backbone_input, timesteps=sigma_processed, 
                                   store_kv=store_kv,
                                   sample_mode=sample_mode)
            logits = output.logits if hasattr(output, 'logits') else output 
          elif is_custom_bd3lm_backbone: 
            logits = self.backbone(**backbone_input, sigma=sigma_processed,
                                   store_kv=store_kv,
                                   sample_mode=sample_mode)
          elif self.config.algo.name == 'ar':
            if self.config.algo.backbone == 'hf_dit': 
              output = self.backbone(**backbone_input, timesteps=None) 
              logits = output.logits if hasattr(output, 'logits') else output
            else: 
              logits = self.backbone(x, None, sample_mode=sample_mode, store_kv=store_kv) 
            if self.mask_index < logits.shape[-1]: # Check if mask_index is valid
                logits[:, :, self.mask_index] = self.neg_infinity
            logits = logits.log_softmax(-1)
          else: 
            logits = self.backbone(**backbone_input, sigma=sigma_processed) 

        x_orig_for_param = x 
        
        if self.cross_attn :
            if not (self.config.algo.backbone == 'hf_dit'):
                x_orig_len = x.shape[1] // 2 
                x_orig_for_param = x[:, :x_orig_len]
                if logits.shape[1] == x.shape[1] : 
                     logits = logits[:, :x_orig_len, :]


        if self.parameterization == 'subs':
          return self._subs_parameterization(logits=logits, xt=x_orig_for_param)
        elif self.parameterization == 'sedd':
          return self._sedd_parameterization(logits=logits, xt=x_orig_for_param, sigma_batch=sigma_processed) 
        
        return logits 


    def on_train_epoch_start(self):
        self.backbone.train()
        self.meta_controller.train() 
        self.noise.train()
        self.metrics.reset()
        main_metric_collection = self.metrics.train_nlls
        if hasattr(main_metric_collection, 'nll') and hasattr(main_metric_collection.nll, 'mean_value'):
             assert main_metric_collection.nll.mean_value == 0
             assert main_metric_collection.nll.weight == 0
        elif hasattr(main_metric_collection, '_forward_cache') and main_metric_collection._forward_cache is not None:
             main_metric_collection.reset() 


    def training_step(self, batch, batch_idx):
        del batch_idx
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'])
        self.metrics.train_nlls.update(losses.nlls.detach(), losses.token_mask.detach())
        self.log(name='trainer/loss',
                 value=losses.loss.item(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return losses.loss

    def on_validation_epoch_start(self):
        self.metrics.reset()
        if self.ema:
          self.ema.store(self._get_parameters()) 
          self.ema.copy_to(self._get_parameters())
        self.eval() 
        self.backbone.eval()
        self.meta_controller.eval() 
        self.noise.eval()
        
        main_val_metric_collection = self.metrics.valid_nlls
        if hasattr(main_val_metric_collection, 'nll') and hasattr(main_val_metric_collection.nll, 'mean_value'):
            assert main_val_metric_collection.nll.mean_value == 0
            assert main_val_metric_collection.nll.weight == 0
        elif hasattr(main_val_metric_collection, '_forward_cache') and main_val_metric_collection._forward_cache is not None:
            main_val_metric_collection.reset()


        if isinstance(self.config.training.sampling_eps, tuple):
            self.sampling_eps_min_val_current = torch.tensor(self.config.training.sampling_eps[0], device=self.device, dtype=torch.float32)
            self.sampling_eps_max_val_current = torch.tensor(self.config.training.sampling_eps[1], device=self.device, dtype=torch.float32)
        else: 
            self.sampling_eps_min_val_current = torch.tensor(self.config.training.sampling_eps, device=self.device, dtype=torch.float32)
            self.sampling_eps_max_val_current = torch.tensor(1.0, device=self.device, dtype=torch.float32)


    def on_validation_epoch_end(self):
        for k, v_metric in self.metrics.valid_nlls.items():
          if hasattr(v_metric, 'compute'):
              computed_val = v_metric.compute()
              if torch.is_tensor(computed_val) and computed_val.numel() == 1: 
                self.log(name=k, value=computed_val.item(), on_step=False,
                         on_epoch=True, sync_dist=True)
              elif isinstance(computed_val, float):
                 self.log(name=k, value=computed_val, on_step=False,
                         on_epoch=True, sync_dist=True)
          elif isinstance(v_metric, torch.Tensor) and v_metric.numel() == 1: 
               self.log(name=k, value=v_metric.item(), on_step=False,
                       on_epoch=True, sync_dist=True)


        if self.ema:
          self.ema.restore(self._get_parameters())
        if self.var_min and hasattr(self, 'trainer') and self.trainer is not None and not self.trainer.sanity_checking:
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
        s_eps_min_f = sampling_eps_min.item() if isinstance(sampling_eps_min, torch.Tensor) else sampling_eps_min
        s_eps_max_f = sampling_eps_max.item() if isinstance(sampling_eps_max, torch.Tensor) else sampling_eps_max
            
        is_elbo_range = abs(s_eps_min_f - 1e-3) < 1e-6 and abs(s_eps_max_f - 1.0) < 1e-6
        
        eval_nll_configured = getattr(self.config.training, 'eval_nll', False)

        is_nll_bs1_eval_config = eval_nll_configured
        is_nll_bs1_case = self.block_size == 1 and is_nll_bs1_eval_config

        if is_elbo_range and not is_nll_bs1_case:
          return True 
        
        is_nll_bs1_condition_strict = self.block_size == 1 and s_eps_min_f >= (1.0 - 1e-6) 
        if is_nll_bs1_condition_strict and is_nll_bs1_eval_config:
            return True
            
        return False 


    def validation_step(self, batch, batch_idx):
        current_sampling_eps_min_for_val = self.sampling_eps_min_val_current
        current_sampling_eps_max_for_val = self.sampling_eps_max_val_current
        loss_to_return = None 

        if self.var_min:
          main_interval_processed = False
          for noise_clip_start_tuple, collected_nlls_for_var in self.metrics.valid_vars.items(): 
            sampling_eps_min_tensor = torch.tensor(noise_clip_start_tuple[0], device=self.device, dtype=torch.float32)
            sampling_eps_max_tensor = torch.tensor(noise_clip_start_tuple[1], device=self.device, dtype=torch.float32)

            is_main_interval_for_logging = self._check_val_sampling_intvl(sampling_eps_min_tensor, sampling_eps_max_tensor)
            needs_collection_for_var = len(collected_nlls_for_var) < 100

            if is_main_interval_for_logging or needs_collection_for_var:
                losses_clip = self._loss(batch['input_ids'],
                                  batch['attention_mask'],
                                  sampling_eps_min=sampling_eps_min_tensor,
                                  sampling_eps_max=sampling_eps_max_tensor)
                if is_main_interval_for_logging: 
                    self.metrics.valid_nlls.update(losses_clip.nlls.detach(), losses_clip.token_mask.detach())
                    loss_to_return = losses_clip.loss 
                    main_interval_processed = True
                
                if needs_collection_for_var: 
                    nlls_for_var_calc = losses_clip.nlls 
                    if nlls_for_var_calc.numel() > 0 and self.block_size > 0: 
                        nlls_reshaped_for_var = nlls_for_var_calc.view(nlls_for_var_calc.shape[0], -1, self.block_size).mean(-1)
                        collected_nlls_for_var.append(nlls_reshaped_for_var.detach().cpu()) 
            
          if not main_interval_processed and loss_to_return is None: # Ensure loss_to_return is set if no var_min interval was the main one.
              losses_main = self._loss(batch['input_ids'], batch['attention_mask'],
                                          sampling_eps_min=current_sampling_eps_min_for_val,
                                          sampling_eps_max=current_sampling_eps_max_for_val)
              self.metrics.valid_nlls.update(losses_main.nlls.detach(), losses_main.token_mask.detach())
              loss_to_return = losses_main.loss


        elif self.block_size == 1 and getattr(self.config.training, 'eval_nll', False) : 
          losses = self._loss(batch['input_ids'],
                              batch['attention_mask'],
                              sampling_eps_min=torch.tensor(1.0, device=self.device, dtype=torch.float32),
                              sampling_eps_max=torch.tensor(1.0, device=self.device, dtype=torch.float32))
          self.metrics.valid_nlls.update(losses.nlls.detach(), losses.token_mask.detach())
          loss_to_return = losses.loss
        else: 
          losses = self._loss(batch['input_ids'],
                              batch['attention_mask'],
                              sampling_eps_min=current_sampling_eps_min_for_val,
                              sampling_eps_max=current_sampling_eps_max_for_val)
          self.metrics.valid_nlls.update(losses.nlls.detach(), losses.token_mask.detach())
          loss_to_return = losses.loss
        
        if loss_to_return is None: 
            losses_fallback = self._loss(batch['input_ids'], batch['attention_mask'],
                                     sampling_eps_min=current_sampling_eps_min_for_val,
                                     sampling_eps_max=current_sampling_eps_max_for_val)
            self.metrics.valid_nlls.update(losses_fallback.nlls.detach(), losses_fallback.token_mask.detach())
            loss_to_return = losses_fallback.loss
            # print("Warning: loss_to_return was None in validation_step, used fallback.")


        return loss_to_return 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
          self._get_parameters(), 
          lr=self.config.optim.lr,
          betas=(self.config.optim.beta1,
                 self.config.optim.beta2),
          eps=self.config.optim.eps,
          weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
          self.config.lr_scheduler, optimizer=optimizer)
        
        monitor_metric = 'val/loss' 
        if self.parameterization == 'bd3lm':
            monitor_metric = 'val/nll' 
        elif self.block_size == 1 and getattr(self.config.training, 'eval_nll', False):
            monitor_metric = 'val/nll'


        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step',
                          'monitor': monitor_metric, 
                          'name': 'trainer/lr'}

        return [optimizer], [scheduler_dict]

    def _resample_q_xt(self, x_blocks, xt_blocks_initial,
                         block_size_resample, sampling_eps_min_val, sampling_eps_max_val):
        B, N_Blk, BlkS = x_blocks.shape
        xt_blocks = xt_blocks_initial.clone() 

        max_resample_iters = 10 
        for _iter in range(max_resample_iters):
            current_mask_count = (xt_blocks == self.mask_index).long().sum(dim=-1) 
            current_mask_rate = current_mask_count.float() / BlkS

            target_min_masks_float = sampling_eps_min_val * BlkS
            target_max_masks_float = sampling_eps_max_val * BlkS

            active_min_bound = abs(sampling_eps_min_val - 1e-3) > 1e-6
            too_few_masks_indices = (current_mask_rate < sampling_eps_min_val - 1e-6) & active_min_bound 

            if too_few_masks_indices.any():
                for b_idx in range(B):
                    for blk_jdx in range(N_Blk):
                        if too_few_masks_indices[b_idx, blk_jdx]:
                            # Convert float to tensor for torch.ceil
                            num_to_add = torch.ceil(torch.tensor(target_min_masks_float, device=current_mask_count.device)).long() - current_mask_count[b_idx, blk_jdx]
                            if num_to_add.item() <= 0: continue

                            non_masked_indices_in_block = (xt_blocks[b_idx, blk_jdx] != self.mask_index).nonzero(as_tuple=False).squeeze(-1)
                            num_available_to_mask = non_masked_indices_in_block.numel()
                            
                            actual_num_to_add = min(num_to_add.item(), num_available_to_mask)
                            if actual_num_to_add > 0:
                                indices_to_mask = non_masked_indices_in_block[torch.randperm(num_available_to_mask, device=self.device)[:actual_num_to_add]]
                                xt_blocks[b_idx, blk_jdx, indices_to_mask] = self.mask_index
            
            current_mask_count = (xt_blocks == self.mask_index).long().sum(dim=-1)
            current_mask_rate = current_mask_count.float() / BlkS

            active_max_bound = abs(sampling_eps_max_val - 1.0) > 1e-6
            too_many_masks_indices = (current_mask_rate > sampling_eps_max_val + 1e-6) & active_max_bound 

            if too_many_masks_indices.any():
                for b_idx in range(B):
                    for blk_jdx in range(N_Blk):
                        if too_many_masks_indices[b_idx, blk_jdx]:
                            # Convert float to tensor for torch.floor
                            num_to_remove = current_mask_count[b_idx, blk_jdx] - torch.floor(torch.tensor(target_max_masks_float, device=current_mask_count.device)).long()
                            if num_to_remove.item() <= 0: continue
                            
                            masked_indices_in_block = (xt_blocks[b_idx, blk_jdx] == self.mask_index).nonzero(as_tuple=False).squeeze(-1)
                            num_available_to_unmask = masked_indices_in_block.numel()

                            actual_num_to_remove = min(num_to_remove.item(), num_available_to_unmask)
                            if actual_num_to_remove > 0:
                                indices_to_unmask = masked_indices_in_block[torch.randperm(num_available_to_unmask, device=self.device)[:actual_num_to_remove]]
                                xt_blocks[b_idx, blk_jdx, indices_to_unmask] = x_blocks[b_idx, blk_jdx, indices_to_unmask]
            
            final_mask_rate = (xt_blocks == self.mask_index).float().sum(dim=-1) / BlkS
            
            violations = torch.zeros_like(final_mask_rate, dtype=torch.bool)
            if active_min_bound:
                violations |= (final_mask_rate < sampling_eps_min_val - 1e-6)
            if active_max_bound:
                violations |= (final_mask_rate > sampling_eps_max_val + 1e-6)
                
            if not violations.any():
                break 
            if _iter == max_resample_iters - 1 and violations.any():
                # print(f"Warning: Max resamples ({max_resample_iters}) reached in _resample_q_xt, bounds may still be violated.")
                # print(f"Rates: {final_mask_rate}, Min: {sampling_eps_min_val}, Max: {sampling_eps_max_val}")
                pass
        return xt_blocks


    def q_xt(self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
        block_size_resample = self.block_size if block_size is None else block_size
        if block_size_resample == 0 : 
            block_size_resample = x.shape[1] if x.shape[1] > 0 else 1 

        if p.ndim == 3 and p.shape[-1] == 1: 
            p_broadcastable = p.squeeze(-1) 
        elif p.ndim == 2 and p.shape[1] == x.shape[1]: 
            p_broadcastable = p
        elif p.ndim == 2 and p.shape[1] == 1: 
            p_broadcastable = p.expand_as(x) 
        else:
            raise ValueError(f"Unsupported shape for p: {p.shape}, x shape: {x.shape}")

        move_indices = torch.rand_like(x, dtype=torch.float32) <= p_broadcastable
        xt = torch.where(move_indices, self.mask_index, x)

        current_sampling_eps_min = sampling_eps_min if sampling_eps_min is not None else torch.tensor(1e-3, device=x.device)
        current_sampling_eps_max = sampling_eps_max if sampling_eps_max is not None else torch.tensor(1.0, device=x.device)
        
        sampling_eps_min_val = current_sampling_eps_min.item() if isinstance(current_sampling_eps_min, torch.Tensor) else current_sampling_eps_min
        sampling_eps_max_val = current_sampling_eps_max.item() if isinstance(current_sampling_eps_max, torch.Tensor) else current_sampling_eps_max

        if sampling_eps_min_val >= (1.0 - 1e-6): 
            return torch.full_like(x, self.mask_index)

        is_standard_elbo_range = abs(sampling_eps_min_val - 1e-3) < 1e-6 and abs(sampling_eps_max_val - 1.0) < 1e-6
        
        if self.config.training.resample and not is_standard_elbo_range:
            if x.shape[1] % block_size_resample != 0 or x.shape[1]==0:
                 pass
            else:
                xt_blocks_view = xt.view(xt.shape[0], -1, block_size_resample)
                x_blocks_view = x.view_as(xt_blocks_view)
                
                xt_blocks_resampled = self._resample_q_xt( 
                    x_blocks_view, xt_blocks_view, 
                    block_size_resample,
                    sampling_eps_min_val,
                    sampling_eps_max_val
                )
                xt = xt_blocks_resampled.reshape(xt.shape[0], -1)
        return xt
    
    # Methods from _sample_prior downwards can be copied from the previous correct version.
    # Ensure they are consistent with any changes made above, particularly device handling
    # and how `self.forward` is called.
    @torch.no_grad()
    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(
          * batch_dims, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def _nucleus_sample(self, p_x0_probs): 
        p = self.config.sampling.nucleus_p
        if abs(p - 1.0) < 1e-6: 
          return p_x0_probs
        
        is_full_sequence = p_x0_probs.shape[1] > self.block_size and self.block_size > 0
        
        if is_full_sequence:
            p_x0_target_block = p_x0_probs[:, -self.block_size:].clone()
        else:
            p_x0_target_block = p_x0_probs.clone()

        sorted_probs, sorted_indices = p_x0_target_block.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        
        remove_mask = cum_probs > p
        remove_mask_shifted = torch.zeros_like(remove_mask)
        if remove_mask.shape[-1] > 1: 
            remove_mask_shifted[..., 1:] = remove_mask[..., :-1]
        
        final_keep_mask = ~remove_mask_shifted
        final_keep_mask[..., 0] = True 

        probs_in_nucleus_sorted_order = sorted_probs * final_keep_mask
        
        probs_after_nucleus = torch.zeros_like(p_x0_target_block)
        probs_after_nucleus.scatter_(-1, sorted_indices, probs_in_nucleus_sorted_order)

        probs_after_nucleus = probs_after_nucleus / probs_after_nucleus.sum(-1, keepdim=True).clamp(min=1e-9) 
        
        if is_full_sequence:
            p_x0_probs_out = p_x0_probs.clone()
            p_x0_probs_out[:, -self.block_size:] = probs_after_nucleus
            return p_x0_probs_out
        else:
            return probs_after_nucleus

    @torch.no_grad()
    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        _, move_chance_t = self.noise(t) 
        _, move_chance_s = self.noise(t - dt)
        sigma_t_global = self._sigma_from_p(move_chance_t) 
        
        move_chance_t_expanded = move_chance_t[:, None] 
        move_chance_s_expanded = move_chance_s[:, None] 
        mask_prob = move_chance_s_expanded / move_chance_t_expanded.clamp(min=1e-9) 

        current_block_len = x.shape[1] 

        if p_x0 is None:
          sigma_for_fwd = sigma_t_global.unsqueeze(-1).expand(-1, current_block_len) 
          
          model_input_for_fwd = x 
          
          raw_logits_or_logprobs = self.forward(model_input_for_fwd, 
                                                sigma=sigma_for_fwd, 
                                                sample_mode=True).to(torch.float64) 
          
          if self.parameterization == 'subs' or self.parameterization == 'ar': 
              p_x0 = raw_logits_or_logprobs.exp()
          else: 
              p_x0 = F.softmax(raw_logits_or_logprobs, dim=-1)

          p_x0 = self._nucleus_sample(p_x0) 

        if self.config.sampling.first_hitting:
          x_block_sampled = _sample_categorical(p_x0) 
          
          one_change_mask = torch.zeros_like(x_block_sampled, dtype=torch.bool)
          for b_idx in range(x.shape[0]):
              masked_positions = (x[b_idx] == self.mask_index).nonzero(as_tuple=False).squeeze(-1)
              num_masked_in_this_sample_block = masked_positions.shape[0]
              if num_masked_in_this_sample_block > 0:
                  chosen_idx_in_masked_list = torch.randint(0, num_masked_in_this_sample_block, (1,), device=self.device).item()
                  actual_idx_in_block = masked_positions[chosen_idx_in_masked_list]
                  one_change_mask[b_idx, actual_idx_in_block] = True
          
          x_block_new = torch.where(one_change_mask, x_block_sampled, x)

        else: 
          q_xs = p_x0 * (1.0 - mask_prob.unsqueeze(-1)) 
          if self.mask_index < q_xs.shape[-1]:
            q_xs[..., self.mask_index] += mask_prob.squeeze(-1).unsqueeze(-1) 
          x_block_new = _sample_categorical(q_xs) 

        copy_flag = (x != self.mask_index) 
        x_block_final = torch.where(copy_flag, x, x_block_new)
        
        if self.config.sampling.kv_cache and self.mask_index not in x_block_final: 
          sigma_for_kv_cache = sigma_t_global.unsqueeze(-1).expand(-1, current_block_len)
          _ = self.forward(x_block_final, sigma_for_kv_cache, sample_mode=True, store_kv=True)

        if not torch.allclose(x_block_final, x): 
          return None, x_block_final 
        else:
          return p_x0, x_block_final 

    @torch.no_grad()
    def _ar_sampler(self, bsz, context_len=1024):
        if self.config.sampling.kv_cache:
          self.backbone.reset_kv_cache(eval_batch_size=bsz) 

        with torch.amp.autocast('cuda', dtype=torch.float32):
          num_pred_tokens = self.num_tokens - 1
          if num_pred_tokens < 0: num_pred_tokens = 0 

          x = torch.zeros(
            (bsz, num_pred_tokens + 1),
            dtype=torch.long,
            device=self.device)
          if x.numel() > 0 : x[:, 0] = self.tokenizer.bos_token_id
          
          stop_flags = torch.zeros(bsz, dtype=torch.bool, device=self.device) 
          final_lengths = torch.full((bsz,), num_pred_tokens + 1, dtype=torch.long, device=self.device)


          for i in tqdm(range(num_pred_tokens), desc="AR Sampling"):
            if stop_flags.all(): break 

            active_indices = (~stop_flags).nonzero(as_tuple=True)[0]
            if len(active_indices) == 0: break

            x_active = x[active_indices, :i + 1][:, -context_len:]
            if x_active.shape[1] == 0: 
                x_active = x[active_indices, :i+1] 
            
            noise_active = (torch.distributions.Gumbel(0, 1)
                    .sample((len(active_indices), self.vocab_size))
                    .to(self.device))
            
            next_logits_active = self.forward(
              x_active, 
              None, 
              store_kv=self.config.sampling.kv_cache,
              sample_mode=True)[:, -1:].to(torch.float64) 

            next_probs_active = next_logits_active.exp() 
            next_probs_active_nuclear = self._nucleus_sample(next_probs_active) 
            
            y_active = (next_probs_active_nuclear.log().clamp(min=self.neg_infinity) + noise_active).argmax(-1) 
            
            x[active_indices, i + 1] = y_active

            if (i + 1) > 256: 
                 current_x_for_stop_check = x[active_indices, :i+2] 
                 stop_now_active_flags, x_truncated_active_list = self._check_stop_conds_ar_batch(current_x_for_stop_check)
                 
                 for idx_in_active_list, original_batch_idx_tensor in enumerate(active_indices):
                     original_batch_idx = original_batch_idx_tensor.item()
                     if stop_now_active_flags[idx_in_active_list]:
                         stop_flags[original_batch_idx] = True
                         final_lengths[original_batch_idx] = x_truncated_active_list[idx_in_active_list].shape[0]
          
          if not self.config.sampling.var_length:
              return x 
          else: 
                output_samples_list_of_tensors = []
                for b_idx_final in range(bsz):
                    output_samples_list_of_tensors.append(x[b_idx_final, :final_lengths[b_idx_final]])
                return output_samples_list_of_tensors 


    def _check_stop_conds_ar_batch(self, x_batch_active): 
        B_active = x_batch_active.shape[0]
        stop_flags_for_active = torch.zeros(B_active, dtype=torch.bool, device=x_batch_active.device)
        truncated_x_list_for_active = list(x_batch_active.unbind(0)) 

        for i in range(B_active):
            current_x_sample = x_batch_active[i].clone() 
            stop_this_sample_now = False
            current_truncate_idx_for_sample = current_x_sample.shape[0]

            check_entropy_len = min(256, current_x_sample.shape[0])
            if check_entropy_len > 0:
                entropy_val = self._compute_entropy(current_x_sample[-check_entropy_len:])
                if entropy_val < 4.0: 
                    stop_this_sample_now = True
                    if self.config.sampling.var_length: 
                        current_truncate_idx_for_sample = max(1, current_x_sample.shape[0] - check_entropy_len)


            if self.config.sampling.var_length:
                eos_indices = (current_x_sample == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    first_valid_eos_pos = -1
                    for eos_idx_val_item in eos_indices:
                        eos_idx = eos_idx_val_item.item()
                        is_bos_token_val = (current_x_sample.numel()>0 and current_x_sample[0] == self.tokenizer.bos_token_id)
                        is_eos_at_bos_val = (eos_idx == 0)
                        
                        if is_eos_at_bos_val and is_bos_token_val and (self.tokenizer.bos_token_id == self.tokenizer.eos_token_id):
                            if current_x_sample.shape[0] == 1: 
                                first_valid_eos_pos = 1 
                                break
                            elif len(eos_indices) > 1: 
                                first_valid_eos_pos = eos_indices[1].item() + 1
                                break
                        elif not is_eos_at_bos_val : 
                            first_valid_eos_pos = eos_idx + 1
                            break
                        elif is_eos_at_bos_val and not (is_bos_token_val and self.tokenizer.bos_token_id == self.tokenizer.eos_token_id):
                            first_valid_eos_pos = 1
                            break
                            
                    if first_valid_eos_pos != -1:
                        stop_this_sample_now = True
                        current_truncate_idx_for_sample = min(current_truncate_idx_for_sample, first_valid_eos_pos)
            
            if stop_this_sample_now:
                stop_flags_for_active[i] = True
                truncated_x_list_for_active[i] = current_x_sample[:current_truncate_idx_for_sample]
                if truncated_x_list_for_active[i].numel() == 0: 
                    truncated_x_list_for_active[i] = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long, device=x_batch_active.device)
        
        return stop_flags_for_active, truncated_x_list_for_active


    @torch.no_grad()
    def _sample(
      self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
        if seqlen is None:
          seqlen = self.config.model.length
        if batch_size_per_gpu is None:
          batch_size_per_gpu = self.config.loader.eval_batch_size 
        
        samples_list_of_decoded_texts = [] 
        
        if self.parameterization == 'ar':
          for _ in range(self.config.sampling.num_sample_batches):
            ar_output = self._ar_sampler(batch_size_per_gpu) 
            if isinstance(ar_output, list): 
                for s_tensor in ar_output:
                    if s_tensor is not None and s_tensor.numel() > 0:
                         samples_list_of_decoded_texts.append(self.tokenizer.decode(s_tensor, skip_special_tokens=True))
            elif ar_output is not None and ar_output.numel() > 0 : 
                samples_list_of_decoded_texts.extend(self.tokenizer.batch_decode(ar_output, skip_special_tokens=True))
            self.metrics.gen_nfes.append(self.config.model.length) 
          return samples_list_of_decoded_texts 
        
        if num_steps is None: num_steps = self.T 

        tensor_samples_list = []

        if self.sampler == 'semi_ar': 
          for _ in range(self.config.sampling.num_sample_batches):
            sample_i_tensor, num_tries = None, 0
            while sample_i_tensor is None:
              num_tries += 1
              if self.block_size == 0 : num_strides = 1 
              else: num_strides = seqlen // self.block_size
              
              seqlen_eff = seqlen
              if seqlen % self.block_size != 0 and self.parameterization == 'bd3lm' and self.block_size > 0:
                  seqlen_eff = num_strides * self.block_size 
                  if seqlen_eff == 0 : 
                      seqlen_eff = self.block_size
                      num_strides = 1
              
              if num_strides == 0 and seqlen_eff > 0: num_strides = 1 

              sample_i_tensor, nfes = self._semi_ar_sampler(
                n_samples=batch_size_per_gpu,
                num_strides=num_strides, 
                num_steps=num_steps, 
                seqlen=seqlen_eff) 
              if num_tries > 10 and sample_i_tensor is None: 
                print(f'Warning: {self.sampler} Sampling failed after multiple tries. Returning empty list for this batch.')
                break 
            if sample_i_tensor is not None: 
                tensor_samples_list.append(sample_i_tensor) 
                self.metrics.nfes.update(nfes if nfes is not None else 0)
                self.metrics.gen_nfes.append(nfes if nfes is not None else 0)
        else: 
          nfes = num_steps
          for _ in range(self.config.sampling.num_sample_batches):
            sample_i_tensor, num_tries = None, 0
            while sample_i_tensor is None:
              sample_i_tensor = self._analytic_sampler(
                n_samples=batch_size_per_gpu,
                num_steps=num_steps,
                seqlen=seqlen,
                eps=eps)
              num_tries += 1
              if num_tries > 10 and sample_i_tensor is None:
                print('Warning: Analytic Sampling failed. Returning empty list for this batch.')
                break
            if sample_i_tensor is not None:
                tensor_samples_list.append(sample_i_tensor)
                self.metrics.nfes.update(nfes)
                self.metrics.gen_nfes.append(nfes)
        
        if not tensor_samples_list: return [] 
        
        for s_batch_tensor in tensor_samples_list:
            if s_batch_tensor is None or s_batch_tensor.numel() == 0: continue
            if s_batch_tensor.ndim == 1: s_batch_tensor = s_batch_tensor.unsqueeze(0)
            samples_list_of_decoded_texts.extend(self.tokenizer.batch_decode(s_batch_tensor, skip_special_tokens=True))
            
        return samples_list_of_decoded_texts


    def _sigma_from_p(self, p):
        p_device = p.device
        log_val = torch.log(1.0 - p + 1e-9) 
        
        sigma_max_val = getattr(self.noise, 'sigma_max', 10.0) 
        if isinstance(sigma_max_val, torch.Tensor):
            sigma_max_tensor = sigma_max_val.to(p_device)
        else: 
            sigma_max_tensor = torch.tensor(sigma_max_val, device=p_device, dtype=p.dtype) 
            
        return torch.min(-log_val, sigma_max_tensor)


    def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
        if self.ema:
          self.ema.store(self._get_parameters())
          self.ema.copy_to(self._get_parameters())
        self.backbone.eval()
        self.meta_controller.eval() 
        self.noise.eval()
        
        actual_seqlen = seqlen if seqlen is not None else self.config.model.length
        
        samples_text = self._sample( 
          seqlen=actual_seqlen,
          batch_size_per_gpu=self.config.loader.eval_batch_size,
          num_steps=num_steps,
          eps=eps)
        
        if samples_text: 
            self.metrics.record_generative_perplexity(
              samples_text, 
              actual_seqlen, 
              self.config.loader.eval_batch_size, 
              self.device)
        return samples_text


    def get_score(self, x_token_ids, sigma_batch): 
        sigma_for_fwd = sigma_batch.unsqueeze(-1).expand(-1, x_token_ids.shape[1]) 

        model_output_logits = self.forward(x_token_ids, sigma_for_fwd).to(torch.float64) 
        
        if self.parameterization == 'ar' or self.parameterization == 'subs': 
            model_probs = model_output_logits.exp()
        elif self.parameterization == 'sedd':
             model_probs = F.softmax(model_output_logits, dim=-1)
        else: 
             model_probs = F.softmax(model_output_logits, dim=-1)

        if abs(self.config.sampling.nucleus_p - 1.0) < 1e-6:
          return model_probs
        
        model_probs_nuclear = self._nucleus_sample(model_probs) 
        return model_probs_nuclear 

    def _staggered_score(self, score_probs, dsigma_batch): 
        score_probs_clone = score_probs.clone()
        
        dsigma_reshaped_for_sum = dsigma_batch.view(-1,1,1) 
        dsigma_reshaped_for_mult = dsigma_batch.view(-1,1,1) 

        sum_probs_over_vocab = score_probs_clone.sum(dim=-1, keepdim=True) 
        extra_const = (1.0 - torch.exp(dsigma_reshaped_for_sum)) * sum_probs_over_vocab 
        
        score_probs_clone = score_probs_clone * torch.exp(dsigma_reshaped_for_mult) 
        
        if self.mask_index < score_probs_clone.shape[-1]: 
             score_probs_clone[..., self.mask_index] += extra_const.squeeze(-1) 
        return score_probs_clone

    def _analytic_update(self, x, t_batch, dt_scalar): 
        sigma_t = self._sigma_from_p(self.noise(t_batch)[1]) 
        sigma_s = self._sigma_from_p(self.noise(t_batch - dt_scalar)[1]) 
        dsigma = sigma_t - sigma_s 
        
        score_probs = self.get_score(x, sigma_t) 
        stag_score_probs = self._staggered_score(score_probs, dsigma) 
        
        dsigma_per_token = dsigma.unsqueeze(-1).expand(-1, x.shape[1]) 
        trans_probs = self._transp_transition(x, dsigma_per_token) 
        
        probs_final = stag_score_probs * trans_probs
        return _sample_categorical(probs_final)


    def _denoiser_update(self, x, t_batch): 
        sigma = self._sigma_from_p(self.noise(t_batch)[1]) 
        score_probs = self.get_score(x, sigma) 
        stag_score_probs = self._staggered_score(score_probs, sigma) 
        
        sigma_per_token = sigma.unsqueeze(-1).expand_as(x)
        trans_probs = self._transp_transition(x, sigma_per_token)
        
        probs_final = stag_score_probs * trans_probs
        if self.mask_index < probs_final.shape[-1]:
            probs_final[..., self.mask_index] = 0.0 
        samples = _sample_categorical(probs_final)
        return samples


    def _transp_transition(self, i_tokens, sigma_per_token): 
        sigma_expanded = sigma_per_token.unsqueeze(-1) 
        
        one_hot_i = F.one_hot(i_tokens, num_classes=self.vocab_size).type_as(sigma_expanded)
        edge = torch.exp(-sigma_expanded) * one_hot_i
        
        mask_is_mask_index_bool = (i_tokens == self.mask_index) 
        term_to_add_for_mask = (1.0 - torch.exp(-sigma_per_token)) 
        
        if self.mask_index < edge.shape[-1]: 
            edge[mask_is_mask_index_bool, self.mask_index] = term_to_add_for_mask[mask_is_mask_index_bool]
        
        return edge

    def _sample_t(self, batch_dims_for_blocks, device, sampling_eps_min, sampling_eps_max, block_size=None): 
        _eps_b = torch.rand(batch_dims_for_blocks, device=device, dtype=torch.float32) 

        if self.antithetic_sampling:
            num_total_block_samples = batch_dims_for_blocks[0] * batch_dims_for_blocks[1]
            if num_total_block_samples > 0: 
                offset_b = torch.arange(num_total_block_samples, device=device, dtype=torch.float32) / num_total_block_samples
                offset_b = offset_b.view(batch_dims_for_blocks[0], batch_dims_for_blocks[1])
                _eps_b = (_eps_b / num_total_block_samples + offset_b) % 1.0
        
        t_per_block = _eps_b 

        s_eps_max_val = sampling_eps_max.item() if isinstance(sampling_eps_max, torch.Tensor) else sampling_eps_max
        s_eps_min_val = sampling_eps_min.item() if isinstance(sampling_eps_min, torch.Tensor) else sampling_eps_min
        
        is_nll_case_for_block_level_t = (block_size == 1 and s_eps_max_val >= (1.0-1e-6) and s_eps_min_val >= (1.0-1e-6))
        if is_nll_case_for_block_level_t :
            return torch.ones_like(t_per_block, dtype=torch.float32)

        current_sampling_eps_min_tensor = sampling_eps_min.to(device=device, dtype=torch.float32) if isinstance(sampling_eps_min, torch.Tensor) else torch.tensor(sampling_eps_min, device=device, dtype=torch.float32)
        current_sampling_eps_max_tensor = sampling_eps_max.to(device=device, dtype=torch.float32) if isinstance(sampling_eps_max, torch.Tensor) else torch.tensor(sampling_eps_max, device=device, dtype=torch.float32)

        t_per_block = t_per_block * (current_sampling_eps_max_tensor - current_sampling_eps_min_tensor) + current_sampling_eps_min_tensor
        return t_per_block

    def _maybe_sub_sample(self, x0, attention_mask):
        seqlen = x0.shape[1]
        if seqlen == 0 : 
            return x0, None, attention_mask

        if seqlen > self.num_tokens and self.num_tokens > 0 : 
          start_max = seqlen - self.num_tokens
          start = np.random.randint(0, start_max + 1) if start_max >=0 else 0
          
          end = start + self.num_tokens
          if end > seqlen: 
              end = seqlen
              start = max(0, end - self.num_tokens)


          input_tokens = x0[:, start: end]
          output_tokens = x0[:, start + 1: end + 1] if self.parameterization == 'ar' and (end+1 <= seqlen) else None
          new_attention_mask = attention_mask[:, start: end]

          if hasattr(self.config.data, 'insert_train_special') and self.config.data.insert_train_special == True:
            if input_tokens.numel() > 0 and input_tokens.shape[1]>0 : input_tokens[:, 0] = self.tokenizer.bos_token_id
            if output_tokens is not None and output_tokens.numel() > 0 and output_tokens.shape[1]>0: output_tokens[:, -1] = self.tokenizer.eos_token_id
        elif self.parameterization == 'ar':
          if seqlen <= 1: 
              input_tokens = x0 
              output_tokens = None
              new_attention_mask = attention_mask[:, :seqlen] # Match input_tokens length
          else:
              input_tokens = x0[:, :-1]
              output_tokens = x0[:, 1:]
              new_attention_mask = attention_mask[:, :-1] 
        else: 
          input_tokens = x0
          output_tokens = None 
          new_attention_mask = attention_mask

        return input_tokens, output_tokens, new_attention_mask


    def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
        if sampling_eps_min is None:
            if hasattr(self, 'sampling_eps_min') and isinstance(self.sampling_eps_min, torch.Tensor):
                current_sampling_eps_min = self.sampling_eps_min
                current_sampling_eps_max = self.sampling_eps_max
            # MODIFIED PART HERE
            elif isinstance(self.config.training.sampling_eps, (tuple, ListConfig)) and len(self.config.training.sampling_eps) == 2:
                current_sampling_eps_min = torch.tensor(self.config.training.sampling_eps[0], device=x0.device, dtype=torch.float32)
                current_sampling_eps_max = torch.tensor(self.config.training.sampling_eps[1], device=x0.device, dtype=torch.float32)
            elif not isinstance(self.config.training.sampling_eps, (tuple, ListConfig)): # scalar case
                current_sampling_eps_min = torch.tensor(self.config.training.sampling_eps, device=x0.device, dtype=torch.float32)
                current_sampling_eps_max = torch.tensor(1.0, device=x0.device, dtype=torch.float32) # Default max if only min is scalar
            else: # Should not happen if config is (float, float) or float
                raise ValueError(f"Unexpected type/format for self.config.training.sampling_eps: {self.config.training.sampling_eps}")

        else:
            current_sampling_eps_min = sampling_eps_min.to(x0.device) if isinstance(sampling_eps_min, torch.Tensor) else torch.tensor(sampling_eps_min, device=x0.device, dtype=torch.float32)
            current_sampling_eps_max = sampling_eps_max.to(x0.device) if isinstance(sampling_eps_max, torch.Tensor) else torch.tensor(sampling_eps_max, device=x0.device, dtype=torch.float32)



        input_tokens, _, new_attention_mask = self._maybe_sub_sample(x0, attention_mask)
        B, N_Tokens = input_tokens.shape
        if N_Tokens == 0: 
            return Loss(loss=torch.tensor(0.0, device=x0.device, requires_grad=True), 
                        nlls=torch.empty((B,0), device=x0.device), 
                        token_mask=torch.empty((B,0), device=x0.device))


        if self.parameterization == 'bd3lm':
            if self.block_size == 0 or N_Tokens < self.block_size or N_Tokens % self.block_size != 0: 
                 _dummy_t = torch.full((B,1), 0.5, device=input_tokens.device, dtype=torch.float32)
                 _, _dummy_move_chance = self.noise(_dummy_t)
                 _dummy_sigma = self._sigma_from_p(_dummy_move_chance)
                 _dummy_xt = self.q_xt(input_tokens, _dummy_move_chance, block_size=N_Tokens if N_Tokens > 0 else 1, sampling_eps_min=current_sampling_eps_min, sampling_eps_max=current_sampling_eps_max)
                 _dummy_model_input = _dummy_xt
                 if self.cross_attn: _dummy_model_input = torch.cat((_dummy_xt, input_tokens), dim=-1)
                
                 _dummy_sigma_fwd = _dummy_sigma.unsqueeze(-1).expand(-1,N_Tokens) if N_Tokens > 0 else _dummy_sigma.unsqueeze(-1)
                 _dummy_logits = self.forward(_dummy_model_input, sigma=_dummy_sigma_fwd)
                 if self.cross_attn and self.config.algo.backbone != 'hf_dit' and N_Tokens > 0 : _dummy_logits = _dummy_logits[:,:N_Tokens,:]

                 if N_Tokens > 0:
                    _dummy_log_probs = F.log_softmax(_dummy_logits, dim=-1)
                    _dummy_nll_term = -torch.gather(_dummy_log_probs, -1, input_tokens.unsqueeze(-1)).squeeze(-1)
                    _dummy_masked_nll = (_dummy_nll_term * new_attention_mask)
                    _dummy_mean_nll = _dummy_masked_nll.sum() / new_attention_mask.sum().clamp(min=1.0)
                 else: # Should not happen given N_Tokens == 0 check earlier
                    _dummy_mean_nll = torch.tensor(0.0, device=x0.device, requires_grad=True)
                    _dummy_nll_term = torch.empty((B,0), device=x0.device)

                 return Loss(loss=_dummy_mean_nll, nlls=_dummy_nll_term, token_mask=new_attention_mask)


            N_Blk = N_Tokens // self.block_size
            x0_block_features = self._get_block_features(input_tokens, new_attention_mask) 

            u_samples_per_block = self._sample_t(
                (B, N_Blk), 
                input_tokens.device,
                current_sampling_eps_min,
                current_sampling_eps_max,
                block_size=1 
            ) 

            alpha_b_u_all_blocks, loss_scale_b_u_all_blocks, p_b_u_all_blocks, s_b_for_penalty, log_s_tilde_b = \
                self._get_warped_noise_outputs_for_block_batch(x0_block_features, u_samples_per_block)

            p_per_token = p_b_u_all_blocks.repeat_interleave(self.block_size, dim=1) 
            loss_scale_per_token = loss_scale_b_u_all_blocks.repeat_interleave(self.block_size, dim=1)
            
            sigma_all_tokens_for_backbone = self._sigma_from_p(p_per_token) 
            # Pass sigma_all_tokens_for_backbone (B,N_T) to _process_sigma, it expects (B,N_T) or (B,N_T,1)
            processed_sigma_for_backbone = self._process_sigma(sigma_all_tokens_for_backbone) 


            xt = self.q_xt(input_tokens,
                           p_per_token.unsqueeze(-1), 
                           block_size=self.block_size, 
                           sampling_eps_min=current_sampling_eps_min, 
                           sampling_eps_max=current_sampling_eps_max)

            if self.ignore_bos and N_Tokens > 0:
                xt[:, 0] = input_tokens[:, 0]

            x_model_input = xt
            if self.cross_attn:
                x_model_input = torch.cat((xt, input_tokens), dim=-1)

            # Pass sigma_all_tokens_for_backbone to self.forward, it will call _process_sigma internally
            model_output_logits = self.forward(x_model_input, sigma=sigma_all_tokens_for_backbone) 
            
            log_probs_x0 = F.log_softmax(model_output_logits, dim=-1)
            log_p_theta_x0_given_xt_per_token = torch.gather(log_probs_x0, -1, input_tokens.unsqueeze(-1)).squeeze(-1)
            
            nelbo_term_per_token = loss_scale_per_token * log_p_theta_x0_given_xt_per_token 
            
            s_b_for_penalty_device = s_b_for_penalty.to(self.device)
            base_alpha_bar_at_1_for_penalty = self.base_noise_schedule.get_alpha_bar(
                torch.ones_like(s_b_for_penalty_device.squeeze(-1)) 
            ) 
            
            _alpha_b_at_1_warped = noise_schedule.get_warped_alpha_b_t(
                base_alpha_bar_at_1_for_penalty,
                self.base_log_alpha_bar_at_0,        
                s_b_for_penalty_device,       
                torch.logit(self.target_alpha_at_0.to(base_alpha_bar_at_1_for_penalty.device)) 
            ) 

            surrogate_penalty_per_block = noise_schedule.compute_surrogate_steps_penalty(
                _alpha_b_at_1_warped, 
                self.min_alpha_1_target,
                self.lambda_min_alpha_1_penalty,
                self.alpha_1_clamp_min,
                self.alpha_1_clamp_max
            ) 
            mean_surrogate_penalty = surrogate_penalty_per_block.mean()

            s_b_l2_penalty = torch.tensor(0.0, device=input_tokens.device)
            if self.lambda_s_b_l2_penalty > 0:
                s_b_l2_penalty = self.lambda_s_b_l2_penalty * (log_s_tilde_b**2).mean()

            masked_nelbo_term = (nelbo_term_per_token * new_attention_mask)
            mean_nelbo = masked_nelbo_term.sum() / new_attention_mask.sum().clamp(min=1.0)

            total_loss = mean_nelbo + self.lambda_steps_penalty * mean_surrogate_penalty + s_b_l2_penalty
            
            if hasattr(self, 'trainer') and self.trainer is not None and not self.trainer.sanity_checking:
                self.log('train/nelbo', mean_nelbo, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
                if self.parameterization == 'bd3lm': 
                     self.log('train/surrogate_penalty', mean_surrogate_penalty, on_step=True, on_epoch=False, sync_dist=True)
                     if self.lambda_s_b_l2_penalty > 0:
                         self.log('train/s_b_l2_penalty', s_b_l2_penalty, on_step=True, on_epoch=False, sync_dist=True)
            
            nlls_for_metric = nelbo_term_per_token 
            current_loss_for_return = total_loss

        elif self.parameterization == 'ar':
            if input_tokens.shape[1] <= 1 and x0.shape[1] <=1 : 
                 return Loss(loss=torch.tensor(0.0, device=x0.device, requires_grad=True), nlls=torch.zeros_like(input_tokens, dtype=torch.float32), token_mask=new_attention_mask)

            output_ar = self.forward(input_tokens, None) 
            
            # Determine target based on original x0 and how input_tokens was derived
            if x0.shape[1] > input_tokens.shape[1]: # input_tokens was result of subsampling
                # This case is complex for AR, as output_ar corresponds to input_tokens
                # Need to ensure target aligns with input_tokens context.
                # Simple approach: target is next token for each token in input_tokens, from original x0
                # This requires knowing the start index of input_tokens within x0 if _maybe_sub_sample changed it.
                # For now, assume _maybe_sub_sample for AR makes input_tokens = x0[:, :-1], output_tokens = x0[:, 1:]
                # So, target is effectively x0[:, 1:]
                 target_ar = x0[:, 1:input_tokens.shape[1]+1] # Target for x0[0] is x0[1], for x0[L-1] is x0[L]
            else: # No subsampling or input_tokens is already x0[:,:-1]
                 target_ar = x0[:, 1:]


            if target_ar.shape[1] == 0: 
                 return Loss(loss=torch.tensor(0.0, device=x0.device, requires_grad=True), nlls=torch.zeros_like(input_tokens, dtype=torch.float32), token_mask=new_attention_mask)
            
            min_len_ar = min(output_ar.shape[1], target_ar.shape[1])
            output_ar_matched = output_ar[:,:min_len_ar,:]
            target_ar_matched = target_ar[:,:min_len_ar]
            mask_ar = new_attention_mask[:, :min_len_ar]


            loss_ar = - output_ar_matched.gather(-1, target_ar_matched.unsqueeze(-1)).squeeze(-1)
            
            mean_loss_ar = (loss_ar * mask_ar).sum() / mask_ar.sum().clamp(min=1.0)
            nlls_for_metric = loss_ar 
            current_loss_for_return = mean_loss_ar
        else: 
            if t is None:
                _eps = torch.rand(input_tokens.shape[0], 1, device=input_tokens.device, dtype=torch.float32)
                if self.antithetic_sampling:
                    offset = torch.arange(
                        input_tokens.shape[0], device=input_tokens.device, dtype=torch.float32) / max(1,input_tokens.shape[0]) # Avoid div by zero
                    _eps = (_eps / max(1,input_tokens.shape[0]) + offset[:, None]) % 1.0
                t_for_loss = _eps * (current_sampling_eps_max - current_sampling_eps_min) + current_sampling_eps_min
            else: 
                t_val = t.item() if isinstance(t,torch.Tensor) and t.numel()==1 else t
                t_for_loss = t_val * torch.ones(input_tokens.shape[0], 1, device=input_tokens.device, dtype=torch.float32)
            
            loss_scale, move_chance = self.noise(t_for_loss) 
            sigma_for_loss = self._sigma_from_p(move_chance) 
            
            xt = self.q_xt(input_tokens, move_chance, 
                           block_size=self.block_size, 
                           sampling_eps_min=current_sampling_eps_min, 
                           sampling_eps_max=current_sampling_eps_max)
            
            if self.ignore_bos and N_Tokens > 0:
                xt[:, 0] = input_tokens[:, 0]

            x_model_input_other = xt
            if self.cross_attn:
                x_model_input_other = torch.cat((xt, input_tokens), dim=-1)
            
            sigma_for_fwd_other = sigma_for_loss.unsqueeze(-1).expand(-1, N_Tokens) if N_Tokens > 0 else sigma_for_loss.unsqueeze(-1)
            model_output_logits_other = self.forward(x_model_input_other, sigma=sigma_for_fwd_other) 
            
            if self.parameterization == 'sedd':
                loss_other = self._score_entropy(model_output_logits_other, sigma_for_loss, xt, input_tokens)
            else: 
                log_probs_x0_other = F.log_softmax(model_output_logits_other, dim=-1) 
                loss_other = -torch.gather(log_probs_x0_other, -1, input_tokens.unsqueeze(-1)).squeeze(-1)
            
            if self.mdlm_loss_scale: 
                loss_other *= loss_scale.squeeze(-1) 
            
            masked_loss_other = (loss_other * new_attention_mask)
            mean_loss_other = masked_loss_other.sum() / new_attention_mask.sum().clamp(min=1.0)
            
            nlls_for_metric = loss_other 
            current_loss_for_return = mean_loss_other
            
        return Loss(loss=current_loss_for_return,
                    nlls=nlls_for_metric, 
                    token_mask=new_attention_mask)


    def _clipped_schedule_search(self):
        best_var = float('inf')
        if not hasattr(self, 'sampling_eps_min') or not hasattr(self, 'sampling_eps_max'):
             self.sampling_eps_min = torch.tensor(self.config.training.sampling_eps_min, device=self.device, dtype=torch.float32)
             self.sampling_eps_max = torch.tensor(self.config.training.sampling_eps_max, device=self.device, dtype=torch.float32)

        sampling_eps_min_best = self.sampling_eps_min.item() 
        sampling_eps_max_best = self.sampling_eps_max.item()

        for (eps_min_float, eps_max_float), var_list_cpu_tensors in self.metrics.valid_vars.items():
          if not var_list_cpu_tensors: continue 
          
          var_list_cpu_tensors_on_cpu = [t.cpu() for t in var_list_cpu_tensors if t is not None and t.numel() > 0] # Filter None and empty
          if not var_list_cpu_tensors_on_cpu: continue # Skip if list becomes empty after filtering

          all_vars_tensor_cpu = torch.cat(var_list_cpu_tensors_on_cpu, dim=0) 
          if all_vars_tensor_cpu.numel() == 0: continue # Skip if concatenated tensor is empty

          all_vars_tensor_device = all_vars_tensor_cpu.to(self.device)
          
          gathered_vars = self.all_gather(all_vars_tensor_device) 

          if isinstance(gathered_vars, list) and all(isinstance(t, torch.Tensor) for t in gathered_vars):
            if not gathered_vars : continue # if list is empty after all_gather
            gathered_vars_flat = torch.cat([gv.view(-1) for gv in gathered_vars if gv.numel() > 0]) # filter empty tensors before cat
          elif isinstance(gathered_vars, torch.Tensor): 
            gathered_vars_flat = gathered_vars.view(-1)
          else: 
            print(f"Warning: Unexpected output from all_gather: {type(gathered_vars)}")
            continue
          
          if gathered_vars_flat.numel() == 0: continue # Skip if empty after flattening

          if gathered_vars_flat.numel() > 1 : 
            current_total_variance = gathered_vars_flat.var()
          else: 
            current_total_variance = torch.tensor(0.0, device=self.device, dtype=torch.float32)
          
          if current_total_variance < best_var:
            best_var = current_total_variance
            sampling_eps_min_best = eps_min_float
            sampling_eps_max_best = eps_max_float
          self.log(f'valid_var_{round(eps_min_float, 3)}_{round(eps_max_float, 3)}', 
                    current_total_variance.item() if isinstance(current_total_variance, torch.Tensor) else current_total_variance, 
                    on_epoch=True,
                    on_step=False,
                    sync_dist=False) 
        if self.config.algo.fix_clipping == False:
          self.sampling_eps_min.fill_(sampling_eps_min_best)
          self.sampling_eps_max.fill_(sampling_eps_max_best)

    def _score_entropy(self, log_score, sigma_batch, xt, x0): 
        masked_indices = xt == self.mask_index 
        sigma_expanded_like_xt = sigma_batch.unsqueeze(-1).expand_as(xt) 

        expsig_minus_1 = torch.expm1(sigma_expanded_like_xt) 
        
        entropy = torch.zeros_like(xt, dtype=torch.float32) 

        if masked_indices.any(): 
            q_ratio = 1.0 / expsig_minus_1[masked_indices].clamp(min=1e-9) 
            words_that_were_masked = x0[masked_indices]
            
            log_score_masked = log_score[masked_indices] 
            
            neg_term = q_ratio * torch.gather(
              log_score_masked,
              -1,
              words_that_were_masked.unsqueeze(-1)).squeeze(-1)
            
            score_exp_masked = log_score_masked.exp()

            pos_term_sum_parts = []
            if self.mask_index > 0: 
                pos_term_sum_parts.append(score_exp_masked[..., :self.mask_index].sum(dim=-1))
            if self.mask_index < score_exp_masked.shape[-1] -1 : 
                pos_term_sum_parts.append(score_exp_masked[..., self.mask_index + 1:].sum(dim=-1))
            
            if not pos_term_sum_parts: 
                 pos_term = torch.zeros_like(neg_term)
            else:
                 pos_term = torch.sum(torch.stack(pos_term_sum_parts, dim=0), dim=0) if len(pos_term_sum_parts) > 1 else pos_term_sum_parts[0]

            const = q_ratio * (q_ratio.log() - 1.0) 
            
            entropy[masked_indices] = pos_term - neg_term + const
        return entropy
    
    # Methods from _analytic_sampler downwards remain largely unchanged from the previous version.
    # Assuming they were correct or less impacted by the specific errors seen.
    # They use other corrected methods like get_score, _sigma_from_p, etc.
    @torch.no_grad()
    def _analytic_sampler(
      self, n_samples, num_steps, seqlen, eps=1e-5):
        x = self._sample_prior(
          n_samples,
          seqlen).to(self.device)
        if x.numel() > 0 : 
            x[:, 0] = self.tokenizer.bos_token_id
        
        timesteps = torch.linspace(
          1.0, eps, num_steps + 1, device=self.device) 
        dt = (1.0 - eps) / num_steps 
        
        for i in tqdm(range(num_steps), desc='Analytic Step'):
          t_scalar = timesteps[i] 
          t_batch = t_scalar * torch.ones(x.shape[0], 1, device=self.device, dtype=torch.float32) 
          x = self._analytic_update(x=x, t=t_batch, dt=dt)
        
        t_scalar_final = timesteps[-1]
        t_batch_final = t_scalar_final * torch.ones(x.shape[0], 1, device=self.device, dtype=torch.float32)
        x = self._denoiser_update(x=x, t=t_batch_final)

        stop_signal, x_final_or_list = self._check_stop_conds(x) 
        
        if not self.config.sampling.var_length: 
            if stop_signal and x_final_or_list is None: 
                return None 
            return x_final_or_list 
        else: 
            if stop_signal and x_final_or_list is None: 
                 return None 
            return x_final_or_list 


    @torch.no_grad()
    def _semi_ar_sampler(
      self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
        if seqlen is None:
          seqlen = self.config.model.length
        if seqlen == 0: 
            return torch.empty((n_samples, 0), dtype=torch.long, device=self.device), 0

        sampling_steps = 0

        mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
        current_context_size_for_model = self.block_size if mdlm_semi_ar else context_size


        ones_dtype = torch.float32 
        if hasattr(self, 'dtype'): ones_dtype = self.dtype
        ones = torch.ones((n_samples,1), dtype=ones_dtype, device=self.device)

        if self.config.sampling.kv_cache:
          self.backbone.reset_kv_cache(eval_batch_size=n_samples) 

        x_accum = None
        if num_strides == 0 and seqlen > 0: num_strides = 1 

        for stride_num in tqdm(range(num_strides), desc="Semi-AR Strides"):
          current_block_gen_len = 512 if mdlm_semi_ar and stride_num > 0 else self.block_size
          if x_accum is not None and x_accum.shape[1] + current_block_gen_len > seqlen :
              current_block_gen_len = seqlen - x_accum.shape[1]
          if current_block_gen_len <=0 : break


          if stride_num == 0:
            x_current_block_prior = self._sample_prior(n_samples, current_block_gen_len).to(self.device)
            if x_current_block_prior.numel() > 0 and x_current_block_prior.shape[1] > 0 :
                x_current_block_prior[:, 0] = self.tokenizer.bos_token_id
            x_accum = x_current_block_prior
          else:
            x_new_block_prior = self._sample_prior(n_samples, current_block_gen_len).to(self.device)
            if x_accum is None: x_accum = x_new_block_prior 
            else: x_accum = torch.cat((x_accum, x_new_block_prior), dim=1)

          
          current_block_actual_end_in_accum = x_accum.shape[1]
          window_start = max(0, current_block_actual_end_in_accum - current_context_size_for_model)
          x_model_input_window = x_accum[:, window_start:current_block_actual_end_in_accum]
          
          x_to_denoise_this_stride = x_accum[:, -current_block_gen_len:] 


          dt_per_step = 1.0 / num_steps 
          p_x0_cache_block = None 
          
          timesteps_for_block_ddpm = torch.linspace(1.0, dt_per_step, num_steps, device=self.device) 
          
          for step_idx_in_block in range(num_steps):
            if x_to_denoise_this_stride.numel() == 0: break 

            current_t_for_block = timesteps_for_block_ddpm[step_idx_in_block] * ones 
            
            # The `x` passed to _ddpm_caching_update should be the current state of the block being refined.
            # If BD3LM, this is x_to_denoise_this_stride.
            # If other semi-ar, it's the relevant part of x_model_input_window.
            # For simplicity, we assume x_to_denoise_this_stride is the primary target for _ddpm_caching_update.
            # The `self.forward` call inside _ddpm_caching_update will get the right context (e.g. full x_model_input_window)
            # if needed for non-kv_cache models, or just x_to_denoise_this_stride if kv_cache is on.
            # This is a subtle part. Let's assume _ddpm_caching_update's `x` is the current version of what was `x_to_denoise_this_stride`.
            
            p_x0_cache_block, x_to_denoise_this_stride_updated = self._ddpm_caching_update(
                x=x_to_denoise_this_stride, # Pass the block being actively denoised
                t=current_t_for_block, 
                dt=dt_per_step, 
                p_x0=p_x0_cache_block)
            
            x_to_denoise_this_stride = x_to_denoise_this_stride_updated

            if p_x0_cache_block is None: 
              sampling_steps += 1
            
            if self.mask_index not in x_to_denoise_this_stride: 
                break 
          
          x_accum[:, -current_block_gen_len:] = x_to_denoise_this_stride
          
          if x_accum.shape[1] > 256 : 
            stop_batch_flag, x_accum_maybe_truncated_list_or_tensor = self._check_stop_conds(x_accum)
            if self.config.sampling.var_length:
                if stop_batch_flag : 
                    if isinstance(x_accum_maybe_truncated_list_or_tensor, list):
                        x_accum = x_accum_maybe_truncated_list_or_tensor[0].unsqueeze(0) 
                        if x_accum.numel() == 0 or (x_accum.shape[1] == 1 and x_accum.shape[0]>0 and x_accum[0,0] == self.tokenizer.bos_token_id):
                            return None, None 
                    elif x_accum_maybe_truncated_list_or_tensor is None: 
                        return None, None
                    break 

            elif stop_batch_flag and x_accum_maybe_truncated_list_or_tensor is None: 
                 return None, None
        
        if x_accum is None or x_accum.numel() == 0 : return None, None 
        
        if self.parameterization != 'bd3lm' and x_to_denoise_this_stride.numel() > 0 :
            final_t_for_denoise = torch.tensor(dt_per_step, device=self.device, dtype=torch.float32) * ones 
            final_sigma = self._sigma_from_p(self.noise(final_t_for_denoise)[1])
            processed_final_sigma = self._process_sigma(final_sigma.unsqueeze(-1).expand(-1,x_to_denoise_this_stride.shape[1]))

            backbone_input_arg_name_final = 'input_ids' if self.config.algo.backbone == 'hf_dit' else 'indices'
            final_denoise_input = {backbone_input_arg_name_final: x_to_denoise_this_stride}


            if self.config.algo.backbone == 'hf_dit':
                 denoised_logits = self.backbone(**final_denoise_input, timesteps=processed_final_sigma).logits
                 x_denoised_final_block = denoised_logits.argmax(-1)
            else: 
                 x_denoised_final_block = self.backbone(**final_denoise_input, sigma=processed_final_sigma).argmax(-1)

            x_accum[:, -x_to_denoise_this_stride.shape[1]:] = x_denoised_final_block 
        
        return x_accum, sampling_steps


    def _compute_entropy(self, x_sample): 
        if x_sample.numel() == 0: return torch.tensor(0.0, device=x_sample.device, dtype=torch.float32)
        _, counts = torch.unique(x_sample, return_counts=True, sorted=False)
        if counts.sum() == 0: return torch.tensor(0.0, device=x_sample.device, dtype=torch.float32)
        probs = counts.float() / counts.sum()
        entropy = torch.special.entr(probs).sum()
        return entropy


    def _check_stop_conds(self, x_batch): 
        B, SeqLen = x_batch.shape
        stop_flags_batch = torch.zeros(B, dtype=torch.bool, device=x_batch.device)
        final_x_batch_list_of_tensors = list(x_batch.clone().unbind(0)) 

        for b_idx in range(B):
            current_x_sample = x_batch[b_idx].clone() 
            stop_this_sample = False
            current_truncate_idx = current_x_sample.shape[0] 

            check_entropy_len = min(256, current_truncate_idx)
            if check_entropy_len > 0:
                entropy_val = self._compute_entropy(current_x_sample[-check_entropy_len:])
                if entropy_val < 4.0: 
                    stop_this_sample = True
                    if self.config.sampling.var_length: 
                        current_truncate_idx = max(1, current_truncate_idx - check_entropy_len)


            if self.config.sampling.var_length:
                eos_indices = (current_x_sample == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    first_valid_eos_pos = -1
                    for eos_idx_val_item in eos_indices:
                        eos_idx = eos_idx_val_item.item()
                        is_bos_token_val = (current_x_sample.numel()>0 and current_x_sample[0] == self.tokenizer.bos_token_id)
                        is_eos_at_bos_val = (eos_idx == 0)
                        
                        if is_eos_at_bos_val and is_bos_token_val and (self.tokenizer.bos_token_id == self.tokenizer.eos_token_id):
                            if current_x_sample.shape[0] == 1: 
                                first_valid_eos_pos = 1 
                                break
                            elif len(eos_indices) > 1: 
                                first_valid_eos_pos = eos_indices[1].item() + 1
                                break
                        elif not is_eos_at_bos_val : 
                            first_valid_eos_pos = eos_idx + 1
                            break
                        elif is_eos_at_bos_val and not (is_bos_token_val and self.tokenizer.bos_token_id == self.tokenizer.eos_token_id):
                            first_valid_eos_pos = 1
                            break
                            
                    if first_valid_eos_pos != -1:
                        stop_this_sample = True
                        current_truncate_idx = min(current_truncate_idx, first_valid_eos_pos)
            
            if stop_this_sample:
                stop_flags_batch[b_idx] = True
                final_x_batch_list_of_tensors[b_idx] = current_x_sample[:current_truncate_idx]
                if final_x_batch_list_of_tensors[b_idx].numel() == 0:
                    final_x_batch_list_of_tensors[b_idx] = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long, device=x_batch.device)


        if not self.config.sampling.var_length: 
            if stop_flags_batch.any(): 
                return True, None 
            return False, x_batch 
        else: 
            all_effectively_empty_or_just_bos = True
            if B > 0 : 
                for s_tensor in final_x_batch_list_of_tensors:
                    is_just_bos = (s_tensor.shape[0] == 1 and s_tensor.numel()>0 and s_tensor[0] == self.tokenizer.bos_token_id)
                    if s_tensor.numel() > 0 and not is_just_bos :
                        all_effectively_empty_or_just_bos = False
                        break
                if all_effectively_empty_or_just_bos :
                    return True, None 
            elif B == 0: 
                 return False, []


            return stop_flags_batch.all(), final_x_batch_list_of_tensors