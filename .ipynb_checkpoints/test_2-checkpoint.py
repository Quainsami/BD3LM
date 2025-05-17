import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig, ListConfig
import math
import os
import sys
from collections import OrderedDict

# Add project root to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.meta_controller import MetaController
import noise_schedule
from diffusion import Diffusion, Loss 
import dataloader

class MockTokenizer(dataloader.Text8Tokenizer):
    def __init__(self, vocab_size_override=50, bos_token_id=0, eos_token_id=1, pad_token_id=2, mask_token_id=3, unk_token_id=4):
        self._vocab_size_override = vocab_size_override

        _special_tokens_map = {
            '[BOS]': bos_token_id,
            '[EOS]': eos_token_id,
            '[PAD]': pad_token_id,
            '[MASK]': mask_token_id,
            '[UNK]': unk_token_id,
        }

        ids_set = set()
        for token_id_val in _special_tokens_map.values():
            if token_id_val in ids_set:
                pass 
            ids_set.add(token_id_val)


        self._vocab_str_to_int_override = {}
        self._vocab_int_to_str_override = {}
        current_ids_assigned = set()

        for token, token_id in _special_tokens_map.items():
            if token_id < self._vocab_size_override and token_id not in current_ids_assigned:
                self._vocab_str_to_int_override[token] = token_id
                self._vocab_int_to_str_override[token_id] = token
                current_ids_assigned.add(token_id)

        next_available_id = 0
        generic_token_idx = 0
        
        needed_generic_tokens = self._vocab_size_override - len(self._vocab_str_to_int_override)

        for _ in range(needed_generic_tokens):
            if len(self._vocab_str_to_int_override) >= self._vocab_size_override:
                break 
            
            while next_available_id in current_ids_assigned or next_available_id >= self._vocab_size_override:
                next_available_id += 1
            
            if next_available_id >= self._vocab_size_override:
                break

            token_str = f'token{generic_token_idx}'
            if token_str not in self._vocab_str_to_int_override: 
                self._vocab_str_to_int_override[token_str] = next_available_id
                self._vocab_int_to_str_override[next_available_id] = token_str
                current_ids_assigned.add(next_available_id)
            generic_token_idx += 1
        
        super_kwargs = {
            'bos_token': '[BOS]', 'eos_token': '[EOS]', 'sep_token': '[SEP]',
            'cls_token': '[CLS]', 'pad_token': '[PAD]', 'mask_token': '[MASK]',
            'unk_token': '[UNK]'
        }
        for token_str_super, default_id_super in [('[BOS]', bos_token_id), ('[EOS]', eos_token_id), 
                                               ('[PAD]', pad_token_id), ('[MASK]', mask_token_id), 
                                               ('[UNK]', unk_token_id)]:
            if token_str_super in self._vocab_str_to_int_override: 
                 super_kwargs[token_str_super.lower().strip('[]') + "_token_id"] = self._vocab_str_to_int_override[token_str_super]


        super().__init__(**super_kwargs)

        final_vocab_str_to_int = {}
        final_vocab_int_to_str = {}
        
        for token, token_id in self._vocab_str_to_int_override.items(): # Start with our custom ones
            final_vocab_str_to_int[token] = token_id
            final_vocab_int_to_str[token_id] = token
            
        for token, token_id in self._vocab_str_to_int.items(): # Add from super's default if not conflicting
            if len(final_vocab_str_to_int) >= self._vocab_size_override:
                break
            if token not in final_vocab_str_to_int and token_id not in final_vocab_int_to_str:
                 if token_id < self._vocab_size_override : # ensure ID is valid for override size
                    final_vocab_str_to_int[token] = token_id
                    final_vocab_int_to_str[token_id] = token
        
        self._vocab_str_to_int = final_vocab_str_to_int
        self._vocab_int_to_str = final_vocab_int_to_str

        if len(self._vocab_str_to_int) > self._vocab_size_override:
            trimmed_vocab_str = {}
            trimmed_vocab_int = {}
            sorted_items_by_id = sorted(self._vocab_str_to_int.items(), key=lambda item: item[1])
            
            special_tokens_priority = {k:v for k,v in _special_tokens_map.items() if v in self._vocab_str_to_int.values()} # only those actually present
            
            for token, token_id in special_tokens_priority.items():
                 if token_id < self._vocab_size_override and len(trimmed_vocab_str) < self._vocab_size_override:
                     trimmed_vocab_str[token] = token_id
                     trimmed_vocab_int[token_id] = token
            
            for token, token_id in sorted_items_by_id:
                if len(trimmed_vocab_str) >= self._vocab_size_override:
                    break
                if token_id not in trimmed_vocab_int: # if not already added (e.g. as a special token)
                    trimmed_vocab_str[token] = token_id
                    trimmed_vocab_int[token_id] = token

            self._vocab_str_to_int = trimmed_vocab_str
            self._vocab_int_to_str = trimmed_vocab_int


        self.bos_token_id = self._vocab_str_to_int.get('[BOS]', bos_token_id)
        self.eos_token_id = self._vocab_str_to_int.get('[EOS]', eos_token_id)
        self.pad_token_id = self._vocab_str_to_int.get('[PAD]', pad_token_id)
        self.mask_token_id = self._vocab_str_to_int.get('[MASK]', mask_token_id)
        self.unk_token_id = self._vocab_str_to_int.get('[UNK]', unk_token_id)


    @property
    def vocab_size(self) -> int:
        return self._vocab_size_override

    def get_vocab(self) -> dict:
        return self._vocab_str_to_int.copy()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str.get(index, self.unk_token)

    def _tokenize(self, text, **kwargs): return list(text)


class MockOutput: 
    def __init__(self, logits):
        self.logits = logits

class MockBackbone(nn.Module):
    def __init__(self, vocab_size, hidden_dim, length, is_hf_model=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.length = length
        self.is_hf_model = is_hf_model

        self.dummy_embed = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        if is_hf_model:
            self.config = OmegaConf.create({
                'hidden_size': hidden_dim,
                'attn_backend': 'sdpa', 
                'cross_attn': False,    
            })

    def forward(self, 
                x=None, 
                input_ids=None, 
                indices=None,   
                # x_embeds=None, # This kwarg is not explicitly used by Diffusion.forward's call to backbone
                sigma=None,     
                timesteps=None, 
                # sigma_or_timesteps=None, # Positional arg for time is less common for HF models
                store_kv=False, sample_mode=False, attention_mask=None, **kwargs):

        final_x_processed = None
        
        if self.is_hf_model:
            # HF model expects input_ids
            current_input_ids = x if x is not None and input_ids is None else input_ids
            if current_input_ids is None:
                raise ValueError("MockBackbone (HF): missing input_ids or positional x")
            final_x_processed = self.dummy_embed(current_input_ids)
        else: # Custom DiT model
            current_indices = x if x is not None and x.dtype == torch.long and indices is None else indices # Assume positional x is indices if long
            current_embeddings = x if x is not None and x.dtype != torch.long and indices is None else None # Assume positional x is embeddings if not long
            
            if current_indices is not None:
                final_x_processed = self.dummy_embed(current_indices)
            elif current_embeddings is not None:
                final_x_processed = current_embeddings
            else:
                raise ValueError("MockBackbone (Custom DiT): missing 'indices', or positional 'x' (as IDs or embeds)")
        
        logits = self.fc(final_x_processed)

        if self.is_hf_model:
            return MockOutput(logits=logits)
        else:
            return logits


class MockBaseNoiseSchedule(noise_schedule.Noise):
    def __init__(self, eps=1e-3, schedule_clamp_epsilon=1e-6):
        super().__init__(eps, schedule_clamp_epsilon)

    def get_alpha_bar_base(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = torch.clamp(t, 0.0, 1.0)
        return (1.0 - self.eps) * torch.cos(t_clamped * torch.pi / 2.0)

    def get_log_alpha_bar_base_derivative_t(self, t: torch.Tensor) -> torch.Tensor:
        t_with_grad = t.clone().detach().requires_grad_(True)
        alpha_bar_base_val_clamped = self.get_alpha_bar(t_with_grad)
        log_alpha_bar_base_val = torch.logit(alpha_bar_base_val_clamped)
        grad_outputs_val = torch.ones_like(log_alpha_bar_base_val, device=log_alpha_bar_base_val.device)
        derivative = torch.autograd.grad(
            outputs=log_alpha_bar_base_val,
            inputs=t_with_grad,
            grad_outputs=grad_outputs_val,
            create_graph=False,
            retain_graph=False,
        )[0]
        return derivative.detach()

    def compute_loss_scaling_and_move_chance(self, t):
        t_clamped = torch.clamp(t, 0.0, 1.0)
        move_chance = (1.0 - self.eps) * (1.0 - torch.cos(0.5 * torch.pi * t_clamped)) + self.eps
        loss_scaling = (0.5 * torch.pi) * torch.tan(0.5 * torch.pi * t_clamped)
        loss_scaling = torch.clamp(loss_scaling, max=1e5)
        return loss_scaling, move_chance

    def get_t_from_move_chance(self, p: torch.Tensor) -> torch.Tensor:
        p_clamped = torch.clamp(p, min=self.eps, max=1.0)
        safe_arg = torch.clamp((1.0 - p_clamped) / (1.0 - self.eps + 1e-9), min=0.0, max=1.0)
        t = (2.0 / torch.pi) * torch.acos(safe_arg)
        return t

class MockLightningTrainer:
    def __init__(self, accumulate_grad_batches=1):
        class MockAcceleratorConnector:
            def __init__(self):
                self.use_distributed_sampler = False
                self.is_distributed = False
        class MockFitLoop:
            def __init__(self):
                class MockCombinedLoader:
                    def __init__(self):
                        self.flattened = [] 
                self._combined_loader = MockCombinedLoader()
        self._accelerator_connector = MockAcceleratorConnector()
        self.fit_loop = MockFitLoop()
        self.accumulate_grad_batches = accumulate_grad_batches
        self.sanity_checking = False
        self.ckpt_path = None
        self.barebones = False # <--- ADD THIS LINE

class TestNewFeaturesBD3LM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cls.meta_controller_config_dict = {
            'feature_dim': 1,
            'hidden_dim': 32,
            's_tilde_squash_factor': 4.0,
            's_min_epsilon': 0.01,
        }
        cls.algo_config_dict = {
            'name': 'bd3lm',
            'backbone': 'hf_dit',
            'parameterization': 'bd3lm',
            'base_noise_type': 'cosine',
            'schedule_clamp_epsilon': 1.0e-6,
            'lambda_steps_penalty': 0.01,
            'lambda_s_b_l2_penalty': 0.001,
            'min_alpha_1_target': 0.005,
            'lambda_min_alpha_1_penalty': 1.0,
            'alpha_1_clamp_min': 1.0e-5,
            'alpha_1_clamp_max': 0.99999,
            'T': 1000,
            'ignore_bos': False,
            'time_conditioning': True,
            'var_min': False,
            'cross_attn': False,
            'sampler': 'semi_ar',
            'mdlm_loss_scale': False,
            'clip_search_delta': 0.1,
            'clip_search_widths': [0.5], 
            'fix_clipping': False,
            'meta_controller': cls.meta_controller_config_dict,
        }
        cls.model_config_dict = {
            'length': 64,
            'hidden_size': 16,
            'attn_backend': 'sdpa'
        }
        cls.training_config_dict = {
            'antithetic_sampling': False,
            'resample': True,
            'ema': 0.0,
            'stage': 'joint',
            'sampling_eps_min': 1e-3,
            'sampling_eps_max': 1.0,
            'sampling_eps': [1e-3, 1.0], 
            'eval_nll': False,
        }
        cls.eval_config_dict = {
            'perplexity_batch_size':1,
            'gen_ppl_eval_model_name_or_path': 'gpt2',
            'checkpoint_path': "prajjwal1/bert-tiny"
        }
        cls.sampling_config_dict = {
            'first_hitting': False, 'kv_cache': False, 'nucleus_p': 1.0,
            'var_length': False, 'logdir': './sample_logs/test_samples.csv',
            'num_sample_batches': 1,
        }

        cls.mock_tokenizer = MockTokenizer(vocab_size_override=20)

        cls.diffusion_config = OmegaConf.create({
            'mode': 'train',
            'algo': cls.algo_config_dict,
            'model': cls.model_config_dict,
            'training': cls.training_config_dict,
            'sampling': cls.sampling_config_dict,
            'block_size': 16,
            'vocab_size': cls.mock_tokenizer.vocab_size,
            'data': {'tokenizer_name_or_path': 'mock', 'wrap': True},
            'loader': {'eval_batch_size': 1, 'batch_size':2, 'num_workers': 0},
            'noise': {
                'type': 'loglinear',
                'eps': 1e-3
            },
            'eval': cls.eval_config_dict,
        })
        cls.diffusion_config.vocab_size = cls.mock_tokenizer.vocab_size


    def _create_mock_diffusion_model(self, config_overrides=None):
        current_config = self.diffusion_config.copy()
        if config_overrides:
            if isinstance(config_overrides, dict):
                config_overrides = OmegaConf.create(config_overrides)
            current_config = OmegaConf.merge(current_config, config_overrides)

        is_hf_model = current_config.algo.backbone == 'hf_dit'
        current_config.vocab_size = self.mock_tokenizer.vocab_size

        model = Diffusion(current_config, tokenizer=self.mock_tokenizer).to(self.device)

        effective_length = current_config.model.length
        if current_config.algo.cross_attn and not is_hf_model :
             effective_length += current_config.model.length

        model.backbone = MockBackbone(
            vocab_size=model.vocab_size,
            hidden_dim=current_config.model.hidden_size,
            length=effective_length,
            is_hf_model=is_hf_model
        ).to(self.device)

        model.base_noise_schedule = MockBaseNoiseSchedule(
            eps=current_config.noise.eps,
            schedule_clamp_epsilon=current_config.algo.schedule_clamp_epsilon
        ).to(self.device)
        _dummy_zero_t = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        clamped_base_alpha_at_0 = model.base_noise_schedule.get_alpha_bar(_dummy_zero_t)
        model.base_log_alpha_bar_at_0 = torch.logit(clamped_base_alpha_at_0).detach()
        model.trainer = MockLightningTrainer()
        return model, current_config

    def test_meta_controller(self):
        print("\nTesting MetaController (from test_1.py)...")
        config = OmegaConf.create({'algo': {'meta_controller': self.meta_controller_config_dict}})
        controller = MetaController(config).to(self.device)
        B, N_Blk, FeatDim = 2, 4, config.algo.meta_controller.feature_dim
        block_features = torch.randn(B, N_Blk, FeatDim, device=self.device)
        log_s_tilde = controller(block_features)
        self.assertEqual(log_s_tilde.shape, (B, N_Blk, 1))
        s_b = controller.get_s_b(log_s_tilde)
        self.assertEqual(s_b.shape, (B, N_Blk, 1))
        self.assertTrue(torch.all(s_b >= config.algo.meta_controller.s_min_epsilon))
        large_log_s_tilde = torch.full_like(log_s_tilde, 100.0)
        s_b_max_tanh = controller.get_s_b(large_log_s_tilde)
        expected_s_b_max_tanh = F.softplus(torch.tensor(1.0 * config.algo.meta_controller.s_tilde_squash_factor)) + config.algo.meta_controller.s_min_epsilon
        self.assertTrue(torch.allclose(s_b_max_tanh, expected_s_b_max_tanh.to(self.device), atol=1e-5))
        small_log_s_tilde = torch.full_like(log_s_tilde, -100.0)
        s_b_min_tanh = controller.get_s_b(small_log_s_tilde)
        expected_s_b_min_tanh = F.softplus(torch.tensor(-1.0 * config.algo.meta_controller.s_tilde_squash_factor)) + config.algo.meta_controller.s_min_epsilon
        self.assertTrue(torch.allclose(s_b_min_tanh, expected_s_b_min_tanh.to(self.device), atol=1e-5))
        print("MetaController tests passed.")

    def test_noise_schedule_base_methods_via_mock(self):
        print("\nTesting NoiseSchedule base methods (MockBaseNoiseSchedule example)...")
        schedule_clamp_eps = self.algo_config_dict['schedule_clamp_epsilon']
        base_schedule = MockBaseNoiseSchedule(eps=self.diffusion_config.noise.eps, schedule_clamp_epsilon=schedule_clamp_eps)
        base_schedule = base_schedule.to(self.device)
        t_vals = torch.tensor([0.0, 0.5, 1.0], device=self.device)
        alpha_bar_base = base_schedule.get_alpha_bar_base(t_vals)
        expected_alpha_base_0 = (1.0 - base_schedule.eps) * math.cos(0.0 * math.pi / 2.0)
        expected_alpha_base_0_5 = (1.0 - base_schedule.eps) * math.cos(0.5 * math.pi / 2.0)
        expected_alpha_base_1 = (1.0 - base_schedule.eps) * math.cos(1.0 * math.pi / 2.0)
        self.assertAlmostEqual(alpha_bar_base[0].item(), expected_alpha_base_0, places=5)
        self.assertAlmostEqual(alpha_bar_base[1].item(), expected_alpha_base_0_5, places=5)
        self.assertAlmostEqual(alpha_bar_base[2].item(), expected_alpha_base_1, places=5)

        alpha_bar_clamped = base_schedule.get_alpha_bar(t_vals)
        self.assertTrue(torch.all(alpha_bar_clamped >= schedule_clamp_eps))
        self.assertTrue(torch.all(alpha_bar_clamped <= 1.0 - schedule_clamp_eps))

        expected_clamped_t0 = torch.clamp(torch.tensor(1.0 - base_schedule.eps, device=self.device),
                                          min=schedule_clamp_eps,
                                          max=1.0 - schedule_clamp_eps).item()
        self.assertAlmostEqual(alpha_bar_clamped[0].item(), expected_clamped_t0, places=5)
        self.assertAlmostEqual(alpha_bar_clamped[2].item(), schedule_clamp_eps, places=5)

        t_for_deriv = torch.tensor([0.25, 0.5, 0.75], device=self.device, dtype=torch.float64, requires_grad=True)
        def func_to_check(t_input):
            return torch.logit(base_schedule.get_alpha_bar(t_input))

        gradcheck_passed = torch.autograd.gradcheck(func_to_check, t_for_deriv, eps=1e-6, atol=1e-4, nondet_tol=1e-4)
        self.assertTrue(gradcheck_passed, "Numerical gradcheck for logit(alpha_bar(t)) failed.")

        L_prime_t = base_schedule.get_log_alpha_bar_base_derivative_t(t_vals.to(torch.float64))
        self.assertEqual(L_prime_t.shape, t_vals.shape)
        self.assertTrue(torch.all(L_prime_t <= 1e-6), f"L_prime_t should be <= 0 (or very small positive due to numerics), got {L_prime_t}")

        p_vals = base_schedule.compute_loss_scaling_and_move_chance(t_vals)[1]
        t_reconstructed = base_schedule.get_t_from_move_chance(p_vals)
        self.assertTrue(torch.allclose(t_vals, t_reconstructed, atol=1e-5),
                        f"t_from_move_chance failed. Original: {t_vals}, Reconstructed: {t_reconstructed}")
        print("NoiseSchedule base methods (Mock) tests passed.")

    def test_warped_schedule_functions_bd3lm(self):
        print("\nTesting warped schedule functions for BD3LM...")
        schedule_clamp_eps = self.algo_config_dict['schedule_clamp_epsilon']
        base_schedule = MockBaseNoiseSchedule(eps=self.diffusion_config.noise.eps, schedule_clamp_epsilon=schedule_clamp_eps).to(self.device)
        B, N_Blk, N_t_points = 2, 3, 5
        s_b = (torch.rand(B, N_Blk, 1, device=self.device) + 0.5) * 2.0
        s_b_squeezed = s_b.squeeze(-1)

        t_points_raw = torch.linspace(0, 1, N_t_points, device=self.device)
        t_points_for_base = t_points_raw.unsqueeze(0).unsqueeze(0).expand(B, N_Blk, -1)

        base_alpha_bar_t_for_warping = base_schedule.get_alpha_bar(t_points_for_base)
        base_log_alpha_bar_at_0_scalar = torch.logit(base_schedule.get_alpha_bar(torch.tensor(0.0, device=self.device)))
        target_alpha_at_0_scalar = torch.tensor(1.0 - schedule_clamp_eps, device=self.device)
        target_log_alpha_at_0_scalar = torch.logit(target_alpha_at_0_scalar)

        alpha_b_t_direct = noise_schedule.get_warped_alpha_b_t(
            base_alpha_bar_t_for_warping, base_log_alpha_bar_at_0_scalar, s_b, target_log_alpha_at_0_scalar
        )
        self.assertEqual(alpha_b_t_direct.shape, (B, N_Blk, N_t_points))
        self.assertTrue(torch.all((alpha_b_t_direct >= 0) & (alpha_b_t_direct <= 1)))

        alpha_b_t_at_0 = noise_schedule.get_warped_alpha_b_t(
            base_schedule.get_alpha_bar(torch.zeros_like(t_points_for_base[..., :1])),
            base_log_alpha_bar_at_0_scalar, s_b, target_log_alpha_at_0_scalar
        )
        self.assertTrue(torch.allclose(alpha_b_t_at_0, target_alpha_at_0_scalar, atol=1e-4),
                        f"alpha_b(0) was not target_alpha_at_0. Got {alpha_b_t_at_0.mean()}, expected {target_alpha_at_0_scalar}")

        base_log_alpha_deriv_t_for_warping = base_schedule.get_log_alpha_bar_base_derivative_t(t_points_for_base)
        alpha_b_t, loss_scale_b_t, p_b_t = noise_schedule.get_warped_schedule_outputs(
            base_alpha_bar_t_for_warping, base_log_alpha_deriv_t_for_warping,
            base_log_alpha_bar_at_0_scalar,
            s_b_squeezed,
            target_log_alpha_at_0_scalar
        )
        self.assertEqual(alpha_b_t.shape, (B, N_Blk, N_t_points))
        self.assertEqual(loss_scale_b_t.shape, (B, N_Blk, N_t_points))
        self.assertEqual(p_b_t.shape, (B, N_Blk, N_t_points))
        self.assertTrue(torch.allclose(p_b_t, 1.0 - alpha_b_t, atol=1e-6))
        self.assertTrue(torch.all(loss_scale_b_t <= 1e-6),
                        f"Expected loss_scale_b_t <= 0, got max {loss_scale_b_t.max()}")

        s_b_one = torch.ones_like(s_b_squeezed)
        alpha_b_t_identity, _, p_b_t_identity = noise_schedule.get_warped_schedule_outputs(
            base_alpha_bar_t_for_warping, base_log_alpha_deriv_t_for_warping,
            base_log_alpha_bar_at_0_scalar,
            s_b_one,
            base_log_alpha_bar_at_0_scalar
        )
        self.assertTrue(torch.allclose(alpha_b_t_identity, base_alpha_bar_t_for_warping, atol=1e-5))
        self.assertTrue(torch.allclose(p_b_t_identity, 1.0 - base_alpha_bar_t_for_warping, atol=1e-5))
        print("Warped schedule functions for BD3LM tests passed.")

    def test_compute_surrogate_steps_penalty_bd3lm(self):
        print("\nTesting surrogate_steps_penalty for BD3LM...")
        B, N_Blk = 2, 3
        alpha_b_at_1_small = torch.full((B, N_Blk, 1), 0.001, device=self.device)
        cfg_algo = self.algo_config_dict
        penalty_small = noise_schedule.compute_surrogate_steps_penalty(
            alpha_b_at_1_small, cfg_algo['min_alpha_1_target'], cfg_algo['lambda_min_alpha_1_penalty'],
            cfg_algo['alpha_1_clamp_min'], cfg_algo['alpha_1_clamp_max']
        )
        expected_log_term_small = -torch.log(torch.clamp(alpha_b_at_1_small, min=cfg_algo['alpha_1_clamp_min']))
        expected_floor_penalty_small = cfg_algo['lambda_min_alpha_1_penalty'] * \
                                F.relu(cfg_algo['min_alpha_1_target'] - alpha_b_at_1_small)**2
        self.assertTrue(torch.allclose(penalty_small, expected_log_term_small + expected_floor_penalty_small, atol=1e-5))

        alpha_b_at_1_violates_floor = torch.full((B, N_Blk, 1), cfg_algo['min_alpha_1_target'] / 2, device=self.device)
        alpha_b_at_1_violates_floor = torch.max(alpha_b_at_1_violates_floor, torch.tensor(cfg_algo['alpha_1_clamp_min'] * 1.1, device=self.device))

        penalty_violating = noise_schedule.compute_surrogate_steps_penalty(
             alpha_b_at_1_violates_floor, cfg_algo['min_alpha_1_target'], cfg_algo['lambda_min_alpha_1_penalty'],
             cfg_algo['alpha_1_clamp_min'], cfg_algo['alpha_1_clamp_max']
        )
        expected_log_term_violating = -torch.log(torch.clamp(alpha_b_at_1_violates_floor, min=cfg_algo['alpha_1_clamp_min']))
        expected_floor_term_violating = cfg_algo['lambda_min_alpha_1_penalty'] * (cfg_algo['min_alpha_1_target'] - alpha_b_at_1_violates_floor)**2
        self.assertTrue(torch.allclose(penalty_violating, expected_log_term_violating + expected_floor_term_violating, atol=1e-5))

        alpha_b_at_1_respects_floor = torch.full((B, N_Blk, 1), cfg_algo['min_alpha_1_target'] * 2, device=self.device)
        penalty_respecting = noise_schedule.compute_surrogate_steps_penalty(
             alpha_b_at_1_respects_floor, cfg_algo['min_alpha_1_target'], cfg_algo['lambda_min_alpha_1_penalty'],
             cfg_algo['alpha_1_clamp_min'], cfg_algo['alpha_1_clamp_max']
        )
        expected_log_term_respecting = -torch.log(torch.clamp(alpha_b_at_1_respects_floor, min=cfg_algo['alpha_1_clamp_min']))
        self.assertTrue(torch.allclose(penalty_respecting, expected_log_term_respecting, atol=1e-5))
        print("Surrogate_steps_penalty for BD3LM tests passed.")

    def test_diffusion_initialization_bd3lm(self):
        print("\nTesting Diffusion initialization for BD3LM (copied)...")
        model, config = self._create_mock_diffusion_model()
        self.assertIsInstance(model.meta_controller, MetaController)
        self.assertIsInstance(model.base_noise_schedule, noise_schedule.Noise)
        self.assertTrue(hasattr(model, 'target_alpha_at_0'))
        self.assertTrue(hasattr(model, 'base_log_alpha_bar_at_0'))
        expected_target_alpha_0 = 1.0 - config.algo.schedule_clamp_epsilon
        self.assertAlmostEqual(model.target_alpha_at_0.item(), expected_target_alpha_0, places=5)

        alpha_bar_base_at_0_val = 1.0 - model.base_noise_schedule.eps
        expected_base_alpha_0_clamped = torch.clamp(
            torch.tensor(alpha_bar_base_at_0_val, device=self.device),
            min=config.algo.schedule_clamp_epsilon,
            max=1.0-config.algo.schedule_clamp_epsilon
        )
        expected_base_log_alpha_0 = torch.logit(expected_base_alpha_0_clamped)
        self.assertTrue(torch.allclose(model.base_log_alpha_bar_at_0, expected_base_log_alpha_0, atol=1e-5))
        print("Diffusion initialization for BD3LM (copied) tests passed.")

    def test_diffusion_get_parameters_stages_bd3lm(self):
        print("\nTesting Diffusion._get_parameters() for stages (BD3LM context)...")
        model_co, _ = self._create_mock_diffusion_model({'training': {'stage': 'controller_only'}})
        params_co = list(model_co._get_parameters())
        mc_params_ids_co = {id(p) for p in model_co.meta_controller.parameters()}
        lm_params_ids_co = {id(p) for p in model_co.backbone.parameters()}
        for p_opt in params_co:
            self.assertIn(id(p_opt), mc_params_ids_co)
            if lm_params_ids_co:
                self.assertNotIn(id(p_opt), lm_params_ids_co)

        model_lm, _ = self._create_mock_diffusion_model({'training': {'stage': 'lm_only'}})
        params_lm = list(model_lm._get_parameters())
        mc_params_ids_lm = {id(p) for p in model_lm.meta_controller.parameters()}
        lm_params_ids_lm = {id(p) for p in model_lm.backbone.parameters()}
        for p_opt in params_lm:
            if lm_params_ids_lm: self.assertIn(id(p_opt), lm_params_ids_lm)
            self.assertNotIn(id(p_opt), mc_params_ids_lm)

        model_jt, _ = self._create_mock_diffusion_model({'training': {'stage': 'joint'}})
        params_jt = list(model_jt._get_parameters())
        mc_params_ids_jt = {id(p) for p in model_jt.meta_controller.parameters()}
        lm_params_ids_jt = {id(p) for p in model_jt.backbone.parameters()}
        param_ids_jt_set = {id(p) for p in params_jt}
        self.assertTrue(mc_params_ids_jt.issubset(param_ids_jt_set))
        if lm_params_ids_jt: self.assertTrue(lm_params_ids_jt.issubset(param_ids_jt_set))
        print("Diffusion._get_parameters() for stages (BD3LM context) tests passed.")

    def test_diffusion_get_block_features_bd3lm(self):
        print("\nTesting Diffusion._get_block_features() (BD3LM context)...")
        feature_dim_used_by_method = 1
        model, config = self._create_mock_diffusion_model({'algo': {'meta_controller': {'feature_dim': feature_dim_used_by_method}}})
        B, N_Tokens = 2, config.model.length
        N_Blk = N_Tokens // config.block_size
        tokens = torch.randint(0, config.vocab_size, (B, N_Tokens), device=self.device)
        attn_mask = torch.ones_like(tokens)
        block_features = model._get_block_features(tokens, attn_mask)
        self.assertEqual(block_features.shape, (B, N_Blk, feature_dim_used_by_method))
        if feature_dim_used_by_method == 1:
            expected_indices = torch.arange(N_Blk, device=self.device, dtype=torch.float32)
            if N_Blk > 1: expected_indices = expected_indices / (N_Blk - 1 + 1e-9)
            else: expected_indices = torch.zeros_like(expected_indices)
            self.assertTrue(torch.allclose(block_features[0, :, 0], expected_indices))
        print("Diffusion._get_block_features() (BD3LM context) tests passed.")

    def test_diffusion_q_xt_and_sample_t_bd3lm(self):
        print("\nTesting Diffusion.q_xt and _sample_t (BD3LM context)...")
        torch.manual_seed(123)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(123)

        B_stat, N_Tokens_stat, BlkSize_stat = 8, 256, self.diffusion_config.block_size
        model_stat, config_stat = self._create_mock_diffusion_model({
            'model': {'length': N_Tokens_stat}, 'block_size': BlkSize_stat,
            'training': {'resample': False}
        })
        x0_stat = torch.randint(0, config_stat.vocab_size, (B_stat, N_Tokens_stat), device=self.device)
        p_region1 = torch.full((B_stat, N_Tokens_stat // 2), 0.1, device=self.device)
        p_region2 = torch.full((B_stat, N_Tokens_stat // 2), 0.9, device=self.device)
        p_distinct_per_token = torch.cat([p_region1, p_region2], dim=1)
        xt_distinct = model_stat.q_xt(x0_stat, p_distinct_per_token.unsqueeze(-1))
        masks_r1 = (xt_distinct[:, :N_Tokens_stat//2] == model_stat.mask_index).float().mean()
        masks_r2 = (xt_distinct[:, N_Tokens_stat//2:] == model_stat.mask_index).float().mean()
        self.assertAlmostEqual(masks_r1.item(), 0.1, delta=0.05)
        self.assertAlmostEqual(masks_r2.item(), 0.9, delta=0.05)

        B, N_Tokens, BlkSize = 2, 64, 16
        N_Blk = N_Tokens // BlkSize
        model, config = self._create_mock_diffusion_model({
            'model': {'length': N_Tokens}, 'block_size': BlkSize,
            'training': {'resample': False}
        })
        x0 = torch.randint(0, config.vocab_size, (B, N_Tokens), device=self.device)
        
        sampling_eps_cfg = config.training.sampling_eps
        if isinstance(sampling_eps_cfg, ListConfig): # OmegaConf list
            sampling_eps_min_val = sampling_eps_cfg[0]
            sampling_eps_max_val = sampling_eps_cfg[1]
        elif isinstance(sampling_eps_cfg, (tuple, list)): # Python native tuple/list
            sampling_eps_min_val = sampling_eps_cfg[0]
            sampling_eps_max_val = sampling_eps_cfg[1]
        else: # Scalar, implies range [scalar, 1.0]
            sampling_eps_min_val = sampling_eps_cfg
            sampling_eps_max_val = 1.0


        sampling_eps_min_tensor = torch.tensor(sampling_eps_min_val, device=self.device)
        sampling_eps_max_tensor = torch.tensor(sampling_eps_max_val, device=self.device)


        u_per_block = model._sample_t((B, N_Blk), self.device, sampling_eps_min_tensor, sampling_eps_max_tensor, block_size=1)
        self.assertEqual(u_per_block.shape, (B, N_Blk))
        self.assertTrue(torch.all((u_per_block >= sampling_eps_min_val) & (u_per_block <= sampling_eps_max_val)))


        p_block_constant = torch.tensor([[0.2]*BlkSize + [0.8]*BlkSize + [0.2]*BlkSize + [0.8]*BlkSize], device=self.device).float()
        p_block_constant = p_block_constant.expand(B, -1)
        xt_block_p = model.q_xt(x0, p_block_constant.unsqueeze(-1))
        xt_block_p_reshaped = xt_block_p.view(B, N_Blk, BlkSize)
        mask_rates_block_p_0 = (xt_block_p_reshaped[:,0,:] == model.mask_index).float().mean()
        mask_rates_block_p_1 = (xt_block_p_reshaped[:,1,:] == model.mask_index).float().mean()
        self.assertAlmostEqual(mask_rates_block_p_0.item(), 0.2, delta=0.20) 
        self.assertAlmostEqual(mask_rates_block_p_1.item(), 0.8, delta=0.20) 

        xt_nll = model.q_xt(x0, torch.ones_like(p_block_constant).unsqueeze(-1),
                            block_size=1, sampling_eps_min=torch.tensor(1.0))
        self.assertTrue(torch.all(xt_nll == model.mask_index))
        print("Diffusion.q_xt and _sample_t (BD3LM context) tests passed.")

    def test_diffusion_process_sigma_bd3lm(self):
        print("\nTesting Diffusion._process_sigma (BD3LM context)...")
        model, config = self._create_mock_diffusion_model()
        B, N_T = 2, config.model.length
        sigma_in_bnt = torch.rand(B, N_T, device=self.device)
        processed_b = model._process_sigma(sigma_in_bnt)
        self.assertEqual(processed_b.shape, (B,))
        self.assertTrue(torch.allclose(processed_b, sigma_in_bnt.mean(dim=-1)))

        sigma_in_bnt1 = torch.rand(B, N_T, 1, device=self.device)
        processed_b_from_bnt1 = model._process_sigma(sigma_in_bnt1)
        self.assertEqual(processed_b_from_bnt1.shape, (B,))
        self.assertTrue(torch.allclose(processed_b_from_bnt1, sigma_in_bnt1.squeeze(-1).mean(dim=-1)))

        model_no_tc, _ = self._create_mock_diffusion_model({'algo': {'time_conditioning': False}})
        processed_no_tc = model_no_tc._process_sigma(sigma_in_bnt)
        self.assertTrue(torch.all(processed_no_tc == 0.0))
        print("Diffusion._process_sigma (BD3LM context) tests passed.")

    def test_diffusion_loss_bd3lm_full(self):
        print("\nTesting Diffusion._loss for BD3LM (full path)...")
        model, config = self._create_mock_diffusion_model({
            'algo': {'parameterization': 'bd3lm', 'var_min': False, 'lambda_s_b_l2_penalty': 0.01},
            'training': {'stage': 'joint'}
        })
        B, N_Tokens = 2, config.model.length
        x0 = torch.randint(0, model.vocab_size, (B, N_Tokens), device=self.device)
        attention_mask = torch.ones_like(x0)
        
        model.on_train_start() 

        # Temporarily mock the self.log method for this test
        original_log_method = model.log
        model.log = lambda name, value, **kwargs: None # Dummy log method

        try:
            loss_obj = model._loss(x0, attention_mask)

            self.assertIsInstance(loss_obj, Loss)
            self.assertIsNotNone(loss_obj.loss)
            self.assertTrue(loss_obj.loss.requires_grad, "Loss should require gradients for backprop.")
            self.assertEqual(loss_obj.nlls.shape, (B, N_Tokens))
            self.assertEqual(loss_obj.token_mask.shape, (B, N_Tokens))
            self.assertEqual(loss_obj.loss.ndim, 0)

            loss_obj.loss.backward() # Check backward pass
            mc_grad_exists = any(p.grad is not None for p in model.meta_controller.parameters() if p.requires_grad)
            bb_grad_exists = any(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)
            
            self.assertTrue(mc_grad_exists, "No gradients for meta_controller in joint stage.")
            self.assertTrue(bb_grad_exists, "No gradients for backbone in joint stage.")
            model.zero_grad()
        except RuntimeError as e:
            self.fail(f"_loss.backward() failed: {e}")
        finally:
            model.log = original_log_method # Restore original log method

        print("Diffusion._loss for BD3LM (full path) tests passed.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)