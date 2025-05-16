import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
import math
import os
import sys

# Add project root to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.meta_controller import MetaController
import noise_schedule 
from diffusion import Diffusion, Loss 
import dataloader 

class MockTokenizer(dataloader.Text8Tokenizer):
    def __init__(self, vocab_size_override=50, bos_token_id=0, eos_token_id=1, pad_token_id=2, mask_token_id=3):
        self._vocab_size_override = vocab_size_override
        self._vocab_str_to_int_override = {
            '[BOS]': bos_token_id, '[EOS]': eos_token_id, '[PAD]': pad_token_id,
            '[MASK]': mask_token_id, '[UNK]': 4,
            **{f'token{i}': i + 5 for i in range(self._vocab_size_override - 5)}
        }
        self._vocab_int_to_str_override = {v: k for k, v in self._vocab_str_to_int_override.items()}
        super().__init__(bos_token='[BOS]', eos_token='[EOS]', sep_token='[SEP]', 
                         cls_token='[CLS]', pad_token='[PAD]', mask_token='[MASK]', 
                         unk_token='[UNK]')
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        for token, token_id in [('[BOS]', bos_token_id), ('[EOS]', eos_token_id), 
                                ('[PAD]', pad_token_id), ('[MASK]', mask_token_id), ('[UNK]', 4)]:
            if token not in self._vocab_str_to_int_override:
                 self._vocab_str_to_int_override[token] = token_id
                 self._vocab_int_to_str_override[token_id] = token

    @property
    def vocab_size(self) -> int:
        return self._vocab_size_override
    def get_vocab(self) -> dict:
        return self._vocab_str_to_int_override
    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int_override.get(token, self._vocab_str_to_int_override['[UNK]'])
    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str_override.get(index, '[UNK]')
    def _tokenize(self, text, **kwargs): return list(text)

class MockBackbone(nn.Module):
    def __init__(self, vocab_size, hidden_dim, length):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.length = length 
        self.dummy_embed = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size) 

    def forward(self, x_embeds, sigma, **kwargs): 
        logits = self.fc(x_embeds) 
        return logits

    def get_input_embeddings(self): 
        return self.dummy_embed
        
    def reset_kv_cache(self, eval_batch_size=1): 
        pass

class TestNewFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        cls.meta_controller_config_dict = {
            'feature_dim': 1, 
            'hidden_dim': 32,
            's_tilde_squash_factor': 4.0,
            's_min_epsilon': 0.01,
            'min_alpha_1_target': 0.005,
            'lambda_min_alpha_1_penalty': 1.0,
            'alpha_1_clamp_min': 1.0e-5,
            'alpha_1_clamp_max': 0.99999,
        }
        cls.algo_config_dict = {
            'meta_controller': cls.meta_controller_config_dict,
            'base_noise_type': 'cosine',
            'schedule_clamp_epsilon': 1.0e-6,
            'lambda_steps_penalty': 0.01,
            'lambda_s_b_l2_penalty': 0.001,
            'parameterization': 'bd3lm', 
            'T': 1000, 
            'ignore_bos': False,
            'name': 'bd3lm', 
            'backbone': 'hf_dit', 
            'time_conditioning': True, 
            'var_min': False,
            'cross_attn': False, 
            'sampler': 'semi_ar', 
            'mdlm_loss_scale': False, 
        }
        cls.model_config_dict = { 
            'length': 64, 
            'hidden_size': 16,
            'attn_backend': 'sdpa' # <--- ADDED for _validate_configuration
        }
        cls.training_config_dict = { 
            'antithetic_sampling': False,
            'resample': False, 
            'ema': 0.0, 
            'stage': 'joint', 
            'sampling_eps_min': 1e-3, 
            'sampling_eps_max': 1.0,   
            'sampling_eps': (1e-3, 1.0), 
        }
        cls.eval_config_dict = { 
            'perplexity_batch_size':1, 
            'gen_ppl_eval_model_name_or_path': 'gpt2',
            'checkpoint_path': "prajjwal1/bert-tiny" 
        }
        cls.sampling_config_dict = { # <--- ADDED for _validate_configuration
            'first_hitting': False,
            'kv_cache': False,
            'nucleus_p': 1.0, # Add other defaults if Diffusion uses them
            'var_length': False,
            'logdir': './sample_logs/test_samples.csv', # For metrics init
            'num_sample_batches': 1, # For metrics init
        }


        cls.mock_tokenizer = MockTokenizer(vocab_size_override=20) 
        
        cls.diffusion_config = OmegaConf.create({
            'mode': 'ppl_eval', # <--- ADDED TOP-LEVEL MODE
            'algo': cls.algo_config_dict,
            'model': cls.model_config_dict,
            'training': cls.training_config_dict,
            'sampling': cls.sampling_config_dict, # <--- ADDED SAMPLING CONFIG
            'block_size': 16, 
            'vocab_size': cls.mock_tokenizer.vocab_size,
            'data': {'tokenizer_name_or_path': 'mock'}, 
            'noise': {
                'type': 'cosine',
                'eps': 1e-3 
            }, 
            'eval': cls.eval_config_dict, 
        })
        assert cls.diffusion_config.vocab_size == 20, f"Config vocab_size is {cls.diffusion_config.vocab_size}, expected 20"

    def _create_mock_diffusion_model(self, config_overrides=None):
        current_config = self.diffusion_config.copy()
        if config_overrides:
            if isinstance(config_overrides, dict):
                config_overrides = OmegaConf.create(config_overrides)
            current_config = OmegaConf.merge(current_config, config_overrides)
        
        current_config_for_init = current_config.copy()
        current_config_for_init.algo.backbone = 'hf_dit' 
        if 'checkpoint_path' not in current_config_for_init.eval: 
             current_config_for_init.eval.checkpoint_path = "prajjwal1/bert-tiny"

        model = Diffusion(current_config_for_init, tokenizer=self.mock_tokenizer).to(self.device)
        
        effective_length = current_config.model.length
        if current_config.algo.cross_attn:
            effective_length += current_config.model.length
            
        model.backbone = MockBackbone(
            vocab_size=current_config.vocab_size,
            hidden_dim=current_config.model.hidden_size,
            length=effective_length 
        ).to(self.device)
        
        return model, current_config


    def test_meta_controller(self):
        print("\nTesting MetaController...")
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

    def test_noise_schedule_base_methods(self):
        print("\nTesting NoiseSchedule base methods (CosineNoise example)...")
        schedule_clamp_eps = self.algo_config_dict['schedule_clamp_epsilon']
        base_schedule = noise_schedule.CosineNoise(eps=self.diffusion_config.noise.eps, schedule_clamp_epsilon=schedule_clamp_eps)
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
        self.assertTrue(torch.all(L_prime_t <= 1e-7), f"L_prime_t should be <= 0, got {L_prime_t}")
        p_vals = base_schedule.compute_loss_scaling_and_move_chance(t_vals)[1]
        t_reconstructed = base_schedule.get_t_from_move_chance(p_vals)
        self.assertTrue(torch.allclose(t_vals, t_reconstructed, atol=1e-5),
                        f"t_from_move_chance failed. Original: {t_vals}, Reconstructed: {t_reconstructed}")
        print("NoiseSchedule base methods tests passed.")

    def test_warped_schedule_functions(self):
        print("\nTesting warped schedule functions...")
        schedule_clamp_eps = self.algo_config_dict['schedule_clamp_epsilon']
        base_schedule = noise_schedule.CosineNoise(eps=self.diffusion_config.noise.eps, schedule_clamp_epsilon=schedule_clamp_eps).to(self.device)
        B, N_Blk, N_t = 2, 3, 5
        s_b = (torch.rand(B, N_Blk, 1, device=self.device) + 0.5) * 2.0 
        s_b_squeezed = s_b.squeeze(-1) 
        t_points_raw = torch.linspace(0, 1, N_t, device=self.device)
        t_points = t_points_raw.unsqueeze(0).unsqueeze(0).expand(B, N_Blk, -1) 
        base_alpha_bar_t = base_schedule.get_alpha_bar(t_points) 
        base_log_alpha_bar_at_0_scalar = torch.logit(base_schedule.get_alpha_bar(torch.tensor(0.0, device=self.device)))
        target_alpha_at_0_scalar = torch.tensor(1.0 - schedule_clamp_eps, device=self.device)
        target_log_alpha_at_0_scalar = torch.logit(target_alpha_at_0_scalar)
        alpha_b_t_direct = noise_schedule.get_warped_alpha_b_t(
            base_alpha_bar_t, base_log_alpha_bar_at_0_scalar, s_b, target_log_alpha_at_0_scalar
        )
        self.assertEqual(alpha_b_t_direct.shape, (B, N_Blk, N_t))
        self.assertTrue(torch.all((alpha_b_t_direct >= 0) & (alpha_b_t_direct <= 1)))
        alpha_b_t_at_0 = noise_schedule.get_warped_alpha_b_t(
            base_schedule.get_alpha_bar(torch.zeros_like(t_points[..., :1])), 
            base_log_alpha_bar_at_0_scalar, s_b, target_log_alpha_at_0_scalar
        )
        self.assertTrue(torch.allclose(alpha_b_t_at_0, target_alpha_at_0_scalar, atol=1e-4))
        base_log_alpha_deriv_t = base_schedule.get_log_alpha_bar_base_derivative_t(t_points) 
        alpha_b_t, loss_scale_b_t, p_b_t = noise_schedule.get_warped_schedule_outputs(
            base_alpha_bar_t, base_log_alpha_deriv_t, base_log_alpha_bar_at_0_scalar, 
            s_b_squeezed, target_log_alpha_at_0_scalar 
        )
        self.assertEqual(alpha_b_t.shape, (B, N_Blk, N_t))
        self.assertEqual(loss_scale_b_t.shape, (B, N_Blk, N_t))
        self.assertEqual(p_b_t.shape, (B, N_Blk, N_t))
        self.assertTrue(torch.allclose(p_b_t, 1.0 - alpha_b_t))
        self.assertTrue(torch.all(loss_scale_b_t <= 1e-6), f"Expected loss_scale_b_t <=0, got max {loss_scale_b_t.max()}")
        s_b_one = torch.ones_like(s_b_squeezed)
        alpha_b_t_identity, _, _ = noise_schedule.get_warped_schedule_outputs(
            base_alpha_bar_t, base_log_alpha_deriv_t, base_log_alpha_bar_at_0_scalar,
            s_b_one, base_log_alpha_bar_at_0_scalar 
        )
        self.assertTrue(torch.allclose(alpha_b_t_identity, base_alpha_bar_t, atol=1e-5))
        print("Warped schedule functions tests passed.")

    def test_compute_surrogate_steps_penalty(self):
        print("\nTesting surrogate_steps_penalty...")
        B, N_Blk = 2, 3
        alpha_b_at_1 = torch.rand(B, N_Blk, 1, device=self.device) * 0.1 
        cfg_mc = self.meta_controller_config_dict
        penalty = noise_schedule.compute_surrogate_steps_penalty(
            alpha_b_at_1, cfg_mc['min_alpha_1_target'], cfg_mc['lambda_min_alpha_1_penalty'],
            cfg_mc['alpha_1_clamp_min'], cfg_mc['alpha_1_clamp_max']
        )
        self.assertEqual(penalty.shape, (B, N_Blk, 1))
        alpha_b_at_1_violating_floor = torch.full_like(alpha_b_at_1, cfg_mc['min_alpha_1_target'] / 2)
        penalty_violating = noise_schedule.compute_surrogate_steps_penalty(
             alpha_b_at_1_violating_floor, cfg_mc['min_alpha_1_target'], cfg_mc['lambda_min_alpha_1_penalty'],
             cfg_mc['alpha_1_clamp_min'], cfg_mc['alpha_1_clamp_max']
        )
        expected_log_term_violating = -torch.log(torch.clamp(alpha_b_at_1_violating_floor, min=cfg_mc['alpha_1_clamp_min']))
        expected_floor_term = cfg_mc['lambda_min_alpha_1_penalty'] * (cfg_mc['min_alpha_1_target'] - cfg_mc['min_alpha_1_target']/2)**2
        self.assertTrue(torch.allclose(penalty_violating, expected_log_term_violating + expected_floor_term, atol=1e-5))
        alpha_b_at_1_respecting_floor = torch.full_like(alpha_b_at_1, cfg_mc['min_alpha_1_target'] * 2)
        penalty_respecting = noise_schedule.compute_surrogate_steps_penalty(
             alpha_b_at_1_respecting_floor, cfg_mc['min_alpha_1_target'], cfg_mc['lambda_min_alpha_1_penalty'],
             cfg_mc['alpha_1_clamp_min'], cfg_mc['alpha_1_clamp_max']
        )
        expected_log_term_respecting = -torch.log(torch.clamp(alpha_b_at_1_respecting_floor, min=cfg_mc['alpha_1_clamp_min']))
        self.assertTrue(torch.allclose(penalty_respecting, expected_log_term_respecting, atol=1e-5))
        print("Surrogate_steps_penalty tests passed.")

    def test_diffusion_initialization(self):
        print("\nTesting Diffusion initialization...")
        model, config = self._create_mock_diffusion_model()
        self.assertIsInstance(model.meta_controller, MetaController)
        self.assertIsInstance(model.base_noise_schedule, noise_schedule.Noise) 
        self.assertTrue(hasattr(model, 'target_alpha_at_0'))
        self.assertTrue(hasattr(model, 'base_log_alpha_bar_at_0'))
        expected_target_alpha_0 = 1.0 - config.algo.schedule_clamp_epsilon
        self.assertAlmostEqual(model.target_alpha_at_0.item(), expected_target_alpha_0, places=5)
        expected_base_alpha_0_clamped = torch.clamp(
            model.base_noise_schedule.get_alpha_bar_base(torch.tensor(0.0, device=self.device)),
            min=config.algo.schedule_clamp_epsilon,
            max=1.0-config.algo.schedule_clamp_epsilon
        )
        expected_base_log_alpha_0 = torch.logit(expected_base_alpha_0_clamped)
        self.assertTrue(torch.allclose(model.base_log_alpha_bar_at_0, expected_base_log_alpha_0, atol=1e-5))
        print("Diffusion initialization tests passed.")

    def test_diffusion_get_parameters(self):
        print("\nTesting Diffusion._get_parameters()...")
        model_co, _ = self._create_mock_diffusion_model({'training': {'stage': 'controller_only'}})
        params_co = list(model_co._get_parameters())
        mc_params_ids_co = {id(p) for p in model_co.meta_controller.parameters()}
        lm_params_ids_co = {id(p) for p in model_co.backbone.parameters()}
        for p_opt in params_co:
            self.assertIn(id(p_opt), mc_params_ids_co)
            self.assertNotIn(id(p_opt), lm_params_ids_co)
        model_lm, _ = self._create_mock_diffusion_model({'training': {'stage': 'lm_only'}})
        params_lm = list(model_lm._get_parameters())
        mc_params_ids_lm = {id(p) for p in model_lm.meta_controller.parameters()}
        lm_params_ids_lm = {id(p) for p in model_lm.backbone.parameters()}
        for p_opt in params_lm:
            self.assertIn(id(p_opt), lm_params_ids_lm)
            self.assertNotIn(id(p_opt), mc_params_ids_lm)
        model_jt, _ = self._create_mock_diffusion_model({'training': {'stage': 'joint'}})
        params_jt = list(model_jt._get_parameters())
        mc_params_ids_jt = {id(p) for p in model_jt.meta_controller.parameters()}
        lm_params_ids_jt = {id(p) for p in model_jt.backbone.parameters()}
        param_ids_jt = {id(p) for p in params_jt}
        self.assertTrue(mc_params_ids_jt.issubset(param_ids_jt))
        self.assertTrue(lm_params_ids_jt.issubset(param_ids_jt))
        print("Diffusion._get_parameters() tests passed.")

    def test_diffusion_get_block_features(self):
        print("\nTesting Diffusion._get_block_features()...")
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
        print("Diffusion._get_block_features() tests passed.")

    def test_diffusion_q_xt_and_sample_t(self):
        print("\nTesting Diffusion.q_xt and _sample_t...")
        B, N_Tokens_model = 2, 64
        block_size_cfg = 16 
        N_Blk = N_Tokens_model // block_size_cfg
        model, config = self._create_mock_diffusion_model({
            'model': {'length': N_Tokens_model}, 
            'block_size': block_size_cfg,
            'training': {'resample': False} 
        })
        x0 = torch.randint(0, config.vocab_size, (B, N_Tokens_model), device=self.device)
        sampling_eps_min = torch.tensor(0.1, device=self.device)
        sampling_eps_max = torch.tensor(0.9, device=self.device)
        t_per_block = model._sample_t((B, N_Blk), self.device, sampling_eps_min, sampling_eps_max, block_size=1)
        self.assertEqual(t_per_block.shape, (B, N_Blk))
        self.assertTrue(torch.all((t_per_block >= sampling_eps_min) & (t_per_block <= sampling_eps_max)))
        t_nll = model._sample_t((B, N_Blk), self.device, torch.tensor(1.0), torch.tensor(1.0), block_size=1)
        self.assertTrue(torch.all(t_nll == 1.0))
        p_per_block = t_per_block 
        p_per_token_for_qxt = p_per_block.repeat_interleave(block_size_cfg, dim=1) 
        xt = model.q_xt(x0, p_per_token_for_qxt.unsqueeze(-1)) 
        self.assertEqual(xt.shape, x0.shape)
        p_region1 = torch.full((B, N_Tokens_model // 2), 0.1, device=self.device)
        p_region2 = torch.full((B, N_Tokens_model // 2), 0.9, device=self.device)
        p_distinct = torch.cat([p_region1, p_region2], dim=1)
        xt_distinct = model.q_xt(x0, p_distinct.unsqueeze(-1))
        masks_region1 = (xt_distinct[:, :N_Tokens_model//2] == model.mask_index).float().mean()
        masks_region2 = (xt_distinct[:, N_Tokens_model//2:] == model.mask_index).float().mean()
        self.assertAlmostEqual(masks_region1.item(), 0.1, delta=0.05) 
        self.assertAlmostEqual(masks_region2.item(), 0.9, delta=0.05)
        xt_nll = model.q_xt(x0, torch.ones_like(p_per_token_for_qxt).unsqueeze(-1), 
                            block_size=1, sampling_eps_min=torch.tensor(1.0))
        self.assertTrue(torch.all(xt_nll == model.mask_index))
        print("Diffusion.q_xt and _sample_t tests passed.")

    def test_diffusion_process_sigma(self):
        print("\nTesting Diffusion._process_sigma...")
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
        print("Diffusion._process_sigma tests passed.")

    @unittest.skip("Skipping full _loss test due to complexity and reliance on many mocks; test components instead.")
    def test_diffusion_loss_and_forward_pass_components(self):
        pass

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)