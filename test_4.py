import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
# import hydra # Not strictly needed for these unit tests if config is manually created
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import os # For os.path.exists and os.remove

# Assuming your project structure allows these imports
from diffusion import Diffusion
from models.meta_controller import MetaController
import noise_schedule # For get_warped_schedule_outputs etc.


# --- Mock Components ---
class MockBackbone(nn.Module):
    def __init__(self, hidden_size, vocab_size, block_size, length):
        super().__init__()
        self.config = OmegaConf.create({ # Mocking a config attribute
            'hidden_size': vocab_size, # For this mock, input to linear is vocab_size
            'length': length
        })
        self.vocab_size = vocab_size
        # Crude projection layer: vocab_size -> vocab_size
        self.output_proj = nn.Linear(vocab_size, vocab_size)
        self.length = length

    def forward(self, indices, sigma=None, timesteps=None, store_kv=False, sample_mode=False, **kwargs):
        B, L = indices.shape
        # Ensure indices are within vocab bounds before one-hot
        clamped_indices = torch.clamp(indices, 0, self.vocab_size - 1)
        one_hot_indices = F.one_hot(clamped_indices, num_classes=self.vocab_size).float()
        
        if sigma is not None: # Minimal effect simulation
            # Ensure sigma_effect can be broadcast to (B, L, 1)
            if sigma.ndim == 1: # (B,)
                sigma_effect = sigma.view(B, 1, 1).expand(-1, L, -1)
            elif sigma.ndim == 2 and sigma.shape[1] == 1: # (B, 1)
                sigma_effect = sigma.view(B, 1, 1).expand(-1, L, -1)
            elif sigma.ndim == 2 and sigma.shape[1] == L: # (B, L)
                sigma_effect = sigma.view(B, L, 1)
            elif sigma.ndim == 2: # (B, N_block_dim) -> needs careful expansion/repeat
                if L > 0 and sigma.shape[1] > 0 and L % sigma.shape[1] == 0:
                    block_len_for_repeat = L // sigma.shape[1]
                    sigma_effect = sigma.repeat_interleave(block_len_for_repeat, dim=1).unsqueeze(-1)
                else: # Fallback: average and expand
                     sigma_effect = sigma.mean(dim=1, keepdim=True).unsqueeze(-1).expand(-1, L, -1)
            else: # Fallback for unexpected shapes
                sigma_effect = 0
            
            one_hot_indices = one_hot_indices + sigma_effect * 0.01 

        logits = self.output_proj(one_hot_indices)
        return logits

    def reset_kv_cache(self, eval_batch_size):
        pass


class MockBaseNoiseSchedule(nn.Module):
    def __init__(self, device='cpu', type="linear"):
        super().__init__()
        self._device_val = torch.device(device) # Store device explicitly
        self.type = type
        self.sigma_min = 1e-4
        self.sigma_max = 20

    def get_alpha_bar(self, t_in):
        t = t_in.to(self._device_val) # Ensure t is on correct device for calculations
        t_clamped_for_logit = t.clamp(min=1e-6, max=1.0 - 1e-6)
        if self.type == "linear":
            return (1.0 - t_clamped_for_logit)
        elif self.type == "cosine":
            return torch.cos(t_clamped_for_logit * torch.pi / 2)**2
        elif self.type == "loglinear":
            sigma_val = self.sigma_min * (self.sigma_max / self.sigma_min)**t # Renamed sigma to sigma_val
            alpha_bar = torch.exp(-0.5 * sigma_val**2)
            return alpha_bar.clamp(min=1e-6, max=1.0 - 1e-6)
        else:
            raise ValueError(f"Unknown mock schedule type: {self.type}")

    def get_log_alpha_bar_base_derivative_t(self, t_in):
        t = t_in.to(self._device_val) # Ensure t is on correct device
        t_clamped_for_calc = t.clamp(min=1e-7, max=1.0 - 1e-7)
        with torch.enable_grad():
            t_var = t_clamped_for_calc.detach().requires_grad_(True)
            alpha_bar_val = self.get_alpha_bar(t_var)
            logit_alpha_bar = torch.logit(alpha_bar_val.clamp(min=1e-6, max=1.0-1e-6))
            grad_outputs = torch.ones_like(logit_alpha_bar)
            derivative = torch.autograd.grad(logit_alpha_bar, t_var, grad_outputs=grad_outputs, create_graph=False)[0]
        return derivative.detach()

    def to(self, device_obj, *args, **kwargs):
        if isinstance(device_obj, (torch.device, str)):
            self._device_val = torch.device(device_obj)
        return super().to(device_obj, *args, **kwargs)

    @property
    def module_device(self): # Renamed for clarity from 'device' to avoid nn.Module conflict
        return self._device_val


class TestDiffusionAdaptiveScheduling(unittest.TestCase):
    tokenizer_path = "test_tokenizer.json"

    @classmethod
    def setUpClass(cls):
        simple_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "c", "hello", "world", ".", "EOS", "BOS"]

        if not os.path.exists(cls.tokenizer_path):
            print(f"Tokenizer file not found at {os.path.abspath(cls.tokenizer_path)}. Creating new one.")
            cls._create_tokenizer_file(simple_vocab, cls.tokenizer_path)

        try:
            print(f"Loading tokenizer from: {os.path.abspath(cls.tokenizer_path)}")
            cls.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=cls.tokenizer_path,
                unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                sep_token="[SEP]", mask_token="[MASK]",
                eos_token="EOS", bos_token="BOS"
            )
        except Exception as e:
            print(f"FATAL: Could not load tokenizer {cls.tokenizer_path}: {e}")
            raise

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {cls.device}")

        base_config = {
            'mode': 'sample_eval',
            'diffusion': 'absorbing_state', 'seed': 42, 'block_size': 4,
            'model': {
                'name': 'test_small', 'type': 'ddit', 'hidden_size': 32, 'cond_dim': 16,
                'length': 16, 'n_blocks': 2, 'n_heads': 2, 'scale_by_sigma': True,
                'dropout': 0.0, 'tie_word_embeddings': True, 'adaln': False, 'attn_backend': 'sdpa'
            },
            'loader': {'eval_batch_size': 2, 'num_workers': 0, 'pin_memory': False},
            'sampling': {
                'noise_removal': False, 'num_sample_batches': 1, 'var_length': False,
                'logdir': './test_samples', 'nucleus_p': 1.0, 'first_hitting': False, 'kv_cache': False
            },
            'training': {
                'ema': 0.0, 'antithetic_sampling': False, 'sampling_eps': 1e-3,
                'resample': False, 'sampling_eps_min': 1e-3, 'sampling_eps_max': 1.0, 'stage': 'joint'
            },
            'eval': { # <<< MODIFIED HERE
                'checkpoint_path': 'dummy/path',
                'perplexity_batch_size': 2, # Added
                'gen_ppl_eval_model_name_or_path': 'gpt2', # Added (mock name, not actually loaded in these tests)
                'compute_perplexity_on_sanity': False, # Added, from main config
                'disable_ema': False, # Added, from main config
                'generate_samples': False # Added, from main config
            },
            'optim': {},
            'noise': { 
                'type': 'loglinear', 
                'sigma_min': 1e-4,
                'sigma_max': 20.0,
                'eps': 1e-5 
            },
            'algo': { 
                'name': 'bd3lm', 'backbone': 'dit', 'parameterization': 'bd3lm',
                'time_conditioning': False, 'T': 100,
                'causal_attention': False, 'dropout': 0.0, 'ignore_bos': True,
                'cross_attn': True, 'var_min': True, 'clip_search_delta': 0.05,
                'clip_search_widths': [], 'fix_clipping': False, 'sampler': 'semi_ar',
                'mdlm_loss_scale': False,
                'base_noise_type': 'linear', 
                'schedule_clamp_epsilon': 1e-6, 'lambda_steps_penalty': 0.01,
                'min_alpha_1_target': 0.005, 'lambda_min_alpha_1_penalty': 1.0,
                'alpha_1_clamp_min': 1e-5, 'alpha_1_clamp_max': 1.0 - 1e-6,
                'dynamic_nfe_grid_size': 100,
                'feature_extractor_model_name': None, 'spacy_model_name': None, 'benepar_model_name': None,
                'meta_controller': {
                    'feature_dim': 3, 'hidden_dim': 16,
                    's_tilde_squash_factor': 4.0, 's_min_epsilon': 0.01
                }
            }
        }
        cls.config = OmegaConf.create(base_config)

    @classmethod
    def _create_tokenizer_file(cls, simple_vocab, tokenizer_path_to_save):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        vocab_map = {token: i for i, token in enumerate(simple_vocab)}
        hf_tokenizer_obj = Tokenizer(WordLevel(vocab=vocab_map, unk_token="[UNK]"))
        hf_tokenizer_obj.pre_tokenizer = Whitespace()
        hf_tokenizer_obj.save(tokenizer_path_to_save)
        print(f"Tokenizer file created at {os.path.abspath(tokenizer_path_to_save)}")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.tokenizer_path):
            try:
                os.remove(cls.tokenizer_path)
                print(f"Cleaned up tokenizer file: {cls.tokenizer_path}")
            except OSError as e:
                print(f"Error cleaning up tokenizer file {cls.tokenizer_path}: {e}")

    def _setup_diffusion_instance(self, config_overrides=None):
        effective_config = self.config.copy()

        intended_final_algo_backbone = effective_config.algo.backbone # Store original intent
        if config_overrides and OmegaConf.is_dict(config_overrides.get('algo')) and 'backbone' in config_overrides['algo']:
            intended_final_algo_backbone = config_overrides['algo']['backbone']
        
        # Create a config copy for initialization to avoid modifying self.config
        config_for_init = self.config.copy() 
        if config_overrides:
            config_for_init = OmegaConf.merge(config_for_init, OmegaConf.create(config_overrides))
        
        # Temporarily set backbone to a known simple type if 'mock' is not directly supported by Diffusion's __init__
        # This is if Diffusion.__init__ does specific logic based on config.algo.backbone string.
        # The actual backbone instance will be mocked afterwards.
        original_init_backbone_val = config_for_init.algo.backbone
        if original_init_backbone_val == 'mock': # If we intend to test 'mock' directly
            config_for_init.algo.backbone = 'dit' # Or any valid known type for init
            print(f"Temporarily setting algo.backbone to 'dit' for Diffusion init, original was 'mock'.")


        diffusion_model = Diffusion(config_for_init, self.tokenizer)

        # Restore the intended config value for algo.backbone after Diffusion init,
        # so that subsequent logic in the test uses the correct intended value.
        diffusion_model.config.algo.backbone = intended_final_algo_backbone

        # Replace actual backbone with MockBackbone
        diffusion_model.backbone = MockBackbone(
            diffusion_model.config.model.hidden_size, 
            diffusion_model.vocab_size,
            diffusion_model.config.block_size, 
            diffusion_model.config.model.length
        ).to(self.device)
        
        # Replace base_noise_schedule with MockBaseNoiseSchedule
        diffusion_model.base_noise_schedule = MockBaseNoiseSchedule(
            device=self.device, type=diffusion_model.config.algo.base_noise_type
        ).to(self.device)
        
        # Nullify real feature extractors as they are not part of these unit tests
        diffusion_model.distilbert_model = None
        diffusion_model.distilbert_tokenizer = None
        diffusion_model.nlp = None

        # Ensure MetaController's feature_dim matches the dynamically calculated one
        calculated_feature_dim = sum(diffusion_model.feature_dim_parts.values())
        if diffusion_model.meta_controller.fc_in.in_features != calculated_feature_dim:
            print(f"Test setup: MetaController re-initialization needed. fc_in.in_features ({diffusion_model.meta_controller.fc_in.in_features}) "
                  f"!= calculated_feature_dim ({calculated_feature_dim}).")
            
            # Create a new config node for meta_controller with updated feature_dim
            mc_reinit_config_node = diffusion_model.config.algo.meta_controller.copy()
            mc_reinit_config_node.feature_dim = calculated_feature_dim
            
            # Create a temporary full config structure just for MC re-initialization
            mc_temp_full_config_for_reinit = OmegaConf.create({'algo': {'meta_controller': mc_reinit_config_node}})
            diffusion_model.meta_controller = MetaController(mc_temp_full_config_for_reinit)

        return diffusion_model.to(self.device)


    def test_extract_features_for_block_generation_shape_and_device(self):
        model = self._setup_diffusion_instance()
        B = 2
        context_len = 10
        x_accum_context = torch.randint(0, len(self.tokenizer.get_vocab()), (B, context_len), device=self.device)
        current_block_idx = 1
        total_num_blocks = 5

        features = model._extract_features_for_block_generation(
            x_accum_context, current_block_idx, total_num_blocks
        )
        expected_feature_dim = model.meta_controller_config.feature_dim
        self.assertEqual(features.shape, (B, 1, expected_feature_dim))
        self.assertEqual(features.device, self.device)


    def test_extract_features_pos_ratio(self):
        model = self._setup_diffusion_instance()
        B = 1
        x_accum_context = torch.empty(B, 0, device=self.device, dtype=torch.long) 
        
        pos_ratio_idx = -1
        current_idx = 0
        # This order must match the concatenation order in _extract_features_for_block_generation
        ordered_feature_keys = ["pos_ratio", "semantic_context", "block_entropy_proxy", 
                                "token_variance_proxy", "syntactic_depth_context", "context_entropy"]
        
        found_pos_ratio = False
        for key in ordered_feature_keys:
            dim_part = model.feature_dim_parts.get(key, 0) # Get dim, default 0 if key might be missing
            if key == "pos_ratio":
                if dim_part > 0: 
                    pos_ratio_idx = current_idx # pos_ratio is at the start of its segment
                    found_pos_ratio = True
                break # Found (or confirmed absence of) pos_ratio segment
            current_idx += dim_part # Accumulate index up to this feature
        
        if not found_pos_ratio or model.feature_dim_parts.get("pos_ratio", 0) == 0:
             self.skipTest("pos_ratio feature is not enabled (dim is 0 or key missing in feature_dim_parts)")
             return

        self.assertNotEqual(pos_ratio_idx, -1, "pos_ratio feature expected but its index not found based on feature_dim_parts ordering.")
            
        features_first = model._extract_features_for_block_generation(x_accum_context, 0, 3)
        self.assertAlmostEqual(features_first[0, 0, pos_ratio_idx].item(), 0.0 / (3 - 1), places=5)

        features_middle = model._extract_features_for_block_generation(x_accum_context, 1, 3)
        self.assertAlmostEqual(features_middle[0, 0, pos_ratio_idx].item(), 1.0 / (3 - 1), places=5)

        features_last = model._extract_features_for_block_generation(x_accum_context, 2, 3)
        self.assertAlmostEqual(features_last[0, 0, pos_ratio_idx].item(), 2.0 / (3 - 1), places=5)

        features_single = model._extract_features_for_block_generation(x_accum_context, 0, 1)
        self.assertAlmostEqual(features_single[0, 0, pos_ratio_idx].item(), 0.0, places=5)


    def test_extract_features_no_extractors(self):
        # _setup_diffusion_instance already disables distilbert and spacy/benepar
        model = self._setup_diffusion_instance()
        
        # Check that corresponding feature_dim_parts are zero
        self.assertEqual(model.feature_dim_parts["semantic_context"], 0)
        self.assertEqual(model.feature_dim_parts["syntactic_depth_context"], 0)
        
        # The meta_controller_config.feature_dim should match the sum of active feature_dim_parts
        expected_dim = sum(model.feature_dim_parts.values())
        self.assertEqual(model.meta_controller_config.feature_dim, expected_dim)

        B = 1
        x_accum_context = torch.empty(B, 0, device=self.device, dtype=torch.long)
        features = model._extract_features_for_block_generation(x_accum_context, 0, 1)
        self.assertEqual(features.shape, (B, 1, expected_dim))
        # Test that it runs without error and produces the correct shape. Specific values would depend on proxy logic (e.g. zeros).

    def test_get_warped_outputs_at_t_shapes_and_alpha0(self):
        model = self._setup_diffusion_instance()
        B = 2
        s_b = torch.rand(B, 1, 1, device=self.device) + 0.5 # Ensure s_b is positive
        t_values_near_0 = torch.tensor([[0.001], [0.002]], device=self.device, dtype=torch.float32) # (B, N_t=1)

        alpha_b_t, beta_b_t, p_b_t = model._get_warped_outputs_at_t(s_b, t_values_near_0)

        self.assertEqual(alpha_b_t.shape, (B, 1, 1)) # Expect (B, N_Blk_eff=1, N_t_eff=1)
        self.assertEqual(beta_b_t.shape, (B, 1, 1))
        self.assertEqual(p_b_t.shape, (B, 1, 1))

        # Check alpha_b(t~0) is close to target_alpha_at_0
        expected_alpha0 = model.target_alpha_at_0.item()
        self.assertTrue(torch.allclose(alpha_b_t.squeeze(), torch.full_like(alpha_b_t.squeeze(), expected_alpha0), atol=1e-2))
        self.assertTrue(torch.allclose(p_b_t, 1.0 - alpha_b_t, atol=1e-6)) # p_b_t = 1 - alpha_b_t
        self.assertTrue((beta_b_t >= -1e-5).all()) # beta_b_t should be non-negative if L_base' is positive

    def test_get_remapped_timesteps_shape_range_monotonicity(self):
        model = self._setup_diffusion_instance()
        B = 2
        s_b = torch.rand(B, 1, 1, device=self.device) + 0.5 # Positive s_b
        total_global_steps = 50

        prev_t_prime_k_batch = torch.full((B,1), -1.0, device=self.device) # To check monotonicity of t_prime_k

        for k_step in [1, total_global_steps // 2, total_global_steps]:
            t_prime_k, t_prime_k_minus_1 = model._get_remapped_timesteps_for_block(s_b, k_step, total_global_steps)
            self.assertEqual(t_prime_k.shape, (B, 1))
            self.assertEqual(t_prime_k_minus_1.shape, (B, 1))
            # Check range [0, 1], allowing for small floating point errors or grid effects
            self.assertTrue((t_prime_k >= -1e-5).all() and (t_prime_k <= 1.0001 + 1e-5).all()) # Wider tolerance for grid search
            self.assertTrue((t_prime_k_minus_1 >= -1e-5).all() and (t_prime_k_minus_1 <= 1.0001 + 1e-5).all())
            # t_k should be >= t_{k-1} (or very close, due to discrete grid)
            self.assertTrue((t_prime_k >= t_prime_k_minus_1 - 1e-5).all()) 
            
            # Check that t_prime_k is non-decreasing with k_step
            self.assertTrue((t_prime_k >= prev_t_prime_k_batch - 1e-5).all()) # t_prime_k(k) >= t_prime_k(k-1)
            prev_t_prime_k_batch = t_prime_k.clone()


    def test_get_remapped_timesteps_edge_cases_k(self):
        model = self._setup_diffusion_instance()
        B = 1
        s_b = torch.tensor([[[1.0]]], device=self.device) # Neutral s_b (approx)
        total_global_steps = 50

        # k=1 (first step)
        t_prime_k_first, t_prime_k_minus_1_first = model._get_remapped_timesteps_for_block(s_b, 1, total_global_steps)
        # t'_{k-1} for k=1 should map to C_{k-1}=0, so t'_{k-1} should be ~0
        self.assertTrue(torch.allclose(t_prime_k_minus_1_first, torch.zeros_like(t_prime_k_minus_1_first), atol=1e-6)) 
        self.assertTrue(t_prime_k_first[0,0] > -1e-5) # t'_k should be > 0 (or ~0 if T_bj is tiny)

        # k = total_global_steps (last step)
        t_prime_k_last, t_prime_k_minus_1_last = model._get_remapped_timesteps_for_block(s_b, total_global_steps, total_global_steps)
        # t'_k for k=N_global should map to C_k = T_bj, so t'_k should be ~1
        # Allow tolerance due to discrete grid search in _get_remapped_timesteps_for_block
        self.assertTrue(torch.allclose(t_prime_k_last, torch.ones_like(t_prime_k_last), atol=0.02 + 1.0/model.dynamic_nfe_grid_size)) 
        self.assertTrue(t_prime_k_minus_1_last[0,0] < t_prime_k_last[0,0] + 1e-5) # t'_{N-1} < t'_N


    def test_semi_ar_sampler_runs_bd3lm(self):
        config_overrides = {'block_size': 4, 'model': {'length': 8}}
        model = self._setup_diffusion_instance(config_overrides)
        model.config.sampling.var_length = False # Fixed length for this test

        n_samples = model.config.loader.eval_batch_size
        num_steps = 10 # Diffusion steps per block
        num_strides = model.config.model.length // model.config.block_size
        seqlen = model.config.model.length

        samples, nfes = model._semi_ar_sampler(n_samples, num_steps, num_strides, seqlen)

        self.assertIsNotNone(samples)
        self.assertEqual(samples.shape, (n_samples, seqlen))
        self.assertTrue(nfes >= 0) # NFE can be 0 if num_steps=0 or block fully clean initially

    def test_analytic_sampler_runs_bd3lm(self):
        config_overrides = {
            'block_size': 4, 
            'model': {'length': 8}, 
            'algo': {'sampler': 'analytic'} # Ensure sampler is set to analytic
        }
        model = self._setup_diffusion_instance(config_overrides)
        model.config.sampling.var_length = False # Fixed length

        n_samples = model.config.loader.eval_batch_size
        num_steps = 10 # Diffusion steps per block
        num_strides = model.config.model.length // model.config.block_size
        seqlen = model.config.model.length

        samples, nfes = model._analytic_sampler(n_samples, num_strides, num_steps, seqlen, eps=0.01)

        self.assertIsNotNone(samples)
        if samples is not None: # Can be None if all samples fail stop conditions early
            self.assertEqual(samples.shape, (n_samples, seqlen))
        self.assertTrue(nfes >= 0) # NFE check


    def test_semi_ar_sampler_var_length_bd3lm(self):
        config_overrides = {'block_size': 2, 'model': {'length': 6}} # Short seq for faster test
        model = self._setup_diffusion_instance(config_overrides)
        model.config.sampling.var_length = True
        
        # Mock _check_stop_conds to force early stop for one sample to test truncation
        original_check_stop_conds = model._check_stop_conds # Store original
        def mock_check_stop_conds_custom(x_batch_input):
            B, L_current = x_batch_input.shape
            stop_flags = torch.zeros(B, dtype=torch.bool, device=model.device)
            final_x_list = list(x_batch_input.clone().unbind(0)) # Create list of tensors
            all_should_stop_flag = True # Assume all stop unless one continues
            if B > 0:
                for i in range(B):
                    if i == 0 and L_current >= model.block_size: # Stop the first sample after 1st block
                        stop_flags[i] = True 
                        final_x_list[i] = final_x_list[i][:model.block_size] # Truncate it
                        if final_x_list[i].numel() == 0: # Ensure not empty
                            final_x_list[i] = torch.tensor([model.tokenizer.bos_token_id], device=model.device, dtype=torch.long)
                    elif L_current < model.config.model.length: # If other samples are not full length yet
                        all_should_stop_flag = False # Mark that not all are stopping

            return all_should_stop_flag if B > 0 else True, final_x_list # Return flag and list
        model._check_stop_conds = mock_check_stop_conds_custom

        n_samples = model.config.loader.eval_batch_size # e.g. 2
        num_steps = 5
        num_strides = model.config.model.length // model.config.block_size
        seqlen = model.config.model.length

        output_tensor_batch, nfes = model._semi_ar_sampler(n_samples, num_steps, num_strides, seqlen)
        
        model._check_stop_conds = original_check_stop_conds # Restore original

        self.assertTrue(nfes >= 0)
        if output_tensor_batch is not None:
            # _semi_ar_sampler for var_length now returns the batch tensor, where individual samples might be truncated
            # or padded if _check_stop_conds output was processed into a batch tensor.
            # The current _semi_ar_sampler's var_length + stop logic returns the *first* sample if it stops early.
            # This needs refinement if batch processing of var_len is key.
            # For this test, if mock stops first sample, output_tensor_batch might be just that one.
            self.assertTrue(output_tensor_batch.shape[0] <= n_samples) # Could be 1 if only first sample returned
            self.assertTrue(output_tensor_batch.shape[1] <= seqlen)
            if n_samples > 0 and output_tensor_batch.shape[0] > 0: # If first sample was returned
                 if output_tensor_batch.shape[0] == 1: # If only the stopped sample was returned
                    self.assertEqual(output_tensor_batch.shape[1], model.block_size) # Check if first sample truncated as per mock
        else:
            # Output can be None if sampling fails critically (e.g., all samples become empty)
            pass


    def test_adaptive_samplers_kv_cache(self):
        config_overrides = {
            'block_size': 2, 
            'model': {'length': 4}, 
            'sampling': {'kv_cache': True},
        }
        model = self._setup_diffusion_instance(config_overrides)
        
        original_forward_method = model.backbone.forward # Keep original for restoration
        forward_calls_store_kv_flags = []
        def mock_forward_for_kv(*args, **kwargs): # Mock to record store_kv calls
            forward_calls_store_kv_flags.append(kwargs.get('store_kv', False))
            return original_forward_method(*args, **kwargs)
        
        model.backbone.forward = mock_forward_for_kv
        model.backbone.reset_kv_cache = MagicMock() # Mock reset_kv_cache

        n_samples = 1
        num_steps = 3
        num_strides = model.config.model.length // model.config.block_size
        seqlen = model.config.model.length

        # Test Semi-AR (default sampler in base_config.algo)
        model.backbone.reset_kv_cache.reset_mock()
        forward_calls_store_kv_flags.clear()
        model._semi_ar_sampler(n_samples, num_steps, num_strides, seqlen)
        model.backbone.reset_kv_cache.assert_called_once()
        self.assertTrue(any(call_flag for call_flag in forward_calls_store_kv_flags if call_flag is True))

        # Test Analytic with KV cache
        model.config.algo.sampler = 'analytic' # Switch sampler type in config
        model.sampler = 'analytic' # Also update the instance attribute if it's used directly

        model.backbone.reset_kv_cache.reset_mock()
        forward_calls_store_kv_flags.clear()
        model._analytic_sampler(n_samples, num_strides, num_steps, seqlen, eps=0.01)
        model.backbone.reset_kv_cache.assert_called_once()
        self.assertTrue(any(call_flag for call_flag in forward_calls_store_kv_flags if call_flag is True))
        
        model.backbone.forward = original_forward_method # Restore original forward


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)