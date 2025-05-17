import abc
import torch
import torch.nn as nn
import math # For torch.pi in newer PyTorch, or math.pi
import torch.nn.functional as F


def get_noise(config, noise_type=None):
  if noise_type is None:
    noise_type = config.noise.type

  if noise_type == 'loglinear':
    return LogLinearNoise(
        eps=config.noise.eps, 
        schedule_clamp_epsilon=config.algo.get('schedule_clamp_epsilon', 1e-6) # Use get for safety
    )
  elif noise_type == 'square':
    return ExpNoise(
        exp=2, 
        eps=config.noise.eps, 
        schedule_clamp_epsilon=config.algo.get('schedule_clamp_epsilon', 1e-6)
    )
  elif noise_type == 'square_root':
    return ExpNoise(
        exp=0.5, 
        eps=config.noise.eps, 
        schedule_clamp_epsilon=config.algo.get('schedule_clamp_epsilon', 1e-6)
    )
  elif noise_type == 'log':
    return LogarithmicNoise(
        eps=config.noise.eps, 
        schedule_clamp_epsilon=config.algo.get('schedule_clamp_epsilon', 1e-6)
    )
  elif noise_type == 'cosine':
    return CosineNoise(
        eps=config.noise.eps, 
        schedule_clamp_epsilon=config.algo.get('schedule_clamp_epsilon', 1e-6)
    )
  else:
    raise ValueError(f'{noise_type} is not a valid noise')

class Noise(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """
  def __init__(self, eps=1e-3, schedule_clamp_epsilon=1e-6): 
    super().__init__()
    self.eps = eps 
    self.schedule_clamp_epsilon = schedule_clamp_epsilon
  
  def forward(self, t):
    return self.compute_loss_scaling_and_move_chance(t)

  @abc.abstractmethod
  def compute_loss_scaling_and_move_chance(self, t):
    pass

  @abc.abstractmethod
  def get_alpha_bar_base(self, t: torch.Tensor) -> torch.Tensor:
    """ Returns alpha_bar_base(t) before clamping for logit. """
    raise NotImplementedError

  def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
    """ Returns alpha_bar_base(t) appropriately clamped for logit. """
    alpha_bar = self.get_alpha_bar_base(t)
    return torch.clamp(alpha_bar, min=self.schedule_clamp_epsilon, max=1.0 - self.schedule_clamp_epsilon)

  def get_log_alpha_bar_base_derivative_t(self, t: torch.Tensor) -> torch.Tensor:
    """ Computes d/dt logit(alpha_bar_base(t)) numerically or analytically. """
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

  @abc.abstractmethod
  def get_t_from_move_chance(self, p: torch.Tensor) -> torch.Tensor:
    """ Returns t_base such that p_base(t_base) = p. Needs schedule-specific inversion. """
    raise NotImplementedError

class CosineNoise(Noise):
  def __init__(self, eps=1e-3, schedule_clamp_epsilon=1e-6): 
    super().__init__(eps, schedule_clamp_epsilon)

  def compute_loss_scaling_and_move_chance(self, t):
    # p(t) = (1-self.eps)*(1 - cos(t*pi/2)) + self.eps -> p(0)=self.eps, p(1)=1
    # alpha_bar(t) = 1 - p(t) = (1-self.eps)*cos(t*pi/2)
    # log alpha_bar(t) = log(1-self.eps) + log(cos(t*pi/2))
    # d/dt log alpha_bar(t) = -tan(t*pi/2) * pi/2
    # loss_scale = - d/dt log alpha_bar(t) = tan(t*pi/2) * pi/2 (positive)
    # Clamp t to avoid issues at boundaries if it can go outside [0,1]
    t_clamped = torch.clamp(t, 0.0, 1.0)
    move_chance = (1.0 - self.eps) * (1.0 - torch.cos(0.5 * torch.pi * t_clamped)) + self.eps
    
    # For loss_scaling = tan(t*pi/2) * pi/2:
    # Handle t_clamped approaching 1 where tan -> inf
    # If t_clamped is exactly 1, tan is undefined. If very close, it's huge.
    # We need to make sure the derivative d/dt log(alpha_bar_base(t)) is well-defined or handled.
    # The get_log_alpha_bar_base_derivative_t uses logit(get_alpha_bar(t)).
    # get_alpha_bar clamps alpha_bar to [eps_clamp, 1-eps_clamp].
    # So logit is well-defined. The derivative should also be.
    # loss_scale = - self.get_log_alpha_bar_base_derivative_t(t)
    # However, the original BD3LM implementation might have a specific formula:
    # Original from provided code:
    # cos_term = - (1 - self.eps) * torch.cos(t * torch.pi / 2)
    # sin_term = - (1 - self.eps) * torch.sin(t * torch.pi / 2)
    # move_chance_orig = cos_term + 1  # This is (1-eps)(1-cos(t*pi/2)). p(0)=0, p(1)=1-eps.
    # loss_scaling_orig = sin_term / (move_chance_orig + self.eps) * torch.pi / 2 # This is negative.

    # Let's use the definition that leads to positive loss scale for a positive CE term
    # and aligns with common diffusion practice where loss_scale is related to -d/dt log(alpha_bar)
    loss_scaling = (0.5 * torch.pi) * torch.tan(0.5 * torch.pi * t_clamped)
    # Clamp loss_scaling to avoid infinity at t=1 for tan
    loss_scaling = torch.clamp(loss_scaling, max=1e5) # Arbitrary large number, tune if needed

    return loss_scaling, move_chance

  def get_alpha_bar_base(self, t: torch.Tensor) -> torch.Tensor:
    # alpha_bar_base(t) = (1-eps) * cos(t * pi/2)
    # Clamp t to [0,1] for cos domain
    t_clamped = torch.clamp(t, 0.0, 1.0)
    return (1.0 - self.eps) * torch.cos(t_clamped * torch.pi / 2.0)
  
  def get_t_from_move_chance(self, p: torch.Tensor) -> torch.Tensor:
    # p(t) = (1-eps)*(1 - cos(t*pi/2)) + eps
    # (p - eps) / (1-eps) = 1 - cos(t*pi/2)
    # cos(t*pi/2) = 1 - (p-eps)/(1-eps) = (1-eps-p+eps)/(1-eps) = (1-p)/(1-eps)
    # t*pi/2 = acos((1-p)/(1-eps))
    # Ensure (1-p)/(1-eps) is in [-1, 1]. Since p in [eps, 1], (1-p) in [0, 1-eps].
    # So (1-p)/(1-eps) is in [0, 1].
    # Clamp p to [self.eps, 1.0] for safety
    p_clamped = torch.clamp(p, min=self.eps, max=1.0)
    safe_arg = torch.clamp((1.0 - p_clamped) / (1.0 - self.eps + 1e-9), min=0.0, max=1.0) # Add epsilon for division
    t = (2.0 / torch.pi) * torch.acos(safe_arg)
    return t

class ExpNoise(Noise):
  def __init__(self, exp=2, eps=1e-3, schedule_clamp_epsilon=1e-6):
    super().__init__(eps, schedule_clamp_epsilon)
    self.exp = exp
  
  def compute_loss_scaling_and_move_chance(self, t):
    # Clamp t to [0,1]
    t_clamped = torch.clamp(t, 0.0, 1.0)
    move_chance = torch.pow(t_clamped, self.exp)
    # Ensure move_chance is at least self.eps.
    # The original `torch.clamp(move_chance, min=self.eps)` would be problematic
    # if t_clamped^exp is very small but non-zero, then clamping it to self.eps changes the function.
    # A better way for p(t) = t^exp, p(0)=0, p(1)=1.
    # To make p(0)=eps, p(1)=1:  p(t) = t^exp * (1-eps) + eps
    move_chance = torch.pow(t_clamped, self.exp) * (1.0 - self.eps) + self.eps

    # alpha_bar(t) = 1 - p(t) = (1-t^exp)(1-eps)
    # log(alpha_bar(t)) = log(1-t^exp) + log(1-eps)
    # d/dt log(alpha_bar(t)) = (1/(1-t^exp)) * (-exp * t^(exp-1))
    # loss_scale = -d/dt = exp * t^(exp-1) / (1-t^exp)
    # Handle t_clamped approaching 1 where denominator -> 0
    numerator = self.exp * torch.pow(t_clamped, self.exp - 1) * (1.0 - self.eps) # from d(alpha_bar)/dt
    denominator_alpha_bar = (1.0 - torch.pow(t_clamped, self.exp)) * (1.0 - self.eps) # This is alpha_bar for p(t)=t^exp
                                                                               # If p(t) = t^exp*(1-eps)+eps, then alpha_bar=(1-t^exp)(1-eps)

    # loss_scale = - (d(alpha_bar)/dt) / alpha_bar
    d_alpha_bar_dt = -self.exp * torch.pow(t_clamped, self.exp - 1) * (1.0 - self.eps)
    alpha_bar = (1.0 - torch.pow(t_clamped, self.exp)) * (1.0 - self.eps)
    
    loss_scaling = -d_alpha_bar_dt / (alpha_bar + 1e-9) # Add epsilon for stability
    loss_scaling = torch.clamp(loss_scaling, max=1e5) # Clamp large values near t=1

    # Original was: loss_scaling = - (self.exp * torch.pow(t, self.exp-1)) / move_chance
    # This assumed move_chance = t^exp. If move_chance includes eps, it needs update.
    # The original move_chance = torch.clamp(torch.pow(t, self.exp), min=self.eps)
    # This implies a piecewise definition for p(t)
    return loss_scaling, move_chance

  def get_alpha_bar_base(self, t: torch.Tensor) -> torch.Tensor:
    t_clamped = torch.clamp(t, 0.0, 1.0)
    # alpha_bar_base(t) = (1 - t^exp)(1-eps)
    return (1.0 - torch.pow(t_clamped, self.exp)) * (1.0 - self.eps)

  def get_t_from_move_chance(self, p: torch.Tensor) -> torch.Tensor:
    # p(t) = t^exp * (1-eps) + eps
    # t^exp = (p-eps)/(1-eps)
    # t = ((p-eps)/(1-eps))^(1/exp)
    p_clamped = torch.clamp(p, min=self.eps, max=1.0)
    # Ensure base of power is non-negative
    base = torch.clamp((p_clamped - self.eps) / (1.0 - self.eps + 1e-9), min=0.0)
    t = torch.pow(base, 1.0 / self.exp)
    return t

class LogarithmicNoise(Noise):
  def __init__(self, eps=1e-3, schedule_clamp_epsilon=1e-6):
    super().__init__(eps, schedule_clamp_epsilon)
    self.log2 = torch.log(torch.tensor(2.0))

  def compute_loss_scaling_and_move_chance(self, t):
    t_clamped = torch.clamp(t, 0.0, 1.0) # Ensure t is in [0,1]
    # p(t) = log(1+t)/log(2) * (1-eps) + eps
    move_chance = (torch.log1p(t_clamped) / self.log2) * (1.0 - self.eps) + self.eps

    # alpha_bar(t) = (1 - log(1+t)/log(2)) * (1-eps)
    # d_alpha_bar_dt = -(1/( (1+t_clamped) * self.log2 )) * (1-eps)
    # loss_scale = -d_alpha_bar_dt / alpha_bar
    d_alpha_bar_dt = -(1.0 / ((1.0 + t_clamped) * self.log2)) * (1.0 - self.eps)
    alpha_bar = (1.0 - (torch.log1p(t_clamped) / self.log2)) * (1.0 - self.eps)
    
    loss_scaling = -d_alpha_bar_dt / (alpha_bar + 1e-9)
    loss_scaling = torch.clamp(loss_scaling, max=1e5)

    # Original:
    # move_chance_orig = torch.log1p(t) / torch.log(torch.tensor(2.0))
    # loss_scaling_orig = - 1 / (move_chance_orig * torch.log(torch.tensor(2.0)) * (1 + t))
    # This assumed move_chance_orig.
    return loss_scaling, move_chance

  def get_alpha_bar_base(self, t: torch.Tensor) -> torch.Tensor:
    t_clamped = torch.clamp(t, 0.0, 1.0)
    # alpha_bar_base(t) = (1 - log(1+t)/log(2)) * (1-eps)
    return (1.0 - (torch.log1p(t_clamped) / self.log2)) * (1.0 - self.eps)

  def get_t_from_move_chance(self, p: torch.Tensor) -> torch.Tensor:
    # p(t) = (log(1+t)/log2) * (1-eps) + eps
    # (p-eps)/(1-eps) = log(1+t)/log2
    # log2 * (p-eps)/(1-eps) = log(1+t)
    # exp(log2 * (p-eps)/(1-eps)) = 1+t
    # t = exp(log2 * (p-eps)/(1-eps)) - 1
    p_clamped = torch.clamp(p, min=self.eps, max=1.0)
    exponent = self.log2 * (p_clamped - self.eps) / (1.0 - self.eps + 1e-9)
    t = torch.expm1(exponent) # exp(exponent) - 1
    return t

class LogLinearNoise(Noise):
  """Log Linear noise schedule.
  Total noise sigma_base(t) = -log(1 - (1 - eps) * t).
  So, alpha_bar_base(t) = exp(-sigma_base(t)) = 1 - (1-eps)*t
  This means p_base(t) = 1 - alpha_bar_base(t) = (1-eps)*t.
  This original definition of p_base(t) goes from 0 to 1-eps.
  If we want p(t) from eps_schedule_min to 1 for t in [0,1]:
  p(t) = eps_schedule_min + t * (1 - eps_schedule_min)
  alpha_bar(t) = (1-t) * (1 - eps_schedule_min)
  """
  def __init__(self, eps=1e-3, schedule_clamp_epsilon=1e-6):
    super().__init__(eps, schedule_clamp_epsilon)
    # self.eps here is the 'eps_schedule_min' for the p(t) mapping
    # The 'eps' in the original formula was for (1-eps)*t
    # Let's keep self.eps as the minimum move_chance value.
    # sigma_max and sigma_min based on the new alpha_bar(t)
    self.sigma_max = -torch.log(self.get_alpha_bar_base(torch.tensor(1.0))) # alpha_bar(1)=0 -> sigma_max=inf. Clamped by get_alpha_bar
    self.sigma_min = -torch.log(self.get_alpha_bar_base(torch.tensor(0.0))) # alpha_bar(0)=1-eps -> sigma_min=-log(1-eps)

  def compute_loss_scaling_and_move_chance(self, t):
    t_clamped = torch.clamp(t, 0.0, 1.0)
    # p(t) = self.eps + t * (1 - self.eps)
    move_chance = self.eps + t_clamped * (1.0 - self.eps)
    
    # alpha_bar(t) = (1-t) * (1-self.eps)
    # d_alpha_bar_dt = -(1-self.eps)
    # loss_scale = -d_alpha_bar_dt / alpha_bar = (1-self.eps) / ((1-t)*(1-self.eps)) = 1 / (1-t)
    loss_scaling = 1.0 / (1.0 - t_clamped + 1e-9) # Add epsilon for t_clamped=1
    loss_scaling = torch.clamp(loss_scaling, max=1e5)
    
    # Original code had:
    # loss_scaling = - 1 / t
    # move_chance = t
    # This implies p(t) = t, and alpha_bar(t) = 1-t. loss_scale = 1/(1-t).
    # If p(t) = t, then loss_scale from BD3LM paper for this p(t) is -1/t (for their specific loss form)
    # Let's stick to p(t)=t for now if that's the BD3LM standard LogLinear.
    # If p(t)=t, then alpha_bar_base(t) = 1-t
    # This means LogLinearNoise's original `total_noise` and `rate_noise` were for a different p(t).
    # Let's use p(t)=t for this class to match existing use, assuming self.eps is ignored for p(t).
    move_chance_orig_bd3lm = t_clamped # p(t) goes from 0 to 1.
    # alpha_bar = 1 - t_clamped
    # loss_scale_orig_bd3lm = 1.0 / (1.0 - t_clamped + 1e-9) # Positive
    
    # The paper's loss L_simple = - (1/t) log p_theta(x0|xt)
    # This implies loss_scaling = -1/t. (t is move_chance here)
    loss_scaling_bd3lm = -1.0 / (move_chance_orig_bd3lm + 1e-9) # add epsilon for t=0
    # Ensure it doesn't go to +inf if t is negative (shouldn't be with t_clamped)
    loss_scaling_bd3lm = torch.clamp(loss_scaling_bd3lm, min=-1e5)


    return loss_scaling_bd3lm, move_chance_orig_bd3lm

  def get_alpha_bar_base(self, t: torch.Tensor) -> torch.Tensor:
    # Based on p(t) = t, alpha_bar_base(t) = 1-t
    t_clamped = torch.clamp(t, 0.0, 1.0)
    return 1.0 - t_clamped

  def get_t_from_move_chance(self, p: torch.Tensor) -> torch.Tensor:
    # If p(t) = t, then t = p
    # Clamp p to [0,1]
    return torch.clamp(p, 0.0, 1.0)

  # Original total_noise and rate_noise are kept for reference but might not align with p(t)=t
  def rate_noise(self, t): # Based on alpha_bar(t) = 1-(1-eps)*t
      return (1 - self.eps) / (1 - (1 - self.eps) * t + 1e-9)

  def total_noise(self, t): # Based on alpha_bar(t) = 1-(1-eps)*t
      return -torch.log1p(-(1 - self.eps) * t + 1e-9) # add eps for t=1/(1-eps) if eps !=0


# Warped schedule functions (remain the same)
def get_warped_alpha_b_t(
    base_alpha_bar_t: torch.Tensor,      
    base_log_alpha_bar_at_0: torch.Tensor, # Scalar tensor
    s_b: torch.Tensor,                   # (B,N_Blk) or (B,N_Blk,1) or (B,N_Blk,N_t) if s_b is also t-dependent
    target_log_alpha_at_0: torch.Tensor  # Scalar tensor
    ) -> torch.Tensor:
    
    L_base_t = torch.logit(base_alpha_bar_t) # Can be (B,N_Blk) or (B,N_Blk,N_t)
    
    # term_in_paren is (L_base_t - base_log_alpha_bar_at_0)
    # Target: s_b_eff * term_in_paren, where shapes are compatible
    
    s_b_eff = s_b
    # Case 1: s_b is (B,N_Blk,1), L_base_t is (B,N_Blk) -> s_b should be (B,N_Blk)
    if s_b.ndim == L_base_t.ndim + 1 and s_b.shape[-1] == 1:
        s_b_eff = s_b.squeeze(-1)
    # Case 2: s_b is (B,N_Blk), L_base_t is (B,N_Blk,N_t) -> s_b should be (B,N_Blk,1)
    elif s_b.ndim == L_base_t.ndim -1 and L_base_t.ndim > s_b.ndim : # Ensure L_base_t actually has more dims
        s_b_eff = s_b.unsqueeze(-1)
    # Case 3: s_b and L_base_t have same ndim (e.g. both (B,N_Blk) or s_b is (B,N_Blk,1) and L_base_t is (B,N_Blk,N_t))
    # or other cases handled by broadcasting. If s_b is (B,N_Blk,1) and L_base_t is (B,N_Blk,N_t),
    # s_b_eff remains (B,N_Blk,1) and broadcasts correctly.
    
    A_b_t = s_b_eff * (L_base_t - base_log_alpha_bar_at_0) + \
            target_log_alpha_at_0 # Scalars base_log_alpha_bar_at_0 and target_log_alpha_at_0 broadcast
            
    alpha_b_t = torch.sigmoid(A_b_t)
    return alpha_b_t

def get_warped_schedule_outputs(
    base_alpha_bar_t: torch.Tensor,
    base_log_alpha_bar_base_derivative_t: torch.Tensor, 
    base_log_alpha_bar_at_0: torch.Tensor,
    s_b: torch.Tensor, # Can be (B,N_Blk) or (B,N_Blk,1)
    target_log_alpha_at_0: torch.Tensor # Scalar
    ):
    
    alpha_b_t = get_warped_alpha_b_t(
        base_alpha_bar_t, base_log_alpha_bar_at_0, s_b, target_log_alpha_at_0
    )
    
    # s_b needs to match alpha_b_t and base_log_alpha_bar_base_derivative_t for element-wise mult
    if s_b.ndim == 3 and s_b.shape[-1] == 1 and alpha_b_t.ndim == base_log_alpha_bar_base_derivative_t.ndim and alpha_b_t.ndim == s_b.ndim -1 : # s_b is (B,N_Blk,1), others (B,N_Blk) or (B,N_Blk,N_t)
        s_b_expanded = s_b
    elif s_b.ndim == 2 and alpha_b_t.ndim > s_b.ndim : # s_b is (B,N_Blk), others (B,N_Blk,N_t)
        s_b_expanded = s_b.unsqueeze(-1)
    else: # s_b and others have same dimensions or s_b is (B,N_Blk,1) and others (B,N_Blk,N_t) which broadcasts
        s_b_expanded = s_b


    loss_scale_b_t = alpha_b_t * s_b_expanded * base_log_alpha_bar_base_derivative_t
    
    p_b_t = 1.0 - alpha_b_t
    return alpha_b_t, loss_scale_b_t, p_b_t

def compute_surrogate_steps_penalty(
    alpha_b_at_1: torch.Tensor, 
    min_alpha_1_target: float,
    lambda_min_alpha_1_penalty: float,
    alpha_1_clamp_min: float = 1e-5,
    alpha_1_clamp_max: float = 1.0 - 1e-6
    ) -> torch.Tensor:
    
    alpha_b_at_1_clamped = torch.clamp(alpha_b_at_1, min=alpha_1_clamp_min, max=alpha_1_clamp_max)
    T_b_surrogate = -torch.log(alpha_b_at_1_clamped)
    
    floor_penalty = lambda_min_alpha_1_penalty * F.relu(min_alpha_1_target - alpha_b_at_1)**2
    
    return T_b_surrogate + floor_penalty