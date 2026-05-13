import torch
from torch import nn

class DDPM(nn.Module):
    def __init__(
        self,
        timesteps: int,
        eps_model: nn.Module,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.eps_model = eps_model
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        # t.shape == (batch_size, ), x_0.shape == (b, c, h, w)
        mean = torch.sqrt(self.alpha_bar[t])[:, None, None, None] * x_0
        var = (1 - self.alpha_bar[t])[:, None, None, None]
        x_t = mean + torch.sqrt(var) * eps
        return x_t
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        eps_theta = self.eps_model(x_t, t)
        alpha = self.alpha[t][:, None, None, None]
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        mean = 1 / (torch.sqrt(alpha)) * (x_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta)
        var = self.beta[t][:, None, None, None]
        return mean + torch.sqrt(var) * z