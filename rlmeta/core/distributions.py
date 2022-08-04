import abc
from typing import List
import numpy as np
import math

import torch

EPSILON = 1e-7  # Small value to avoid divide by zero

class GaussianDistInstance():
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self):
        sample = self.mean + torch.randn_like(self.mean) * self.std
        return sample

    def deterministic_sample(self):
        return self.mean

    def log_prob(self, value):
        var = self.std ** 2
        log_scale = torch.log(self.std + EPSILON)
        log_prob_per_act_dim = (
            -((value - self.mean) ** 2) / (2 * var + EPSILON)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
        return torch.sum(log_prob_per_act_dim, dim=1)

    def pdf(self, value):
        log_prob = self.log_prob(value)
        return torch.exp(log_prob)

    def entropy(self):
        return torch.mean(
            0.5 * torch.log(2 * math.pi * math.e * self.std ** 2 + EPSILON),
            dim=1,
            keepdim=True,
        )


class GaussianDistribution(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_outputs: int,
    ):
        super().__init__()
        self.mu = torch.nn.Linear(hidden_size, num_outputs)
            
        self.log_sigma = torch.nn.Parameter(
            torch.zeros(1, num_outputs, requires_grad=True)
        )

    def forward(self, inputs: torch.Tensor) -> List[GaussianDistInstance]:
        mu = self.mu(inputs)
        log_sigma = self.log_sigma.expand(*mu.size())

        return GaussianDistInstance(mu, torch.exp(log_sigma))
