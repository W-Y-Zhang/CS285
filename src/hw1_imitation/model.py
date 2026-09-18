"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    ### 从专家演示数据里取出一对(si,ai)
    ### 把si 喂进策略网络pi 得到网络自己预测的动作
    ### 预测的动作和真实的动作算mse
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        current_dim = state_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim,h))
            layers.append(nn.ReLU())
            current_dim = h

        layers.append(nn.Linear(current_dim,action_dim*chunk_size))

        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        action_pre = self.net(state)
        action_pre = action_pre.reshape(-1, self.chunk_size,self.action_dim)
        loss = F.mse_loss(action_pre,action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        action_pre = self.net(state)
        action_pre = action_pre.reshape(-1, self.chunk_size,self.action_dim)
        return action_pre
    
class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        current_dim = state_dim + action_dim*chunk_size + 1
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim,h))
            layers.append(nn.ReLU())
            current_dim = h

        layers.append(nn.Linear(current_dim,action_dim*chunk_size))

        self.net = nn.Sequential(*layers)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = state.shape[0]

        x0 = torch.randn_like(action_chunk)
        x1 = action_chunk
        t = torch.rand(batch_size, 1, 1, device=state.device, dtype=state.dtype)
        xt = (1-t)*x0 + t*x1
        xt = xt.reshape(-1, self.chunk_size*self.action_dim)
        t = t.squeeze(-1)
        u = x1-x0
        velocity_field = torch.cat([state,xt,t],dim=1)
        v_pre = self.net(velocity_field)
        loss = F.mse_loss(v_pre, u.flatten(start_dim=1))

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        batch_size = state.shape[0]
        step_length = 1/num_steps
        xk = torch.randn(
            batch_size, self.chunk_size, self.action_dim,
            device=state.device, dtype=state.dtype,
        )
        for k in range(num_steps):
            tk = k*step_length
            tk_tensor = torch.full(
                (batch_size, 1), tk, device=state.device, dtype=state.dtype
            )
            xk_flat = xk.reshape(-1, self.chunk_size*self.action_dim)
            velocity_field = torch.cat([state,xk_flat,tk_tensor],dim = 1)
            v_pre = self.net(velocity_field)
            v_pre = v_pre.reshape(-1,self.chunk_size,self.action_dim)
            xk_1 = xk + step_length*v_pre
            xk = xk_1
        return xk

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
