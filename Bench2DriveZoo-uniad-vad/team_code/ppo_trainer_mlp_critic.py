import copy
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Transition:
    state: torch.Tensor
    reward: float
    done: bool
    local_command_xy: torch.Tensor
    batch: Optional[dict]


class PPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_head(state).squeeze(-1)


class PPOBuffer:
    def __init__(self, gamma: float = 0.99, max_size: int = 32):
        self.gamma = gamma
        self.max_size = max_size
        self.storage: List[Transition] = []

    def add(self, transition: Transition):
        self.storage.append(transition)

    def ready(self) -> bool:
        return len(self.storage) >= self.max_size

    def compute_returns(self, last_value: float = 0.0) -> torch.Tensor:
        returns = []
        running_return = last_value
        for transition in reversed(self.storage):
            running_return = transition.reward + self.gamma * running_return * (1.0 - float(transition.done))
            returns.insert(0, running_return)
        return torch.tensor(returns, dtype=torch.float32)

    def clear(self):
        self.storage.clear()


class PPOTrainer:
    """
    Lightweight PPO-style trainer that wires together the critic and actor (planning head).

    The critic consumes a flattened state vector composed of:
    1) can_bus ego pose/velocity values
    2) the discrete high-level command and the local route displacement (x, y)
    3) the flattened planning trajectory predicted by UniAD's planning head

    This state flow makes the critic aware of both perception and planning context so that
    rewards emitted during closed-loop simulation can be turned into dense value targets.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        gamma: float = 0.99,
        buffer_size: int = 32,
        critic_lr: float = 1e-4,
        actor_lr: float = 5e-5,
    ):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.buffer = PPOBuffer(gamma=gamma, max_size=buffer_size)
        self.critic: Optional[PPOCritic] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.state_dim: Optional[int] = None
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self._freeze_non_planning_parameters()

    def _freeze_non_planning_parameters(self):
        planning_params = []
        for name, param in self.model.named_parameters():
            if "planning" in name:
                param.requires_grad = True
                planning_params.append(param)
            else:
                param.requires_grad = False
        if planning_params:
            self.actor_optimizer = torch.optim.Adam(planning_params, lr=self.actor_lr)

    def maybe_initialize_critic(self, state_dim: int):
        if self.critic is None:
            self.state_dim = state_dim
            self.critic = PPOCritic(state_dim).to(self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def build_state(self, can_bus, command: int, local_command_xy, plan) -> torch.Tensor:
        plan_flat = plan.reshape(-1)
        state = torch.tensor(
            list(can_bus[:8])
            + [float(command)]
            + [float(local_command_xy[0]), float(local_command_xy[1])]
            + plan_flat.tolist(),
            dtype=torch.float32,
            device=self.device,
        )
        if self.critic is None:
            self.maybe_initialize_critic(state.numel())
        return state

    def compute_reward(self, metric_info: dict) -> float:
        progress_reward = -metric_info.get("distance_to_command", 0.0)
        speed_penalty = -abs(metric_info.get("speed", 0.0) - metric_info.get("target_speed", 0.0))
        heading_penalty = -abs(metric_info.get("heading_error", 0.0)) * 0.1
        smoothness_penalty = -(metric_info.get("control_effort", 0.0)) * 0.05
        return float(progress_reward + speed_penalty + heading_penalty + smoothness_penalty)

    def _clone_batch_to_cpu(self, batch: dict) -> dict:
        cloned = {}
        for key, val in batch.items():
            if key == "img_metas":
                cloned[key] = copy.deepcopy(val)
            else:
                cloned[key] = [tensor.detach().cpu() for tensor in val]
        return cloned

    def _batch_to_device(self, batch: dict) -> dict:
        device_batch = {}
        for key, val in batch.items():
            if key == "img_metas":
                device_batch[key] = val
            else:
                device_batch[key] = [tensor.to(self.device) for tensor in val]
        return device_batch

    def add_transition(self, state: torch.Tensor, reward: float, done: bool, local_command_xy, batch: dict):
        stored_batch = self._clone_batch_to_cpu(batch)
        transition = Transition(
            state=state.detach().cpu(),
            reward=reward,
            done=done,
            local_command_xy=torch.tensor(local_command_xy, dtype=torch.float32),
            batch=stored_batch,
        )
        self.buffer.add(transition)
        if self.buffer.ready():
            self.update()

    def _extract_plan(self, forward_output):
        if isinstance(forward_output, list):
            forward_output = forward_output[0]
        if "planning" in forward_output:
            return forward_output["planning"]["result_planning"]["sdc_traj"][0]
        # fall back to BEV plan when running the VAD variant
        if "pts_bbox" in forward_output and "ego_fut_preds" in forward_output["pts_bbox"]:
            return forward_output["pts_bbox"]["ego_fut_preds"][0]
        raise RuntimeError("Cannot extract planning trajectory from model output.")

    def _build_target_traj(self, local_command_xy: torch.Tensor, plan_shape) -> torch.Tensor:
        horizon = plan_shape[0]
        target_xy = local_command_xy.to(self.device)
        target_traj = target_xy.repeat(horizon, 1)
        return target_traj

    def update(self):
        if self.critic is None or not self.buffer.storage:
            return
        self.model.eval()
        states = torch.stack([t.state.to(self.device) for t in self.buffer.storage])
        returns = self.buffer.compute_returns().to(self.device)
        values = self.critic(states)
        critic_loss = F.mse_loss(values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.actor_optimizer is None:
            self.buffer.clear()
            return

        actor_losses = []
        advantages = (returns - values.detach()).clamp(-5.0, 5.0)
        self.model.train()
        for transition, advantage in zip(self.buffer.storage, advantages):
            device_batch = self._batch_to_device(transition.batch)
            forward_output = self.model(device_batch, return_loss=False, rescale=True)
            plan_pred = self._extract_plan(forward_output)
            target_traj = self._build_target_traj(transition.local_command_xy, plan_pred.shape)
            plan_loss = ((plan_pred - target_traj) ** 2).mean()
            actor_losses.append(plan_loss * advantage)
        if actor_losses:
            actor_loss = torch.stack(actor_losses).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        self.model.eval()
        self.buffer.clear()