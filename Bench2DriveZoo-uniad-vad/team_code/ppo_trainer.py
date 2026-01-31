import copy
from dataclasses import dataclass
import importlib
from pathlib import Path
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Transition:
    state_batch: dict
    planning_hints: Dict[str, torch.Tensor]
    reward: float
    done: bool
    local_command_xy: torch.Tensor
    batch: Optional[dict]


class UniADCritic(nn.Module):
    """
    Critic that mirrors the end-to-end UniAD stack and regresses a single
    scalar value from the planning-context inputs.

    The critic owns a deep copy of the UniAD model to ensure its perception
    and planning inputs are processed identically to the actor. A lightweight
    value head converts the produced planning trajectory into a scalar value.
    Additional planning hints (command, local route offset, ego kinematics)
    are concatenated to the flattened trajectory to provide richer context
    for value estimation in complex driving scenes.
    """

    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.device = device
        self.encoder = copy.deepcopy(base_model).to(device)
        self._freeze_encoder_non_planning()
        self.value_head = (
            nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
            ).to(device)
        )

    def _freeze_encoder_non_planning(self):
        print('冻结critic参数')
        for name, param in self.encoder.named_parameters():
            param.requires_grad = "planning" in name

    def _extract_plan(self, forward_output):
        if isinstance(forward_output, list):
            forward_output = forward_output[0]
        if "planning" in forward_output:
            return forward_output["planning"]["result_planning"]["sdc_traj"]
        if "pts_bbox" in forward_output and "ego_fut_preds" in forward_output["pts_bbox"]:
            return forward_output["pts_bbox"]["ego_fut_preds"]
        raise RuntimeError("Cannot extract planning trajectory from model output.")

    def forward(self, batch: dict, planning_hints: Dict[str, torch.Tensor]) -> torch.Tensor:
        forward_output = self.encoder(batch, return_loss=False, rescale=True)
        plan_pred = self._extract_plan(forward_output)
        plan_feat = plan_pred.view(plan_pred.shape[0], -1)
        if planning_hints:
            hint_tensors = []
            batch_size = plan_feat.shape[0]
            for hint in planning_hints.values():
                hint = hint.to(plan_feat.device)
                if hint.dim() == 0:
                    hint = hint.unsqueeze(0)
                if hint.dim() == 1:
                    hint = hint.unsqueeze(0)
                if hint.shape[0] == 1 and batch_size > 1:
                    hint = hint.expand(batch_size, -1)
                elif hint.shape[0] != batch_size:
                    hint = hint[:batch_size]
                hint_tensors.append(hint)
            aux_feats = torch.cat(hint_tensors, dim=-1)
            if aux_feats.dim() == 1:
                aux_feats = aux_feats.unsqueeze(0)
            if aux_feats.shape[0] != batch_size:
                aux_feats = aux_feats.expand(batch_size, -1)
            plan_feat = torch.cat([plan_feat, aux_feats], dim=-1)
        value = self.value_head(plan_feat)
        return value.squeeze(-1)
    
class SharedUniADCritic(nn.Module):
    def __init__(self, device: torch.device): 
        super().__init__()
        self.device = device

        
        # Critic 不再持有 encoder，只持有 value_head
        self.value_head = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
            ).to(device)

    def forward(self, planning_hints):
        feats_list = []
        
        # 定义需要提取的 key 的顺序 (必须固定顺序，否则网络学不到东西)
        # 根据你之前提供的信息，通常包含以下几个关键信息
        target_keys = ['can_bus', 'command', 'local_command_xy']

        for key in target_keys:
            if key not in planning_hints:
                # 容错处理：如果某个key不存在，跳过
                continue
            feat = planning_hints[key].to(self.device)
            
            # === 维度修正逻辑 ===
        
            # Case 1: 处理高维特征 (例如 traj: [B, T, 2])
            # 需要展平成 [B, T*2] 才能拼接到 MLP
            if feat.dim() > 2:
                feat = feat.flatten(start_dim=1)
            
            # Case 2: 处理一维特征 (例如 command: [B])
            # 需要变成 [B, 1]
            elif feat.dim() == 1:
                feat = feat.unsqueeze(1)
                
            # (已移除) Case 3: 以前依赖 batch_feat 做的 expand 逻辑
            # 现在假设 planning_hints 里的所有数据都是 Batch 对齐的，
            # 即所有 Tensor 的第一维 (Batch Size) 都是相同的。
            
            # 确保数据类型为 float (MLP 不接受 Long/Int)
            if feat.dtype != torch.float32:
                feat = feat.float()
            
            feats_list.append(feat)

        if len(feats_list) == 0:
            raise ValueError("planning_hints 中没有找到任何匹配 target_keys 的特征！")

        # 3. 特征拼接
        # 将 [B, Traj_Dim], [B, Bus_Dim], [B, 1], ... 拼接成 [B, Total_Dim]
        combined_feat = torch.cat(feats_list, dim=1)

        # 4. 通过 Value Head
        # nn.LazyLinear 会在这里第一次运行时，根据 combined_feat 的形状初始化权重
        value = self.value_head(combined_feat)
        
        # 5. 调整输出维度
        # 输出形状通常是 [Batch, 1]，PPO Loss 计算通常需要 [Batch]
        return value.squeeze(-1)


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
        device: torch.device = torch.device("cuda"),
        gamma: float = 0.99,
        buffer_size: int = 32,
        critic_lr: float = 1e-4,
        actor_lr: float = 5e-5,
        actor_warmup_updates: int = 100,
        initial_critic_updates: int =0,
        actor_updates:int =0,
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.gamma = gamma
        self.buffer = PPOBuffer(gamma=gamma, max_size=buffer_size)
        self.critic: Optional[SharedUniADCritic] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.critic_updates = initial_critic_updates
        if initial_critic_updates >= actor_warmup_updates:
            self.actor_warmup_updates = 0
            print(f"[PPO] 检测到已训练的 Critic (Updates: {initial_critic_updates}), Actor 将立即开始更新。")
        else:
            self.actor_warmup_updates = actor_warmup_updates
            print(f"[PPO] 新训练开始,Actor 将在 Critic 更新 {actor_warmup_updates} 次后激活。")
        self.actor_warmup_updates = max(0, actor_warmup_updates)
        self.actor_updates = actor_updates
        self.episode_reward = 0.0
        self.checkpoint_dir = Path(checkpoint_dir or "ppo_checkpoints_uniad_1231")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = self._init_tensorboard()
        self.episode_index = 0
        self._freeze_non_planning_parameters()

    def _init_tensorboard(self) -> Optional[SummaryWriter]:
        log_dir = self.checkpoint_dir / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))

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

    def _init_critic(self):
        if self.critic is None:
            self.critic = UniADCritic(self.model, device=self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def build_state(self, can_bus, command: int, local_command_xy, batch: dict) -> dict:
        self._init_critic()
        planning_hints = {
            "command": torch.tensor(float(command), dtype=torch.float32).view(1, 1),
            "local_command_xy": torch.tensor(local_command_xy, dtype=torch.float32).view(1, -1),
            "can_bus": torch.tensor(list(can_bus[:8]), dtype=torch.float32).view(1, -1),
        }
        return {
            "state_batch": self._clone_batch_to_cpu(batch),
            "planning_hints": planning_hints,
        }

    # def compute_reward(self, metric_info: dict) -> float:
    #     progress_reward = -metric_info.get("distance_to_command", 0.0)
    #     speed_penalty = -abs(metric_info.get("speed", 0.0) - metric_info.get("target_speed", 0.0))
    #     heading_penalty = -abs(metric_info.get("heading_error", 0.0)) * 0.1
    #     smoothness_penalty = -(metric_info.get("control_effort", 0.0)) * 0.05
    #     return float(progress_reward + speed_penalty + heading_penalty + smoothness_penalty)

    def compute_reward(self, metric_info: dict) -> float:
        # 将每个信号归一化到 [0, 1] 的范围内，其中 1.0 代表最佳结果。
        distance = float(metric_info.get("distance_to_command", 0.0))
        speed_error = float(abs(metric_info.get("speed", 0.0) - metric_info.get("target_speed", 0.0)))
        heading_error = float(abs(metric_info.get("heading_error", 0.0)))
        control_effort = float(metric_info.get("control_effort", 0.0))
        plan_score = float(metric_info.get("plan_score", 0.0))

        # 归一化，用于保持奖励值较小且为正值。
        distance_score = max(0.0, 1.0 - min(distance / 30.0, 1.0))
        speed_score = max(0.0, 1.0 - min(speed_error / 10.0, 1.0))
        heading_score = max(0.0, 1.0 - min(heading_error / 3.14159, 1.0))
        smooth_score = max(0.0, 1.0 - min(control_effort / 3.0, 1.0))
        plan_score_norm = max(0.0, 1.0 - min(plan_score / 15.0, 1.0))

        # 混合均匀
        # blended = 0.2 * distance_score + 0.2 * speed_score + 0.2 * heading_score + 0.2 * smooth_score + 0.2 * plan_score_norm
        blended = plan_score_norm
        reward = 0.1 + blended  
        return float(min(reward, 1.5))

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

    def add_transition(self, state: dict, reward: float, done: bool,last_done: bool, local_command_xy, batch: dict):
        transition = Transition(
            state_batch=state["state_batch"],
            planning_hints={k: v.detach().cpu() for k, v in state["planning_hints"].items()},
            reward=reward,
            done=done,
            local_command_xy=torch.tensor(local_command_xy, dtype=torch.float32),
            batch=self._clone_batch_to_cpu(batch),
        )
        if not last_done or not done:
            self.episode_reward += reward
        if done and not last_done:
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("episode/cumulative_reward", self.episode_reward, self.episode_index)
            self.episode_index += 1
            self.episode_reward = 0.0
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
        returns = self.buffer.compute_returns().to(self.device)
        returns = self._normalize_returns(returns)
        critic_values = []
        self.critic.train()
        for transition in self.buffer.storage:
            device_batch = self._batch_to_device(transition.state_batch)
            hints = {k: v.to(self.device) for k, v in transition.planning_hints.items()}
            critic_values.append(self.critic(device_batch, hints))
        values = torch.stack(critic_values)
        critic_loss = F.mse_loss(values.squeeze(-1), returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_updates += 1

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("loss/critic", critic_loss.item(), self.critic_updates)
            self.tb_writer.add_scalar("value/mean", values.detach().mean().item(), self.critic_updates)
            self.tb_writer.add_scalar("value/std", values.detach().std().item(), self.critic_updates)

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
            if self.critic_updates >= self.actor_warmup_updates:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_updates += 1
                if self.actor_updates % 100 == 0:
                    self._save_checkpoint(self.actor_updates)
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("loss/actor", actor_loss.item(), self.actor_updates)
        self.model.eval()
        self.buffer.clear()


    def _normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        if returns.numel() == 0:
            return returns
        min_val = returns.min()
        max_val = returns.max()
        if (max_val - min_val) < 1e-6:
            return torch.full_like(returns, 0.5)
        normalized = (returns - min_val) / (max_val - min_val + 1e-6)
        return torch.clamp(normalized + 0.05, 0.0, 1.05)

    def _save_checkpoint(self, step: int):
        checkpoint_path = self.checkpoint_dir / f"ppo_finetune_step_{step}.pt"
        latest_path = self.checkpoint_dir / "ppo_latest.pt"
        data_to_save={
                "actor_state_dict": self.model.state_dict(),
                "critic_state_dict": self.critic.state_dict() if self.critic is not None else None,
                "critic_updates": self.critic_updates,
                "actor_updates": self.actor_updates,
                "episode_index": self.episode_index,
            }
        torch.save(data_to_save,checkpoint_path)
        torch.save(data_to_save,latest_path)
            
        