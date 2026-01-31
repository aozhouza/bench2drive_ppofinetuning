import copy
from dataclasses import dataclass
from pathlib import Path
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

class PPOBuffer:
    def __init__(self, gamma: float = 0.99, max_size: int = 32):
        self.gamma = gamma
        self.max_size = max_size
        self.storage: List[Transition] = []

    def add(self, transition: Transition):
        self.storage.append(transition)

    def ready(self) -> bool:
        return len(self.storage) >= self.max_size

    def compute_returns(self) -> torch.Tensor:
        """
        计算蒙特卡洛回报 (Monte Carlo Returns)
        """
        returns = []
        running_return = 0.0 # 在 GRPO 中，通常直接使用 Episode Reward，但为了兼容截断，这里保留衰减逻辑
        # 注意：如果 storage 包含多个 episode，应该在 done 时重置 running_return
        # 这里简化处理，假设 storage 是连续的时间步
        for transition in reversed(self.storage):
            running_return = transition.reward + self.gamma * running_return * (1.0 - float(transition.done))
            returns.insert(0, running_return)
        return torch.tensor(returns, dtype=torch.float32)

    def clear(self):
        self.storage.clear()


class GRPOTrainer:
    """
    GRPO-style trainer for UniAD.
    
    Core Changes from PPO:
    1. Removes the Critic (Value Network).
    2. Uses Group Relative Advantage: A = (R - mean(R)) / std(R).
    3. Adds a Reference Model to compute KL penalty (drift from initial policy).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda"),
        gamma: float = 0.99,
        buffer_size: int = 32,
        actor_lr: float = 5e-5,
        actor_warmup_updates: int = 100,
        beta: float = 0.04,  # KL 惩罚系数 (GRPO 重要参数)
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.gamma = gamma
        
        # 1. 初始化 Reference Model (冻结副本)
        print("Initializing Reference Model for GRPO...")
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        self.buffer = PPOBuffer(gamma=gamma, max_size=buffer_size)
        
        # 移除 Critic 相关的 Optimizer
        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.actor_lr = actor_lr
        self.actor_warmup_updates = max(0, actor_warmup_updates)
        self.beta = beta # KL penalty coefficient
        
        self.updates = 0 # 统称为 updates，不再区分 actor/critic updates
        self.episode_reward = 0.0
        self.checkpoint_dir = Path(checkpoint_dir or "grpo_checkpoints")
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

    def build_state(self, can_bus, command: int, local_command_xy, batch: dict) -> dict:
        # GRPO 不需要 Critic，这里只构建 Actor 需要的输入
        planning_hints = {
            "command": torch.tensor(float(command), dtype=torch.float32).view(1, 1),
            "local_command_xy": torch.tensor(local_command_xy, dtype=torch.float32).view(1, -1),
            "can_bus": torch.tensor(list(can_bus[:8]), dtype=torch.float32).view(1, -1),
        }
        return {
            "state_batch": self._clone_batch_to_cpu(batch),
            "planning_hints": planning_hints,
        }

    def compute_reward(self, metric_info: dict) -> float:
        # 保持原有的 Reward 逻辑不变
        distance = float(metric_info.get("distance_to_command", 0.0))
        speed_error = float(abs(metric_info.get("speed", 0.0) - metric_info.get("target_speed", 0.0)))
        heading_error = float(abs(metric_info.get("heading_error", 0.0)))
        control_effort = float(metric_info.get("control_effort", 0.0))
        plan_score = float(metric_info.get("plan_score", 0.0))

        distance_score = max(0.0, 1.0 - min(distance / 30.0, 1.0))
        speed_score = max(0.0, 1.0 - min(speed_error / 10.0, 1.0))
        heading_score = max(0.0, 1.0 - min(heading_error / 3.14159, 1.0))
        smooth_score = max(0.0, 1.0 - min(control_effort / 3.0, 1.0))
        plan_score_norm = max(0.0, 1.0 - min(plan_score / 15.0, 1.0))

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

    def add_transition(self, state: dict, reward: float, done: bool, last_done: bool, local_command_xy, batch: dict):
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
        if "pts_bbox" in forward_output and "ego_fut_preds" in forward_output["pts_bbox"]:
            return forward_output["pts_bbox"]["ego_fut_preds"][0]
        raise RuntimeError("Cannot extract planning trajectory from model output.")

    def _build_target_traj(self, local_command_xy: torch.Tensor, plan_shape) -> torch.Tensor:
        horizon = plan_shape[0]
        target_xy = local_command_xy.to(self.device)
        target_traj = target_xy.repeat(horizon, 1)
        return target_traj

    def update(self):
        if not self.buffer.storage or self.actor_optimizer is None:
            return

        # 1. 计算 Returns (Cumulated Rewards)
        # 维度: [Batch_Size]
        returns = self.buffer.compute_returns().to(self.device)

        # 2. GRPO 核心逻辑: Group Relative Advantage
        # A = (r - mean(r)) / std(r)
        # 这里我们将整个 Buffer 视为一个 Group (或者多个 Group 的混合)。
        # 如果能保证 Buffer 内的数据是针对同一/相似场景的多次采样，效果最好。
        if returns.numel() > 1:
            mean_returns = returns.mean()
            std_returns = returns.std() + 1e-4
            advantages = (returns - mean_returns) / std_returns
        else:
            # Fallback if batch size is 1
            advantages = torch.zeros_like(returns)

        # 3. 开始训练循环
        actor_losses = []
        kl_losses = []
        
        self.model.train()
        
        # 遍历 Buffer 中的每一个样本
        for i, transition in enumerate(self.buffer.storage):
            advantage = advantages[i]
            device_batch = self._batch_to_device(transition.batch)
            
            # --- Policy Forward (Actor) ---
            forward_output = self.model(device_batch, return_loss=False, rescale=True)
            plan_pred = self._extract_plan(forward_output)
            
            # --- Reference Model Forward (用于 KL Penalty) ---
            # with torch.no_grad():
            #     ref_output = self.ref_model(device_batch, return_loss=False, rescale=True)
            #     plan_ref = self._extract_plan(ref_output)
            
            # 计算 Policy Loss (回归 Loss 加权优势)
            # 原理: 如果 Advantage > 0 (表现优于平均)，我们要最小化 Loss (即拉近 prediction 和 target_traj)
            # 注意：原始代码逻辑是 MSE * Adv。
            # PPO/GRPO 标准是: loss = - min(ratio * A, clip(ratio) * A)
            # 这里的实现是针对确定性轨迹的 Simplified Policy Gradient
            target_traj = self._build_target_traj(transition.local_command_xy, plan_pred.shape)
            plan_mse = ((plan_pred - target_traj) ** 2).mean()
            
            # 策略梯度近似:
            # 我们希望 maximize (Advantage * log_prob)。对于高斯策略，log_prob 正比于 -MSE。
            # 所以 maximize (Advantage * -MSE) => minimize (Advantage * MSE)
            # 为了数值稳定性，我们通常反转 Advantage 的符号或者只对正 Advantage 优化
            # 这里保持你原有的逻辑: loss = MSE * Advantage 
            # (如果 Adv是正的，MSE越小Loss越小 -> 鼓励；如果Adv是负的，MSE越大Loss越小 -> 抑制)
            policy_loss = plan_mse * advantage
            
            # --- KL Divergence Penalty ---
            # 对于确定性输出，KL 散度可以近似为预测轨迹与参考轨迹之间的 L2 距离
            # ref_loss = ((plan_pred - plan_ref) ** 2).mean()
            # 简化版：也可以直接惩罚 plan_pred 偏离 "初始行为"
            # 如果不开销太大，可以跑一次 ref_model。为了性能，这里演示逻辑，实际由于显存限制可能需要优化
            with torch.no_grad():
                 ref_output = self.ref_model(device_batch, return_loss=False, rescale=True)
                 plan_ref = self._extract_plan(ref_output)
            
            kl_loss = ((plan_pred - plan_ref) ** 2).mean()

            # 总 Loss
            total_sample_loss = policy_loss + self.beta * kl_loss
            
            actor_losses.append(total_sample_loss)
            kl_losses.append(kl_loss)

        if actor_losses:
            # 聚合 Batch Loss
            total_loss = torch.stack(actor_losses).mean()
            mean_kl = torch.stack(kl_losses).mean()
            
            if self.updates >= self.actor_warmup_updates:
                self.actor_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.updates += 1
                
                if self.updates % 100 == 0:
                    self._save_checkpoint(self.updates)
                
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("loss/grpo_total", total_loss.item(), self.updates)
                    self.tb_writer.add_scalar("loss/kl_penalty", mean_kl.item(), self.updates)
                    self.tb_writer.add_scalar("grpo/mean_return", returns.mean().item(), self.updates)
                    self.tb_writer.add_scalar("grpo/std_return", returns.std().item(), self.updates)

        self.model.eval()
        self.buffer.clear()

    def _save_checkpoint(self, step: int):
        checkpoint_path = self.checkpoint_dir / f"grpo_finetune_step_{step}.pt"
        torch.save(
            {
                "actor_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.actor_optimizer.state_dict(),
                "updates": self.updates,
            },
            checkpoint_path,
        )