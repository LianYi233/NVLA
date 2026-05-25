import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS


# ======================= Key code: Gaussian Predictive Sampler =======================

class GaussianPredictiveSampler(nn.Module):
    """
    用经验高斯作为候选聚合器：
      - mu = 样本均值
      - Sigma = 样本（对角/全）协方差 + ridge*I
    训练期默认采样，推理期默认不采样（用 mu）。
    """
    def __init__(
        self,
        d: int,
        K: int,
        diag_cov: bool = True,               # True=对角方差（更稳更快），False=全协方差
        ridge: float = 1e-4,                 # 协方差稳定项
        jitter: float = 1e-6,                # Cholesky 抖动（仅 full 协方差采样用）
        sample_at_inference: bool = False,   # 测试期是否采样（默认 False：更稳）
        sample_at_training: bool = True,     # 训练期是否采样（默认 True：带噪增强）
        force_fp32_compute: bool = True,     # 统计与分解用 FP32，更稳
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.diag_cov = diag_cov
        self.ridge = float(ridge)
        self.jitter = float(jitter)
        self.sample_at_inference = sample_at_inference
        self.sample_at_training = sample_at_training
        self.force_fp32_compute = force_fp32_compute

    def forward(self, candidates: torch.Tensor, phase: str = "Inference") -> torch.Tensor:
        """
        candidates: (B, T, K, d)
        return:     (B, T, d)
        """
        B, T, K, d = candidates.shape
        assert K == self.K and d == self.d, "Candidates shape mismatch."

        out_dtype = candidates.dtype
        device = candidates.device
        x = candidates
        if self.force_fp32_compute and x.dtype != torch.float32:
            x = x.float()

        # 样本均值 mu: (B,T,d)
        mu = x.mean(dim=2)

        # 是否采样（训练: True/False 由构造参数控制；推理: 构造里默认 False）
        do_sample = (phase == "Training" and self.sample_at_training) or \
                    (phase != "Training" and self.sample_at_inference)

        if not do_sample:
            y = mu
        else:
            centered = x - mu.unsqueeze(2)  # (B,T,K,d)

            if self.diag_cov:
                # 经验方差（用 K 或 K-1 都可；这里用均值统计）
                var = centered.pow(2).mean(dim=2) + self.ridge  # (B,T,d)
                eps = torch.randn(B, T, d, device=device, dtype=x.dtype)
                y = mu + eps * torch.sqrt(var.clamp_min(1e-12))
            else:
                # 经验协方差: \sum (x-mu)(x-mu)^T / max(K-1,1) + ridge*I
                C = torch.einsum("btkd,btkc->btdc", centered, centered)
                denom = max(K - 1, 1)
                cov = C / denom
                I = torch.eye(self.d, device=device, dtype=x.dtype)
                cov = cov + self.ridge * I  # (B,T,d,d)

                # Cholesky（数值兜底）
                jitter = self.jitter
                for _ in range(5):
                    L, info = torch.linalg.cholesky_ex(cov + jitter * I)
                    if torch.all(info == 0):
                        break
                    jitter *= 10.0
                else:
                    # 兜底对角分解，避免极端退化
                    diag = torch.clamp(torch.diagonal(cov, dim1=-2, dim2=-1), min=self.jitter)
                    L = torch.diag_embed(torch.sqrt(diag))

                z = torch.randn(B, T, d, device=device, dtype=x.dtype)
                y = mu + (L @ z.unsqueeze(-1)).squeeze(-1)

        if y.dtype != out_dtype:
            y = y.to(out_dtype)
        return y


# ======================= Utils =======================

def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations


def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    assert q.size(-1) % 2 == 0 and k.size(-1) % 2 == 0, "RoPE head_dim must be even"

    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]    # even indices, (..., D/2)
        x2 = x[..., 1::2]   # odd  indices, (..., D/2)
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


# ======================= Main Action Head =======================

class L1RegressionActionHead(nn.Module):
    """
    回到最初思路：每个时间步生成 K=20 个候选，再用经验高斯聚合为最终动作。
    - 训练：默认采样（可改为 False），
    - 推理：把 phase="Inference" 传入 predict_action（默认训练）。
    输出形状：(B, NUM_ACTIONS_CHUNK, action_dim)。
    """
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
        num_candidates: int = 20,
        # Gaussian sampler 超参
        gp_diag_cov: bool = True,
        gp_ridge: float = 1e-4,
        gp_jitter: float = 1e-6,
        gp_sample_at_inference: bool = False,
        gp_sample_at_training: bool = True,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_candidates = num_candidates

        # 让主干输出 K*d（逐时间步）
        output_dim = action_dim * num_candidates

        self.model = MLPResNet(
            num_blocks=24,
            input_dim=input_dim*ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_pro_version=use_pro_version
        )

        # 经验高斯聚合（逐时间步）
        self.gauss_sampler = GaussianPredictiveSampler(
            d=action_dim,
            K=num_candidates,
            diag_cov=gp_diag_cov,
            ridge=gp_ridge,
            jitter=gp_jitter,
            sample_at_inference=gp_sample_at_inference,
            sample_at_training=gp_sample_at_training,
            force_fp32_compute=True,
        )

    def predict_action(
            self,
            actions_hidden_states,
            proprio=None,
            proprio_projector=None,
            phase: str = "Inference"
            ):
        """
        返回 (B, NUM_ACTIONS_CHUNK, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        dtype_tokens = actions_hidden_states.dtype  # 继承上游 dtype（常为 bf16）

        # proprio → projector（沿用你们原逻辑）
        proprio = proprio.reshape(batch_size, -1).to(dtype_tokens)  # (B, PROPRIO_DIM)
        proprio_features = proprio_projector(proprio)               # (B, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)        # (B, 1, llm_dim)

        # 拆分 token（沿用你们原逻辑）
        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states_tail = actions_hidden_states[:, :, self.num_task_tokens:, :]

        # scaffold（沿用你们原逻辑）
        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=dtype_tokens
        ).detach()

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (B, T, action_dim * hidden_dim)

        # 训练时可学习微扰（保持与原逻辑一致）
        if phase == "Training":
            _, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(
                seq_len, dim,
                device=rearranged_actions_hidden_states.device,
                dtype=rearranged_actions_hidden_states.dtype
            )
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations)

        # 主干：输出 (B, T, K*d)
        action_candidates_flat = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states_tail,
            p=proprio_features,
            h_t=task_hidden_states
        )  # (B, T, K*d)

        B, T, KD = action_candidates_flat.shape
        K = self.num_candidates
        d = self.action_dim
        assert KD == K * d, f"Expected last dim {K*d}, got {KD}"

        # reshape → (B, T, K, d)
        candidates = action_candidates_flat.view(B, T, K, d)

        # 经验高斯聚合（训练默认采样；推理传 phase="Inference" 用均值）
        actions_bt = self.gauss_sampler(candidates, phase=phase)  # (B, T, d)

        return actions_bt


# ======================= Backbone (unchanged except for RoPE fix) =======================

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self,
            num_blocks,
            input_dim,
            hidden_dim,
            output_dim,
            use_pro_version=False
            ):

        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 输出 K*d

    def forward(self, x, h_a=None, h_t=None, p= None):
        # x: (B, T, input_dim)
        x = self.layer_norm1(x)  # (B, T, I)
        x = self.fc1(x)          # (B, T, H)
        x = self.relu(x)         # (B, T, H)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # (B, T, H)
        x = self.layer_norm2(x)  # (B, T, H)
        x = self.fc2(x)          # (B, T, K*d)
        return x


class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (B, T, C)
        h_t, h_a, p: (..., C)
        """
        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)
        h = torch.cat(conditions, dim=1)  # (B, cond_len, C)

        B, T, C = x.shape
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h
        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x)              # (B, T, C)
        k_tokens = self.k_proj(x)         # (B, T, C)
        v_tokens = self.v_proj(x)         # (B, T, C)
        k_task = self.k_proj(task_k)      # (B, K_t, C)
        v_task = self.v_proj(task_v)      # (B, K_t, C)
        k_adapter = self.k_proj(adapter_k)# (B, K, C)
        v_adapter = self.v_proj(adapter_v)# (B, K, C)

        # -> (B, H, L, D)
        def split_heads(t, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_1 = split_heads(q_1, T)
        k_tokens, v_tokens = split_heads(k_tokens, T), split_heads(v_tokens, T)
        k_task, v_task = split_heads(k_task, K_t), split_heads(v_task, K_t)
        k_adapter, v_adapter = split_heads(k_adapter, K), split_heads(v_adapter, K)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1))                  # (B, H, T, T)
        attn_scores_task   = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1               # (B, H, T, K_t)
        attn_scores_adapt  = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g      # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapt], dim=-1)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2)                        # (B, H, T+K_t+K, D)
        output = torch.matmul(attn_weights, v_combined)                                     # (B, H, T, D)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x)
        return x


class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # 单例 RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, h_a=None, h_t=None, p=None):
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        h_adapter = torch.cat((h_a, p), dim=1)
        h_task = h_t

        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        q_1 = self.q_proj(x)
        k_tokens = self.k_self(x);    v_tokens = self.v_self(x)
        k_adapter = self.k_adapter(h_adapter); v_adapter = self.v_adapter(h_adapter)
        k_task = self.k_task(h_task); v_task = self.v_task(h_task)

        def split_heads(t, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_1 = split_heads(q_1, T)
        k_tokens, v_tokens = split_heads(k_tokens, T), split_heads(v_tokens, T)
        k_adapter, v_adapter = split_heads(k_adapter, K_a), split_heads(v_adapter, K_a)
        k_task, v_task = split_heads(k_task, K_t), split_heads(v_task, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        if K_a > 0:
            cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
            _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        if K_t > 0:
            cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
            _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        v_list = [v_tokens, v_adapter, v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x)
        return x
