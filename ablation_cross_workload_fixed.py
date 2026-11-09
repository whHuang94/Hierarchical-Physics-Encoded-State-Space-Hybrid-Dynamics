# -*- coding: utf-8 -*-
"""
跨工况创新点消融（修正版）

修正内容
--------
1. C2（Separate LuGre + Bouc-Wen）分支中，hysteresis/friction 状态没有更新。
   原代码只计算 z_h_new / z_f_new，却没有赋值回 z_h / z_f，导致 C2 实际上
   退化为静态模型。本版在每步计算后更新状态。
2. 数据路径改为相对仓库根目录的 data/，不再依赖本地硬编码路径。
3. 增加数据文件存在性检查，便于审稿人直接运行。
4. 轻微整理绘图颜色列表，避免配置数量变化时颜色长度不一致。

运行
----
    python ablation_cross_workload_fixed.py

默认读取：
    data/dataset1.csv   源域（训练）
    data/dataset2.csv   目标域（测试）

也可以通过环境变量覆盖：
    DATA_DIR=/path/to/data SRC_FILE=dataset1.csv TGT_FILE=dataset2.csv \
        python ablation_cross_workload_fixed.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# ==================== 配置 ====================
DATA_DIR = Path(
    os.environ.get(
        "DATA_DIR",
        Path(__file__).resolve().parent.parent / "data",
    )
).expanduser()

FILE_SRC = DATA_DIR / os.environ.get("SRC_FILE", "dataset1.csv")
FILE_TGT = DATA_DIR / os.environ.get("TGT_FILE", "dataset2.csv")

if not FILE_SRC.exists():
    raise FileNotFoundError(f"Source dataset not found: {FILE_SRC}")
if not FILE_TGT.exists():
    raise FileNotFoundError(f"Target dataset not found: {FILE_TGT}")

N_TRAIN = 10000
SEG_LEN = 200
CHUNK_LEN = 20
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
DOF = 6

LAMBDA_C_QUAD = 1e-2
LAMBDA_BLACK_L2 = 5e-2
LAMBDA_BLACK_ORTH = 1e-1
LAMBDA_BW_BOUND = 1e-2
LAMBDA_BW_ENERGY = 1e-3

feat_cols = [f"q{i}" for i in range(1, 7)] + [f"dq{i}" for i in range(1, 7)]
targ_cols = [f"tau_res{i}" for i in range(1, 7)]


# ==================== 统一耗散支路 ====================
class UnifiedDissipation(nn.Module):
    def __init__(self, dof=6, hidden=32, hyst_mode="phys"):
        super().__init__()
        self.dof = dof
        self.hyst_mode = hyst_mode
        self.gate_net = nn.Sequential(
            nn.Linear(4 * dof, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 6 * dof),
        )
        self.z_h_init = nn.Parameter(torch.zeros(dof))
        self.z_f_init = nn.Parameter(torch.zeros(dof))
        self.k_h = nn.Parameter(torch.ones(dof) * 0.5)
        self.k_f = nn.Parameter(torch.ones(dof) * 0.5)

    def init_state(self, B):
        return (
            self.z_h_init.expand(B, self.dof).clone(),
            self.z_f_init.expand(B, self.dof).clone(),
        )

    def step(self, q_t, dq_t, state):
        z_h, z_f = state
        inp = torch.cat([q_t, dq_t, z_h, z_f], dim=-1)
        out = self.gate_net(inp)

        A = F.softplus(out[:, 0 * self.dof:1 * self.dof]) + 0.1
        beta = F.softplus(out[:, 1 * self.dof:2 * self.dof]) + 0.1
        gamma = out[:, 2 * self.dof:3 * self.dof]
        sigma0 = F.softplus(out[:, 3 * self.dof:4 * self.dof]) + 0.1
        sigma1 = F.softplus(out[:, 4 * self.dof:5 * self.dof]) + 0.1
        sigma2 = F.softplus(out[:, 5 * self.dof:6 * self.dof]) + 0.1

        abs_dq = torch.abs(dq_t)

        if self.hyst_mode == "phys":
            abs_z_h = torch.abs(z_h)
            zdot_h = (
                A * dq_t
                - beta * abs_dq * z_h * abs_z_h
                - gamma * dq_t * abs_z_h ** 2
            )
            dt_eff_h = 1.0 / (1.0 + 0.5 * abs_dq * abs_z_h)
            z_h_new = torch.clamp(z_h + dt_eff_h * zdot_h, -2.0, 2.0)
        elif self.hyst_mode == "gru":
            r = torch.sigmoid(out[:, 0 * self.dof:1 * self.dof])
            u = torch.sigmoid(out[:, 1 * self.dof:2 * self.dof])
            z_cand = torch.tanh(out[:, 2 * self.dof:3 * self.dof])
            z_h_new = (1 - u) * z_h + u * (r * z_cand + (1 - r) * z_h)
        elif self.hyst_mode == "mlp":
            z_h_new = torch.tanh(self.gate_net(inp)[:, : self.dof])
        else:
            raise ValueError(f"Unknown hyst_mode: {self.hyst_mode}")

        g = sigma0 / (1 + sigma1 * abs_dq)
        zdot_f = dq_t - sigma0 * abs_dq * z_f / (g + 1e-6)
        dt_eff_f = 0.1 / (1.0 + 0.5 * abs_dq * torch.abs(z_f))
        z_f_new = torch.clamp(z_f + dt_eff_f * zdot_f, -1.0, 1.0)

        tau_h = self.k_h.unsqueeze(0) * z_h_new
        tau_f = self.k_f.unsqueeze(0) * z_f_new
        return tau_h, tau_f, (z_h_new, z_f_new)


# ==================== 主模型 ====================
class AblatableModel(nn.Module):
    def __init__(
        self,
        input_dim=12,
        hidden=64,
        dof=6,
        black_hidden=32,
        hyst_mode="phys",
        use_structure_isolation=True,
        use_unified=True,
    ):
        super().__init__()
        self.dof = dof
        self.hyst_mode = hyst_mode
        self.use_structure_isolation = use_structure_isolation
        self.use_unified = use_unified

        shared_dim = 4 * dof

        in_dim = 2 * dof if use_structure_isolation else shared_dim
        self.gravity = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, dof),
        )

        if use_unified:
            self.dissipation = UnifiedDissipation(dof, 32, hyst_mode)
        else:
            if hyst_mode == "phys":
                self.hyst_gate = nn.Sequential(
                    nn.Linear(3 * dof, 32),
                    nn.Tanh(),
                    nn.Linear(32, 5 * dof),
                )
            elif hyst_mode == "gru":
                self.hyst_gate = nn.Sequential(
                    nn.Linear(3 * dof, 32),
                    nn.Tanh(),
                    nn.Linear(32, 3 * dof),
                )
            elif hyst_mode == "mlp":
                self.hyst_net = nn.Sequential(
                    nn.Linear(2 * dof, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, dof),
                )
            else:
                raise ValueError(f"Unknown hyst_mode: {hyst_mode}")

            self.z_h_init = nn.Parameter(torch.zeros(dof))
            self.k_h = nn.Parameter(torch.ones(dof) * 0.5)

            self.fric_gate = nn.Sequential(
                nn.Linear(3 * dof, 32),
                nn.Tanh(),
                nn.Linear(32, 3 * dof),
            )
            self.z_f_init = nn.Parameter(torch.zeros(dof))
            self.k_f = nn.Parameter(torch.ones(dof) * 0.5)

        n_cor = dof * (dof + 1) // 2
        in_dim_c = n_cor if use_structure_isolation else shared_dim
        self.coriolis = nn.Sequential(
            nn.Linear(in_dim_c, hidden),
            nn.SELU(),
            nn.Linear(hidden, dof),
        )

        self.black = nn.Sequential(
            nn.Linear(2 * dof, black_hidden),
            nn.SELU(),
            nn.Linear(black_hidden, black_hidden),
            nn.SELU(),
            nn.Linear(black_hidden, dof),
        )

        self.register_buffer("x_mu", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

    def set_norm(self, x_mu, x_std):
        self.x_mu.data = torch.tensor(x_mu, dtype=torch.float32)
        self.x_std.data = torch.tensor(x_std, dtype=torch.float32)

    def _shared_input(self, q_t, dq_t):
        return torch.cat([q_t, dq_t, torch.sin(q_t), torch.cos(q_t)], dim=-1)

    def forward_chunk(self, x_chunk, state_h, state_f=None):
        B, L, _ = x_chunk.shape
        x_orig = x_chunk * self.x_std + self.x_mu
        q, dq = x_orig[:, :, : self.dof], x_orig[:, :, self.dof:2 * self.dof]

        z_h = state_h
        z_f = (
            state_f
            if state_f is not None
            else torch.zeros(B, self.dof, device=device)
        )

        outs = []
        y_g_l, y_f_l, y_h_l, y_c_l, y_b_l = [], [], [], [], []
        z_h_l, z_f_l = [], []

        for t in range(L):
            q_t, dq_t = q[:, t], dq[:, t]
            shared = self._shared_input(q_t, dq_t)
            out = 0

            iso = torch.cat([torch.sin(q_t), torch.cos(q_t)], -1)
            inp_g = iso if self.use_structure_isolation else shared
            y_g = self.gravity(inp_g)
            out = out + y_g

            if self.use_unified:
                y_h, y_f, (z_h, z_f) = self.dissipation.step(q_t, dq_t, (z_h, z_f))
                out = out + y_h + y_f
            else:
                if self.hyst_mode == "phys":
                    inp_h = torch.cat([q_t, dq_t, z_h], -1)
                    out_h = self.hyst_gate(inp_h)
                    r = torch.sigmoid(out_h[:, 0 * self.dof:1 * self.dof])
                    u = torch.sigmoid(out_h[:, 1 * self.dof:2 * self.dof])
                    A = F.softplus(out_h[:, 2 * self.dof:3 * self.dof]) + 0.1
                    beta = F.softplus(out_h[:, 3 * self.dof:4 * self.dof]) + 0.1
                    gamma = out_h[:, 4 * self.dof:5 * self.dof]
                    abs_dq = torch.abs(dq_t)
                    abs_z = torch.abs(z_h)
                    zdot = (
                        A * dq_t
                        - beta * abs_dq * z_h * abs_z
                        - gamma * dq_t * abs_z ** 2
                    )
                    dt_eff = 1.0 / (1.0 + 0.5 * abs_dq * abs_z)
                    z_h_new = (1 - u) * z_h + u * (
                        r * (z_h + dt_eff * zdot) + (1 - r) * z_h
                    )
                    z_h_new = torch.clamp(z_h_new, -2.0, 2.0)
                    y_h = self.k_h.unsqueeze(0) * z_h_new
                elif self.hyst_mode == "gru":
                    inp_h = torch.cat([q_t, dq_t, z_h], -1)
                    out_h = self.hyst_gate(inp_h)
                    r = torch.sigmoid(out_h[:, 0 * self.dof:1 * self.dof])
                    u = torch.sigmoid(out_h[:, 1 * self.dof:2 * self.dof])
                    z_cand = torch.tanh(out_h[:, 2 * self.dof:3 * self.dof])
                    z_h_new = (1 - u) * z_h + u * (
                        r * z_cand + (1 - r) * z_h
                    )
                    y_h = self.k_h.unsqueeze(0) * z_h_new
                elif self.hyst_mode == "mlp":
                    inp_h = torch.cat([q_t, dq_t], -1)
                    z_h_new = self.hyst_net(inp_h)
                    y_h = self.k_h.unsqueeze(0) * z_h_new
                else:
                    raise ValueError(f"Unknown hyst_mode: {self.hyst_mode}")

                # 关键修正：分离支路的状态必须向前推进。
                z_h = z_h_new
                out = out + y_h

                inp_f = torch.cat([dq_t, torch.abs(dq_t), z_f], -1)
                out_f = self.fric_gate(inp_f)
                sigma0 = F.softplus(out_f[:, 0 * self.dof:1 * self.dof]) + 0.1
                sigma1 = F.softplus(out_f[:, 1 * self.dof:2 * self.dof]) + 0.1
                sigma2 = F.softplus(out_f[:, 2 * self.dof:3 * self.dof]) + 0.1
                abs_dq = torch.abs(dq_t)
                g = sigma0 / (1 + sigma1 * abs_dq)
                zdot_f = dq_t - sigma0 * abs_dq * z_f / (g + 1e-6)
                dt_eff_f = 0.1 / (1.0 + 0.5 * abs_dq * torch.abs(z_f))
                z_f_new = torch.clamp(z_f + dt_eff_f * zdot_f, -1.0, 1.0)

                # 关键修正：摩擦状态也必须向前推进。
                z_f = z_f_new
                y_f = self.k_f.unsqueeze(0) * z_f_new
                out = out + y_f

            cor = [
                dq_t[:, i] * dq_t[:, j]
                for i in range(self.dof)
                for j in range(i, self.dof)
            ]
            iso_c = torch.stack(cor, -1)
            inp_c = iso_c if self.use_structure_isolation else shared
            y_c = self.coriolis(inp_c)
            out = out + y_c

            y_b = self.black(torch.cat([q_t, dq_t], dim=-1))
            out = out + y_b

            outs.append(out.unsqueeze(1))
            y_g_l.append(y_g.unsqueeze(1))
            y_f_l.append(y_f.unsqueeze(1))
            y_h_l.append(y_h.unsqueeze(1))
            y_c_l.append(y_c.unsqueeze(1))
            y_b_l.append(y_b.unsqueeze(1))
            z_h_l.append(z_h.unsqueeze(1))
            z_f_l.append(z_f.unsqueeze(1))

        return (
            torch.cat(outs, 1),
            torch.cat(y_g_l, 1),
            torch.cat(y_f_l, 1),
            torch.cat(y_h_l, 1),
            torch.cat(y_c_l, 1),
            torch.cat(y_b_l, 1),
            torch.cat(z_h_l, 1),
            torch.cat(z_f_l, 1),
            z_h,
            z_f,
        )

    def losses(
        self,
        x_chunk,
        y_chunk,
        state_h,
        state_f=None,
        use_orth=True,
        use_bw=True,
        use_c_quad=True,
    ):
        (
            out,
            y_g,
            y_f,
            y_h,
            y_c,
            y_b,
            z_h_seq,
            z_f_seq,
            z_h_f,
            z_f_f,
        ) = self.forward_chunk(x_chunk, state_h, state_f)
        loss_main = F.mse_loss(out, y_chunk)

        x_orig = x_chunk * self.x_std + self.x_mu
        q_l = x_orig[:, -1, : self.dof]
        dq_l = x_orig[:, -1, self.dof:2 * self.dof]
        shared_l = self._shared_input(q_l, dq_l)

        loss_c = torch.tensor(0.0, device=x_chunk.device)
        if use_c_quad:
            alpha = 0.5
            co = [
                dq_l[:, i] * dq_l[:, j]
                for i in range(self.dof)
                for j in range(i, self.dof)
            ]
            cs = [
                alpha ** 2 * dq_l[:, i] * dq_l[:, j]
                for i in range(self.dof)
                for j in range(i, self.dof)
            ]
            co_v = torch.stack(co, -1)
            cs_v = torch.stack(cs, -1)
            if not self.use_structure_isolation:
                co_v = shared_l
                cs_v = shared_l
            loss_c = (
                self.coriolis(cs_v) - alpha ** 2 * self.coriolis(co_v)
            ).pow(2).mean()

        loss_bl2 = y_b.pow(2).mean()

        loss_orth = torch.tensor(0.0, device=x_chunk.device)
        if use_orth:
            def cen(x):
                return x - x.mean(0, keepdim=True)

            yb_c = cen(y_b.reshape(-1, self.dof))
            for y_phys in [y_g, y_f, y_h, y_c]:
                yp_c = cen(y_phys.reshape(-1, self.dof))
                cos = (yb_c * yp_c).sum(0) / (
                    yb_c.norm(dim=0) * yp_c.norm(dim=0) + 1e-8
                )
                loss_orth = loss_orth + (cos ** 2).mean()
            loss_orth = loss_orth / 4.0

        loss_bwb = torch.tensor(0.0, device=x_chunk.device)
        loss_bwe = torch.tensor(0.0, device=x_chunk.device)
        if use_bw:
            loss_bwb = F.relu(torch.abs(z_h_seq) - 1.8).pow(2).mean()
            z_dot = z_h_seq[:, 1:, :] - z_h_seq[:, :-1, :]
            z_mid = 0.5 * (z_h_seq[:, 1:, :] + z_h_seq[:, :-1, :])
            loss_bwe = F.relu(z_mid * z_dot).pow(2).mean()

        total = (
            loss_main
            + LAMBDA_C_QUAD * loss_c
            + LAMBDA_BLACK_L2 * loss_bl2
            + LAMBDA_BLACK_ORTH * loss_orth
            + LAMBDA_BW_BOUND * loss_bwb
            + LAMBDA_BW_ENERGY * loss_bwe
        )
        return total, loss_main.item(), z_h_f, z_f_f


# ==================== 数据 ====================
def load_csv(path):
    df = pd.read_csv(path, comment="%")
    df.columns = [str(c).strip().strip("'").strip('"') for c in df.columns]
    missing = [c for c in feat_cols + targ_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[feat_cols + targ_cols]


def make_segments(X, Y, seg_len, stride=None):
    if stride is None:
        stride = seg_len
    ss, sy = [], []
    for s in range(0, len(X) - seg_len + 1, stride):
        ss.append(X[s:s + seg_len])
        sy.append(Y[s:s + seg_len])
    return np.array(ss, np.float32), np.array(sy, np.float32)


def moving_average(x, window=15):
    k = np.ones(window) / window
    out = np.empty_like(x)
    for i in range(x.shape[1]):
        out[:, i] = np.convolve(x[:, i], k, mode="same")
    return out


# 源域
df_src = load_csv(FILE_SRC)
X_src = df_src[feat_cols].values.astype(np.float32)
Y_src = df_src[targ_cols].values.astype(np.float32)
X_tr_src = X_src[:N_TRAIN]
Y_tr_src = Y_src[:N_TRAIN]

# 归一化参数：只从源域训练集统计
x_mu = X_tr_src.mean(0)
x_std = X_tr_src.std(0) + 1e-8
y_mu = Y_tr_src.mean(0)
y_std = Y_tr_src.std(0) + 1e-8

X_tr_src_n = (X_tr_src - x_mu) / x_std
Y_tr_src_n = (Y_tr_src - y_mu) / y_std

# 目标域（用源域归一化参数）
df_tgt = load_csv(FILE_TGT)
X_tgt = df_tgt[feat_cols].values.astype(np.float32)
Y_tgt = df_tgt[targ_cols].values.astype(np.float32)
X_tgt_n = (X_tgt - x_mu) / x_std
Y_tgt_n = (Y_tgt - y_mu) / y_std

# 源域测试集（源域后部分）
X_te_src = X_src[N_TRAIN:]
Y_te_src = Y_src[N_TRAIN:]
X_te_src_n = (X_te_src - x_mu) / x_std
Y_te_src_n = (Y_te_src - y_mu) / y_std

# 切段
Xtr_src, Ytr_src = make_segments(
    X_tr_src_n, Y_tr_src_n, SEG_LEN, stride=SEG_LEN // 2
)
Xte_src, Yte_src = make_segments(
    X_te_src_n, Y_te_src_n, SEG_LEN, stride=SEG_LEN // 2
)
Xte_tgt, Yte_tgt = make_segments(
    X_tgt_n, Y_tgt_n, SEG_LEN, stride=SEG_LEN // 2
)

Xtr_src_t = torch.tensor(Xtr_src).to(device)
Ytr_src_t = torch.tensor(Ytr_src).to(device)
Xte_src_t = torch.tensor(Xte_src).to(device)
Yte_src_t = torch.tensor(Yte_src).to(device)
Xte_tgt_t = torch.tensor(Xte_tgt).to(device)
Yte_tgt_t = torch.tensor(Yte_tgt).to(device)

print("=" * 78)
print(f"Source: {FILE_SRC}  |  Train: {len(X_tr_src)}  Test: {len(X_te_src)}")
print(f"Target: {FILE_TGT}  |  All: {len(X_tgt)}")
print(f"Drift: {np.linalg.norm(Y_tgt.mean(0) - Y_tr_src.mean(0)):.4f}")
print("=" * 78)


# ==================== 训练与跨域评估 ====================
def train_eval_cross(model_kwargs, train_kwargs, tag, epochs=EPOCHS):
    """源域训练 → 源域测试 + 目标域测试"""
    torch.manual_seed(0)
    np.random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)
    loader = DataLoader(
        TensorDataset(Xtr_src_t, Ytr_src_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )

    model = AblatableModel(
        input_dim=12,
        hidden=64,
        dof=DOF,
        black_hidden=32,
        **model_kwargs,
    )
    model.set_norm(x_mu, x_std)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    use_stateful = train_kwargs.get("use_stateful", True)
    use_orth = train_kwargs.get("use_orth", True)
    use_bw = train_kwargs.get("use_bw", True)
    use_c_quad = train_kwargs.get("use_c_quad", True)
    use_unified = model_kwargs.get("use_unified", True)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for xb_seg, yb_seg in loader:
            B, L, _ = xb_seg.shape
            if use_unified:
                z_h, z_f = model.dissipation.init_state(B)
            else:
                z_h = model.z_h_init.expand(B, DOF).clone()
                z_f = model.z_f_init.expand(B, DOF).clone()

            for c in range((L + CHUNK_LEN - 1) // CHUNK_LEN):
                s = c * CHUNK_LEN
                e = min(s + CHUNK_LEN, L)
                opt.zero_grad()
                loss, _, z_h_new, z_f_new = model.losses(
                    xb_seg[:, s:e, :],
                    yb_seg[:, s:e, :],
                    z_h,
                    z_f,
                    use_orth=use_orth,
                    use_bw=use_bw,
                    use_c_quad=use_c_quad,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if use_stateful:
                    z_h = z_h_new.detach()
                    z_f = z_f_new.detach()
                else:
                    if use_unified:
                        z_h, z_f = model.dissipation.init_state(B)
                    else:
                        z_h = model.z_h_init.expand(B, DOF).clone()
                        z_f = model.z_f_init.expand(B, DOF).clone()
        sch.step()
    train_time = time.time() - t0

    def eval_on(X_segs_t, Y_segs_t):
        model.eval()
        loader_e = DataLoader(
            TensorDataset(X_segs_t, Y_segs_t),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader_e:
                xb = xb.to(device)
                B = xb.shape[0]
                if use_unified:
                    z_h, z_f = model.dissipation.init_state(B)
                else:
                    z_h = model.z_h_init.expand(B, DOF).clone()
                    z_f = model.z_f_init.expand(B, DOF).clone()
                out, *_ = model.forward_chunk(xb, z_h, z_f)
                preds.append(out.cpu().numpy())
                trues.append(yb.cpu().numpy())
        p = np.concatenate(preds, 0).reshape(-1, DOF)
        t = np.concatenate(trues, 0).reshape(-1, DOF)
        p_phys = p * y_std + y_mu
        t_phys = t * y_std + y_mu
        mse_raw = np.mean((p_phys - t_phys) ** 2)
        mse_sm = np.mean(
            (moving_average(p_phys, 15) - moving_average(t_phys, 15)) ** 2
        )
        return mse_raw, mse_sm

    mse_src_raw, mse_src_sm = eval_on(Xte_src_t, Yte_src_t)
    mse_tgt_raw, mse_tgt_sm = eval_on(Xte_tgt_t, Yte_tgt_t)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  [{tag:<38}] {train_time:>5.1f}s  "
        f"src={mse_src_raw:.5f}/{mse_src_sm:.5f}  "
        f"tgt={mse_tgt_raw:.5f}/{mse_tgt_sm:.5f}"
    )
    return (
        mse_src_raw,
        mse_src_sm,
        mse_tgt_raw,
        mse_tgt_sm,
        n_params,
    )


# ==================== 消融配置 ====================
BASE_MODEL = dict(
    hyst_mode="phys",
    use_structure_isolation=True,
    use_unified=True,
)
BASE_TRAIN = dict(
    use_stateful=True,
    use_orth=True,
    use_bw=True,
    use_c_quad=True,
)

ABLATIONS = [
    ("Baseline (Unified+Phys+Iso+Stateful)", {}, {}),
    ("C1: Hyst as GRU", {"hyst_mode": "gru"}, {}),
    ("C1: Hyst as MLP", {"hyst_mode": "mlp"}, {}),
    ("C2: Separate LuGre + Bouc-Wen", {"use_unified": False}, {}),
    ("C3: Shared input", {"use_structure_isolation": False}, {}),
    ("C4: Window-independent", {}, {"use_stateful": False}),
]


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n" + "=" * 78)
    print("  Cross-Workload Ablation on Core Innovations")
    print("=" * 78)
    print(
        f"  {'Configuration':<38}{'time':>7}"
        f"{'SRC raw':>10}{'SRC sm':>10}"
        f"{'TGT raw':>10}{'TGT sm':>10}"
    )
    print("  " + "-" * 86)

    results = {}
    for tag, mkw, tkw in ABLATIONS:
        model_kwargs = {**BASE_MODEL, **mkw}
        train_kwargs = {**BASE_TRAIN, **tkw}
        res = train_eval_cross(model_kwargs, train_kwargs, tag)
        results[tag] = res

    # 汇总
    print("\n" + "=" * 78)
    print("  Summary: Cross-Workload Degradation")
    print("=" * 78)
    base = results["Baseline (Unified+Phys+Iso+Stateful)"]
    base_src_raw, base_src_sm, base_tgt_raw, base_tgt_sm, _ = base

    print(
        f"  {'Configuration':<38}"
        f"{'SRC raw':>10}{'Δsrc':>8}"
        f"{'TGT raw':>10}{'Δtgt':>8}"
    )
    print("  " + "-" * 74)
    for tag, (sr, ss, tr, ts, n) in results.items():
        d_src = (sr / base_src_raw - 1) * 100
        d_tgt = (tr / base_tgt_raw - 1) * 100
        if "Baseline" in tag:
            print(f"  {tag:<38}{sr:>10.5f}{'—':>8}{tr:>10.5f}{'—':>8}")
        else:
            print(
                f"  {tag:<38}{sr:>10.5f}{d_src:>+7.1f}%"
                f"{tr:>10.5f}{d_tgt:>+7.1f}%"
            )
    print("=" * 78)

    # 跨域退化比（target/src）
    print("\n  Cross-domain degradation ratio (target/src):")
    for tag, (sr, ss, tr, ts, n) in results.items():
        ratio = tr / sr
        print(f"    {tag:<38} ratio = {ratio:.2f}×")

    # ==================== 可视化 ====================
    tags = list(results.keys())
    src_raw = [results[t][0] for t in tags]
    tgt_raw = [results[t][2] for t in tags]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # 图 1：源域 vs 目标域
    ax = axes[0]
    x = np.arange(len(tags))
    w = 0.38
    b1 = ax.bar(
        x - w / 2,
        src_raw,
        w,
        label="Source (same domain)",
        color="steelblue",
    )
    b2 = ax.bar(
        x + w / 2,
        tgt_raw,
        w,
        label="Target (cross domain)",
        color="coral",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MSE (raw)")
    ax.set_title("Source vs Target Domain Performance", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    # 图 2：退化比
    ax = axes[1]
    ratios = [results[t][2] / results[t][0] for t in tags]
    colors = ["gray"] + ["crimson", "crimson", "purple", "steelblue", "steelblue"]
    colors = colors[: len(tags)]
    bars = ax.bar(range(len(tags)), ratios, color=colors, alpha=0.85)
    ax.axhline(1.0, color="k", ls="--", alpha=0.5)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Degradation ratio (target/src)")
    ax.set_title("Cross-Workload Degradation", fontsize=11)
    for b, r in zip(bars, ratios):
        ax.text(
            b.get_x() + b.get_width() / 2,
            r,
            f"{r:.2f}×",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Cross-Workload Ablation: Innovation Points", fontsize=13)
    plt.tight_layout()
    plt.savefig("cross_workload_innovation_ablation.png", dpi=150)
    plt.close()
    print("\nSaved: cross_workload_innovation_ablation.png")
