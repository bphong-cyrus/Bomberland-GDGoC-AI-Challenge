"""
GDGoC-HCMUS AI Challenge 2026 — Bomberland DQN Agent
4-player (13×13 grid) | CPU inference <100ms | Double DQN + TorchScript
Flat submission: agent.py + model.pt (or model.pth) in same directory.
"""
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

# ── Game constants ─────────────────────────────────────────────────────────────
BOMB_TIMER_MAX  = 7
BOMB_RADIUS_MAX = 5
BOMB_CAP_MAX    = 5
NUM_ACTIONS     = 6   # 0=STOP 1=LEFT 2=RIGHT 3=UP 4=DOWN 5=PLACE_BOMB

# State encoding dimensions (change these only if you change encode_obs)
N_MAP_CH = 10   # terrain×5, my_pos, enemy_pos(merged), bomb_timer, my_bomb, foe_bomb
N_AUX    = 3    # bombs_left/max, radius/max, enemies_alive_ratio


# ── State encoder ──────────────────────────────────────────────────────────────
def encode_obs(obs: dict, agent_id: int):
    """
    Encode a 4-player (or 2-player) observation into tensors.

    Returns
    -------
    map_feat : np.ndarray, float32, shape (N_MAP_CH, H, W)
    aux_feat : np.ndarray, float32, shape (N_AUX,)
    """
    uid     = int(agent_id)
    grid    = np.asarray(obs["map"])              # (H, W) int
    players = np.asarray(obs["players"])          # (N, 5): x,y,alive,bombs_left,radius
    H, W    = grid.shape
    N       = players.shape[0]

    my_x       = players[uid, 0]
    my_y       = players[uid, 1]
    my_alive   = players[uid, 2]
    my_bleft   = players[uid, 3]
    my_radius  = players[uid, 4]

    # ── terrain one-hot (cell types 0..4: grass, wall, box, item_rad, item_cap)
    ch_terrain = [(grid == v).astype(np.float32) for v in range(5)]   # 5 channels

    # ── self position
    ch_me = np.zeros((H, W), dtype=np.float32)
    if int(my_alive):
        ch_me[int(my_x), int(my_y)] = 1.0

    # ── all enemy positions merged into one channel
    ch_foes = np.zeros((H, W), dtype=np.float32)
    n_alive_foes = 0
    for pid in range(N):
        if pid == uid:
            continue
        fx, fy, fa = players[pid, 0], players[pid, 1], players[pid, 2]
        if int(fa):
            ch_foes[int(fx), int(fy)] = 1.0
            n_alive_foes += 1

    # ── bomb channels: normalised timer, my bombs, enemy bombs
    bombs_raw = obs["bombs"]
    if len(bombs_raw) == 0:
        bombs = np.zeros((0, 4), dtype=np.float32)
    else:
        bombs = np.asarray(bombs_raw, dtype=np.float32)
        if bombs.ndim == 1:
            bombs = bombs.reshape(1, -1)

    ch_timer  = np.zeros((H, W), dtype=np.float32)
    ch_mybomb = np.zeros((H, W), dtype=np.float32)
    ch_fobomb = np.zeros((H, W), dtype=np.float32)
    for b in bombs:
        bx, by = int(b[0]), int(b[1])
        tmr, own = float(b[2]), int(b[3])
        t = tmr / BOMB_TIMER_MAX
        ch_timer[bx, by]  = max(ch_timer[bx, by], t)
        if own == uid:
            ch_mybomb[bx, by] = 1.0
        else:
            ch_fobomb[bx, by] = 1.0

    # ── stack → (10, H, W)
    map_feat = np.stack(
        ch_terrain + [ch_me, ch_foes, ch_timer, ch_mybomb, ch_fobomb],
        axis=0,
    ).astype(np.float32)

    # ── auxiliary scalars
    aux_feat = np.array([
        float(my_bleft)  / BOMB_CAP_MAX,
        float(my_radius) / BOMB_RADIUS_MAX,
        n_alive_foes     / max(N - 1, 1),
    ], dtype=np.float32)

    return map_feat, aux_feat


# ── Neural network ─────────────────────────────────────────────────────────────
class DQNModel(nn.Module):
    """
    2-layer Conv + small MLP head.
    batch=1 on 13×13 CPU: ~2 ms  →  well within 100ms budget.
    TorchScript-compatible (no dynamic shapes, no Python-only ops).
    """

    def __init__(self, map_shape: tuple, aux_dim: int, num_actions: int) -> None:
        super().__init__()
        c, h, w = map_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = int(self.conv(torch.zeros(1, c, h, w)).reshape(1, -1).size(1))
        self.aux_net = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU())
        self.head    = nn.Sequential(
            nn.Linear(flat + 32, 128), nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, map_x: torch.Tensor, aux_x: torch.Tensor) -> torch.Tensor:
        spatial = self.conv(map_x).flatten(1)
        aux     = self.aux_net(aux_x)
        return self.head(torch.cat([spatial, aux], dim=1))


# ── Replay buffer ──────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Pre-allocated numpy circular buffer — O(1) push, O(batch) sample."""

    def __init__(self, cap: int, map_shape: tuple, aux_dim: int):
        self.cap  = cap
        self.pos  = 0
        self.size = 0
        ms, ad    = tuple(map_shape), int(aux_dim)
        self.sm   = np.zeros((cap, *ms), dtype=np.float32)
        self.sa   = np.zeros((cap, ad),  dtype=np.float32)
        self.nsm  = np.zeros((cap, *ms), dtype=np.float32)
        self.nsa  = np.zeros((cap, ad),  dtype=np.float32)
        self.acts = np.zeros(cap,        dtype=np.int64)
        self.rews = np.zeros(cap,        dtype=np.float32)
        self.done = np.zeros(cap,        dtype=np.float32)

    def push(self, sm, sa, a, r, nsm, nsa, d):
        p         = self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        self.sm[p]=sm;   self.sa[p]=sa
        self.nsm[p]=nsm; self.nsa[p]=nsa
        self.acts[p]=a;  self.rews[p]=r;  self.done[p]=d

    def sample(self, n: int):
        idx = np.random.randint(0, self.size, n)
        return (self.sm[idx], self.sa[idx], self.nsm[idx], self.nsa[idx],
                self.acts[idx], self.rews[idx], self.done[idx])

    def __len__(self):
        return self.size


# ── Training agent ─────────────────────────────────────────────────────────────
class TrainingAgent:
    team_id = "DQNAgent"

    def __init__(self, agent_id: int, input_spec, num_actions: int,
                 lr: float = 3e-4, device: str = "cpu",
                 pretrained_model: str = None):
        self.agent_id    = agent_id
        self.num_actions = num_actions
        self.device      = device
        self.gamma       = 0.99
        self.global_step = 0
        self.epsilon     = 1.0
        self.lr          = lr

        if pretrained_model:
            self._load_checkpoint(pretrained_model)
        else:
            self.map_shape = tuple(input_spec[0])
            self.aux_dim   = int(input_spec[1])
            self.q_net     = DQNModel(self.map_shape, self.aux_dim,
                                      num_actions).to(device)
            self.optimizer = optim.Adam(
                self.q_net.parameters(), lr=lr, eps=1e-8, weight_decay=1e-5)

        self.target_net = DQNModel(
            self.map_shape, self.aux_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.loss_fn = nn.SmoothL1Loss()   # Huber loss — robust to reward outliers

    # ── inference ──────────────────────────────────────────────────────────────
    def act(self, map_s: np.ndarray, aux_s: np.ndarray,
            epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            mt = torch.from_numpy(map_s).unsqueeze(0).to(self.device)
            at = torch.from_numpy(aux_s).unsqueeze(0).to(self.device)
            return int(self.q_net(mt, at).argmax().item())

    # ── Double DQN update ──────────────────────────────────────────────────────
    def train_step(self, sm, sa, nsm, nsa, acts, rews, dones) -> float:
        dev = self.device

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(x).to(dev)

        sm_t  = t(sm);   sa_t  = t(sa)
        nsm_t = t(nsm);  nsa_t = t(nsa)
        a_t   = t(acts).unsqueeze(1)        # (B,1) int64
        r_t   = t(rews).unsqueeze(1)        # (B,1) float32
        d_t   = t(dones).unsqueeze(1)       # (B,1) float32

        # Current Q(s,a)
        q_curr = self.q_net(sm_t, sa_t).gather(1, a_t)

        with torch.no_grad():
            # Double DQN: select with q_net, evaluate with target_net
            best_a  = self.q_net(nsm_t, nsa_t).argmax(1, keepdim=True)
            q_next  = self.target_net(nsm_t, nsa_t).gather(1, best_a)
            q_target = r_t + self.gamma * q_next * (1.0 - d_t)

        loss = self.loss_fn(q_curr, q_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.global_step += 1
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── export ─────────────────────────────────────────────────────────────────
    def export_torchscript(self, path: str):
        """Save as TorchScript for fastest possible CPU inference."""
        self.q_net.eval()
        scripted = torch.jit.script(self.q_net)
        scripted.save(path)
        print(f"[TorchScript] saved → {path}")

    # ── checkpoint I/O ─────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str, epsilon: float, input_spec):
        from utils import save_model_fn
        save_model_fn(self.q_net, self.optimizer, self.global_step,
                      epsilon, self.lr, input_spec, self.num_actions, path)

    def _load_checkpoint(self, path: str):
        ck = torch.load(path, map_location=self.device)
        sp = ck.get("input_spec", ck.get("input_shape", ck.get("input_dim")))
        self.map_shape   = tuple(sp[0])
        self.aux_dim     = int(sp[1])
        self.num_actions = int(ck["num_actions"])
        self.q_net       = DQNModel(
            self.map_shape, self.aux_dim, self.num_actions).to(self.device)
        self.q_net.load_state_dict(ck["model_state_dict"])
        self.lr          = float(ck.get("lr", 3e-4))
        self.optimizer   = optim.Adam(
            self.q_net.parameters(), lr=self.lr, eps=1e-8, weight_decay=1e-5)
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.global_step = int(ck.get("global_step", 0))
        self.epsilon     = float(ck.get("epsilon", 0.1))


# ── Training loop ──────────────────────────────────────────────────────────────
def train_dqn(
    user_id: int      = 0,
    enemy_types       = ("simple",),   # up to 3 enemies
    num_episodes: int = 200,
    max_steps: int    = 500,
    seed: int         = 86,
    save_model: bool  = True,
    pretrained_model  = None,
):
    """
    Train DQN against 1-3 rule-based enemies.
    Enemies get IDs 1, 2, 3 (user always gets ID 0).
    """
    import sys, os
    from pathlib import Path as _P
    _root = _P(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from reward import compute_reward
    from utils  import (plot_loss, plot_rewards, plot_win_rates,
                        plot_moving_average, save_model_fn)
    from agent  import (SimpleRuleAgent, SmarterRuleAgent,
                        TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent)
    from engine import BomberEnv

    _CLS = {
        "simple":     SimpleRuleAgent,
        "smarter":    SmarterRuleAgent,
        "tactical":   TacticalRuleAgent,
        "genius":     GeniusRuleAgent,
        "box_farmer": BoxFarmerAgent,
    }

    env     = BomberEnv(max_steps=max_steps, seed=seed)
    enemies = [_CLS[et](i + 1) for i, et in enumerate(list(enemy_types)[:3])]

    # ── Hyperparams
    epsilon       = 1.0
    epsilon_min   = 0.05
    epsilon_decay = 0.9995
    batch_size    = 128
    lr            = 3e-4
    buffer_cap    = 100_000
    target_freq   = 500    # sync target every N gradient steps
    ckpt_freq     = 500    # save checkpoint every N episodes

    # ── Bootstrap
    dummy_obs          = env.reset(seed=seed)
    ms, xs             = encode_obs(dummy_obs, user_id)
    input_spec         = (ms.shape, xs.shape[0])
    device             = "cuda" if torch.cuda.is_available() else "cpu"

    agent   = TrainingAgent(user_id, input_spec, NUM_ACTIONS, lr=lr,
                            device=device, pretrained_model=pretrained_model)
    epsilon = agent.epsilon if pretrained_model else epsilon
    buf     = ReplayBuffer(buffer_cap, input_spec[0], input_spec[1])

    tag = "dqn_" + "_".join(enemy_types) + f"_{num_episodes}ep_{seed}seed"
    os.makedirs(f"ckpts/{tag}", exist_ok=True)

    loss_h, rew_h, win_h = [], [], []

    with tqdm(total=num_episodes, desc=f"Training vs {list(enemy_types)}") as pbar:
        for ep in range(num_episodes):
            obs       = env.reset(seed=seed + ep)
            prev_obs  = None
            ms, xs    = encode_obs(obs, user_id)
            ep_rew    = 0.0
            n_players = int(np.asarray(obs["players"]).shape[0])

            for _ in range(max_steps):
                # ── Actions for all players
                act      = agent.act(ms, xs, epsilon)
                all_acts = [None] * n_players
                all_acts[user_id] = act
                for e in enemies:
                    if e.agent_id < n_players:
                        all_acts[e.agent_id] = e.act(obs)

                nobs, terminated, truncated = env.step(all_acts)
                done = terminated or truncated

                r      = compute_reward(prev_obs, nobs, agent_id=user_id)
                ep_rew += r

                nms, nxs = encode_obs(nobs, user_id)
                buf.push(ms, xs, act, r, nms, nxs, float(done))

                # ── Learn
                if len(buf) >= batch_size:
                    loss = agent.train_step(*buf.sample(batch_size))
                    loss_h.append(loss)

                # ── Sync target (step-based)
                if agent.global_step % target_freq == 0:
                    agent.sync_target()

                prev_obs = obs;  obs = nobs
                ms = nms;        xs  = nxs
                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            alive = int(np.asarray(nobs["players"])[user_id, 2])
            win_h.append(alive)
            rew_h.append(ep_rew)

            pbar.update(1)
            wr100 = sum(win_h[-100:]) / min(100, len(win_h))
            pbar.set_postfix(rew=f"{ep_rew:.1f}", eps=f"{epsilon:.3f}",
                             wr=f"{wr100:.2f}", step=agent.global_step)

            # ── Mid-training checkpoint
            if save_model and (ep + 1) % ckpt_freq == 0:
                cp = f"ckpts/{tag}/ep{ep+1}_{agent.global_step}s.pth"
                agent.save_checkpoint(cp, epsilon, input_spec)
                print(f"\n[ckpt] {cp}")

    # ── Final save
    final_pth = f"ckpts/{tag}/model.pth"
    final_pt  = f"ckpts/{tag}/model.pt"
    if save_model:
        agent.save_checkpoint(final_pth, epsilon, input_spec)
        agent.export_torchscript(final_pt)
        print(f"\n✓ Checkpoint : {final_pth}"
              f"\n✓ TorchScript: {final_pt}  ← copy this to submission folder")

    # ── Plots
    plot_loss(loss_h,      f"ckpts/{tag}/loss.png")
    plot_rewards(rew_h,    f"ckpts/{tag}/rewards.png")
    plot_win_rates(win_h,  f"ckpts/{tag}/win_rate.png")
    plot_moving_average(rew_h, 50, f"ckpts/{tag}/ma_reward.png")

    return final_pth, final_pt


# ── CLI + Curriculum ───────────────────────────────────────────────────────────
def training():
    from utils import seed_everything

    p = argparse.ArgumentParser(description="Bomberland DQN trainer")
    p.add_argument("--enemy_types", nargs="+", default=["simple"],
                   choices=["simple","smarter","tactical","genius","box_farmer"],
                   help="1-3 enemy types (e.g. --enemy_types tactical genius)")
    p.add_argument("--num_episodes",  type=int, default=200)
    p.add_argument("--max_steps",     type=int, default=500)
    p.add_argument("--seed",          type=int, default=86)
    p.add_argument("--save_model",    action="store_true", default=True)
    p.add_argument("--no_save",       dest="save_model", action="store_false")
    p.add_argument("--load_model",    type=str, default=None,
                   help="Path to .pth checkpoint to resume from")
    p.add_argument("--curriculum",    action="store_true",
                   help="Run 4-stage curriculum: simple→smarter→tactical→tactical+genius")
    p.add_argument("--curriculum_episodes", nargs=4, type=int,
                   default=[2000, 2000, 3000, 3000],
                   help="Episodes per curriculum stage")
    args = p.parse_args()
    seed_everything(args.seed)

    if args.curriculum:
        # Stage 3 uses 1 tactical; Stage 4 uses tactical + genius (2 enemies)
        STAGES = [
            (["simple"],              args.curriculum_episodes[0]),
            (["smarter"],             args.curriculum_episodes[1]),
            (["tactical"],            args.curriculum_episodes[2]),
            (["tactical", "genius"],  args.curriculum_episodes[3]),
        ]
        ckpt = args.load_model
        for i, (etypes, n_ep) in enumerate(STAGES):
            print(f"\n{'='*55}"
                  f"\nCurriculum Stage {i+1}/{len(STAGES)}: {etypes} × {n_ep} ep"
                  f"\n{'='*55}")
            pth, _pt = train_dqn(
                enemy_types=etypes, num_episodes=n_ep,
                max_steps=args.max_steps, seed=args.seed + i,
                save_model=True, pretrained_model=ckpt,
            )
            ckpt = pth   # load checkpoint (not TorchScript) for next stage
            print(f"  → next stage loads: {ckpt}")
    else:
        train_dqn(
            enemy_types=args.enemy_types,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            save_model=args.save_model,
            pretrained_model=args.load_model,
        )


# ── Submission Agent ───────────────────────────────────────────────────────────
class Agent:
    """
    Mandatory submission class.

    File priority (all in Path(__file__).parent):
      1. model.pt   — TorchScript (fastest CPU inference, ~2 ms/step)
      2. model.pth  — regular checkpoint
      3. *.pth      — any .pth file (legacy fallback)

    The act() method must return an int in [0, 5] within 100 ms.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        base = Path(__file__).parent

        ts  = base / "model.pt"
        pth = base / "model.pth"
        leg = next(base.glob("*.pth"), None)

        if ts.exists():
            self._net = torch.jit.load(str(ts), map_location="cpu")
            self._net.eval()
        elif pth.exists():
            self._net = self._from_pth(str(pth))
        elif leg:
            self._net = self._from_pth(str(leg))
        else:
            raise FileNotFoundError(
                f"No model file found in {base}. "
                "Expected model.pt or model.pth after training."
            )

    @staticmethod
    def _from_pth(path: str) -> nn.Module:
        ck  = torch.load(path, map_location="cpu")
        sp  = ck.get("input_spec", ck.get("input_shape", ck.get("input_dim")))
        ms  = tuple(sp[0])
        ad  = int(sp[1])
        na  = int(ck["num_actions"])
        if ms[0] != N_MAP_CH or ad != N_AUX:
            raise ValueError(
                f"Checkpoint channels ({ms[0]}) or aux ({ad}) don't match "
                f"current encoder ({N_MAP_CH} channels, {N_AUX} aux). "
                "Please retrain with the updated agent.py."
            )
        net = DQNModel(ms, ad, na)
        net.load_state_dict(ck["model_state_dict"])
        net.eval()
        return net

    def act(self, obs: dict) -> int:
        try:
            ms, xs = encode_obs(obs, self.agent_id)
            with torch.no_grad():
                return int(
                    self._net(
                        torch.from_numpy(ms).unsqueeze(0),
                        torch.from_numpy(xs).unsqueeze(0),
                    ).argmax(1).item()
                )
        except Exception as e:
            print(f"[Agent.act ERROR] {e}")
            return 0   # STOP — safe fallback


if __name__ == "__main__":
    training()
