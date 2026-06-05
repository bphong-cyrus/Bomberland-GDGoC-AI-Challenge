"""
Self-play trainer for the Bomberland DQN agent.

Upgrades over the baseline trainer (all toggleable):
  * ACTIVE reward shaping  — potential-based pull toward the nearest box/item
    (and enemies when the map is farmed out) so the agent stops loitering and
    goes farming -> snowballs items -> pressures kills. Plus a small per-episode
    novelty bonus for map coverage.
  * SELF-PLAY on all 4 seats — every seat is independently the learner (current
    net) or an opponent (frozen self snapshot / strong rule baseline), so the
    network learns every corner and stays robust across match-ups.
  * Prioritized Experience Replay (PER) — sample high-TD-error transitions more
    (proportional, sum-tree), with importance-sampling correction.
  * n-step Double DQN + Dueling network (architecture lives in model.py).

Run from the repository root:

    python -m agent.dqn_agent.train --episodes 12000 --opponents mix

Reward is read straight from env.players[i].stats, so it matches the match
ranking (survival, then kills > boxes > items > bombs).
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── make the repo importable however this file is launched ───────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.dqn_agent.model import (              # noqa: E402
    DQNModel, encode_obs, safe_action, legal_safe_actions,
    compute_danger, NUM_ACTIONS,
)
from agent.dqn_agent.utils import (              # noqa: E402
    seed_everything, plot_loss, plot_rewards, plot_win_rates, plot_moving_average,
)
from engine.game import BomberEnv                # noqa: E402
from agent import (                              # noqa: E402  (top-level baselines)
    RandomAgent, SimpleRuleAgent, SmarterRuleAgent,
    TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent,
)

RULE_CLASSES = {
    "random": RandomAgent, "simple": SimpleRuleAgent, "smarter": SmarterRuleAgent,
    "tactical": TacticalRuleAgent, "genius": GeniusRuleAgent, "box_farmer": BoxFarmerAgent,
}
STRONG_RULES = ["tactical", "genius", "smarter", "box_farmer"]

GRASS, WALL, BOX, ITEM_R, ITEM_C = 0, 1, 2, 3, 4


# ── reward shaping (aligned with the match ranking) ──────────────────────────
REWARD = {
    "death":        -3.0,   # being eliminated — by far the worst outcome
    "sole_winner":   3.0,   # last agent standing
    "survive_500":   0.0,   # reached the step cap alive — REMOVED. Surviving is
                            # already enforced by death=-3 + the safety mask, and
                            # any bonus here just rewards turtling, which WINS vs
                            # weak rule baselines but LOSES vs real opponents.
    "kill":          1.5,   # an opponent you eliminated (your stats['kills']) — up
    "opp_died":      0.4,   # any opponent removed this step (rank improves) — up
    "box":           0.5,   # box you destroyed (opens the map + spawns items) — up
    "item":          0.7,   # item collected = MORE bombs / bigger blast = the only
                            # way to out-power real opponents. Farm aggressively.
    "bomb":          0.0,   # no flat reward for placing a bomb (avoids spam)
    "bomb_target":   0.15,  # placed a bomb whose blast threatens a box or enemy —
                            # dense signal that teaches PURPOSEFUL aggression — up
    "bomb_waste":   -0.02,  # placed a bomb that threatens nothing (discourages spam)
    "camp":         -0.03,  # per-step penalty while stuck in an already-explored
                            # pocket (see run_episode) — directly punishes the
                            # "loiter in ~5 cells, never advance" failure mode
    "escape":        0.05,  # left a tile that was about to explode
    "step":         -0.01,  # mild time pressure (no idle-corner exploit)
    "novelty":       0.03,  # first visit to a tile this episode (map coverage) — up
}


def _bombs(obs):
    raw = obs.get("bombs")
    if raw is None:
        return np.zeros((0, 4), dtype=np.int64)
    arr = np.asarray(raw)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int64)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def _min_instant_at(instants, pos):
    s = instants.get(pos)
    return min(s) if s else None


def target_potential(grid, players, agent_id, lam, dmax=24.0):
    """
    Potential Phi(s) for shaping: higher when the agent is CLOSER (BFS distance)
    to a useful target. Targets = item tiles + grass tiles adjacent to a box
    (i.e. good bomb spots); if the map is farmed out, fall back to enemies.
    Returns a value in [0, lam]. Potential-based shaping (gamma*Phi' - Phi)
    provably does not change the optimal policy — it only adds a gradient toward
    objectives, curing the "loiter near spawn" behaviour.
    """
    if lam <= 0:
        return 0.0
    uid = int(agent_id)
    if int(players[uid][2]) != 1:
        return 0.0
    H, W = grid.shape
    sx, sy = int(players[uid][0]), int(players[uid][1])

    targets = set()
    for x in range(H):
        for y in range(W):
            c = grid[x, y]
            if c == ITEM_R or c == ITEM_C:
                targets.add((x, y))
            elif c == BOX:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 < nx < H - 1 and 0 < ny < W - 1 and grid[nx, ny] in (GRASS, ITEM_R, ITEM_C):
                        targets.add((nx, ny))
    if not targets:
        for pid in range(len(players)):
            if pid != uid and int(players[pid][2]) == 1:
                targets.add((int(players[pid][0]), int(players[pid][1])))
    if not targets:
        return 0.0

    q = deque([(sx, sy, 0)])
    seen = {(sx, sy)}
    while q:
        x, y, d = q.popleft()
        if (x, y) in targets:
            return lam * (1.0 - min(d, dmax) / dmax)
        if d >= dmax:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 < nx < H - 1 and 0 < ny < W - 1 and (nx, ny) not in seen and grid[nx, ny] in (GRASS, ITEM_R, ITEM_C):
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))
    return 0.0


def enemy_potential(players, agent_id, lam_e, dmax=24.0):
    """Potential that is higher when the agent is CLOSER (Manhattan) to the
    nearest LIVE enemy. Manhattan (not BFS) so the pull exists even when boxes
    block the path — the agent then bombs its way toward the fight instead of
    farming one corner forever. Potential-based, so policy-invariant in theory:
    it only adds a gradient that says "go engage", curing the
    'bombs+dodges locally but never advances' local optimum."""
    if lam_e <= 0:
        return 0.0
    uid = int(agent_id)
    if int(players[uid][2]) != 1:
        return 0.0
    sx, sy = int(players[uid][0]), int(players[uid][1])
    dists = [abs(sx - int(players[p][0])) + abs(sy - int(players[p][1]))
             for p in range(len(players))
             if p != uid and int(players[p][2]) == 1]
    if not dists:
        return 0.0
    return lam_e * (1.0 - min(min(dists), dmax) / dmax)


def total_potential(grid, players, agent_id, shaping_lam, enemy_w):
    """Box/item-seeking shaping + always-on enemy-seeking shaping."""
    return (target_potential(grid, players, agent_id, shaping_lam)
            + enemy_potential(players, agent_id, enemy_w))


def _bomb_threatens_target(grid, players, agent_id, bx, by, radius):
    """True if a cross-shaped blast from (bx,by) of `radius` reaches a box or a
    live enemy. Walls block the blast; a box is itself a valid target (and stops
    the ray). Used to reward PURPOSEFUL bomb placement (vs. spamming bombs)."""
    H, W = grid.shape
    enemies = {(int(players[p][0]), int(players[p][1]))
               for p in range(len(players))
               if p != agent_id and int(players[p][2]) == 1}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            nx, ny = bx + dx * r, by + dy * r
            if not (0 <= nx < H and 0 <= ny < W):
                break
            if (nx, ny) in enemies:
                return True
            c = grid[nx, ny]
            if c == WALL:        # solid wall blocks the blast
                break
            if c == BOX:         # box is a target AND stops further blast
                return True
    return False


def event_reward(prev_obs, obs, agent_id, prev_stats, stats,
                 prev_alive, alive, n_opp_died, terminal, sole_winner, truncated):
    """Stat/event reward for the transition prev_obs -> obs (no shaping)."""
    if not prev_alive:
        return 0.0
    r = REWARD["step"]
    r += REWARD["kill"] * max(0, stats["kills"] - prev_stats["kills"])
    r += REWARD["box"]  * max(0, stats["boxes"] - prev_stats["boxes"])
    r += REWARD["item"] * max(0, stats["items"] - prev_stats["items"])
    r += REWARD["bomb"] * max(0, stats["bombs"] - prev_stats["bombs"])
    r += REWARD["opp_died"] * n_opp_died

    grid = np.asarray(prev_obs["map"])
    pl_prev = np.asarray(prev_obs["players"])
    pl_now = np.asarray(obs["players"])
    px, py = int(pl_prev[agent_id][0]), int(pl_prev[agent_id][1])
    cx, cy = int(pl_now[agent_id][0]), int(pl_now[agent_id][1])

    # purposeful-bomb shaping: a bomb was placed this step (bombs stat rose) at the
    # agent's previous tile. Reward it only if its blast threatens a box/enemy;
    # otherwise mildly penalise to discourage aimless bomb spam.
    n_bomb = max(0, stats["bombs"] - prev_stats["bombs"])
    if n_bomb > 0:
        radius = 1 + int(pl_prev[agent_id][4])
        if _bomb_threatens_target(grid, pl_prev, agent_id, px, py, radius):
            r += REWARD["bomb_target"] * n_bomb
        else:
            r += REWARD["bomb_waste"] * n_bomb

    inst_prev, _ = compute_danger(grid, _bombs(prev_obs), pl_prev)
    prev_t = _min_instant_at(inst_prev, (px, py))
    if prev_t is not None and (px, py) != (cx, cy):
        inst_now, _ = compute_danger(np.asarray(obs["map"]), _bombs(obs), pl_now)
        if _min_instant_at(inst_now, (cx, cy)) is None:
            r += REWARD["escape"] * (1.5 if prev_t <= 2 else 1.0)

    if not alive:
        r += REWARD["death"]
    elif terminal and sole_winner:
        r += REWARD["sole_winner"]
    elif truncated and alive:
        r += REWARD["survive_500"]
    return float(r)


# ── Prioritized Experience Replay (proportional, sum-tree) ───────────────────
class SumTree:
    def __init__(self, cap):
        self.cap = cap
        self.tree = np.zeros(2 * cap, dtype=np.float64)

    def total(self):
        return float(self.tree[1])

    def update(self, data_idx, p):
        idx = data_idx + self.cap
        delta = p - self.tree[idx]
        self.tree[idx] = p
        idx >>= 1
        while idx >= 1:
            self.tree[idx] += delta
            idx >>= 1

    def get(self, s):
        idx = 1
        while idx < self.cap:
            left = idx << 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - self.cap


class PERBuffer:
    def __init__(self, cap, map_shape, aux_dim, alpha=0.6, prioritized=True):
        self.cap, self.pos, self.size = cap, 0, 0
        self.alpha, self.prioritized, self.max_p = alpha, prioritized, 1.0
        ms = tuple(map_shape)
        self.sm = np.zeros((cap, *ms), np.float16)
        self.sa = np.zeros((cap, aux_dim), np.float16)
        self.nsm = np.zeros((cap, *ms), np.float16)
        self.nsa = np.zeros((cap, aux_dim), np.float16)
        self.act = np.zeros(cap, np.int64)
        self.ret = np.zeros(cap, np.float32)
        self.disc = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.float32)
        self.tree = SumTree(cap) if prioritized else None

    def push(self, sm, sa, a, ret, nsm, nsa, disc, done):
        p = self.pos
        self.sm[p], self.sa[p], self.nsm[p], self.nsa[p] = sm, sa, nsm, nsa
        self.act[p], self.ret[p], self.disc[p], self.done[p] = a, ret, disc, done
        if self.prioritized:
            self.tree.update(p, self.max_p)
        self.pos = (p + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n, beta=0.4):
        if self.prioritized:
            total = self.tree.total()
            seg = total / n
            idx = np.empty(n, np.int64)
            for i in range(n):
                s = random.uniform(seg * i, seg * (i + 1))
                idx[i] = min(self.tree.get(s), self.size - 1)
            leaf = self.tree.tree[idx + self.cap]
            probs = leaf / max(total, 1e-8)
            w = (self.size * probs) ** (-beta)
            w /= w.max() + 1e-8
            w = w.astype(np.float32)
        else:
            idx = np.random.randint(0, self.size, n)
            w = np.ones(n, np.float32)
        batch = (self.sm[idx].astype(np.float32), self.sa[idx].astype(np.float32),
                 self.nsm[idx].astype(np.float32), self.nsa[idx].astype(np.float32),
                 self.act[idx], self.ret[idx], self.disc[idx], self.done[idx])
        return batch, idx, w

    def update_priorities(self, idx, td):
        if not self.prioritized:
            return
        p = (np.abs(td) + 1e-5) ** self.alpha
        self.max_p = max(self.max_p, float(p.max()))
        for i, pi in zip(idx, p):
            self.tree.update(int(i), float(pi))

    def __len__(self):
        return self.size


# ── policies for opponent seats ──────────────────────────────────────────────
class NetPolicy:
    """Wraps a frozen network snapshot so it can play an opponent seat."""
    def __init__(self, state_dict, map_shape, aux_dim, agent_id, epsilon=0.0):
        self.net = DQNModel(map_shape, aux_dim, NUM_ACTIONS)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.agent_id = agent_id
        self.epsilon = epsilon

    def act(self, obs):
        if random.random() < self.epsilon:
            return random.choice(legal_safe_actions(obs, self.agent_id))
        ms, xs = encode_obs(obs, self.agent_id)
        with torch.no_grad():
            q = self.net(torch.from_numpy(ms).unsqueeze(0),
                         torch.from_numpy(xs).unsqueeze(0))[0].numpy()
        return safe_action(obs, self.agent_id, q)


# ── ranking (mirror competition.evaluation.match_runner) ─────────────────────
def final_ranks(env, death_order):
    n = len(env.players)
    ranks = [0] * n
    order = [list(g) for g in death_order]
    alive = [i for i, p in enumerate(env.players) if p.alive]
    if alive:
        def key(i):
            s = env.players[i].stats
            return (s["kills"], s["boxes"], s["items"], s["bombs"])
        alive.sort(key=key, reverse=True)
        groups, cur, cstat = [], [alive[0]], key(alive[0])
        for i in alive[1:]:
            if key(i) == cstat:
                cur.append(i)
            else:
                groups.append(cur); cur, cstat = [i], key(i)
        groups.append(cur)
        order.extend(reversed(groups))
    for rank, group in enumerate(reversed(order)):
        for i in group:
            ranks[i] = rank
    return ranks


# ── one self-play episode ────────────────────────────────────────────────────
def run_episode(env, seed, q_net, learner_seats, opponents, epsilon, gamma,
                n_step, shaping_lam, novelty_w, enemy_w):
    obs = env.reset(seed=seed)
    n = len(env.players)
    traj = {s: [] for s in learner_seats}
    prev_stats = {i: dict(env.players[i].stats) for i in range(n)}
    prev_alive = [bool(env.players[i].alive) for i in range(n)]
    death_order = []
    enc = {s: encode_obs(obs, s) for s in learner_seats}
    phi = {s: total_potential(np.asarray(obs["map"]), np.asarray(obs["players"]), s, shaping_lam, enemy_w)
           for s in learner_seats}
    visited = {s: {(int(obs["players"][s][0]), int(obs["players"][s][1]))} for s in learner_seats}
    stale = {s: 0 for s in learner_seats}     # consecutive steps without a new tile
    dev = next(q_net.parameters()).device

    while True:
        actions = [0] * n
        for s in learner_seats:
            if env.players[s].alive:
                if random.random() < epsilon:
                    actions[s] = random.choice(legal_safe_actions(obs, s))
                else:
                    ms, xs = enc[s]
                    with torch.no_grad():
                        q = q_net(torch.from_numpy(ms).unsqueeze(0).to(dev),
                                  torch.from_numpy(xs).unsqueeze(0).to(dev))[0].cpu().numpy()
                    actions[s] = safe_action(obs, s, q)
        for seat, pol in opponents.items():
            if env.players[seat].alive:
                try:
                    actions[seat] = int(pol.act(obs))
                except Exception:
                    actions[seat] = 0

        nobs, terminated, truncated = env.step(actions)
        done = terminated or truncated
        alive_now = [bool(env.players[i].alive) for i in range(n)]
        died = [i for i in range(n) if prev_alive[i] and not alive_now[i]]
        if died:
            death_order.append(died)
        survivors = [i for i in range(n) if alive_now[i]]

        npl = np.asarray(nobs["map"]), np.asarray(nobs["players"])
        for s in learner_seats:
            if not prev_alive[s]:
                continue
            stats = env.players[s].stats
            n_opp_died = sum(1 for d in died if d != s)
            r = event_reward(obs, nobs, s, prev_stats[s], stats,
                             prev_alive[s], alive_now[s], n_opp_died,
                             terminated, terminated and survivors == [s], truncated)
            # potential-based shaping (Phi'=0 once dead/terminal)
            phi_next = 0.0
            if alive_now[s] and not done:
                phi_next = total_potential(npl[0], npl[1], s, shaping_lam, enemy_w)
            r += gamma * phi_next - phi[s]
            phi[s] = phi_next
            # novelty bonus for new tiles + anti-camp penalty for being stuck.
            if alive_now[s]:
                cpos = (int(npl[1][s][0]), int(npl[1][s][1]))
                if cpos not in visited[s]:
                    visited[s].add(cpos)
                    stale[s] = 0
                    if novelty_w > 0:
                        r += novelty_w
                else:
                    stale[s] += 1
                    # only punish camping while genuinely trapped in a small pocket
                    # (<25 tiles seen): forces it to bomb its way out, not nag it
                    # late-game when it's fighting in an already-explored arena.
                    if stale[s] > 12 and len(visited[s]) < 25:
                        r += REWARD["camp"]

            nenc = encode_obs(nobs, s)
            traj[s].append((enc[s][0], enc[s][1], actions[s], r,
                            nenc[0], nenc[1], float(not alive_now[s] or done)))
            enc[s] = nenc

        prev_stats = {i: dict(env.players[i].stats) for i in range(n)}
        prev_alive = alive_now
        obs = nobs
        if done:
            break

    return traj, final_ranks(env, death_order)


def push_nstep(buf, records, gamma, n_step):
    T = len(records)
    for t in range(T):
        m = min(n_step, T - t)
        ret, disc, done = 0.0, gamma ** m, 0.0
        for k in range(m):
            ret += (gamma ** k) * records[t + k][3]
            if records[t + k][6]:
                done, disc = 1.0, 0.0
                break
        boot = records[t + m - 1]
        buf.push(records[t][0], records[t][1], records[t][2], ret,
                 boot[4], boot[5], disc, done)


# ── evaluation vs strong baselines ───────────────────────────────────────────
def evaluate(q_net, map_shape, aux_dim, n_matches=40, seed=10_000, max_steps=500):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    cpu_sd = {k: v.cpu() for k, v in q_net.state_dict().items()}
    pol = NetPolicy(cpu_sd, map_shape, aux_dim, agent_id=0, epsilon=0.0)
    wins, rank_sum = 0, 0
    for m in range(n_matches):
        opps = {i: RULE_CLASSES[random.choice(STRONG_RULES)](i) for i in (1, 2, 3)}
        ranks = _play_eval_match(env, seed + m, pol, opps)
        rank_sum += ranks[0]
        if ranks[0] == 0 and ranks.count(0) == 1:
            wins += 1
    return wins / n_matches, rank_sum / n_matches


def _play_eval_match(env, seed, learner_pol, opps):
    obs = env.reset(seed=seed)
    n = len(env.players)
    prev_alive = [bool(p.alive) for p in env.players]
    death_order = []
    while True:
        actions = [0] * n
        for i in range(n):
            if not env.players[i].alive:
                continue
            pol = learner_pol if i == 0 else opps.get(i)
            try:
                actions[i] = int(pol.act(obs))
            except Exception:
                actions[i] = 0
        obs, terminated, truncated = env.step(actions)
        alive_now = [bool(p.alive) for p in env.players]
        died = [i for i in range(n) if prev_alive[i] and not alive_now[i]]
        if died:
            death_order.append(died)
        prev_alive = alive_now
        if terminated or truncated:
            break
    return final_ranks(env, death_order)


# ── main training loop ───────────────────────────────────────────────────────
def train(
    episodes=12000, max_steps=500, seed=86, opponents="mix",
    lr=3e-4, gamma=0.99, n_step=3, batch_size=256, buffer_cap=100_000,
    target_freq=2000, eps_start=1.0, eps_end=0.05, eps_decay_episodes=8000,
    self_play_after=2000, snapshot_every=1000, pool_cap=10, learner_prob=0.6,
    shaping_lam=0.1, novelty_w=0.03, enemy_w=0.08, per=True, alpha=0.6, beta0=0.4,
    eval_every=500, save_dir="ckpts", resume=None, warm_from=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} episodes={episodes} n_step={n_step} "
          f"per={per} shaping={shaping_lam} novelty={novelty_w} learner_prob={learner_prob}")

    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs0 = env.reset(seed=seed)
    ms0, xs0 = encode_obs(obs0, 0)
    map_shape, aux_dim = ms0.shape, xs0.shape[0]
    input_spec = (tuple(map_shape), int(aux_dim))

    q_net = DQNModel(map_shape, aux_dim, NUM_ACTIONS).to(device)
    tgt_net = DQNModel(map_shape, aux_dim, NUM_ACTIONS).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=lr, eps=1e-5)
    global_step, start_ep, epsilon = 0, 0, eps_start

    if resume:
        ck = torch.load(resume, map_location=device)
        q_net.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        global_step = int(ck.get("global_step", 0))
        epsilon = float(ck.get("epsilon", eps_start))
        start_ep = int(ck.get("episode", 0))
        print(f"[resume] {resume} @ ep{start_ep} step{global_step} eps{epsilon:.3f}")
    elif warm_from:
        # Warm start from a TorchScript (model.pt) or .pth weights file: keep the
        # learned weights but start a FRESH optimizer/epsilon/step counter (those
        # were lost). Set --eps_start lower (e.g. 0.3) so it doesn't re-explore wildly.
        if warm_from.endswith(".pt"):
            scripted = torch.jit.load(warm_from, map_location=device)
            src_sd = scripted.state_dict()
        else:
            ck = torch.load(warm_from, map_location=device)
            src_sd = ck.get("model_state_dict", ck)
        missing, unexpected = q_net.load_state_dict(src_sd, strict=False)
        epsilon = eps_start            # fresh exploration schedule, controlled by --eps_start
        print(f"[warm_from] {warm_from}  eps={epsilon:.3f}  "
              f"(missing={len(missing)} unexpected={len(unexpected)} keys; "
              f"fresh optimizer + step counter)")
        if missing or unexpected:
            print(f"[warm_from] WARNING key mismatch -> missing={missing[:4]} "
                  f"unexpected={unexpected[:4]}")
    tgt_net.load_state_dict(q_net.state_dict())
    tgt_net.eval()

    buf = PERBuffer(buffer_cap, map_shape, aux_dim, alpha=alpha, prioritized=per)
    learner_prob = float(learner_prob)
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{save_dir}/selfplay_{opponents}_{episodes}ep"
    os.makedirs(tag, exist_ok=True)

    pool = []
    loss_h, rew_h, win_h = [], [], []
    eps_drop = (eps_start - eps_end) / max(1, eps_decay_episodes)

    try:
        from tqdm import trange
        bar = trange(start_ep, episodes, desc="self-play")
    except Exception:
        bar = range(start_ep, episodes)

    for ep in bar:
        beta = min(1.0, beta0 + (1.0 - beta0) * ep / max(1, episodes))

        # ---- assign each seat to learner or opponent ------------------------
        learner_seats, opponents_map = [], {}
        for seat in range(len(env.players)):
            if random.random() < learner_prob:
                learner_seats.append(seat)
            else:
                use_self = pool and ep >= self_play_after and random.random() < 0.5
                if use_self:
                    opponents_map[seat] = NetPolicy(
                        random.choice(pool), map_shape, aux_dim, seat, epsilon=0.05)
                else:
                    name = opponents if opponents in RULE_CLASSES else random.choice(STRONG_RULES)
                    opponents_map[seat] = RULE_CLASSES[name](seat)
        if not learner_seats:                       # guarantee >=1 learner seat
            seat = random.randrange(len(env.players))
            learner_seats.append(seat)
            opponents_map.pop(seat, None)

        traj, ranks = run_episode(env, seed + ep, q_net, learner_seats,
                                  opponents_map, epsilon, gamma, n_step,
                                  shaping_lam, novelty_w, enemy_w)

        ep_rew, n_records = 0.0, 0
        for s in learner_seats:
            if traj[s]:
                ep_rew += sum(rec[3] for rec in traj[s])
                n_records += len(traj[s])
                push_nstep(buf, traj[s], gamma, n_step)

        # ---- learn -----------------------------------------------------------
        if len(buf) >= batch_size:
            n_updates = min(max(1, n_records // 8), 64)
            for _ in range(n_updates):
                loss = _learn(q_net, tgt_net, optimizer, buf, batch_size,
                              gamma, device, beta)
                loss_h.append(loss)
                global_step += 1
                if global_step % target_freq == 0:
                    tgt_net.load_state_dict(q_net.state_dict())

        epsilon = max(eps_end, epsilon - eps_drop)
        win_h.append(1 if min(ranks[s] for s in learner_seats) == 0 else 0)
        rew_h.append(ep_rew / max(1, len(learner_seats)))

        if (ep + 1) >= self_play_after and (ep + 1) % snapshot_every == 0:
            pool.append(copy.deepcopy({k: v.cpu() for k, v in q_net.state_dict().items()}))
            if len(pool) > pool_cap:
                pool.pop(0)

        if hasattr(bar, "set_postfix") and (ep % 20 == 0):
            wr = sum(win_h[-200:]) / max(1, len(win_h[-200:]))
            bar.set_postfix(rew=f"{rew_h[-1]:6.1f}", eps=f"{epsilon:.3f}",
                            top1=f"{wr:.2f}", buf=len(buf), pool=len(pool))

        if (ep + 1) % eval_every == 0:
            wr, avg_rank = evaluate(q_net, map_shape, aux_dim, n_matches=40)
            print(f"\n[eval ep{ep+1}] vs strong baselines: "
                  f"top1_rate={wr:.2f} avg_rank={avg_rank:.2f} (0=best,3=worst)")
            _save(q_net, optimizer, global_step, epsilon, ep + 1, input_spec,
                  f"{tag}/ep{ep+1}.pth", export_pt=f"{tag}/model.pt")

    _save(q_net, optimizer, global_step, epsilon, episodes, input_spec,
          f"{tag}/model.pth", export_pt=f"{tag}/model.pt")
    try:
        plot_loss(loss_h, f"{tag}/loss.png")
        plot_rewards(rew_h, f"{tag}/rewards.png")
        plot_win_rates(win_h, f"{tag}/win_rate.png")
        if len(rew_h) > 50:
            plot_moving_average(rew_h, 50, f"{tag}/ma_reward.png")
    except Exception as e:
        print(f"[plot] skipped: {e}")
    print(f"\n[done] Copy {tag}/model.pt next to agent.py + model.py for submission.")
    return f"{tag}/model.pth", f"{tag}/model.pt"


def _learn(q_net, tgt_net, optimizer, buf, batch_size, gamma, device, beta):
    (sm, sa, nsm, nsa, act, ret, disc, done), idx, w = buf.sample(batch_size, beta)
    t = lambda x: torch.from_numpy(x).to(device)
    sm, sa, nsm, nsa = t(sm), t(sa), t(nsm), t(nsa)
    act = t(act).unsqueeze(1)
    ret = t(ret).unsqueeze(1)
    disc = t(disc).unsqueeze(1)
    done = t(done).unsqueeze(1)
    w = t(w).unsqueeze(1)

    q = q_net(sm, sa).gather(1, act)
    with torch.no_grad():
        best = q_net(nsm, nsa).argmax(1, keepdim=True)          # Double DQN
        q_next = tgt_net(nsm, nsa).gather(1, best)
        target = ret + disc * q_next * (1.0 - done)
    td = target - q
    loss = (w * F.smooth_l1_loss(q, target, reduction="none")).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
    buf.update_priorities(idx, td.detach().squeeze(1).cpu().numpy())
    return float(loss.item())


def _save(q_net, optimizer, global_step, epsilon, episode, input_spec, path, export_pt=None):
    ck = {
        "model_state_dict": q_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step, "epsilon": epsilon, "episode": episode,
        "input_spec": input_spec, "input_shape": input_spec, "input_dim": input_spec,
        "num_actions": NUM_ACTIONS,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ck, path)
    if export_pt:
        was_cuda = next(q_net.parameters()).is_cuda
        scripted = torch.jit.script(q_net.cpu().eval())
        scripted.save(export_pt)
        if was_cuda:
            q_net.cuda()
        print(f"[save] {path}  +  TorchScript {export_pt}")
    else:
        print(f"[save] {path}")


def _b(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def main():
    p = argparse.ArgumentParser("Bomberland self-play DQN trainer (active shaping + PER)")
    p.add_argument("--episodes", type=int, default=12000)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--opponents", default="mix",
                   help="'mix' (strong rules + self), or a rule name "
                        "(simple/smarter/tactical/genius/box_farmer/random)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_cap", type=int, default=100_000)
    p.add_argument("--eps_decay_episodes", type=int, default=8000)
    p.add_argument("--self_play_after", type=int, default=2000)
    p.add_argument("--snapshot_every", type=int, default=1000)
    p.add_argument("--learner_prob", type=float, default=0.6,
                   help="prob each seat is the learner (rest = opponents). "
                        "1.0 = pure 4-seat self-play")
    p.add_argument("--shaping_lam", type=float, default=0.1, help="0 disables potential shaping")
    p.add_argument("--novelty_w", type=float, default=0.03, help="0 disables novelty bonus")
    p.add_argument("--enemy_w", type=float, default=0.08,
                   help="enemy-seeking shaping weight (pull toward nearest live enemy; "
                        "0 disables). Cures 'farms one corner, never advances'.")
    p.add_argument("--per", type=_b, default=True)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_dir", default="ckpts")
    p.add_argument("--resume", default=None,
                   help="resume FULL state from a .pth checkpoint (model+optimizer+epsilon+step)")
    p.add_argument("--warm_from", default=None,
                   help="warm start weights only from model.pt/.pth (fresh optimizer+epsilon; "
                        "set --eps_start lower e.g. 0.3)")
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    args = p.parse_args()

    seed_everything(args.seed)
    train(
        episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
        opponents=args.opponents, lr=args.lr, gamma=args.gamma, n_step=args.n_step,
        batch_size=args.batch_size, buffer_cap=args.buffer_cap,
        eps_decay_episodes=args.eps_decay_episodes, self_play_after=args.self_play_after,
        snapshot_every=args.snapshot_every, learner_prob=args.learner_prob,
        shaping_lam=args.shaping_lam, novelty_w=args.novelty_w, enemy_w=args.enemy_w, per=args.per,
        eval_every=args.eval_every, save_dir=args.save_dir, resume=args.resume,
        warm_from=args.warm_from, eps_start=args.eps_start, eps_end=args.eps_end,
    )


if __name__ == "__main__":
    main()
