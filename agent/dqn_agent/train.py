"""
Self-play trainer for the Bomberland DQN agent (Dueling Double DQN + n-step).

Run from the repository root:

    python -m agent.dqn_agent.train --episodes 20000 --opponents mix

Design choices (see docs/COMPETITION_GUIDE.md for the scoring rules):
  * Survival-first reward — match rank is decided by survival, then
    kills > boxes > items > bombs, so those are exactly the shaped events.
  * Safe exploration — both random and greedy actions are filtered through the
    same safety planner used at inference, so the network spends its capacity
    learning *strategy* while the mask guarantees it rarely dies by accident.
  * Self-play — opponents are a mix of rule baselines and frozen snapshots of
    the learner, which keeps the policy robust against unseen styles.
  * n-step Double DQN with a dueling network for fast, stable credit assignment.
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
import torch.optim as optim

# ── make the repo importable however this file is launched ───────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.dqn_agent.model import (              # noqa: E402
    DQNModel, encode_obs, safe_action, legal_safe_actions,
    compute_danger, N_MAP_CH, N_AUX, NUM_ACTIONS, BOMB_TIMER_MAX,
)
from agent.dqn_agent.utils import (              # noqa: E402
    seed_everything, save_model_fn, plot_loss, plot_rewards,
    plot_win_rates, plot_moving_average,
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


# ── reward shaping (aligned with the match ranking) ──────────────────────────
REWARD = {
    "death":        -3.0,   # being eliminated — by far the worst outcome
    "sole_winner":   3.0,   # last agent standing
    "survive_500":   0.5,   # reached the step cap alive
    "kill":          1.0,   # an opponent you eliminated (your stats['kills'])
    "opp_died":      0.3,   # any opponent removed this step (rank improves)
    "box":           0.3,   # box you destroyed
    "item":          0.2,   # item you collected
    "bomb":          0.02,  # bomb placed (encourage activity, not spam)
    "escape":        0.05,  # left a tile that was about to explode
    "step":         -0.005, # mild time pressure (no idle-corner exploit)
}


def _min_instant_at(instants, pos):
    s = instants.get(pos)
    return min(s) if s else None


def step_reward(prev_obs, obs, agent_id, prev_stats, stats,
                prev_alive, alive, n_opp_died, terminal, sole_winner, truncated):
    """Reward for the transition prev_obs -> obs for `agent_id`."""
    if not prev_alive:
        return 0.0
    r = REWARD["step"]
    r += REWARD["kill"]   * max(0, stats["kills"]  - prev_stats["kills"])
    r += REWARD["box"]    * max(0, stats["boxes"]  - prev_stats["boxes"])
    r += REWARD["item"]   * max(0, stats["items"]  - prev_stats["items"])
    r += REWARD["bomb"]   * max(0, stats["bombs"]  - prev_stats["bombs"])
    r += REWARD["opp_died"] * n_opp_died

    # danger shaping: reward fleeing an imminent blast
    grid = np.asarray(prev_obs["map"])
    pl_prev = np.asarray(prev_obs["players"])
    pl_now = np.asarray(obs["players"])
    px, py = int(pl_prev[agent_id][0]), int(pl_prev[agent_id][1])
    cx, cy = int(pl_now[agent_id][0]), int(pl_now[agent_id][1])
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


def _bombs(obs):
    raw = obs.get("bombs")
    if raw is None:
        return np.zeros((0, 4), dtype=np.int64)
    arr = np.asarray(raw)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int64)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


# ── replay buffer (float16 storage, n-step ready) ────────────────────────────
class ReplayBuffer:
    def __init__(self, cap, map_shape, aux_dim):
        self.cap, self.pos, self.size = cap, 0, 0
        ms = tuple(map_shape)
        self.sm = np.zeros((cap, *ms), np.float16)
        self.sa = np.zeros((cap, aux_dim), np.float16)
        self.nsm = np.zeros((cap, *ms), np.float16)
        self.nsa = np.zeros((cap, aux_dim), np.float16)
        self.act = np.zeros(cap, np.int64)
        self.ret = np.zeros(cap, np.float32)   # n-step return
        self.disc = np.zeros(cap, np.float32)  # gamma**n (0 if terminal)
        self.done = np.zeros(cap, np.float32)

    def push(self, sm, sa, a, ret, nsm, nsa, disc, done):
        p = self.pos
        self.sm[p], self.sa[p] = sm, sa
        self.nsm[p], self.nsa[p] = nsm, nsa
        self.act[p], self.ret[p], self.disc[p], self.done[p] = a, ret, disc, done
        self.pos = (p + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (self.sm[idx].astype(np.float32), self.sa[idx].astype(np.float32),
                self.nsm[idx].astype(np.float32), self.nsa[idx].astype(np.float32),
                self.act[idx], self.ret[idx], self.disc[idx], self.done[idx])

    def __len__(self):
        return self.size


# ── policies for the non-learner seats ───────────────────────────────────────
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
def run_episode(env, seed, q_net, learner_seats, opponents, epsilon, gamma, n_step):
    """Play one match; return (list of per-seat trajectories, ranks)."""
    obs = env.reset(seed=seed)
    n = len(env.players)
    traj = {s: [] for s in learner_seats}     # seat -> list of step records
    prev_stats = {i: dict(env.players[i].stats) for i in range(n)}
    prev_alive = [bool(env.players[i].alive) for i in range(n)]
    death_order = []
    enc = {s: encode_obs(obs, s) for s in learner_seats}
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
        sole_winner = terminated and len(survivors) == 1

        for s in learner_seats:
            if not prev_alive[s]:
                continue
            stats = env.players[s].stats
            n_opp_died = sum(1 for d in died if d != s)
            r = step_reward(obs, nobs, s, prev_stats[s], stats,
                            prev_alive[s], alive_now[s], n_opp_died,
                            terminated, sole_winner and survivors == [s], truncated)
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
    """Convert a seat trajectory into n-step transitions and store them."""
    T = len(records)
    for t in range(T):
        m = min(n_step, T - t)
        ret, disc, done = 0.0, gamma ** m, 0.0
        for k in range(m):
            ret += (gamma ** k) * records[t + k][3]
            if records[t + k][6]:           # terminal reached within the window
                done, disc = 1.0, 0.0
                break
        boot = records[t + m - 1]
        buf.push(records[t][0], records[t][1], records[t][2], ret,
                 boot[4], boot[5], disc, done)


# ── evaluation vs strong baselines ───────────────────────────────────────────
def evaluate(q_net, map_shape, aux_dim, n_matches=30, seed=10_000, max_steps=500):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    # snapshot to CPU so the eval policy is independent of the (possibly CUDA) learner
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
    episodes=20000, max_steps=500, seed=86, learner_seats=(0,),
    opponents="mix", lr=3e-4, gamma=0.99, n_step=3, batch_size=256,
    buffer_cap=100_000, target_freq=2000, train_per_step=1,
    eps_start=1.0, eps_end=0.05, eps_decay_episodes=8000,
    self_play_after=2000, snapshot_every=1000, pool_cap=8,
    eval_every=1000, save_dir="ckpts", resume=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  episodes={episodes}  n_step={n_step}")

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
    tgt_net.load_state_dict(q_net.state_dict())
    tgt_net.eval()

    buf = ReplayBuffer(buffer_cap, map_shape, aux_dim)
    loss_fn = nn.SmoothL1Loss()
    learner_seats = list(learner_seats)
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{save_dir}/selfplay_{opponents}_{episodes}ep"
    os.makedirs(tag, exist_ok=True)

    pool = []                       # list of frozen state_dicts for self-play
    loss_h, rew_h, win_h = [], [], []
    eps_drop = (eps_start - eps_end) / max(1, eps_decay_episodes)

    try:
        from tqdm import trange
        bar = trange(start_ep, episodes, desc="self-play")
    except Exception:
        bar = range(start_ep, episodes)

    for ep in bar:
        # ---- assign opponent seats -------------------------------------------
        opp_seats = [s for s in range(len(env.players)) if s not in learner_seats]
        opponents_map = {}
        for seat in opp_seats:
            use_self = pool and ep >= self_play_after and random.random() < 0.5
            if opponents == "self" and pool:
                use_self = True
            if use_self:
                opponents_map[seat] = NetPolicy(
                    random.choice(pool), map_shape, aux_dim, seat,
                    epsilon=0.05)
            else:
                if opponents in RULE_CLASSES:
                    name = opponents
                else:                       # "mix" / "self": sample a strong rule
                    name = random.choice(STRONG_RULES)
                opponents_map[seat] = RULE_CLASSES[name](seat)

        traj, ranks = run_episode(env, seed + ep, q_net, learner_seats,
                                  opponents_map, epsilon, gamma, n_step)

        ep_rew = 0.0
        for s in learner_seats:
            if traj[s]:
                ep_rew += sum(rec[3] for rec in traj[s])
                push_nstep(buf, traj[s], gamma, n_step)

        # ---- learn -----------------------------------------------------------
        if len(buf) >= batch_size:
            n_updates = max(1, train_per_step) * max(1, sum(len(traj[s]) for s in learner_seats) // 8)
            for _ in range(min(n_updates, 64)):
                loss = _learn(q_net, tgt_net, optimizer, loss_fn, buf,
                              batch_size, gamma, device)
                loss_h.append(loss)
                global_step += 1
                if global_step % target_freq == 0:
                    tgt_net.load_state_dict(q_net.state_dict())

        epsilon = max(eps_end, epsilon - eps_drop)
        learner_rank = min(ranks[s] for s in learner_seats)
        win_h.append(1 if learner_rank == 0 else 0)
        rew_h.append(ep_rew)

        if (ep + 1) >= self_play_after and (ep + 1) % snapshot_every == 0:
            pool.append(copy.deepcopy({k: v.cpu() for k, v in q_net.state_dict().items()}))
            if len(pool) > pool_cap:
                pool.pop(0)

        if hasattr(bar, "set_postfix") and (ep % 20 == 0):
            wr = sum(win_h[-200:]) / max(1, len(win_h[-200:]))
            bar.set_postfix(rew=f"{ep_rew:6.1f}", eps=f"{epsilon:.3f}",
                            top1=f"{wr:.2f}", buf=len(buf), step=global_step,
                            pool=len(pool))

        if (ep + 1) % eval_every == 0:
            wr, avg_rank = evaluate(q_net, map_shape, aux_dim, n_matches=30)
            print(f"\n[eval ep{ep+1}] vs strong baselines: "
                  f"top1_rate={wr:.2f}  avg_rank={avg_rank:.2f} (0=best,3=worst)")
            _save(q_net, optimizer, global_step, epsilon, ep + 1, input_spec,
                  f"{tag}/ep{ep+1}.pth", export_pt=f"{tag}/model.pt")

    # ---- final save ----------------------------------------------------------
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


def _learn(q_net, tgt_net, optimizer, loss_fn, buf, batch_size, gamma, device):
    sm, sa, nsm, nsa, act, ret, disc, done = buf.sample(batch_size)
    t = lambda x: torch.from_numpy(x).to(device)
    sm, sa, nsm, nsa = t(sm), t(sa), t(nsm), t(nsa)
    act = t(act).unsqueeze(1)
    ret = t(ret).unsqueeze(1)
    disc = t(disc).unsqueeze(1)
    done = t(done).unsqueeze(1)

    q = q_net(sm, sa).gather(1, act)
    with torch.no_grad():
        best = q_net(nsm, nsa).argmax(1, keepdim=True)         # Double DQN
        q_next = tgt_net(nsm, nsa).gather(1, best)
        target = ret + disc * q_next * (1.0 - done)
    loss = loss_fn(q, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
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
        q_net_cpu = q_net.cpu().eval()
        scripted = torch.jit.script(q_net_cpu)
        scripted.save(export_pt)
        if was_cuda:
            q_net.cuda()
        print(f"[save] {path}  +  TorchScript {export_pt}")
    else:
        print(f"[save] {path}")


def main():
    p = argparse.ArgumentParser("Bomberland self-play DQN trainer")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--opponents", default="mix",
                   help="'mix' (sample strong rules + self), 'self', or a rule name "
                        "(simple/smarter/tactical/genius/box_farmer/random)")
    p.add_argument("--learner_seats", nargs="+", type=int, default=[0])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_cap", type=int, default=100_000)
    p.add_argument("--self_play_after", type=int, default=2000)
    p.add_argument("--snapshot_every", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_dir", default="ckpts")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    train(
        episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
        learner_seats=tuple(args.learner_seats), opponents=args.opponents,
        lr=args.lr, gamma=args.gamma, n_step=args.n_step, batch_size=args.batch_size,
        buffer_cap=args.buffer_cap, self_play_after=args.self_play_after,
        snapshot_every=args.snapshot_every, eval_every=args.eval_every,
        save_dir=args.save_dir, resume=args.resume,
    )


if __name__ == "__main__":
    main()
