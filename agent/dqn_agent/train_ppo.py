"""
PPO (Proximal Policy Optimization) self-play trainer for the Bomberland agent.

WHY PPO (and why this fixes the camping problem the DQN had):
  The DQN agent camped in its spawn corner. Root cause was NOT the algorithm but
  the training regime:
    1. A hard "guaranteed-survivable" safety mask (model.safe_action) meant survival
       never depended on the policy -> the net never had to learn to play, and at
       inference it would only go where survival was already proven (= stay home).
    2. The shaping rewarded being CLOSE to the nearest box; at spawn the agent is
       already boxed-in (adjacent to boxes), so the potential was already maxed and
       gave ZERO pull outward -> it was literally pinned to the spawn cluster.
    3. The reward was dominated by tiny shaping/novelty terms, barely tied to the
       match ranking, and eval reused the same weak opponents (over-optimistic).

  This trainer attacks all three:
    * NO safety mask while learning. The policy SAMPLES actions (PPO is on-policy),
      experiences death (death = -3) directly, and an ENTROPY bonus keeps it
      exploring so it cannot collapse onto a single degenerate "stay" behaviour.
      (Only a light, cheap shield is used at *inference* — see ppo.shielded_action.)
    * Reward aligned with the TrueSkill match ranking (survive >> kills > boxes >
      items > bombs), with potential-based shaping that PULLS THE AGENT OUTWARD
      (toward the board centre, then toward enemies as it powers up) instead of
      pinning it to spawn.
    * A camping-punishing opponent league (Hunter-heavy) + frozen self snapshots,
      and a behaviour-based eval (cells explored / boxes / items / survival) that
      actually predicts leaderboard strength instead of rewarding turtling.

Run from the repository ROOT:
    python -m agent.dqn_agent.train_ppo --iters 1500 --epi_per_iter 8

The exported `model.pt` (TorchScript of the actor-critic) drops straight into the
flat submission zip next to agent.py + model.py + ppo.py.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from collections import deque
from functools import partial
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
    encode_obs, compute_danger, NUM_ACTIONS,
)
from agent.dqn_agent.ppo import (                # noqa: E402
    PPOActorCritic, physical_action_mask, safe_action_mask, shielded_action,
    survivable_action_mask, DEFAULT_SHIELD_HORIZON,
)
from engine.game import BomberEnv                # noqa: E402
from agent import (                              # noqa: E402
    RandomAgent, SimpleRuleAgent, SmarterRuleAgent,
    TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent, HunterAgent,
)

GRASS, WALL, BOX, ITEM_R, ITEM_C = 0, 1, 2, 3, 4

RULE_CLASSES = {
    "random": RandomAgent, "simple": SimpleRuleAgent, "smarter": SmarterRuleAgent,
    "tactical": TacticalRuleAgent, "genius": GeniusRuleAgent,
    "box_farmer": BoxFarmerAgent, "hunter": HunterAgent,
}
# Hunter-heavy: in 4-player FFA a learner can win cheaply by turtling while the
# others fight and die. Only an opponent that actively HUNTS a camper makes
# passivity lose, so Hunter is over-represented in the training mix.
STRONG_RULES = ["hunter", "hunter", "hunter", "genius", "genius",
                "tactical", "box_farmer", "smarter"]


# ── reward shaping (aligned with the match ranking) ──────────────────────────
REWARD = {
    "death":       -4.0,    # eliminated — by far the worst outcome (survival DOMINATES
                            #   the TrueSkill rank); bumped -3 -> -4 so the policy values
                            #   staying alive over a greedy farm that gets it cornered.
    "sole_winner":  5.0,    # last agent standing — the top of the ranking (draw_prob=0.1
                            #   => a SOLE win is worth far more than a shared survival).
    "kill":         2.5,    # opponent you eliminated (ranking tiebreak #1 among survivors)
    "opp_died":     0.3,    # any opponent removed this step (your rank improves)
    "box":          0.5,    # box destroyed (opens the map + spawns items)
    "item":         1.0,    # item collected = more bombs / bigger blast = real power
    "bomb_box":     0.10,   # placed a bomb whose blast reaches a BOX (farming)
    "bomb_enemy":   0.20,   # placed a bomb whose blast reaches an ENEMY -- a safe-strike
                            #   SETUP, DOUBLED when farmed-out (the late-game hunt). Rewards
                            #   the OUTCOME (threatening a foe from wherever you stand) so
                            #   PPO learns the survivable distance, NOT a proximity pull that
                            #   just ran it onto the enemy's bomb (that died ~step 369).
    "bomb_waste":  -0.05,   # placed a bomb that threatens nothing (anti-spam)
    "escape":       0.05,   # left a tile that was about to explode
    "step":        -0.005,  # mild time pressure (was -0.01: over 500 steps that summed to
                            #   -5, i.e. it PENALISED long survival — wrong for a survival
                            #   game; anti-dawdle is handled by `camp` + `novelty`).
    "novelty":      0.02,   # first visit to a tile this episode (map coverage)
    "camp":        -0.05,   # stuck in a tiny already-explored pocket (anti-camp)
    "idle_farmed": -0.03,   # farmed-out (few boxes left) but sitting still while enemies
                            #   live -> the "doesn't know what to do after farming" bug.
}

# Boards with <= this many boxes left count as "farmed out" -> switch to HUNT (pull
# hard toward enemies + penalise idling). This encodes the top-8 arc the agent missed:
# farm boxes -> eat radius -> once nothing's left to farm, go kill people.
FARMED_OUT_BOXES = 15

# Multiplier on the bomb_enemy reward once the board is farmed out (the late-game hunt
# incentive). 1.0 = no late boost (~ proven, no extra hunting). Tunable via --hunt_boost.
HUNT_BOOST = 3.0


# ── small utilities ──────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _bomb_hits(grid, players, agent_id, bx, by, radius):
    """Return (hits_box, hits_enemy): does a cross blast from (bx,by) reach a BOX /
    a live enemy? Walls block the ray; a box is counted then blocks further."""
    H, W = grid.shape
    enemies = {(int(players[p][0]), int(players[p][1]))
               for p in range(len(players))
               if p != agent_id and int(players[p][2]) == 1}
    hits_box = hits_enemy = False
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            nx, ny = bx + dx * r, by + dy * r
            if not (0 <= nx < H and 0 <= ny < W):
                break
            if (nx, ny) in enemies:
                hits_enemy = True
            c = grid[nx, ny]
            if c == WALL:
                break
            if c == BOX:
                hits_box = True
                break
    return hits_box, hits_enemy


# ── potential-based shaping: PULL OUTWARD, then toward enemies ───────────────
def advance_potential(players, uid, lam_adv, center, dmax):
    """Higher when CLOSER to the board centre. At a spawn corner this is ~0, so
    every step toward the centre yields positive (telescoping) shaping -> the
    agent is pulled OUT of its corner. Replaces the old box-proximity potential
    that pinned a boxed-in agent to spawn."""
    if lam_adv <= 0 or int(players[uid][2]) != 1:
        return 0.0
    d = abs(int(players[uid][0]) - center[0]) + abs(int(players[uid][1]) - center[1])
    return lam_adv * (1.0 - min(d, dmax) / dmax)


def enemy_potential(players, uid, lam_e, dmax=20.0):
    """Higher when CLOSER (Manhattan) to the nearest live enemy. Scaled up with the
    agent's power so a powered-up agent is pulled into the fight (farm early, fight
    late) — but with a floor so even a weak agent still pressures opponents.

    This is a MILD positional pull only (kept at the proven low weight). The actual
    hunt incentive lives in the OUTCOME reward `bomb_enemy` (see event_reward): a
    strong farmed-out PROXIMITY boost was tried and it just ran the agent onto enemy
    bombs and died (~step 369). Rewarding the bomb-on-enemy OUTCOME instead lets PPO
    learn to strike from a survivable distance."""
    if lam_e <= 0 or int(players[uid][2]) != 1:
        return 0.0
    sx, sy = int(players[uid][0]), int(players[uid][1])
    dists = [abs(sx - int(players[p][0])) + abs(sy - int(players[p][1]))
             for p in range(len(players))
             if p != uid and int(players[p][2]) == 1]
    if not dists:
        return 0.0
    power = int(players[uid][4])
    scale = min(1.0, 0.2 + 0.3 * power)
    return lam_e * scale * (1.0 - min(min(dists), dmax) / dmax)


def total_potential(players, uid, lam_adv, lam_e, center, dmax_adv):
    return (advance_potential(players, uid, lam_adv, center, dmax_adv)
            + enemy_potential(players, uid, lam_e))


def event_reward(prev_obs, obs, uid, prev_stats, stats,
                 prev_alive, alive, n_opp_died, terminal, sole_winner):
    """Stat/event reward for prev_obs -> obs (no potential shaping)."""
    if not prev_alive:
        return 0.0
    r = REWARD["step"]
    r += REWARD["kill"] * max(0, stats["kills"] - prev_stats["kills"])
    r += REWARD["box"] * max(0, stats["boxes"] - prev_stats["boxes"])
    r += REWARD["item"] * max(0, stats["items"] - prev_stats["items"])
    r += REWARD["opp_died"] * n_opp_died

    grid = np.asarray(prev_obs["map"])
    pl_prev = np.asarray(prev_obs["players"])
    pl_now = np.asarray(obs["players"])
    px, py = int(pl_prev[uid][0]), int(pl_prev[uid][1])
    cx, cy = int(pl_now[uid][0]), int(pl_now[uid][1])

    n_bomb = max(0, stats["bombs"] - prev_stats["bombs"])
    if n_bomb > 0:
        radius = 1 + int(pl_prev[uid][4])
        hits_box, hits_enemy = _bomb_hits(grid, pl_prev, uid, px, py, radius)
        if hits_enemy:
            # OUTCOME-based hunt reward, DOUBLED once farmed out (the late-game phase
            # where it used to wander). PPO learns to do this from a survivable spot
            # (death=-4 + the shield), not by rushing onto the enemy.
            farmed_out = int((grid == BOX).sum()) <= FARMED_OUT_BOXES
            r += REWARD["bomb_enemy"] * (HUNT_BOOST if farmed_out else 1.0) * n_bomb
        elif hits_box:
            r += REWARD["bomb_box"] * n_bomb
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
    return float(r)


# ── final ranking (mirror competition.evaluation.match_runner) ───────────────
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


# ── frozen PPO snapshot as an opponent ───────────────────────────────────────
class NetPPOPolicy:
    def __init__(self, state_dict, map_shape, aux_dim, agent_id,
                 horizon=DEFAULT_SHIELD_HORIZON):
        self.net = PPOActorCritic(map_shape, aux_dim, NUM_ACTIONS)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.agent_id = agent_id
        self.horizon = horizon

    def act(self, obs):
        ms, xs = encode_obs(obs, self.agent_id)
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(ms).unsqueeze(0),
                                 torch.from_numpy(xs).unsqueeze(0))
        return shielded_action(obs, self.agent_id, logits[0].numpy(),
                               horizon=self.horizon)


# ── masked log-prob / entropy used in the PPO update ─────────────────────────
def _logp_entropy(logits, mask, actions):
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg))
    logp_all = F.log_softmax(masked, dim=1)
    logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    p_all = logp_all.exp()
    ent = -(p_all * logp_all)
    ent = torch.where(mask, ent, torch.zeros_like(ent)).sum(1)
    return logp, ent


# ── one self-play episode: collect on-policy trajectories for learner seats ──
def run_episode(env, seed, net, device, learner_seats, opponents, cfg, center,
                dmax_adv, mask_fn=physical_action_mask):
    obs = env.reset(seed=seed)
    n = len(env.players)
    neg = torch.finfo(torch.float32).min

    traj = {s: {"map": [], "aux": [], "act": [], "logp": [], "val": [],
                "mask": [], "rew": [], "done": []} for s in learner_seats}
    prev_stats = {i: dict(env.players[i].stats) for i in range(n)}
    prev_alive = [bool(env.players[i].alive) for i in range(n)]
    death_order = []
    phi = {s: total_potential(np.asarray(obs["players"]), s,
                              cfg["lam_adv"], cfg["lam_e"], center, dmax_adv)
           for s in learner_seats}
    visited = {s: {(int(obs["players"][s][0]), int(obs["players"][s][1]))}
               for s in learner_seats}
    stale = {s: 0 for s in learner_seats}
    gamma = cfg["gamma"]

    while True:
        actions = [0] * n

        # ---- learner seats: batched forward, masked sampling -----------------
        act_seats = [s for s in learner_seats if env.players[s].alive]
        step_info = {}
        if act_seats:
            ms_list, xs_list, masks = [], [], []
            for s in act_seats:
                ms, xs = encode_obs(obs, s)
                ms_list.append(ms); xs_list.append(xs)
                masks.append(mask_fn(obs, s))
            M = torch.from_numpy(np.stack(ms_list)).to(device)
            X = torch.from_numpy(np.stack(xs_list)).to(device)
            mask_t = torch.from_numpy(np.stack(masks)).to(device)
            with torch.no_grad():
                logits, value = net(M, X)
                masked = torch.where(mask_t, logits, torch.full_like(logits, neg))
                logp_all = F.log_softmax(masked, dim=1)
                probs = logp_all.exp()
                sampled = torch.multinomial(probs, 1).squeeze(1)
                logp = logp_all.gather(1, sampled.unsqueeze(1)).squeeze(1)
            sampled_np = sampled.cpu().numpy()
            logp_np = logp.cpu().numpy()
            value_np = value.cpu().numpy()
            for j, s in enumerate(act_seats):
                actions[s] = int(sampled_np[j])
                step_info[s] = (ms_list[j], xs_list[j], masks[j],
                                int(sampled_np[j]), float(logp_np[j]),
                                float(value_np[j]))

        # ---- opponent seats --------------------------------------------------
        for seat, pol in opponents.items():
            if env.players[seat].alive:
                try:
                    actions[seat] = int(pol.act(obs))
                except Exception:
                    actions[seat] = 0

        nobs, terminated, truncated = env.step(actions)
        done_env = terminated or truncated
        alive_now = [bool(env.players[i].alive) for i in range(n)]
        died = [i for i in range(n) if prev_alive[i] and not alive_now[i]]
        if died:
            death_order.append(died)
        survivors = [i for i in range(n) if alive_now[i]]
        npl = np.asarray(nobs["players"])

        for s in act_seats:
            stats = env.players[s].stats
            n_opp_died = sum(1 for d in died if d != s)
            r = event_reward(obs, nobs, s, prev_stats[s], stats,
                             prev_alive[s], alive_now[s], n_opp_died,
                             terminated, terminated and survivors == [s])
            # potential-based shaping (Phi' = 0 once dead / true terminal)
            phi_next = 0.0
            if alive_now[s] and not (terminated):
                phi_next = total_potential(npl, s, cfg["lam_adv"], cfg["lam_e"],
                                           center, dmax_adv)
            r += gamma * phi_next - phi[s]
            phi[s] = phi_next
            # novelty + anti-camp
            if alive_now[s]:
                cpos = (int(npl[s][0]), int(npl[s][1]))
                if cpos not in visited[s]:
                    visited[s].add(cpos)
                    stale[s] = 0
                    r += REWARD["novelty"]
                else:
                    stale[s] += 1
                    if stale[s] > 12 and len(visited[s]) < 25:
                        r += REWARD["camp"]
                    # farmed-out idle: nothing left to farm, enemies alive, but sitting
                    # still -> the exact passivity to punish (pushes it to go hunt).
                    elif stale[s] > 8 and len(survivors) > 1 \
                            and int((np.asarray(nobs["map"]) == BOX).sum()) <= FARMED_OUT_BOXES:
                        r += REWARD["idle_farmed"]
            # true terminal for GAE = died OR env terminated (NOT plain truncation)
            done_gae = 1.0 if (not alive_now[s] or terminated) else 0.0
            mi = step_info[s]
            traj[s]["map"].append(mi[0]); traj[s]["aux"].append(mi[1])
            traj[s]["mask"].append(mi[2]); traj[s]["act"].append(mi[3])
            traj[s]["logp"].append(mi[4]); traj[s]["val"].append(mi[5])
            traj[s]["rew"].append(r); traj[s]["done"].append(done_gae)

        prev_stats = {i: dict(env.players[i].stats) for i in range(n)}
        prev_alive = alive_now
        obs = nobs
        if done_env:
            break

    # bootstrap value for seats cut off by the time limit while still alive
    bootstrap = {}
    for s in learner_seats:
        if not traj[s]["done"]:
            bootstrap[s] = 0.0
            continue
        if traj[s]["done"][-1] >= 1.0:
            bootstrap[s] = 0.0
        else:
            ms, xs = encode_obs(obs, s)
            with torch.no_grad():
                _, v = net(torch.from_numpy(ms).unsqueeze(0).to(device),
                           torch.from_numpy(xs).unsqueeze(0).to(device))
            bootstrap[s] = float(v.item())

    ranks = final_ranks(env, death_order)
    return traj, bootstrap, ranks


def compute_gae(rew, val, done, bootstrap, gamma, lam):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        v_next = bootstrap if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * v_next * nonterminal - val[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + np.asarray(val, dtype=np.float32)
    return adv, ret


# ── PPO update ────────────────────────────────────────────────────────────────
def ppo_update(net, optimizer, data, cfg, device, ent_coef):
    sm = torch.from_numpy(data["map"]).to(device)
    sa = torch.from_numpy(data["aux"]).to(device)
    act = torch.from_numpy(data["act"]).to(device)
    mask = torch.from_numpy(data["mask"]).to(device)
    logp_old = torch.from_numpy(data["logp"]).to(device)
    ret = torch.from_numpy(data["ret"]).to(device)
    adv = torch.from_numpy(data["adv"]).to(device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = sm.shape[0]
    mb = cfg["minibatch"]
    idx = np.arange(N)
    clip = cfg["clip"]
    pg_log, v_log, ent_log, kl_log = [], [], [], []

    for _ in range(cfg["epochs"]):
        np.random.shuffle(idx)
        for start in range(0, N, mb):
            b = idx[start:start + mb]
            bt = torch.from_numpy(b).to(device)
            logits, value = net(sm[bt], sa[bt])
            logp, ent = _logp_entropy(logits, mask[bt], act[bt])
            ratio = torch.exp(logp - logp_old[bt])
            a = adv[bt]
            s1 = ratio * a
            s2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a
            pg_loss = -torch.min(s1, s2).mean()
            v_loss = F.smooth_l1_loss(value, ret[bt])
            ent_loss = -ent.mean()
            loss = pg_loss + cfg["vf_coef"] * v_loss + ent_coef * ent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            with torch.no_grad():
                kl_log.append(float((logp_old[bt] - logp).mean().item()))
            pg_log.append(float(pg_loss.item()))
            v_log.append(float(v_loss.item()))
            ent_log.append(float(ent.mean().item()))

    return (np.mean(pg_log), np.mean(v_log), np.mean(ent_log), np.mean(kl_log))


# ── behaviour-based evaluation (predicts leaderboard, unlike win-rate vs weak) ─
def behaviour_eval(net, map_shape, aux_dim, device, opponents="genius",
                   games=12, max_steps=500, seed0=50_000,
                   horizon=DEFAULT_SHIELD_HORIZON):
    """Behaviour eval for seat 0 vs a fixed opponent field.

    `opponents` is either a single rule name (all 3 seats use it) or a list of 3
    rule names (a MIXED strong panel, e.g. ["hunter","genius","tactical"] — the most
    leaderboard-like field, since the real board is 4 diverse strong agents, not 3
    clones that conveniently kill each other)."""
    pol = NetPPOPolicy({k: v.cpu() for k, v in net.state_dict().items()},
                       map_shape, aux_dim, agent_id=0, horizon=horizon)
    if isinstance(opponents, (list, tuple)):
        seat_cls = {seat: RULE_CLASSES[opponents[k]] for k, seat in enumerate((1, 2, 3))}
    else:
        seat_cls = {seat: RULE_CLASSES[opponents] for seat in (1, 2, 3)}
    cells = boxes = items = bombs = alive_end = wins = 0
    rank_sum = 0
    death_steps = []
    for g in range(games):
        env = BomberEnv(max_steps=max_steps, seed=seed0 + g)
        obs = env.reset(seed=seed0 + g)
        opps = {seat: seat_cls[seat](seat) for seat in (1, 2, 3)}
        visited = set()
        prev_alive = [True] * 4
        death_order = []
        t = 0
        for t in range(max_steps):
            acts = [0, 0, 0, 0]
            if env.players[0].alive:
                acts[0] = int(pol.act(obs))
                if acts[0] == 5:
                    bombs += 1
            for i in (1, 2, 3):
                if env.players[i].alive:
                    try:
                        acts[i] = int(opps[i].act(obs))
                    except Exception:
                        acts[i] = 0
            obs, term, trunc = env.step(acts)
            alive_now = [bool(p.alive) for p in env.players]
            d = [i for i in range(4) if prev_alive[i] and not alive_now[i]]
            if d:
                death_order.append(d)
            prev_alive = alive_now
            if env.players[0].alive:
                visited.add((int(obs["players"][0][0]), int(obs["players"][0][1])))
            if term or trunc:
                break
        s = env.players[0].stats
        cells += len(visited); boxes += int(s["boxes"]); items += int(s["items"])
        me_alive = bool(env.players[0].alive)
        opp_alive = sum(1 for i in (1, 2, 3) if env.players[i].alive)
        alive_end += int(me_alive)
        if me_alive and opp_alive == 0:
            wins += 1
        if not me_alive:
            death_steps.append(t)
        rank_sum += final_ranks(env, death_order)[0]
    n = games
    died_at = (sum(death_steps) / len(death_steps)) if death_steps else None
    return {
        "cells": cells / n, "boxes": boxes / n, "items": items / n,
        "alive_end": alive_end, "wins": wins, "avg_rank": rank_sum / n,
        "bombs": bombs / n, "died_at": died_at, "games": n,
    }


# ── main training loop ───────────────────────────────────────────────────────
def train(
    iters=1500, epi_per_iter=8, max_steps=500, seed=86,
    lr=2.5e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4, minibatch=1024,
    vf_coef=0.5, ent_start=0.03, ent_end=0.005, max_grad_norm=0.5,
    lam_adv=0.05, lam_e=0.1, learner_prob=0.5, safe_mask=True,
    shield_horizon=DEFAULT_SHIELD_HORIZON,
    self_play_after=50, snapshot_every=100, pool_cap=12, self_opp_prob=0.5,
    bc_pretrain=False, bc_games=300, bc_epochs=4, bc_lr=1e-3, bc_sources=None,
    bomb_enemy=None, bomb_box=None, idle_w=None, hunt_boost=None,
    eval_every=100, save_dir="ckpts_ppo", resume=None, warm_from=None,
    opponent_ckpts=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global HUNT_BOOST
    if bomb_enemy is not None:
        REWARD["bomb_enemy"] = float(bomb_enemy)
    if bomb_box is not None:
        REWARD["bomb_box"] = float(bomb_box)
    if idle_w is not None:
        REWARD["idle_farmed"] = -abs(float(idle_w))
    if hunt_boost is not None:
        HUNT_BOOST = float(hunt_boost)
    print(f"[reward] bomb_enemy={REWARD['bomb_enemy']:.2f} bomb_box={REWARD['bomb_box']:.2f} "
          f"idle_farmed={REWARD['idle_farmed']:.3f} hunt_boost={HUNT_BOOST:.1f} lam_e={lam_e}")
    total_epi = iters * epi_per_iter
    print(f"[ppo] device={device} iters={iters} epi/iter={epi_per_iter} "
          f"(~{total_epi} episodes) lr={lr} clip={clip} ent={ent_start}->{ent_end} "
          f"lam_adv={lam_adv} lam_e={lam_e} learner_prob={learner_prob} "
          f"safe_mask={safe_mask} shield_horizon={shield_horizon}")

    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs0 = env.reset(seed=seed)
    ms0, xs0 = encode_obs(obs0, 0)
    map_shape, aux_dim = ms0.shape, xs0.shape[0]
    input_spec = (tuple(int(d) for d in map_shape), int(aux_dim))
    H, W = np.asarray(obs0["map"]).shape
    center = (H // 2, W // 2)
    dmax_adv = float(center[0] + center[1])

    net = PPOActorCritic(map_shape, aux_dim, NUM_ACTIONS).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    start_iter = 0

    if resume:
        ck = torch.load(resume, map_location=device)
        net.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        # optimizer.load_state_dict OVERWRITES lr with the value saved in the
        # checkpoint, silently ignoring --lr. Re-apply the requested lr so a resume
        # can fine-tune at a gentler step size (Adam moment estimates are kept).
        for g in optimizer.param_groups:
            g["lr"] = lr
        start_iter = int(ck.get("iter", 0))
        print(f"[resume] {resume} @ iter{start_iter} lr={lr} (Adam state restored)")
    elif warm_from:
        if warm_from.endswith(".pt"):
            src = torch.jit.load(warm_from, map_location=device).state_dict()
        else:
            ck = torch.load(warm_from, map_location=device)
            src = ck.get("model_state_dict", ck)
        missing, unexpected = net.load_state_dict(src, strict=False)
        print(f"[warm_from] {warm_from} (missing={len(missing)} unexpected={len(unexpected)})")

    # Behaviour-Cloning warm-start: imitate a competent farming/fighting rule agent
    # so PPO starts OUTSIDE the camping basin (with the coupled bomb-then-flee skill).
    # See bc.py for why this is the key fix. Skipped when resuming.
    if bc_pretrain and not resume:
        from agent.dqn_agent.bc import bc_pretrain as _bc
        _bc(net, device, games=bc_games, epochs=bc_epochs, lr=bc_lr,
            max_steps=max_steps, seed0=seed + 999, gamma=gamma,
            lam_adv=lam_adv, lam_e=lam_e, sources=bc_sources)

    cfg = {"gamma": gamma, "lam": lam, "clip": clip, "epochs": epochs,
           "minibatch": minibatch, "vf_coef": vf_coef, "max_grad_norm": max_grad_norm,
           "lam_adv": lam_adv, "lam_e": lam_e}
    # Bounded-horizon survivable shield during training, IDENTICAL to inference.
    # safe_action_mask falls back to the physical mask when genuinely trapped, so the
    # sampler never sees an all-False mask (which would NaN the log-softmax).
    mask_fn = (partial(safe_action_mask, horizon=shield_horizon)
               if safe_mask else physical_action_mask)

    os.makedirs(save_dir, exist_ok=True)
    tag = f"{save_dir}/ppo_selfplay_{iters}it"
    os.makedirs(tag, exist_ok=True)

    pool = []
    # seed the self-play pool with the (BC/warm) initial policy so a camper faces an
    # ACTIVE opponent from iteration 0 (camping then yields ties, not free wins).
    if bc_pretrain or warm_from:
        pool.append(copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()}))
    # Variant C: FIXED strong opponents (e.g. your own champion models). Loading
    # them into the opponent set makes the policy train AGAINST real learned agents
    # from iteration 0 -- not just rule bots + early-weak self snapshots -- so it
    # learns to beat the kind of opponent it actually faces on the leaderboard.
    fixed_opponents = []
    if opponent_ckpts:
        for cp in [s.strip() for s in opponent_ckpts.split(",") if s.strip()]:
            if cp.endswith(".pt"):
                sd = torch.jit.load(cp, map_location="cpu").state_dict()
            else:
                ck_o = torch.load(cp, map_location="cpu")
                sd = ck_o.get("model_state_dict", ck_o)
            fixed_opponents.append({k: v.cpu() for k, v in sd.items()})
        print(f"[opponent_ckpts] loaded {len(fixed_opponents)} fixed champion opponent(s)")
    n_players = len(env.players)
    try:
        from tqdm import trange
        bar = trange(start_iter, iters, desc="ppo")
    except Exception:
        bar = range(start_iter, iters)

    for it in bar:
        frac = it / max(1, iters)
        ent_coef = ent_start + (ent_end - ent_start) * frac

        # collect a rollout of `epi_per_iter` episodes ------------------------
        rollouts = {k: [] for k in ("map", "aux", "act", "mask", "logp", "adv", "ret")}
        ep_rew_sum, ep_count, win_count = 0.0, 0, 0
        for e in range(epi_per_iter):
            # assign seats to learner / opponent
            learner_seats, opponents_map = [], {}
            for seat in range(n_players):
                if random.random() < learner_prob:
                    learner_seats.append(seat)
                else:
                    opp_candidates = pool + fixed_opponents
                    use_self = opp_candidates and it >= self_play_after and random.random() < self_opp_prob
                    if use_self:
                        opponents_map[seat] = NetPPOPolicy(
                            random.choice(opp_candidates), map_shape, aux_dim, seat)
                    else:
                        opponents_map[seat] = RULE_CLASSES[random.choice(STRONG_RULES)](seat)
            if not learner_seats:
                seat = random.randrange(n_players)
                learner_seats.append(seat)
                opponents_map.pop(seat, None)

            traj, bootstrap, ranks = run_episode(
                env, seed + it * epi_per_iter + e, net, device,
                learner_seats, opponents_map, cfg, center, dmax_adv, mask_fn=mask_fn)

            for s in learner_seats:
                tr = traj[s]
                if not tr["rew"]:
                    continue
                adv, ret = compute_gae(tr["rew"], tr["val"], tr["done"],
                                       bootstrap[s], gamma, lam)
                rollouts["map"].append(np.asarray(tr["map"], dtype=np.float32))
                rollouts["aux"].append(np.asarray(tr["aux"], dtype=np.float32))
                rollouts["act"].append(np.asarray(tr["act"], dtype=np.int64))
                rollouts["mask"].append(np.asarray(tr["mask"], dtype=bool))
                rollouts["logp"].append(np.asarray(tr["logp"], dtype=np.float32))
                rollouts["adv"].append(adv)
                rollouts["ret"].append(ret)
                ep_rew_sum += float(np.sum(tr["rew"]))
            ep_count += 1
            if min(ranks[s] for s in learner_seats) == 0:
                win_count += 1

        if not rollouts["map"]:
            continue
        data = {k: np.concatenate(v, axis=0) for k, v in rollouts.items()}
        pg, vl, ent, kl = ppo_update(net, optimizer, data, cfg, device, ent_coef)

        # snapshot for the self-play pool
        if (it + 1) >= self_play_after and (it + 1) % snapshot_every == 0:
            pool.append(copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()}))
            if len(pool) > pool_cap:
                pool.pop(0)

        if hasattr(bar, "set_postfix"):
            bar.set_postfix(rew=f"{ep_rew_sum / max(1, ep_count):5.1f}",
                            ent=f"{ent:.2f}", pg=f"{pg:+.3f}", v=f"{vl:.2f}",
                            kl=f"{kl:+.3f}", win=f"{win_count}/{ep_count}",
                            pool=len(pool), n=data["map"].shape[0])

        if (it + 1) % eval_every == 0:
            ev = behaviour_eval(net, map_shape, aux_dim, device, opponents="genius",
                                horizon=shield_horizon)
            evh = behaviour_eval(net, map_shape, aux_dim, device, opponents="hunter",
                                 seed0=70_000, horizon=shield_horizon)
            # MIXED strong panel = the most leaderboard-predictive metric (3 DIFFERENT
            # strong bots, like the real board) -> low avg_rank + high survive here is
            # what actually correlates with climbing, NOT win-rate vs 3 weak clones.
            evm = behaviour_eval(net, map_shape, aux_dim, device,
                                 opponents=["hunter", "genius", "tactical"],
                                 seed0=90_000, horizon=shield_horizon)
            da = f"{ev['died_at']:.0f}" if ev['died_at'] is not None else "-"
            dam = f"{evm['died_at']:.0f}" if evm['died_at'] is not None else "-"
            print(f"\n[eval it{it+1}] vs genius: cells={ev['cells']:.1f} "
                  f"boxes={ev['boxes']:.1f} items={ev['items']:.1f} "
                  f"survive={ev['alive_end']}/{ev['games']} wins={ev['wins']} "
                  f"avg_rank={ev['avg_rank']:.2f} died@{da} | "
                  f"vs hunter: survive={evh['alive_end']}/{evh['games']} "
                  f"wins={evh['wins']} avg_rank={evh['avg_rank']:.2f} | "
                  f"vs MIX: survive={evm['alive_end']}/{evm['games']} "
                  f"boxes={evm['boxes']:.1f} items={evm['items']:.1f} "
                  f"wins={evm['wins']} avg_rank={evm['avg_rank']:.2f} died@{dam}")
            _save(net, optimizer, it + 1, input_spec, f"{tag}/it{it+1}.pth",
                  export_pt=f"{tag}/model.pt")

    _save(net, optimizer, iters, input_spec, f"{tag}/model.pth",
          export_pt=f"{tag}/model.pt")
    print(f"\n[done] Copy {tag}/model.pt next to agent.py + model.py + ppo.py for submission.")
    return f"{tag}/model.pth", f"{tag}/model.pt"


def _save(net, optimizer, it, input_spec, path, export_pt=None):
    ck = {
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": it, "input_spec": input_spec, "input_shape": input_spec,
        "input_dim": input_spec, "num_actions": NUM_ACTIONS, "arch": "ppo_actor_critic",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ck, path)
    if export_pt:
        was_cuda = next(net.parameters()).is_cuda
        scripted = torch.jit.script(net.cpu().eval())
        scripted.save(export_pt)
        if was_cuda:
            net.cuda()
        net.train()
        print(f"[save] {path}  +  TorchScript {export_pt}")
    else:
        print(f"[save] {path}")


def _b(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def main():
    p = argparse.ArgumentParser("Bomberland PPO self-play trainer")
    p.add_argument("--iters", type=int, default=1500,
                   help="number of PPO iterations (each = epi_per_iter episodes + 1 update)")
    p.add_argument("--epi_per_iter", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_start", type=float, default=0.03)
    p.add_argument("--ent_end", type=float, default=0.005)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--lam_adv", type=float, default=0.05,
                   help="centre-pull shaping weight. LOW by default: with a BC active "
                        "prior the agent already leaves spawn, and the board CENTRE is "
                        "the deadliest tile in a 4-player FFA, so a strong pull there "
                        "just gets it killed. Raise only if it still camps.")
    p.add_argument("--lam_e", type=float, default=0.1,
                   help="enemy-seeking shaping weight (scaled up with power)")
    p.add_argument("--learner_prob", type=float, default=0.5,
                   help="prob each seat is a learner (rest = league opponents)")
    p.add_argument("--safe_mask", type=_b, default=True,
                   help="use the bounded-horizon survivable shield during training "
                        "(recommended). Stops the multi-step bomb traps that kill an "
                        "active policy; identical to the inference shield.")
    p.add_argument("--shield_horizon", type=int, default=DEFAULT_SHIELD_HORIZON,
                   help="how many steps ahead the survivable shield verifies escape. "
                        "1=old light shield (dies in 2-3 step traps); ~6=sweet spot; "
                        "large=DQN-style full BFS (over-conservative -> camping).")
    p.add_argument("--self_play_after", type=int, default=50)
    p.add_argument("--snapshot_every", type=int, default=100)
    p.add_argument("--pool_cap", type=int, default=12)
    p.add_argument("--self_opp_prob", type=float, default=0.5,
                   help="prob an opponent seat is a frozen self snapshot (vs a rule bot)")
    p.add_argument("--bc_pretrain", type=_b, default=False,
                   help="Behaviour-Cloning warm-start from rule agents BEFORE PPO "
                        "(the key fix for camping — strongly recommended)")
    p.add_argument("--bc_games", type=int, default=300)
    p.add_argument("--bc_epochs", type=int, default=4)
    p.add_argument("--bc_lr", type=float, default=1e-3)
    p.add_argument("--bc_sources", default=None,
                   help="comma-separated rule names to clone for the BC prior "
                        "(default genius-majority + 1 hunter, see bc.BC_SOURCES). "
                        "e.g. 'genius,genius,hunter' for more hunting in the prior.")
    # reward knobs (None = keep defaults). For a PROVEN no-hunt run set
    # --bomb_enemy 0.10 --idle_w 0 --hunt_boost 1.0 (matches the it6000 regime).
    p.add_argument("--bomb_enemy", type=float, default=None,
                   help="reward for a bomb whose blast reaches an ENEMY (default 0.20)")
    p.add_argument("--bomb_box", type=float, default=None,
                   help="reward for a bomb whose blast reaches a BOX (default 0.10)")
    p.add_argument("--idle_w", type=float, default=None,
                   help="magnitude of the farmed-out idle penalty (default 0.03; 0=off)")
    p.add_argument("--hunt_boost", type=float, default=None,
                   help="bomb_enemy multiplier when farmed-out (default 2.0; 1.0=off)")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_dir", default="ckpts_ppo")
    p.add_argument("--resume", default=None)
    p.add_argument("--warm_from", default=None)
    p.add_argument("--opponent_ckpts", default=None,
                   help="comma-separated .pth/.pt checkpoints to add as FIXED strong "
                        "opponents in the self-play pool (Variant C: train vs your own "
                        "champion models so it learns to beat real learned agents).")
    args = p.parse_args()

    seed_everything(args.seed)
    train(
        iters=args.iters, epi_per_iter=args.epi_per_iter, max_steps=args.max_steps,
        seed=args.seed, lr=args.lr, gamma=args.gamma, lam=args.lam, clip=args.clip,
        epochs=args.epochs, minibatch=args.minibatch, vf_coef=args.vf_coef,
        ent_start=args.ent_start, ent_end=args.ent_end, max_grad_norm=args.max_grad_norm,
        lam_adv=args.lam_adv, lam_e=args.lam_e, learner_prob=args.learner_prob,
        safe_mask=args.safe_mask, shield_horizon=args.shield_horizon,
        self_play_after=args.self_play_after,
        snapshot_every=args.snapshot_every, pool_cap=args.pool_cap,
        self_opp_prob=args.self_opp_prob, bc_pretrain=args.bc_pretrain,
        bc_games=args.bc_games, bc_epochs=args.bc_epochs, bc_lr=args.bc_lr,
        bc_sources=(args.bc_sources.split(",") if args.bc_sources else None),
        bomb_enemy=args.bomb_enemy, bomb_box=args.bomb_box,
        idle_w=args.idle_w, hunt_boost=args.hunt_boost,
        eval_every=args.eval_every, save_dir=args.save_dir,
        resume=args.resume, warm_from=args.warm_from,
        opponent_ckpts=args.opponent_ckpts,
    )


if __name__ == "__main__":
    main()
