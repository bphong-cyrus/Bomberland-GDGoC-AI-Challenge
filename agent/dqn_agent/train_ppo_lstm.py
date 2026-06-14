"""
Recurrent (LSTM) PPO self-play trainer for the Bomberland agent.

SEPARATE TRACK from `train_ppo.py` -- it does NOT touch the feedforward pipeline.
It REUSES, by import (no copy-paste of logic):

  * the reward + shaping (`REWARD`, `event_reward`, `total_potential`) and the
    final-ranking helper, GAE (`compute_gae`), the masked log-prob/entropy
    (`_logp_entropy`), the shield mask functions, and the STRONG_RULES opponent
    league -- all from `train_ppo.py`;
  * the danger-aware encoder + constants from `model.py`;
  * the recurrent net + the bounded-horizon shield from `ppo_lstm.py` / `ppo.py`;
  * the SAME behaviour-cloning teacher dataset collector from `bc.py`.

What is genuinely NEW here (and only here) is the SEQUENCE handling:
  * rollouts are collected as FULL-EPISODE sequences, with the LSTM hidden state
    carried across steps within an episode (and reset between episodes);
  * the PPO update runs `net.seq()` over each episode from a ZERO hidden state,
    recomputes masked log-probs / entropy / values, and applies the same clipped
    surrogate + value loss + entropy bonus as `ppo_update`.

BC WARM-START CHOICE (v1, documented):
  We run behaviour cloning in FEEDFORWARD mode -- i.e. each teacher (state,
  action, return) sample is treated as an INDEPENDENT length-1 sequence, so the
  LSTM sees a zero hidden state on every BC sample. This warm-starts ONLY the
  encoder + actor/critic heads (the parts shared with the feedforward model) to
  escape the camping basin (from-scratch PPO camps -- see bc.py). It deliberately
  does NOT try to teach the recurrence from the rule teacher (the teacher is
  memoryless, so there is no temporal signal to clone). PPO then learns the
  recurrent dynamics on top of the BC-warm-started encoder. This mirrors bc.py's
  proven warm-start and keeps v1 simple and bug-free.

Run from the repository ROOT:
    python -m agent.dqn_agent.train_ppo_lstm --iters 1500 --epi_per_iter 8 \
        --bc_pretrain 1 --bc_games 400 --bc_epochs 5

The exported `model.pt` (TorchScript of the recurrent actor-critic) drops into
the flat submission zip next to agent.py + model.py + ppo.py + ppo_lstm.py.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
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

from agent.dqn_agent.model import encode_obs, NUM_ACTIONS                 # noqa: E402
from agent.dqn_agent.ppo import (                                         # noqa: E402
    physical_action_mask, safe_action_mask, shielded_action,
    DEFAULT_SHIELD_HORIZON,
)
from agent.dqn_agent.ppo_lstm import PPOLSTMActorCritic                   # noqa: E402
# REUSE reward, shaping, ranking, GAE, masked logp/entropy, opponents -- imported,
# never duplicated.
from agent.dqn_agent.train_ppo import (                                   # noqa: E402
    REWARD, event_reward, total_potential, final_ranks, compute_gae,
    _logp_entropy, seed_everything, RULE_CLASSES, STRONG_RULES,
)
from engine.game import BomberEnv                                         # noqa: E402


# ── frozen recurrent snapshot as a self-play opponent ─────────────────────────
class NetLSTMPolicy:
    """A frozen recurrent policy used as a self-play opponent. Holds its OWN
    hidden state across the episode; create a fresh instance per episode (so the
    hidden state starts zeroed, matching game start)."""

    def __init__(self, state_dict, map_shape, aux_dim, agent_id, lstm_hidden=256,
                 horizon=DEFAULT_SHIELD_HORIZON):
        self.net = PPOLSTMActorCritic(map_shape, aux_dim, NUM_ACTIONS,
                                      lstm_hidden=lstm_hidden)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.agent_id = int(agent_id)
        self.horizon = horizon
        self.hidden = None

    def act(self, obs):
        ms, xs = encode_obs(obs, self.agent_id)
        with torch.no_grad():
            logits, _v, self.hidden = self.net.step(
                torch.from_numpy(ms).unsqueeze(0),
                torch.from_numpy(xs).unsqueeze(0), self.hidden)
        return shielded_action(obs, self.agent_id, logits[0].numpy(),
                               horizon=self.horizon)


# ── one self-play episode: collect FULL-EPISODE sequences for learner seats ──
def run_episode_lstm(env, seed, net, device, learner_seats, opponents, cfg,
                     center, dmax_adv, mask_fn=physical_action_mask):
    """Mirror of train_ppo.run_episode, but the learner carries a PER-SEAT LSTM
    hidden state across the whole episode (reset to zero at episode start). The
    reward / shaping / novelty / anti-camp logic is IDENTICAL (reused helpers).

    Returns per-seat trajectory dicts (lists in time order), a bootstrap value
    per seat, and the final ranks. Sequences are kept intact (not flattened) so
    the PPO update can replay them through the recurrence.
    """
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

    # PER-SEAT recurrent hidden state for the learner (carried across steps).
    hidden = {s: None for s in learner_seats}

    while True:
        actions = [0] * n

        # ---- learner seats: per-seat recurrent forward, masked sampling ------
        act_seats = [s for s in learner_seats if env.players[s].alive]
        step_info = {}
        for s in act_seats:
            ms, xs = encode_obs(obs, s)
            mask = mask_fn(obs, s)
            mt = torch.from_numpy(ms).unsqueeze(0).to(device)        # [1,C,H,W]
            xt = torch.from_numpy(xs).unsqueeze(0).to(device)        # [1,A]
            mk = torch.from_numpy(mask).unsqueeze(0).to(device)      # [1,A]
            with torch.no_grad():
                logits, value, hidden[s] = net.step(mt, xt, hidden[s])
                masked = torch.where(mk, logits, torch.full_like(logits, neg))
                logp_all = F.log_softmax(masked, dim=1)
                probs = logp_all.exp()
                sampled = torch.multinomial(probs, 1).squeeze(1)
                logp = logp_all.gather(1, sampled.unsqueeze(1)).squeeze(1)
            a = int(sampled.item())
            actions[s] = a
            step_info[s] = (ms, xs, mask, a, float(logp.item()), float(value.item()))

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
            phi_next = 0.0
            if alive_now[s] and not terminated:
                phi_next = total_potential(npl, s, cfg["lam_adv"], cfg["lam_e"],
                                           center, dmax_adv)
            r += gamma * phi_next - phi[s]
            phi[s] = phi_next
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

    # bootstrap value for seats cut off by the time limit while still alive.
    # NOTE: the correct recurrent bootstrap uses the hidden state AFTER the last
    # collected step, which we kept in hidden[s] -- so the value estimate is on
    # the same recurrent context the policy actually had.
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
                _, v, _ = net.step(torch.from_numpy(ms).unsqueeze(0).to(device),
                                   torch.from_numpy(xs).unsqueeze(0).to(device),
                                   hidden[s])
            bootstrap[s] = float(v.item())

    ranks = final_ranks(env, death_order)
    return traj, bootstrap, ranks


# ── PPO update over full-episode sequences (replays the recurrence) ──────────
def ppo_update_seq(net, optimizer, episodes, cfg, device, ent_coef):
    """`episodes` is a list of per-episode dicts with keys
    map/aux/act/mask/logp/adv/ret (each a numpy array, time-ordered). Each
    episode is replayed through `net.seq` from a ZERO hidden state, exactly as a
    recurrent PPO update requires. Losses mirror train_ppo.ppo_update.

    Advantage normalisation is done GLOBALLY across all episodes (as in the
    feedforward trainer), then sliced per episode for the sequence forward.
    """
    # global advantage normalisation (match feedforward trainer's statistics)
    all_adv = np.concatenate([e["adv"] for e in episodes], axis=0)
    adv_mean = float(all_adv.mean())
    adv_std = float(all_adv.std()) + 1e-8

    clip = cfg["clip"]
    mb_epi = cfg["seq_minibatch_epi"]
    n_epi = len(episodes)
    order = np.arange(n_epi)
    pg_log, v_log, ent_log, kl_log = [], [], [], []

    for _ in range(cfg["epochs"]):
        np.random.shuffle(order)
        for start in range(0, n_epi, mb_epi):
            batch_ids = order[start:start + mb_epi]
            optimizer.zero_grad(set_to_none=True)
            # accumulate the (length-weighted) loss across the episodes in the
            # minibatch, then take one optimizer step.
            tot_loss = torch.zeros((), device=device)
            tot_steps = 0
            for ei in batch_ids:
                ep = episodes[int(ei)]
                T = ep["map"].shape[0]
                if T == 0:
                    continue
                # shape to (T, B=1, ...)
                m = torch.from_numpy(ep["map"]).unsqueeze(1).to(device)   # [T,1,C,H,W]
                x = torch.from_numpy(ep["aux"]).unsqueeze(1).to(device)   # [T,1,A]
                logits_tb, value_tb, _ = net.seq(m, x, None)              # [T,1,A],[T,1]
                logits = logits_tb.squeeze(1)                            # [T,A]
                value = value_tb.squeeze(1)                              # [T]

                mask = torch.from_numpy(ep["mask"]).to(device)           # [T,A] bool
                act = torch.from_numpy(ep["act"]).to(device)             # [T]
                logp_old = torch.from_numpy(ep["logp"]).to(device)       # [T]
                ret = torch.from_numpy(ep["ret"]).to(device)             # [T]
                adv = torch.from_numpy(ep["adv"]).to(device)             # [T]
                adv = (adv - adv_mean) / adv_std

                logp, ent = _logp_entropy(logits, mask, act)
                ratio = torch.exp(logp - logp_old)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
                pg_loss = -torch.min(s1, s2).mean()
                v_loss = F.smooth_l1_loss(value, ret)
                ent_loss = -ent.mean()
                loss = pg_loss + cfg["vf_coef"] * v_loss + ent_coef * ent_loss

                tot_loss = tot_loss + loss * T
                tot_steps += T

                pg_log.append(float(pg_loss.item()))
                v_log.append(float(v_loss.item()))
                ent_log.append(float(ent.mean().item()))
                with torch.no_grad():
                    kl_log.append(float((logp_old - logp).mean().item()))

            if tot_steps == 0:
                continue
            (tot_loss / tot_steps).backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
            optimizer.step()

    if not pg_log:
        return 0.0, 0.0, 0.0, 0.0
    return (float(np.mean(pg_log)), float(np.mean(v_log)),
            float(np.mean(ent_log)), float(np.mean(kl_log)))


# ── BC warm-start in FEEDFORWARD mode (hidden=zeros per sample) ──────────────
def bc_pretrain_lstm(net, device, games=400, epochs=4, lr=1e-3, batch=512,
                     max_steps=500, seed0=12345, gamma=0.99, lam_adv=0.05,
                     lam_e=0.1, sources=None):
    """REUSE bc.collect (the rule-teacher dataset) and clone into the recurrent
    net WITHOUT recurrence: every sample is a length-1 sequence (zero hidden), so
    only the encoder + heads are warm-started. See module docstring for why."""
    from agent.dqn_agent.bc import collect as _bc_collect

    env = BomberEnv(max_steps=max_steps, seed=seed0)
    obs0 = env.reset(seed=seed0)
    H, W = np.asarray(obs0["map"]).shape
    center = (H // 2, W // 2)
    dmax_adv = float(center[0] + center[1])
    print(f"[bc-lstm] collecting {games} games of rule self-play (feedforward "
          f"clone; hidden=zeros per sample) ...")
    data = _bc_collect(games, max_steps, seed0, gamma, lam_adv, lam_e, center,
                       dmax_adv, sources=sources)
    N = data["map"].shape[0]
    print(f"[bc-lstm] collected {N} (state,action) pairs; training BC ...")

    opt = optim.Adam(net.parameters(), lr=lr)
    sm = torch.from_numpy(data["map"])
    sa = torch.from_numpy(data["aux"])
    act = torch.from_numpy(data["act"])
    ret = torch.from_numpy(data["ret"])
    idx = np.arange(N)
    vf_coef = 0.5
    net.train()
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot_ce, tot_v, tot_acc, nb = 0.0, 0.0, 0.0, 0
        for s in range(0, N, batch):
            b = idx[s:s + batch]
            bt = torch.from_numpy(b)
            m = sm[bt].to(device); x = sa[bt].to(device)
            a = act[bt].to(device); g = ret[bt].to(device)
            # length-1 sequence: [T=1, B=batch, ...] from a zero hidden state.
            logits_tb, value_tb, _ = net.seq(m.unsqueeze(0), x.unsqueeze(0), None)
            logits = logits_tb.squeeze(0)            # [B, A]
            value = value_tb.squeeze(0)              # [B]
            ce = F.cross_entropy(logits, a)
            vloss = F.smooth_l1_loss(value, g)
            loss = ce + vf_coef * vloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_ce += float(ce.item()); tot_v += float(vloss.item())
            tot_acc += float((logits.argmax(1) == a).float().mean().item()); nb += 1
        print(f"[bc-lstm] epoch {ep+1}/{epochs}  CE={tot_ce/nb:.3f}  "
              f"acc={tot_acc/nb:.3f}  Vloss={tot_v/nb:.3f}  (N={N})")
    net.eval()
    return net


# ── behaviour-based eval for the recurrent policy (mirrors train_ppo) ─────────
def behaviour_eval_lstm(net, map_shape, aux_dim, device, lstm_hidden,
                        opponents="genius", games=12, max_steps=500,
                        seed0=50_000, horizon=DEFAULT_SHIELD_HORIZON):
    """Behaviour eval for seat 0 vs a fixed opponent field. Uses a fresh
    LSTMShieldedActor per game (hidden reset at game start)."""
    from agent.dqn_agent.ppo_lstm import LSTMShieldedActor

    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    eval_net = PPOLSTMActorCritic(map_shape, aux_dim, NUM_ACTIONS,
                                  lstm_hidden=lstm_hidden)
    eval_net.load_state_dict(sd)
    eval_net.eval()

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
        actor = LSTMShieldedActor(eval_net, 0, horizon=horizon)
        actor.reset()
        opps = {seat: seat_cls[seat](seat) for seat in (1, 2, 3)}
        visited = set()
        prev_alive = [True] * 4
        death_order = []
        t = 0
        for t in range(max_steps):
            acts = [0, 0, 0, 0]
            if env.players[0].alive:
                acts[0] = int(actor.act(obs))
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


# ── checkpoint save + TorchScript export (mirror train_ppo._save) ─────────────
def _save_lstm(net, optimizer, it, input_spec, lstm_hidden, path, export_pt=None):
    ck = {
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": it, "input_spec": input_spec, "input_shape": input_spec,
        "input_dim": input_spec, "num_actions": NUM_ACTIONS,
        "arch": "ppo_lstm_actor_critic", "lstm_hidden": int(lstm_hidden),
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


# ── main training loop ───────────────────────────────────────────────────────
def train(
    iters=1500, epi_per_iter=8, max_steps=500, seed=86,
    lr=2.5e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4, seq_minibatch_epi=4,
    vf_coef=0.5, ent_start=0.03, ent_end=0.005, max_grad_norm=0.5,
    lam_adv=0.05, lam_e=0.1, learner_prob=0.5, safe_mask=True,
    shield_horizon=DEFAULT_SHIELD_HORIZON, lstm_hidden=256,
    self_play_after=50, snapshot_every=100, pool_cap=12, self_opp_prob=0.5,
    bc_pretrain=False, bc_games=300, bc_epochs=4, bc_lr=1e-3,
    eval_every=100, save_dir="ckpts_lstm", resume=None, warm_from=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_epi = iters * epi_per_iter
    print(f"[ppo-lstm] device={device} iters={iters} epi/iter={epi_per_iter} "
          f"(~{total_epi} episodes) lr={lr} clip={clip} ent={ent_start}->{ent_end} "
          f"lstm_hidden={lstm_hidden} lam_adv={lam_adv} lam_e={lam_e} "
          f"learner_prob={learner_prob} safe_mask={safe_mask} "
          f"shield_horizon={shield_horizon}")

    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs0 = env.reset(seed=seed)
    ms0, xs0 = encode_obs(obs0, 0)
    map_shape, aux_dim = ms0.shape, xs0.shape[0]
    input_spec = (tuple(int(d) for d in map_shape), int(aux_dim))
    H, W = np.asarray(obs0["map"]).shape
    center = (H // 2, W // 2)
    dmax_adv = float(center[0] + center[1])

    net = PPOLSTMActorCritic(map_shape, aux_dim, NUM_ACTIONS,
                             lstm_hidden=lstm_hidden).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    start_iter = 0

    if resume:
        ck = torch.load(resume, map_location=device)
        net.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        # honor --lr on resume (optimizer.load_state_dict restores the saved lr).
        for grp in optimizer.param_groups:
            grp["lr"] = lr
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

    # BC warm-start (feedforward mode). Skipped on resume.
    if bc_pretrain and not resume:
        bc_pretrain_lstm(net, device, games=bc_games, epochs=bc_epochs, lr=bc_lr,
                         max_steps=max_steps, seed0=seed + 999, gamma=gamma,
                         lam_adv=lam_adv, lam_e=lam_e)

    cfg = {"gamma": gamma, "lam": lam, "clip": clip, "epochs": epochs,
           "seq_minibatch_epi": seq_minibatch_epi, "vf_coef": vf_coef,
           "max_grad_norm": max_grad_norm, "lam_adv": lam_adv, "lam_e": lam_e}
    mask_fn = (partial(safe_action_mask, horizon=shield_horizon)
               if safe_mask else physical_action_mask)

    os.makedirs(save_dir, exist_ok=True)
    tag = f"{save_dir}/ppo_lstm_{iters}it"
    os.makedirs(tag, exist_ok=True)

    pool = []
    if bc_pretrain or warm_from:
        pool.append(copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()}))

    n_players = len(env.players)
    try:
        from tqdm import trange
        bar = trange(start_iter, iters, desc="ppo-lstm")
    except Exception:
        bar = range(start_iter, iters)

    for it in bar:
        frac = it / max(1, iters)
        ent_coef = ent_start + (ent_end - ent_start) * frac

        episodes = []           # list of per-episode dicts (sequences kept intact)
        ep_rew_sum, ep_count, win_count = 0.0, 0, 0
        for e in range(epi_per_iter):
            learner_seats, opponents_map = [], {}
            for seat in range(n_players):
                if random.random() < learner_prob:
                    learner_seats.append(seat)
                else:
                    use_self = pool and it >= self_play_after and random.random() < self_opp_prob
                    if use_self:
                        opponents_map[seat] = NetLSTMPolicy(
                            random.choice(pool), map_shape, aux_dim, seat,
                            lstm_hidden=lstm_hidden, horizon=shield_horizon)
                    else:
                        opponents_map[seat] = RULE_CLASSES[random.choice(STRONG_RULES)](seat)
            if not learner_seats:
                seat = random.randrange(n_players)
                learner_seats.append(seat)
                opponents_map.pop(seat, None)

            traj, bootstrap, ranks = run_episode_lstm(
                env, seed + it * epi_per_iter + e, net, device,
                learner_seats, opponents_map, cfg, center, dmax_adv, mask_fn=mask_fn)

            for s in learner_seats:
                tr = traj[s]
                if not tr["rew"]:
                    continue
                adv, ret = compute_gae(tr["rew"], tr["val"], tr["done"],
                                       bootstrap[s], gamma, lam)
                episodes.append({
                    "map": np.asarray(tr["map"], dtype=np.float32),
                    "aux": np.asarray(tr["aux"], dtype=np.float32),
                    "act": np.asarray(tr["act"], dtype=np.int64),
                    "mask": np.asarray(tr["mask"], dtype=bool),
                    "logp": np.asarray(tr["logp"], dtype=np.float32),
                    "adv": adv, "ret": ret,
                })
                ep_rew_sum += float(np.sum(tr["rew"]))
            ep_count += 1
            if min(ranks[s] for s in learner_seats) == 0:
                win_count += 1

        if not episodes:
            continue
        pg, vl, ent, kl = ppo_update_seq(net, optimizer, episodes, cfg, device, ent_coef)

        if (it + 1) >= self_play_after and (it + 1) % snapshot_every == 0:
            pool.append(copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()}))
            if len(pool) > pool_cap:
                pool.pop(0)

        n_steps = sum(e["map"].shape[0] for e in episodes)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(rew=f"{ep_rew_sum / max(1, ep_count):5.1f}",
                            ent=f"{ent:.2f}", pg=f"{pg:+.3f}", v=f"{vl:.2f}",
                            kl=f"{kl:+.3f}", win=f"{win_count}/{ep_count}",
                            pool=len(pool), n=n_steps)

        if (it + 1) % eval_every == 0:
            ev = behaviour_eval_lstm(net, map_shape, aux_dim, device, lstm_hidden,
                                     opponents="genius", horizon=shield_horizon)
            evm = behaviour_eval_lstm(net, map_shape, aux_dim, device, lstm_hidden,
                                      opponents=["hunter", "genius", "tactical"],
                                      seed0=90_000, horizon=shield_horizon)
            da = f"{ev['died_at']:.0f}" if ev['died_at'] is not None else "-"
            dam = f"{evm['died_at']:.0f}" if evm['died_at'] is not None else "-"
            print(f"\n[eval it{it+1}] vs genius: cells={ev['cells']:.1f} "
                  f"boxes={ev['boxes']:.1f} items={ev['items']:.1f} "
                  f"survive={ev['alive_end']}/{ev['games']} wins={ev['wins']} "
                  f"avg_rank={ev['avg_rank']:.2f} died@{da} | "
                  f"vs MIX: survive={evm['alive_end']}/{evm['games']} "
                  f"boxes={evm['boxes']:.1f} items={evm['items']:.1f} "
                  f"wins={evm['wins']} avg_rank={evm['avg_rank']:.2f} died@{dam}")
            _save_lstm(net, optimizer, it + 1, input_spec, lstm_hidden,
                       f"{tag}/it{it+1}.pth", export_pt=f"{tag}/model.pt")

    _save_lstm(net, optimizer, iters, input_spec, lstm_hidden,
               f"{tag}/model.pth", export_pt=f"{tag}/model.pt")
    print(f"\n[done] Copy {tag}/model.pt next to agent.py + model.py + ppo.py + "
          f"ppo_lstm.py for submission.")
    return f"{tag}/model.pth", f"{tag}/model.pt"


def _b(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def main():
    p = argparse.ArgumentParser("Bomberland recurrent (LSTM) PPO self-play trainer")
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--epi_per_iter", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam_adv", type=float, default=0.05,
                   help="centre-pull shaping weight (same meaning as train_ppo).")
    p.add_argument("--lam_e", type=float, default=0.1,
                   help="enemy-seeking shaping weight (scaled up with power).")
    # NOTE: --lam (GAE lambda) and --lam_adv/--lam_e (shaping) are distinct, as in
    # train_ppo.py. GAE lambda is --lam_gae here to avoid the clash; default 0.95.
    p.add_argument("--lam_gae", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seq_minibatch_epi", type=int, default=4,
                   help="how many full episodes per PPO minibatch (sequence PPO "
                        "batches whole episodes, not flat steps)")
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_start", type=float, default=0.03)
    p.add_argument("--ent_end", type=float, default=0.005)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--learner_prob", type=float, default=0.5)
    p.add_argument("--safe_mask", type=_b, default=True,
                   help="use the bounded-horizon survivable shield during training "
                        "(identical to inference). Recommended.")
    p.add_argument("--shield_horizon", type=int, default=DEFAULT_SHIELD_HORIZON)
    p.add_argument("--lstm_hidden", type=int, default=256)
    p.add_argument("--self_play_after", type=int, default=50)
    p.add_argument("--snapshot_every", type=int, default=100)
    p.add_argument("--pool_cap", type=int, default=12)
    p.add_argument("--self_opp_prob", type=float, default=0.5)
    p.add_argument("--bc_pretrain", type=_b, default=False,
                   help="Behaviour-Cloning warm-start (feedforward mode) BEFORE "
                        "PPO -- the key fix for camping. Strongly recommended.")
    p.add_argument("--bc_games", type=int, default=300)
    p.add_argument("--bc_epochs", type=int, default=4)
    p.add_argument("--bc_lr", type=float, default=1e-3)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_dir", default="ckpts_lstm")
    p.add_argument("--resume", default=None)
    p.add_argument("--warm_from", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    train(
        iters=args.iters, epi_per_iter=args.epi_per_iter, max_steps=args.max_steps,
        seed=args.seed, lr=args.lr, gamma=args.gamma, lam=args.lam_gae, clip=args.clip,
        epochs=args.epochs, seq_minibatch_epi=args.seq_minibatch_epi,
        vf_coef=args.vf_coef, ent_start=args.ent_start, ent_end=args.ent_end,
        max_grad_norm=args.max_grad_norm, lam_adv=args.lam_adv, lam_e=args.lam_e,
        learner_prob=args.learner_prob, safe_mask=args.safe_mask,
        shield_horizon=args.shield_horizon, lstm_hidden=args.lstm_hidden,
        self_play_after=args.self_play_after, snapshot_every=args.snapshot_every,
        pool_cap=args.pool_cap, self_opp_prob=args.self_opp_prob,
        bc_pretrain=args.bc_pretrain, bc_games=args.bc_games,
        bc_epochs=args.bc_epochs, bc_lr=args.bc_lr, eval_every=args.eval_every,
        save_dir=args.save_dir, resume=args.resume, warm_from=args.warm_from,
    )


if __name__ == "__main__":
    main()
