"""
Behaviour-Cloning warm-start for the PPO agent (docs section 4.5).

WHY: PPO trained from scratch collapses onto CAMPING — because (a) with the safety
mask removed it learns "bombing => death" (it bombs then stands in its own blast),
so it suppresses productive bombing, and (b) the rule opponents die early so passive
survival already "wins" => no pressure to farm. Starting PPO from a RANDOM policy, it
never escapes that camping basin.

FIX: pre-train the actor to IMITATE a competent rule agent (GeniusRuleAgent: escapes
danger, collects items, bombs enemies, farms boxes, pressures opponents). This drops
the policy OUTSIDE the camping basin with the coupled "bomb-then-flee" skill already
baked in. PPO then fine-tunes from this active prior.

We clone BOTH heads:
  * policy head  <- cross-entropy on the rule agent's chosen action (the behaviour),
  * value head   <- regression on Monte-Carlo discounted returns under the SAME
    reward the PPO trainer uses, so PPO starts with a sane critic (avoids the large
    early-advantage errors a random value head would cause).

Usage is via train_ppo.py --bc_pretrain 1 (recommended), or standalone:
    python -m agent.dqn_agent.bc --games 400 --epochs 4 --out ckpts_ppo/bc_init.pth
then PPO:  python -m agent.dqn_agent.train_ppo --warm_from ckpts_ppo/bc_init.pth ...
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.dqn_agent.model import encode_obs, NUM_ACTIONS              # noqa: E402
from agent.dqn_agent.ppo import PPOActorCritic                          # noqa: E402
from agent.dqn_agent.train_ppo import (                                 # noqa: E402
    event_reward, total_potential, REWARD, RULE_CLASSES, _save, seed_everything,
)
from engine.game import BomberEnv                                       # noqa: E402


# default cloning field: genius-MAJORITY (farm-then-survive, the base behaviour we
# want) PLUS one HunterAgent so the prior also contains the coupled "approach enemy ->
# bomb it -> flee" skill. Without a hunter in the clone set the policy only ever sees
# farming frames and goes passive/idle once the board is farmed out (the reported bug);
# the hunter injects the late-game kill behaviour. Still genius-majority so it stays
# farm-biased early (reckless pure-hunter priors die ~step 100). Tune via --bc_sources.
BC_SOURCES = ["genius", "genius", "genius", "hunter", "box_farmer", "tactical"]


def collect(games, max_steps, seed0, gamma, lam_adv, lam_e, center, dmax_adv,
            sources=None, progress=True):
    """Run rule agents on all 4 seats and collect (state, action, MC-return) for
    every alive seat at every step. Reward matches train_ppo (event + potential
    shaping + novelty/anti-camp) so the value targets are PPO-consistent."""
    sources = sources or BC_SOURCES
    maps, auxs, acts, rets = [], [], [], []
    bar = range(games)
    if progress:
        try:
            from tqdm import trange
            bar = trange(games, desc="bc-collect")
        except Exception:
            pass

    for g in bar:
        env = BomberEnv(max_steps=max_steps, seed=seed0 + g)
        obs = env.reset(seed=seed0 + g)
        n = len(env.players)
        pol = {i: RULE_CLASSES[random.choice(sources)](i) for i in range(n)}
        prev_stats = {i: dict(env.players[i].stats) for i in range(n)}
        prev_alive = [True] * n
        phi = {i: total_potential(np.asarray(obs["players"]), i, lam_adv, lam_e,
                                  center, dmax_adv, grid=np.asarray(obs["map"]))
               for i in range(n)}
        visited = {i: {(int(obs["players"][i][0]), int(obs["players"][i][1]))}
                   for i in range(n)}
        stale = {i: 0 for i in range(n)}
        # per-seat lists of (map, aux, act, reward)
        traj = {i: [] for i in range(n)}

        while True:
            actions = [0] * n
            step_enc = {}
            for i in range(n):
                if env.players[i].alive:
                    ms, xs = encode_obs(obs, i)
                    try:
                        a = int(pol[i].act(obs))
                    except Exception:
                        a = 0
                    actions[i] = a
                    step_enc[i] = (ms, xs, a)
            nobs, terminated, truncated = env.step(actions)
            alive_now = [bool(env.players[i].alive) for i in range(n)]
            died = [i for i in range(n) if prev_alive[i] and not alive_now[i]]
            survivors = [i for i in range(n) if alive_now[i]]
            npl = np.asarray(nobs["players"])

            for i in step_enc:
                stats = env.players[i].stats
                n_opp_died = sum(1 for d in died if d != i)
                r = event_reward(obs, nobs, i, prev_stats[i], stats,
                                 prev_alive[i], alive_now[i], n_opp_died,
                                 terminated, terminated and survivors == [i])
                phi_next = 0.0
                if alive_now[i] and not terminated:
                    phi_next = total_potential(npl, i, lam_adv, lam_e, center, dmax_adv,
                                               grid=np.asarray(nobs["map"]))
                r += gamma * phi_next - phi[i]
                phi[i] = phi_next
                if alive_now[i]:
                    cpos = (int(npl[i][0]), int(npl[i][1]))
                    if cpos not in visited[i]:
                        visited[i].add(cpos); stale[i] = 0; r += REWARD["novelty"]
                    else:
                        stale[i] += 1
                        if stale[i] > 12 and len(visited[i]) < 25:
                            r += REWARD["camp"]
                ms, xs, a = step_enc[i]
                traj[i].append((ms, xs, a, float(r)))

            prev_stats = {i: dict(env.players[i].stats) for i in range(n)}
            prev_alive = alive_now
            obs = nobs
            if terminated or truncated:
                break

        # MC discounted returns per seat
        for i in range(n):
            G = 0.0
            recs = traj[i]
            for t in reversed(range(len(recs))):
                G = recs[t][3] + gamma * G
                maps.append(recs[t][0]); auxs.append(recs[t][1])
                acts.append(recs[t][2]); rets.append(G)

    data = {
        "map": np.asarray(maps, dtype=np.float32),
        "aux": np.asarray(auxs, dtype=np.float32),
        "act": np.asarray(acts, dtype=np.int64),
        "ret": np.asarray(rets, dtype=np.float32),
    }
    return data


def train_bc(net, data, epochs, lr, batch, device, vf_coef=0.5, log=True):
    """Supervised: policy head <- cross-entropy(action); value head <- MSE(return)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sm = torch.from_numpy(data["map"])
    sa = torch.from_numpy(data["aux"])
    act = torch.from_numpy(data["act"])
    ret = torch.from_numpy(data["ret"])
    N = sm.shape[0]
    idx = np.arange(N)
    net.train()
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot_ce, tot_v, tot_acc, nb = 0.0, 0.0, 0.0, 0
        for s in range(0, N, batch):
            b = idx[s:s + batch]
            bt = torch.from_numpy(b)
            m = sm[bt].to(device); x = sa[bt].to(device)
            a = act[bt].to(device); g = ret[bt].to(device)
            logits, value = net(m, x)
            ce = F.cross_entropy(logits, a)
            vloss = F.smooth_l1_loss(value, g)
            loss = ce + vf_coef * vloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_ce += float(ce.item()); tot_v += float(vloss.item())
            tot_acc += float((logits.argmax(1) == a).float().mean().item()); nb += 1
        if log:
            print(f"[bc] epoch {ep+1}/{epochs}  CE={tot_ce/nb:.3f}  "
                  f"acc={tot_acc/nb:.3f}  Vloss={tot_v/nb:.3f}  (N={N})")
    net.eval()
    return net


def bc_pretrain(net, device, games=400, epochs=4, lr=1e-3, batch=512, max_steps=500,
                seed0=12345, gamma=0.99, lam_adv=0.1, lam_e=0.1, sources=None):
    """Convenience: collect a cloning dataset and train `net` in place."""
    env = BomberEnv(max_steps=max_steps, seed=seed0)
    obs0 = env.reset(seed=seed0)
    H, W = np.asarray(obs0["map"]).shape
    center = (H // 2, W // 2)
    dmax_adv = float(center[0] + center[1])
    print(f"[bc] collecting {games} games of rule self-play ({sources or BC_SOURCES}) ...")
    data = collect(games, max_steps, seed0, gamma, lam_adv, lam_e, center, dmax_adv,
                   sources=sources)
    print(f"[bc] collected {data['map'].shape[0]} (state,action) pairs; training BC ...")
    train_bc(net, data, epochs, lr, batch, device)
    return net


def main():
    p = argparse.ArgumentParser("Behaviour-Cloning warm-start for the PPO actor-critic")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--sources", default=None,
                   help="comma-separated rule names to clone (default genius-heavy mix)")
    p.add_argument("--out", default="ckpts_ppo/bc_init.pth")
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    obs0 = env.reset(seed=args.seed)
    ms0, xs0 = encode_obs(obs0, 0)
    map_shape, aux_dim = ms0.shape, xs0.shape[0]
    input_spec = (tuple(int(d) for d in map_shape), int(aux_dim))
    net = PPOActorCritic(map_shape, aux_dim, NUM_ACTIONS).to(device)

    sources = args.sources.split(",") if args.sources else None
    bc_pretrain(net, device, games=args.games, epochs=args.epochs, lr=args.lr,
                batch=args.batch, max_steps=args.max_steps, seed0=args.seed,
                sources=sources)

    _save(net, torch.optim.Adam(net.parameters()), 0, input_spec, args.out,
          export_pt=None)
    print(f"[bc] saved warm-start checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
