"""
Entropy KILL-GATE probe: measure how PEAKED (over-confident) a PPO policy is.

Why this exists: win/survival vs rule bots does NOT predict the real leaderboard
(it misled us 4x). The discriminating signal is the policy's ENTROPY. Over-trained
models collapse to a near one-hot policy (entropy -> 0): they play brittle / "khung"
(hesitant, freeze when the shield masks their single favoured action) and generalize
WORSE vs the diverse real board. The settled-best it1000 sits at a HEALTHY soft
optimum (~0.33 nats mean / ~0.16 median); every fine-tune that regressed had lower
entropy (it2500 ~0.21, it4000 ~0.14, median near one-hot).

Use it as a pre-submission gate: REJECT a candidate whose mean policy entropy is
below it1000's level -- it has over-peaked like the regressed fine-tunes and will
likely rank BELOW it1000 on the settled board.

Run from the repo ROOT:
    python -m agent.dqn_agent.entropy_probe --model "agent/dqn_agent/tập train/it1000.pth"
    python -m agent.dqn_agent.entropy_probe --model ckpts_runA/ppo_selfplay_1000it/model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from agent.dqn_agent.model import encode_obs
from agent.dqn_agent.ppo import shielded_action, physical_action_mask
from agent.dqn_agent.test_behavior_ppo import load_net, RULES, MIX_PANEL
from engine.game import BomberEnv

# it1000 (the settled best) reference levels, measured over the MIX panel:
REJECT_MEAN = 0.30    # mean legal-action entropy (nats); it1000 ~0.33
REJECT_MEDIAN = 0.12  # median-state entropy (nats); it1000 ~0.16


def run(model: str, games: int = 15, max_steps: int = 500, seed0: int = 0,
        horizon: int = 6):
    net = load_net(model)
    seat_cls = {seat: RULES[MIX_PANEL[k]] for k, seat in enumerate((1, 2, 3))}

    ent = []        # Shannon entropy (nats) of softmax(logits) over LEGAL actions
    surv = 0
    for g in range(games):
        env = BomberEnv(max_steps=max_steps, seed=seed0 + g)
        obs = env.reset(seed=seed0 + g)
        opps = {seat: seat_cls[seat](seat) for seat in (1, 2, 3)}
        for t in range(max_steps):
            acts = [0, 0, 0, 0]
            if env.players[0].alive:
                ms, xs = encode_obs(obs, 0)
                with torch.no_grad():
                    logits, _ = net(torch.from_numpy(ms).unsqueeze(0),
                                    torch.from_numpy(xs).unsqueeze(0))
                lg = logits[0].numpy()
                mask = np.asarray(physical_action_mask(obs, 0), dtype=bool)
                legal = np.where(mask)[0]
                if legal.size >= 2:                  # entropy only meaningful w/ a choice
                    z = lg[legal] - lg[legal].max()
                    p = np.exp(z)
                    p = p / p.sum()
                    ent.append(float(-(p * np.log(p + 1e-12)).sum()))
                acts[0] = int(shielded_action(obs, 0, lg, horizon=horizon))
            for i in (1, 2, 3):
                if env.players[i].alive:
                    try:
                        acts[i] = int(opps[i].act(obs))
                    except Exception:
                        acts[i] = 0
            obs, term, trunc = env.step(acts)
            if term or trunc:
                break
        surv += int(env.players[0].alive)

    ent = np.asarray(ent)
    mean_h, med_h = float(ent.mean()), float(np.median(ent))
    over_peaked = mean_h < REJECT_MEAN or med_h < REJECT_MEDIAN

    # cp1252 Windows console can't print non-ASCII (e.g. the "tap train" dir) -> sanitize
    model_disp = str(model).encode("ascii", "replace").decode("ascii")
    print(f"\n=== entropy probe: {model_disp} ===")
    print(f"  games {games} vs mix | {ent.size} decision steps | survived {surv}/{games}")
    print(f"  MEAN policy entropy : {mean_h:5.3f} nats   (it1000 ~0.33 | REJECT if < {REJECT_MEAN})")
    print(f"  MEDIAN              : {med_h:5.3f} nats   (it1000 ~0.16 | REJECT if < {REJECT_MEDIAN})")
    print("  VERDICT: ", end="")
    if over_peaked:
        print("OVER-PEAKED -> brittle/'khung', likely regresses vs it1000. DO NOT submit.")
    else:
        print("HEALTHY (soft like it1000). OK to submit as a leaderboard candidate.")
    print()
    return mean_h, med_h


def main():
    p = argparse.ArgumentParser("PPO policy entropy kill-gate probe")
    p.add_argument("--model", required=True, help="path to model.pt or a .pth checkpoint")
    p.add_argument("--games", type=int, default=15)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shield_horizon", type=int, default=6)
    args = p.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"[error] model not found: {args.model}")
    run(args.model, args.games, args.max_steps, args.seed, horizon=args.shield_horizon)


if __name__ == "__main__":
    main()
