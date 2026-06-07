"""
Behaviour test for a trained PPO model — is it ACTIVE (farms + fights + survives)
or PASSIVE (camps the spawn pocket) / RECKLESS (advances but dies early)?

Mirror of test_behavior.py but for the PPO actor-critic: it drives the agent with
ppo.shielded_action (the same light-shield inference the submission uses), NOT the
DQN safety mask. Win-rate vs weak baselines is biased toward turtling and does not
predict the leaderboard; the metrics here (cells explored, boxes, items, survival)
do.

Run from the repo ROOT:
    python -m agent.dqn_agent.test_behavior_ppo --model ckpts_ppo/ppo_selfplay_1500it/model.pt
    python -m agent.dqn_agent.test_behavior_ppo --model .../it1000.pth --opponents hunter --games 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from agent.dqn_agent.model import encode_obs, NUM_ACTIONS, N_MAP_CH, N_AUX
from agent.dqn_agent.ppo import PPOActorCritic, shielded_action
from agent import (RandomAgent, SimpleRuleAgent, SmarterRuleAgent,
                   TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent, HunterAgent)
from engine.game import BomberEnv

RULES = {
    "random": RandomAgent, "simple": SimpleRuleAgent, "smarter": SmarterRuleAgent,
    "tactical": TacticalRuleAgent, "genius": GeniusRuleAgent,
    "box_farmer": BoxFarmerAgent, "hunter": HunterAgent,
}
# "mix" = a DIVERSE strong field (one each), the closest local proxy to the real
# leaderboard (4 different strong agents). Surviving + ranking well here predicts
# climbing far better than win-rate vs 3 identical weak bots that kill each other.
MIX_PANEL = ["hunter", "genius", "tactical"]


def load_net(path: str):
    path = str(path)
    if path.endswith(".pt"):
        net = torch.jit.load(path, map_location="cpu")
    else:
        ck = torch.load(path, map_location="cpu")
        spec = ck.get("input_spec") or ck.get("input_shape") or ck.get("input_dim")
        net = PPOActorCritic(tuple(spec[0]), int(spec[1]),
                             int(ck.get("num_actions", NUM_ACTIONS)))
        net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return net


def run(model: str, opponents: str = "genius", games: int = 10,
        max_steps: int = 500, seed0: int = 0, horizon: int = 6):
    net = load_net(model)
    if opponents == "mix":
        seat_cls = {seat: RULES[MIX_PANEL[k]] for k, seat in enumerate((1, 2, 3))}
    else:
        seat_cls = {seat: RULES[opponents] for seat in (1, 2, 3)}

    cells = boxes = items = bombs = alive_end = wins = 0
    death_steps = []
    heat = None
    spawn = None

    for g in range(games):
        env = BomberEnv(max_steps=max_steps, seed=seed0 + g)
        obs = env.reset(seed=seed0 + g)
        if heat is None:
            H, W = obs["map"].shape
            heat = np.zeros((H, W), dtype=np.int64)
            spawn = (int(obs["players"][0][0]), int(obs["players"][0][1]))
        opps = {seat: seat_cls[seat](seat) for seat in (1, 2, 3)}
        visited = set()
        t = 0
        for t in range(max_steps):
            acts = [0, 0, 0, 0]
            if env.players[0].alive:
                ms, xs = encode_obs(obs, 0)
                with torch.no_grad():
                    logits, _ = net(torch.from_numpy(ms).unsqueeze(0),
                                    torch.from_numpy(xs).unsqueeze(0))
                a = int(shielded_action(obs, 0, logits[0].numpy(), horizon=horizon))
                acts[0] = a
                if a == 5:
                    bombs += 1
            for i in (1, 2, 3):
                if env.players[i].alive:
                    try:
                        acts[i] = int(opps[i].act(obs))
                    except Exception:
                        acts[i] = 0
            obs, term, trunc = env.step(acts)
            if env.players[0].alive:
                px, py = int(obs["players"][0][0]), int(obs["players"][0][1])
                visited.add((px, py))
                heat[px, py] += 1
            if term or trunc:
                break

        s = env.players[0].stats
        cells += len(visited)
        boxes += int(s["boxes"])
        items += int(s["items"])
        me_alive = bool(env.players[0].alive)
        opp_alive = sum(1 for i in (1, 2, 3) if env.players[i].alive)
        alive_end += int(me_alive)
        if me_alive and opp_alive == 0:
            wins += 1
        if not me_alive:
            death_steps.append(t)

    n = games
    avg_cells = cells / n
    died_at = np.mean(death_steps) if death_steps else None

    print(f"\n=== {games} games: PPO agent (seat 0) vs '{opponents}' ===")
    print(f"  avg cells visited : {avg_cells:5.1f}   ( >15 active | ~5 still passive )")
    print(f"  avg boxes broken  : {boxes / n:5.1f}   ( >0 = farms boxes )")
    print(f"  avg items grabbed : {items / n:5.1f}   ( >0 = powers up )")
    if died_at is None:
        print(f"  survived to end   : {alive_end}/{n}   ( never died )")
    else:
        print(f"  survived to end   : {alive_end}/{n}   ( when it died: avg step {died_at:.0f} )")
    print(f"  sole-survivor wins: {wins}/{n}")
    print(f"  total bombs placed: {bombs}")

    print("\n  VERDICT: ", end="")
    if avg_cells < 8:
        print("PASSIVE - still camping the spawn pocket. Train longer / raise lam_adv.")
    elif alive_end < n * 0.3 and died_at is not None and died_at < 200:
        print("RECKLESS - advances but dies early. Lower lam_e / ent, or train longer.")
    else:
        print("BALANCED - active AND survives. Good candidate -> submit to leaderboard.")

    if heat is not None:
        reached = int((heat > 0).sum())
        print(f"\n  HEATMAP ({reached} distinct cells reached over {games} games; "
              f"'S'=spawn, '#'=most time, '.'=never, ' '=wall):")
        hi = heat.max()
        ramp = " .:-=+*o%#"
        for x in range(heat.shape[0]):
            row = []
            for y in range(heat.shape[1]):
                if (x, y) == spawn:
                    row.append("S")
                elif heat[x, y] == 0:
                    row.append("." if 0 < x < heat.shape[0] - 1 and 0 < y < heat.shape[1] - 1 else " ")
                else:
                    lvl = min(len(ramp) - 1, 1 + int((heat[x, y] / hi) * (len(ramp) - 2)))
                    row.append(ramp[lvl])
            print("    " + "".join(row))
        print("    -> nhieu '.' + cum '#' quanh S = CAMPING ; trai deu khap = di map tot")
    print()


def main():
    p = argparse.ArgumentParser("Bomberland PPO model behaviour test")
    p.add_argument("--model", required=True, help="path to model.pt or a .pth checkpoint")
    p.add_argument("--opponents", default="mix", choices=list(RULES) + ["mix"],
                   help="'mix' = hunter+genius+tactical (most leaderboard-like); "
                        "or a single rule name for all 3 seats")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shield_horizon", type=int, default=6,
                   help="must match the shield horizon the model was trained with")
    args = p.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"[error] model not found: {args.model}")
    run(args.model, args.opponents, args.games, args.max_steps, args.seed,
        horizon=args.shield_horizon)


if __name__ == "__main__":
    main()
