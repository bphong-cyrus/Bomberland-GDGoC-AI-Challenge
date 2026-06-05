"""
Behaviour test for a trained Bomberland model — is it ACTIVE or PASSIVE/RECKLESS?

Eval-vs-baselines numbers (top1/avg_rank in train.py) are biased toward turtling
and don't predict the leaderboard. This script instead measures the BEHAVIOUR you
actually care about: does the agent leave its spawn pocket, farm boxes/items
(power up), and survive — or does it loiter, or rush in and die early?

Run from the repo ROOT (not from inside agent/):
    python -m agent.dqn_agent.test_behavior --model ckpts/selfplay_mix_5000ep/model.pt
    python -m agent.dqn_agent.test_behavior --model ckpts/selfplay_mix_5000ep/ep4000.pth \
        --opponents genius --games 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from agent.dqn_agent.model import (encode_obs, safe_action, DQNModel,
                                   NUM_ACTIONS, N_MAP_CH, N_AUX)
from agent import (RandomAgent, SimpleRuleAgent, SmarterRuleAgent,
                   TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent)
from engine.game import BomberEnv

RULES = {
    "random": RandomAgent, "simple": SimpleRuleAgent, "smarter": SmarterRuleAgent,
    "tactical": TacticalRuleAgent, "genius": GeniusRuleAgent, "box_farmer": BoxFarmerAgent,
}


def load_net(path: str):
    """Load a TorchScript model.pt OR a .pth checkpoint into an eval-mode net."""
    path = str(path)
    if path.endswith(".pt"):
        net = torch.jit.load(path, map_location="cpu")
    else:
        ck = torch.load(path, map_location="cpu")
        spec = ck.get("input_spec") or ck.get("input_shape") or ck.get("input_dim")
        net = DQNModel(tuple(spec[0]), int(spec[1]),
                       int(ck.get("num_actions", NUM_ACTIONS)))
        net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return net


def run(model: str, opponents: str = "genius", games: int = 10,
        max_steps: int = 500, seed0: int = 0):
    net = load_net(model)
    opp_cls = RULES[opponents]

    cells = boxes = items = bombs = alive_end = wins = 0
    death_steps = []

    for g in range(games):
        env = BomberEnv(max_steps=max_steps, seed=seed0 + g)
        obs = env.reset(seed=seed0 + g)
        opps = {i: opp_cls(i) for i in (1, 2, 3)}
        visited = set()
        t = 0
        for t in range(max_steps):
            acts = [0, 0, 0, 0]
            if env.players[0].alive:
                ms, xs = encode_obs(obs, 0)
                with torch.no_grad():
                    q = net(torch.from_numpy(ms).unsqueeze(0),
                            torch.from_numpy(xs).unsqueeze(0))[0].numpy()
                a = int(safe_action(obs, 0, q))
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
                visited.add((int(obs["players"][0][0]), int(obs["players"][0][1])))
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
            wins += 1                       # sole survivor = a win
        if not me_alive:
            death_steps.append(t)

    n = games
    avg_cells = cells / n
    died_at = np.mean(death_steps) if death_steps else None

    print(f"\n=== {games} games: trained agent (seat 0) vs '{opponents}' ===")
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
        print("PASSIVE - still loitering in the spawn pocket. Push reward harder.")
    elif alive_end < n * 0.3 and died_at is not None and died_at < 200:
        print("RECKLESS - advances but dies early. Rebalance: less aggression, "
              "a little survival back.")
    else:
        print("BALANCED - active AND survives. Good candidate -> submit to leaderboard.")
    print()


def main():
    p = argparse.ArgumentParser("Bomberland model behaviour test")
    p.add_argument("--model", required=True,
                   help="path to model.pt (TorchScript) or a .pth checkpoint")
    p.add_argument("--opponents", default="genius", choices=list(RULES),
                   help="rule baseline to play against (default: genius)")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"[error] model not found: {args.model}")
    run(args.model, args.opponents, args.games, args.max_steps, args.seed)


if __name__ == "__main__":
    main()
