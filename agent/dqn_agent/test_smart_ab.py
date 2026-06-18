"""
Paired A/B: SmartAgent v2 (adversarial survival + cornering + powered-farm, all ON)
vs v1 (all flags OFF) on the SAME seeds vs the MIX panel (hunter+genius+tactical).

The rule opponents use the global RNG, so _play seeds it per game -> v1 and v2 face
the IDENTICAL opponent randomness for each seed (a controlled comparison; see the
project note that un-seeded MIX swings wildly run-to-run).

GATE to ship v2 as a parallel entry:
  * SURVIVE(v2)  >= SURVIVE(v1)   (survival dominates mu -- must NOT regress)
  * died@(v2)    >= died@(v1)     (don't die earlier)
  * WINS(v2)     >  WINS(v1)      (the whole point: convert ties into kills)
  * boxes/items(v2) ~>= v1        (still farms; powered-farm should help)

Run from repo ROOT with system Python 3.10:
    python -m agent.dqn_agent.test_smart_ab --games 60
    python -m agent.dqn_agent.test_smart_ab --games 60 --seed 100   # 2nd seed range
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from agent.dqn_agent.smart_agent import SmartAgent
from agent.dqn_agent.test_search import _play, _summarise
from agent.dqn_agent.test_behavior_ppo import RULES


def run(games=60, max_steps=500, seed0=0, horizon=6, opponents="mix"):
    # 3-way ablation isolating each upgrade group, all on the SAME seeds:
    configs = [
        ("v1 (base)",  dict(adversarial=False, corner=False, farm_powered=False)),
        ("v_surv",     dict(adversarial=True,  corner=False, farm_powered=True)),
        ("v2 (hunt)",  dict(adversarial=True,  corner=True,  farm_powered=True)),
    ]
    rows = {name: [] for name, _ in configs}
    for name, flags in configs:
        ag = SmartAgent(0, horizon=horizon, **flags)
        for gi in range(games):
            rows[name].append(_play(ag.act, opponents, seed0 + gi, max_steps,
                                    reset_fn=ag.reset))

    print(f"\n=== {games} games vs '{opponents}' (same seeds {seed0}..{seed0+games-1}), "
          f"horizon={horizon} ===")
    for name, _ in configs:
        _summarise(name, rows[name])

    def agg(name, key):
        return float(np.mean([r[key] for r in rows[name]]))
    base = "v1 (base)"
    for name, _ in configs[1:]:
        dwins = sum(r["win"] for r in rows[name]) - sum(r["win"] for r in rows[base])
        dsurv = sum(r["alive"] for r in rows[name]) - sum(r["alive"] for r in rows[base])
        print(f"  delta {name}-v1:  wins {dwins:+d}   survive {dsurv:+d}   "
              f"kills {agg(name,'kills')-agg(base,'kills'):+.2f}   "
              f"boxes {agg(name,'boxes')-agg(base,'boxes'):+.2f}   "
              f"items {agg(name,'items')-agg(base,'items'):+.2f}")
    print("  v_surv = survival+farming only; v2 = + cornering. SHIP if survive NOT lower.\n")


def latency(seed=7, horizon=6):
    """Time act() across a REAL game (live mid-game states with bombs/enemies), not a
    single frozen obs -- so the number reflects the worst-case cornering branch."""
    from engine.game import BomberEnv
    v2 = SmartAgent(0, horizon=horizon)
    env = BomberEnv(max_steps=500, seed=seed)
    obs = env.reset(seed=seed)
    ts, n = 0.0, 0
    for _ in range(500):
        acts = [0, 0, 0, 0]
        if env.players[0].alive:
            t0 = time.perf_counter()
            acts[0] = v2.act(obs)
            ts += time.perf_counter() - t0
            n += 1
        for i in (1, 2, 3):                            # opponents hold -> a long live game
            acts[i] = 0
        obs, term, trunc = env.step(acts)
        if term or trunc:
            break
    dt = ts / max(n, 1) * 1000
    print(f"=== LATENCY: SmartAgent v2 -> {dt:.2f} ms/step over {n} live steps "
          f"(budget 100 ms) ===\n")


def main():
    p = argparse.ArgumentParser("SmartAgent v2-vs-v1 A/B")
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--opponents", default="mix", choices=list(RULES) + ["mix"])
    args = p.parse_args()
    run(args.games, args.max_steps, args.seed, args.horizon, args.opponents)
    latency(horizon=args.horizon)


if __name__ == "__main__":
    main()
