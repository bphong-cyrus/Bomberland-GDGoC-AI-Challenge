"""
Reproduce the grader's submission PRECHECK for a flat submission directory.

The grader imports `agent.py` BY FILE PATH with the submission dir NOT on sys.path
(this is what broke an earlier flat zip -> 'attempted relative import with no known
parent package' -> runtime_precheck_exception). This script does exactly that from a
FOREIGN cwd, instantiates Agent(0), and drives a few real steps -- so a green run here
means the zip will load on the server.

Run from repo ROOT with system Python 3.10 (ONE dir per process -> no module caching):
    python -m agent.dqn_agent.verify_submit submit_search
    python -m agent.dqn_agent.verify_submit submit_surv
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from engine.game import BomberEnv


def verify(subdir):
    d = Path(subdir).resolve()
    agent_py = d / "agent.py"
    if not agent_py.exists():
        raise SystemExit(f"[FAIL] {agent_py} not found")
    # Load agent.py by path; deliberately do NOT add `d` to sys.path (the grader doesn't).
    if str(d) in sys.path:
        sys.path.remove(str(d))
    spec = importlib.util.spec_from_file_location("subm_agent_under_test", str(agent_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                       # runs the in-file sys.path fix
    Agent = mod.Agent
    ag = Agent(0)

    env = BomberEnv(max_steps=500, seed=3)
    obs = env.reset(seed=3)
    acts_seen, t0 = [], time.perf_counter()
    steps = 0
    for _ in range(60):
        a = ag.act(obs)
        assert isinstance(a, int) and 0 <= a <= 5, f"bad action {a!r}"
        acts_seen.append(a)
        obs, term, trunc = env.step([a, 1, 2, 0])
        steps += 1
        if term or trunc:
            break
    ms = (time.perf_counter() - t0) / max(steps, 1) * 1000
    print(f"[OK] {subdir}: loaded Agent (team_id={getattr(Agent, 'team_id', '?')}), "
          f"{steps} steps, {ms:.2f} ms/step, sample acts={acts_seen[:12]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m agent.dqn_agent.verify_submit <submit_dir>")
    verify(sys.argv[1])
