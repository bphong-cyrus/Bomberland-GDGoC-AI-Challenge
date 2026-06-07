"""
GDGoC-HCMUS AI Challenge 2026 — Bomberland submission agent.

This is the ONLY file the evaluation server calls. It supports BOTH model types:

  * PPO actor-critic (model.pt whose forward returns a (logits, value) tuple) —
    picked with a light inference shield (see ppo.shielded_action). This is the
    aggressive, non-camping policy.
  * DQN dueling net (model.pt whose forward returns a single Q tensor) — picked
    with the full survivability safety mask (model.safe_action). Kept as a fallback.

Detection is automatic: we just look at whether the network's output is a tuple.

Flat submission zip should contain:
    agent.py        (this file)
    model.py        (shared encoder / danger model / DQN net / safety mask)
    ppo.py          (PPO actor-critic + inference shield)   [needed for a PPO model]
    model.pt        (TorchScript weights — preferred) or model.pth (checkpoint)
"""
from pathlib import Path

import numpy as np
import torch

try:                                  # flat submission (sibling files on sys.path)
    from model import (DQNModel, encode_obs, safe_action,
                       N_MAP_CH, N_AUX, NUM_ACTIONS)
    from ppo import PPOActorCritic, shielded_action
except ImportError:                   # imported as a package (local dev)
    from .model import (DQNModel, encode_obs, safe_action,
                        N_MAP_CH, N_AUX, NUM_ACTIONS)
    from .ppo import PPOActorCritic, shielded_action

torch.set_num_threads(1)              # match the single-threaded eval sandbox


class Agent:
    """Mandatory submission class: Agent(agent_id).act(obs) -> int in [0, 5]."""

    team_id = "DQNAgent"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.is_ppo = False
        base = Path(__file__).parent

        ts = base / "model.pt"
        pth = base / "model.pth"
        legacy = next(iter(sorted(base.glob("*.pth"))), None)

        if ts.exists():
            self._net = torch.jit.load(str(ts), map_location="cpu")
        elif pth.exists():
            self._net = self._from_pth(str(pth))
        elif legacy is not None:
            self._net = self._from_pth(str(legacy))
        else:
            raise FileNotFoundError(f"No model.pt / model.pth found in {base}")
        self._net.eval()
        self._detect_arch()

    def _detect_arch(self):
        """Run one dummy forward to learn whether this is a PPO (tuple output) or
        DQN (single tensor) network — so act() can dispatch correctly."""
        try:
            dm = torch.zeros(1, N_MAP_CH, 13, 13)
            da = torch.zeros(1, N_AUX)
            with torch.no_grad():
                out = self._net(dm, da)
            self.is_ppo = isinstance(out, (tuple, list))
        except Exception:
            self.is_ppo = False

    @staticmethod
    def _from_pth(path: str):
        ck = torch.load(path, map_location="cpu")
        spec = ck.get("input_spec") or ck.get("input_shape") or ck.get("input_dim")
        map_shape = tuple(spec[0])
        aux_dim = int(spec[1])
        num_actions = int(ck.get("num_actions", NUM_ACTIONS))
        if map_shape[0] != N_MAP_CH or aux_dim != N_AUX:
            raise ValueError(
                f"checkpoint encoding ({map_shape[0]}ch/{aux_dim}aux) != current "
                f"encoder ({N_MAP_CH}ch/{N_AUX}aux); retrain with this model.py")
        if ck.get("arch") == "ppo_actor_critic":
            net = PPOActorCritic(map_shape, aux_dim, num_actions)
        else:
            net = DQNModel(map_shape, aux_dim, num_actions)
        net.load_state_dict(ck["model_state_dict"])
        return net

    def act(self, obs: dict) -> int:
        try:
            map_s, aux_s = encode_obs(obs, self.agent_id)
            with torch.no_grad():
                out = self._net(
                    torch.from_numpy(map_s).unsqueeze(0),
                    torch.from_numpy(aux_s).unsqueeze(0),
                )
            if self.is_ppo:
                logits = out[0][0].numpy()
                return int(shielded_action(obs, self.agent_id, logits))
            q = out[0].numpy()
            return int(safe_action(obs, self.agent_id, q))
        except Exception:
            return 0   # STOP — never crash the worker


if __name__ == "__main__":
    # Tiny self-test / timing benchmark (no training here — see train.py / train_ppo.py).
    import time

    base = Path(__file__).parent
    if not (base / "model.pt").exists() and not (base / "model.pth").exists():
        print("No weights found - train first with:  python -m agent.dqn_agent.train_ppo")
    else:
        agent = Agent(0)
        print(f"loaded model: {'PPO actor-critic' if agent.is_ppo else 'DQN dueling'}")
        rng = np.random.default_rng(0)
        obs = {
            "map": rng.integers(0, 3, size=(13, 13)).astype(np.int64),
            "players": np.array([[1, 1, 1, 1, 0], [11, 11, 1, 1, 0],
                                 [1, 11, 1, 1, 0], [11, 1, 1, 1, 0]], dtype=np.int8),
            "bombs": np.zeros((0, 4), dtype=np.int8),
        }
        for _ in range(5):
            agent.act(obs)  # warm up
        t0 = time.perf_counter()
        for _ in range(200):
            agent.act(obs)
        dt = (time.perf_counter() - t0) / 200 * 1000
        print(f"avg act() latency: {dt:.2f} ms/step (budget 100 ms)")
