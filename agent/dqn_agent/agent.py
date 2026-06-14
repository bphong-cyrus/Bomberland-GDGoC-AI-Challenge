"""
GDGoC-HCMUS AI Challenge 2026 — Bomberland submission agent.

This is the ONLY file the evaluation server calls. It supports THREE model types:

  * PPO actor-critic (model.pt whose forward returns a (logits, value) tuple) —
    picked with a light inference shield (see ppo.shielded_action). This is the
    aggressive, non-camping policy.
  * RECURRENT (LSTM) PPO actor-critic (model.pt exposing a `step(map, aux, hidden)`
    method) — driven with a maintained (h, c) hidden state across act() calls,
    reset at game start; logits then go through the SAME shield. See ppo_lstm.py.
  * DQN dueling net (model.pt whose forward returns a single Q tensor) — picked
    with the full survivability safety mask (model.safe_action). Kept as a fallback.

Detection is automatic: a recurrent net is detected by its `step` method; a PPO
net by its tuple output; otherwise it is treated as a DQN.

Flat submission zip should contain:
    agent.py        (this file)
    model.py        (shared encoder / danger model / DQN net / safety mask)
    ppo.py          (PPO actor-critic + inference shield)   [needed for a PPO model]
    ppo_lstm.py     (recurrent PPO net + driver)            [needed for an LSTM model]
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

# Recurrent (LSTM) net is OPTIONAL: only needed for an LSTM model.pt. A plain
# feedforward submission zip need not ship ppo_lstm.py, so import defensively.
try:
    try:
        from ppo_lstm import PPOLSTMActorCritic
    except ImportError:
        from .ppo_lstm import PPOLSTMActorCritic
except ImportError:
    PPOLSTMActorCritic = None

torch.set_num_threads(1)              # match the single-threaded eval sandbox


class Agent:
    """Mandatory submission class: Agent(agent_id).act(obs) -> int in [0, 5]."""

    team_id = "DQNAgent"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.is_ppo = False
        self.is_recurrent = False
        self._hidden = None            # carried (h, c) for a recurrent policy
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
        """Learn whether this is a RECURRENT PPO (has a `step(map,aux,hidden)`
        method), a feedforward PPO (tuple output), or a DQN (single tensor) — so
        act() can dispatch correctly. The feedforward/DQN detection is UNCHANGED."""
        dm = torch.zeros(1, N_MAP_CH, 13, 13)
        da = torch.zeros(1, N_AUX)
        # 1) recurrent? try the single-step recurrent signature.
        try:
            if hasattr(self._net, "step"):
                with torch.no_grad():
                    out = self._net.step(dm, da, None)
                if isinstance(out, (tuple, list)) and len(out) == 3:
                    self.is_recurrent = True
                    self.is_ppo = True
                    return
        except Exception:
            self.is_recurrent = False
        # 2) feedforward PPO (tuple) vs DQN (tensor) — original heuristic.
        try:
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
        if ck.get("arch") == "ppo_lstm_actor_critic":
            if PPOLSTMActorCritic is None:
                raise ImportError("ppo_lstm.py is required to load a recurrent model")
            net = PPOLSTMActorCritic(map_shape, aux_dim, num_actions,
                                     lstm_hidden=int(ck.get("lstm_hidden", 256)))
        elif ck.get("arch") == "ppo_actor_critic":
            net = PPOActorCritic(map_shape, aux_dim, num_actions)
        else:
            net = DQNModel(map_shape, aux_dim, num_actions)
        net.load_state_dict(ck["model_state_dict"])
        return net

    @staticmethod
    def _is_game_start(obs) -> bool:
        """Heuristic for the first step of a fresh game: NO bombs on the field AND
        all players still alive. Robust whether the harness reuses or recreates the
        Agent — used only by the recurrent path to reset its hidden state."""
        try:
            bombs = obs.get("bombs")
            arr = np.asarray(bombs) if bombs is not None else np.zeros((0,))
            n_bombs = 0 if arr.size == 0 else (1 if arr.ndim == 1 else arr.shape[0])
            players = np.asarray(obs["players"])
            all_alive = bool((players[:, 2] == 1).all())
            return n_bombs == 0 and all_alive
        except Exception:
            return False

    def act(self, obs: dict) -> int:
        try:
            map_s, aux_s = encode_obs(obs, self.agent_id)
            mt = torch.from_numpy(map_s).unsqueeze(0)
            xt = torch.from_numpy(aux_s).unsqueeze(0)

            if self.is_recurrent:
                # reset hidden at game start (step-0-like obs); otherwise carry it.
                if self._is_game_start(obs):
                    self._hidden = None
                with torch.no_grad():
                    logits_t, _v, self._hidden = self._net.step(mt, xt, self._hidden)
                logits = logits_t[0].numpy()
                return int(shielded_action(obs, self.agent_id, logits))

            with torch.no_grad():
                out = self._net(mt, xt)
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
        if agent.is_recurrent:
            kind = "PPO LSTM (recurrent)"
        elif agent.is_ppo:
            kind = "PPO actor-critic"
        else:
            kind = "DQN dueling"
        print(f"loaded model: {kind}")
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
