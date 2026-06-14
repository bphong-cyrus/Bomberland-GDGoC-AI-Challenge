"""
Recurrent (LSTM) PPO Actor-Critic for the Bomberland agent.

This is a SEPARATE, self-contained track that runs ALONGSIDE the proven
feedforward PPO (`ppo.py`). It deliberately REUSES, unchanged:

  * the danger-aware state encoder + constants from `model.py`
    (`encode_obs`, `N_MAP_CH`, `N_AUX`, `NUM_ACTIONS`), and
  * the EXACT same CNN encoder architecture as `ppo.PPOActorCritic`
    (conv stack -> flatten, plus the small aux MLP, then the 256-wide trunk),
    and
  * the bounded-horizon SURVIVAL SHIELD from `ppo.py`
    (`shielded_action`, default horizon 6).

The ONLY new thing is an `nn.LSTM` placed BETWEEN the encoder trunk and the
actor/critic heads, so the policy can carry temporal context (opponent
patterns, bomb timing) across steps. The shield is unchanged and per-step: the
LSTM only produces logits, then the SAME shield masks/selects the action.

Why keep the encoder identical: BC warm-start (bc.py) clones the same teacher
into encoder+heads to escape the camping basin; reusing the architecture means
the BC pretraining and the feedforward lessons transfer directly, and the LSTM
learns the recurrence on top.

Hard deps: numpy + torch only (safe to ship in the flat submission zip).

The network is `torch.jit.script`-able so `train_ppo_lstm._save` can export a
TorchScript `model.pt`. The recurrent forward (`step`) carries an (h, c) hidden
state; the submission's `agent.py` maintains it across `act()` calls and resets
it at game start.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:                                  # flat submission (sibling files on sys.path)
    from model import encode_obs, NUM_ACTIONS, N_MAP_CH, N_AUX
    from ppo import shielded_action, DEFAULT_SHIELD_HORIZON
except ImportError:                   # imported as a package (local dev / training)
    from .model import encode_obs, NUM_ACTIONS, N_MAP_CH, N_AUX
    from .ppo import shielded_action, DEFAULT_SHIELD_HORIZON


# ── Recurrent Actor-Critic network (TorchScript-friendly) ─────────────────────
class PPOLSTMActorCritic(nn.Module):
    """Shared conv/aux encoder -> LSTM -> (policy logits, state value).

    Mirrors `ppo.PPOActorCritic`'s encoder EXACTLY (same conv stack + aux MLP +
    256-wide trunk), then inserts a single-layer `nn.LSTM(feature_dim,
    lstm_hidden)` whose output feeds the actor and critic heads.

    Two entry points (do NOT use `forward` for inference dispatch -- see below):
      * `step`  : ONE timestep, batch fixed to 1, carries (h, c). For inference.
      * `seq`   : a full (T, B) sequence from a given hidden state. For training.

    Note on detection: unlike `PPOActorCritic`, this net is driven explicitly via
    `step`/`seq` (agent.py checks for a `step` method / `is_recurrent` flag), so
    `forward` is only a thin convenience wrapper used by nothing critical.
    """

    # marker constant so callers can cheaply detect a recurrent net even after
    # TorchScript export. MUST be a class-level value listed in __constants__ --
    # a bare `is_recurrent: bool` class annotation is NOT scriptable in torch 2.x.
    __constants__ = ["is_recurrent"]
    is_recurrent = True

    def __init__(self, map_shape, aux_dim, num_actions, lstm_hidden: int = 256):
        super().__init__()
        c, h, w = map_shape
        self.lstm_hidden = int(lstm_hidden)
        self.num_actions = int(num_actions)

        # --- encoder: identical to ppo.PPOActorCritic ---------------------------
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat = int(self.conv(torch.zeros(1, c, h, w)).reshape(1, -1).size(1))
        self.aux_net = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(nn.Linear(flat + 32, 256), nn.ReLU(inplace=True))

        # --- recurrence + heads -------------------------------------------------
        self.feature_dim = 256
        self.lstm = nn.LSTM(self.feature_dim, self.lstm_hidden, batch_first=False)
        self.policy_head = nn.Sequential(nn.Linear(self.lstm_hidden, 128),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(128, num_actions))
        self.value_head = nn.Sequential(nn.Linear(self.lstm_hidden, 128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, 1))

    # ── encoder applied to a FLAT batch of frames -> [N, feature_dim] ──────────
    def encode(self, map_x, aux_x):
        spatial = self.conv(map_x).flatten(1)
        aux = self.aux_net(aux_x)
        return self.trunk(torch.cat([spatial, aux], dim=1))

    def initial_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)

    # ── INFERENCE: single timestep, batch == 1, carries (h, c) ─────────────────
    def step(self, map_x, aux_x,
             hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """map_x[1,C,H,W], aux_x[1,A] -> (logits[1,A], value[1], new_hidden).

        `hidden` is the carried (h, c); pass None to start from zeros.
        """
        if hidden is None:
            h = torch.zeros(1, 1, self.lstm_hidden, device=map_x.device)
            c = torch.zeros(1, 1, self.lstm_hidden, device=map_x.device)
            hidden = (h, c)
        feat = self.encode(map_x, aux_x)            # [1, feature_dim]
        feat = feat.unsqueeze(0)                    # [T=1, B=1, feature_dim]
        out, new_hidden = self.lstm(feat, hidden)   # out [1, 1, lstm_hidden]
        out = out.squeeze(0)                        # [1, lstm_hidden]
        logits = self.policy_head(out)              # [1, A]
        value = self.value_head(out).squeeze(-1)    # [1]
        return logits, value, new_hidden

    # ── TRAINING: full sequence (T, B) from a given hidden state ───────────────
    def seq(self, map_x, aux_x,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """map_x[T,B,C,H,W], aux_x[T,B,A], hidden0 -> (logits[T,B,A],
        values[T,B], hidden). Processes the whole sequence; pass hidden=None to
        start from a zero state (the per-episode training convention)."""
        T = map_x.size(0)
        B = map_x.size(1)
        C = map_x.size(2)
        H = map_x.size(3)
        W = map_x.size(4)
        A = aux_x.size(2)
        if hidden is None:
            h = torch.zeros(1, B, self.lstm_hidden, device=map_x.device)
            c = torch.zeros(1, B, self.lstm_hidden, device=map_x.device)
            hidden = (h, c)
        # encode every frame in one batched conv pass, then restore (T, B)
        feat = self.encode(map_x.reshape(T * B, C, H, W),
                           aux_x.reshape(T * B, A))          # [T*B, feature_dim]
        feat = feat.reshape(T, B, self.feature_dim)
        out, hidden = self.lstm(feat, hidden)                # [T, B, lstm_hidden]
        logits = self.policy_head(out)                       # [T, B, A]
        value = self.value_head(out).squeeze(-1)             # [T, B]
        return logits, value, hidden

    def forward(self, map_x, aux_x,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Thin wrapper == single inference step. agent.py does NOT rely on this for
        # detection (it checks for the recurrent signature / is_recurrent), so the
        # tuple-vs-tensor heuristic used for the feedforward model is untouched.
        return self.step(map_x, aux_x, hidden)


# ── Recurrent inference driver: holds (h, c), applies the SAME shield ─────────
class LSTMShieldedActor:
    """Drives a recurrent policy at inference.

    `act(obs, agent_id)` = encode_obs -> net.step (carrying hidden) ->
    `ppo.shielded_action` (the UNCHANGED bounded-horizon survival shield).
    `reset()` zeros the hidden state (call at game start).

    Works with either a plain `PPOLSTMActorCritic` or its TorchScript export.
    """

    def __init__(self, net, agent_id: int, horizon: int = DEFAULT_SHIELD_HORIZON,
                 device=None):
        self.net = net
        self.agent_id = int(agent_id)
        self.horizon = int(horizon)
        self.device = device
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset(self):
        """Zero the recurrent state (new game)."""
        self.hidden = None

    def act(self, obs) -> int:
        ms, xs = encode_obs(obs, self.agent_id)
        mt = torch.from_numpy(ms).unsqueeze(0)
        xt = torch.from_numpy(xs).unsqueeze(0)
        if self.device is not None:
            mt, xt = mt.to(self.device), xt.to(self.device)
        with torch.no_grad():
            logits, _value, self.hidden = self.net.step(mt, xt, self.hidden)
        logits_np = logits[0].cpu().numpy()
        return int(shielded_action(obs, self.agent_id, logits_np, horizon=self.horizon))
