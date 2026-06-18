"""
PPO Actor-Critic head for the Bomberland agent.

This deliberately REUSES the proven, danger-aware encoder, blast/chain simulation
and constants from `model.py` (so training and inference share the exact same state
representation). The only new thing here is an Actor-Critic network plus the
*inference-time* action handling used by the submission.

Design choice that breaks camping (see train_ppo.py for the full rationale):
  * There is NO hard "guaranteed-survivable" safety mask here. The DQN path used a
    full time-expanded BFS mask (`model.safe_action`) that guaranteed survival —
    which removed the agent's need to LEARN positioning and made it timid (it would
    only ever go where survival was already proven, i.e. stay home). PPO instead
    learns danger from the death penalty + entropy-driven exploration.
  * At inference we apply only:
      1. a PHYSICAL mask (don't waste the action distribution on no-op moves into
         walls/boxes/bombs, or PLACE_BOMB with no bombs left), and
      2. a LIGHT shield: among the actions the policy prefers, skip ones that step
         into a tile exploding THIS step (or that place a bomb with no escape) when
         a safer preferred action exists. This stops only the dumbest deaths; it is
         far cheaper and far less conservative than the DQN BFS mask, so the learned
         aggressive play still expresses.

Hard deps: numpy + torch only (safe to ship in the flat submission zip).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:                                  # flat submission (sibling files on sys.path)
    from model import (encode_obs, compute_danger, ACTION_DELTA, NUM_ACTIONS,
                       N_MAP_CH, N_AUX, _walkable, _bombs_array, _survivable)
except ImportError:
    try:                              # imported as a package (local dev / training)
        from .model import (encode_obs, compute_danger, ACTION_DELTA, NUM_ACTIONS,
                            N_MAP_CH, N_AUX, _walkable, _bombs_array, _survivable)
    except ImportError:               # flat but dir NOT on sys.path (grader precheck via
        import os as _os, sys as _sys  # file-path import) -> add our own dir, retry absolute
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from model import (encode_obs, compute_danger, ACTION_DELTA, NUM_ACTIONS,
                           N_MAP_CH, N_AUX, _walkable, _bombs_array, _survivable)

GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY = 0, 1, 2, 3, 4

# How many steps ahead the inference/training shield verifies survival for.
# 1  == the old "don't step into a blast THIS step" light shield (lets 2-3 step
#       traps kill an aggressive agent -> dies ~step 100, see train logs).
# big == the DQN's full-horizon BFS (guarantees survival but is so conservative it
#       pins a from-scratch policy to spawn -> camping).
# A MIDDLE horizon (~6) is the sweet spot: it stops the multi-step bomb traps that
# were killing the BC-active policy, without the over-conservatism that caused
# camping. Tunable from train_ppo via --shield_horizon.
DEFAULT_SHIELD_HORIZON = 8


# ── Actor-Critic network (TorchScript-friendly; mirrors DQNModel's conv trunk) ──
class PPOActorCritic(nn.Module):
    """Shared conv/aux trunk -> (policy logits, state value).

    forward() returns a TUPLE (logits[B,A], value[B]) so the submission's
    `agent.py` can distinguish a PPO model (tuple) from a DQN model (single tensor)
    purely by inspecting the output type.
    """

    def __init__(self, map_shape, aux_dim, num_actions):
        super().__init__()
        c, h, w = map_shape
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
        self.policy_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True),
                                         nn.Linear(128, num_actions))
        self.value_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True),
                                        nn.Linear(128, 1))

    def forward(self, map_x, aux_x):
        spatial = self.conv(map_x).flatten(1)
        aux = self.aux_net(aux_x)
        f = self.trunk(torch.cat([spatial, aux], dim=1))
        logits = self.policy_head(f)
        value = self.value_head(f).squeeze(-1)
        return logits, value


# ── action legality (physical, NOT safety) ───────────────────────────────────
def physical_action_mask(obs, agent_id):
    """bool[NUM_ACTIONS]: actions that actually do something legal.

    Masks out moves into walls/boxes/out-of-bounds/bomb tiles (the engine would
    treat them as STOP anyway) and PLACE_BOMB when no bombs are left / already on a
    bomb. STOP is always allowed. This is the SAME mask used during PPO training so
    that sampled log-probs stay consistent.
    """
    grid = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    uid = int(agent_id)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[0] = True                                   # STOP always legal
    if uid >= len(players) or int(players[uid][2]) != 1:
        return mask
    mx, my = int(players[uid][0]), int(players[uid][1])
    bombs_left = int(players[uid][3])
    bomb_tiles = {(int(b[0]), int(b[1])) for b in _bombs_array(obs)}
    for a in (1, 2, 3, 4):
        dx, dy = ACTION_DELTA[a]
        nx, ny = mx + dx, my + dy
        if _walkable(grid, nx, ny) and (nx, ny) not in bomb_tiles:
            mask[a] = True
    if bombs_left > 0 and (mx, my) not in bomb_tiles:
        mask[5] = True
    return mask


def _imminent(instants, pos):
    s = instants.get(pos)
    return s is not None and min(s) <= 1


def _latest_lethal(instants, pos):
    """For the trapped fallback: the LAST relative step `pos` is lethal (bigger =
    we survive longer heading there). 1e6 if the tile is never in a blast."""
    s = instants.get(pos)
    return max(s) if s else 10 ** 6


def survivable_action_mask(obs, agent_id, horizon=DEFAULT_SHIELD_HORIZON):
    """bool[NUM_ACTIONS]: the BOUNDED-HORIZON safety shield.

    Used IDENTICALLY in PPO training (to sample / score actions) and at inference,
    so the sampled log-probs stay consistent with the deployed policy.

    For each physically-legal action it keeps the action only if, AFTER taking it,
    the agent can still dodge every explosion for the next `horizon` steps
    (time-expanded BFS, `model._survivable`). This is the key upgrade over the old
    1-step shield: it stops the 2-3 step bomb traps that were killing the active
    BC-warm-started policy at ~step 100, WITHOUT the over-conservatism of the
    full-horizon DQN mask (which pinned a from-scratch policy to spawn). PLACE_BOMB
    is allowed only if, after dropping the bomb, a multi-step escape still exists.

    If nothing is survivable (genuinely trapped), returns an empty mask so the
    caller can apply a least-bad fallback.
    """
    grid = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    uid = int(agent_id)
    phys = physical_action_mask(obs, agent_id)
    if uid >= len(players) or int(players[uid][2]) != 1:
        return phys
    mx, my = int(players[uid][0]), int(players[uid][1])
    my_radius = 1 + int(players[uid][4])
    bombs = _bombs_array(obs)
    instants, bomb_tiles = compute_danger(grid, bombs, players)

    safe = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in range(5):                               # STOP + 4 moves
        if not phys[a]:
            continue
        if a == 0:
            npos = (mx, my)
        else:
            dx, dy = ACTION_DELTA[a]
            npos = (mx + dx, my + dy)
        # not lethal THIS step AND a future escape exists within `horizon` steps
        if not _imminent(instants, npos) and \
                _survivable(npos, 1, instants, grid, bomb_tiles, horizon):
            safe[a] = True
    if phys[5]:                                      # PLACE_BOMB needs a real escape
        inst2, bt2 = compute_danger(grid, bombs, players, extra_bomb=(mx, my, my_radius))
        if not _imminent(inst2, (mx, my)) and \
                _survivable((mx, my), 1, inst2, grid, bt2, horizon):
            safe[5] = True
    return safe


# Backwards-compatible alias: callers that imported the old name still work, now
# backed by the bounded-horizon shield.
def safe_action_mask(obs, agent_id, horizon=DEFAULT_SHIELD_HORIZON):
    mask = survivable_action_mask(obs, agent_id, horizon)
    return mask if mask.any() else physical_action_mask(obs, agent_id)


def shielded_action(obs, agent_id, logits, use_shield=True,
                    horizon=DEFAULT_SHIELD_HORIZON):
    """Greedy submission action: argmax of the policy logits over the bounded-horizon
    survivable mask. If genuinely trapped (no survivable action), fall back to the
    physically-legal move that stays alive the LONGEST (gives bomb chains a chance to
    resolve), instead of an arbitrary suicide. Deterministic; the SAME shield the
    policy was trained to sample under."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    if not use_shield:
        mask = physical_action_mask(obs, agent_id)
        logits[~mask] = -1e30
        return int(np.argmax(logits))

    mask = survivable_action_mask(obs, agent_id, horizon)
    if mask.any():
        masked = logits.copy()
        masked[~mask] = -1e30
        return int(np.argmax(masked))

    # Trapped: pick the physically-legal action that dies latest (least-bad).
    phys = physical_action_mask(obs, agent_id)
    if not phys.any():
        return 0
    grid = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    uid = int(agent_id)
    mx, my = int(players[uid][0]), int(players[uid][1])
    bombs = _bombs_array(obs)
    instants, _ = compute_danger(grid, bombs, players)
    best_a, best_score = 0, -1.0
    for a in range(NUM_ACTIONS):
        if not phys[a]:
            continue
        if a in (0, 5):
            npos = (mx, my)
        else:
            dx, dy = ACTION_DELTA[a]
            npos = (mx + dx, my + dy)
        score = _latest_lethal(instants, npos)
        if score > best_score:
            best_score, best_a = score, a
    return int(best_a)


# ── small helpers shared by trainer / eval ───────────────────────────────────
def policy_logits(net, obs, agent_id, device=None):
    """Run the actor and return raw logits as a numpy array (no_grad)."""
    ms, xs = encode_obs(obs, agent_id)
    mt = torch.from_numpy(ms).unsqueeze(0)
    xt = torch.from_numpy(xs).unsqueeze(0)
    if device is not None:
        mt, xt = mt.to(device), xt.to(device)
    with torch.no_grad():
        logits, _ = net(mt, xt)
    return logits[0].cpu().numpy()
