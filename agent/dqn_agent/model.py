"""
Shared inference core for the Bomberland DQN agent.

This module is imported by BOTH the submission (`agent.py`) and the trainer
(`train.py`), so the *exact same* state encoding, network architecture and
safety logic are used at training and at evaluation time.

Hard dependencies: only `numpy` and `torch` (both guaranteed by the evaluation
environment). Nothing here imports the engine, reward.py, tqdm or matplotlib,
so it is safe to ship in the flat submission zip.

4-player (13x13) | CPU inference < 100ms | Dueling Double DQN + safety mask.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ── Game constants ───────────────────────────────────────────────────────────
BOMB_TIMER_MAX   = 7      # bombs are placed with timer 7
BOMB_RADIUS_MAX  = 5      # actual radius = 1 + bonus, capped at 5
RADIUS_BONUS_MAX = 4      # bonus in [0, 4]
CAP_MAX          = 5      # bomb capacity capped at 5
NUM_ACTIONS      = 6      # 0 STOP 1 LEFT 2 RIGHT 3 UP 4 DOWN 5 PLACE_BOMB

# Map cell codes (must match engine.map.Map)
GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY = 0, 1, 2, 3, 4

# action -> (dx, dy) in (row, col); PLACE_BOMB / STOP keep position
ACTION_DELTA = {
    0: (0, 0),    # STOP
    1: (-1, 0),   # LEFT  (row - 1)
    2: (1, 0),    # RIGHT (row + 1)
    3: (0, -1),   # UP    (col - 1)
    4: (0, 1),    # DOWN  (col + 1)
}

# Encoding dimensions — DO NOT change without retraining the network.
# map channels: 5 terrain one-hot + self + foes + my_bomb + foe_bomb + danger + imminent
N_MAP_CH = 11
N_AUX    = 5


# ── Bomb / blast simulation (mirrors engine exactly) ─────────────────────────
def _bombs_array(obs):
    """Return bombs as a clean (M, 4) int array: [row, col, timer, owner]."""
    raw = obs.get("bombs")
    if raw is None:
        return np.zeros((0, 4), dtype=np.int64)
    arr = np.asarray(raw)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.zeros((arr.shape[0], 4), dtype=np.int64)
    out[:, : min(4, arr.shape[1])] = arr[:, :4]
    if arr.shape[1] < 3:                # timer missing -> assume freshly placed
        out[:, 2] = BOMB_TIMER_MAX
    return out


def _bomb_radius(players, owner_id):
    """Best-effort radius (obs does not expose the bomb's frozen radius)."""
    oid = int(owner_id)
    if 0 <= oid < len(players):
        return 1 + int(players[oid][4])
    return 2


def _blast_tiles(grid, bx, by, radius):
    """Cross-shaped blast: stops at walls (excluded) and boxes (included)."""
    H, W = grid.shape
    tiles = [(bx, by)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < H and 0 <= ty < W):
                break
            cell = grid[tx, ty]
            if cell == WALL:
                break
            tiles.append((tx, ty))
            if cell == BOX:
                break
    return tiles


def compute_danger(grid, bombs, players, extra_bomb=None):
    """
    Time-expanded danger model.

    Returns
    -------
    instants : dict[(x, y) -> set[int]]
        For every tile that any blast covers, the set of *relative* steps at
        which it is lethal (1 == explodes at the end of the upcoming step).
    bomb_tiles : set[(x, y)]
        Tiles occupied by a bomb (block movement / placement).

    `extra_bomb`, if given as (x, y, radius), simulates placing a new bomb now
    (timer 7) so the placement-safety check can plan an escape.
    """
    rows = []  # (x, y, steps_until, radius)
    for b in bombs:
        bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        steps = timer if timer > 0 else 1
        rows.append([bx, by, max(1, steps), _bomb_radius(players, owner)])
    if extra_bomb is not None:
        ex, ey, er = extra_bomb
        rows.append([int(ex), int(ey), BOMB_TIMER_MAX, int(er)])

    bomb_tiles = {(r[0], r[1]) for r in rows}
    if not rows:
        return {}, bomb_tiles

    blasts = [_blast_tiles(grid, r[0], r[1], r[3]) for r in rows]

    # Chain reaction: a bomb caught in another's blast detonates at that time.
    changed = True
    while changed:
        changed = False
        for i in range(len(rows)):
            for j in range(len(rows)):
                if i == j:
                    continue
                if (rows[j][0], rows[j][1]) in blasts[i] and rows[i][2] < rows[j][2]:
                    rows[j][2] = rows[i][2]
                    changed = True

    instants: dict = {}
    for r, blast in zip(rows, blasts):
        t = r[2]
        for tile in blast:
            s = instants.get(tile)
            if s is None:
                instants[tile] = {t}
            else:
                s.add(t)
    return instants, bomb_tiles


# ── State encoder ────────────────────────────────────────────────────────────
def encode_obs(obs, agent_id):
    """
    Encode an observation into (map_feat, aux_feat).

    map_feat : float32 (N_MAP_CH, H, W)
    aux_feat : float32 (N_AUX,)
    """
    uid = int(agent_id)
    grid = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    H, W = grid.shape
    N = players.shape[0]

    mx, my = int(players[uid][0]), int(players[uid][1])
    my_alive = int(players[uid][2])
    my_bleft = int(players[uid][3])
    my_bonus = int(players[uid][4])

    # terrain one-hot (5)
    terrain = [(grid == v).astype(np.float32) for v in range(5)]

    ch_me = np.zeros((H, W), np.float32)
    if my_alive:
        ch_me[mx, my] = 1.0

    ch_foes = np.zeros((H, W), np.float32)
    n_foes = 0
    for pid in range(N):
        if pid == uid:
            continue
        if int(players[pid][2]) == 1:
            ch_foes[int(players[pid][0]), int(players[pid][1])] += 1.0
            n_foes += 1

    bombs = _bombs_array(obs)
    ch_mybomb = np.zeros((H, W), np.float32)
    ch_fobomb = np.zeros((H, W), np.float32)
    for b in bombs:
        bx, by, owner = int(b[0]), int(b[1]), int(b[3])
        if owner == uid:
            ch_mybomb[bx, by] = 1.0
        else:
            ch_fobomb[bx, by] = 1.0

    # danger channels from the time-expanded model
    instants, _ = compute_danger(grid, bombs, players)
    ch_danger = np.zeros((H, W), np.float32)   # urgency: higher == sooner
    ch_imminent = np.zeros((H, W), np.float32)  # lethal this step
    my_in_danger = 0.0
    for (tx, ty), s in instants.items():
        m = min(s)
        ch_danger[tx, ty] = (BOMB_TIMER_MAX - m + 1) / BOMB_TIMER_MAX
        if m <= 1:
            ch_imminent[tx, ty] = 1.0
        if (tx, ty) == (mx, my):
            my_in_danger = 1.0

    map_feat = np.stack(
        terrain + [ch_me, ch_foes, ch_mybomb, ch_fobomb, ch_danger, ch_imminent],
        axis=0,
    ).astype(np.float32)

    on_own_bomb = 1.0 if ch_mybomb[mx, my] > 0 else 0.0
    aux_feat = np.array([
        my_bleft / CAP_MAX,
        my_bonus / RADIUS_BONUS_MAX,
        n_foes / max(N - 1, 1),
        my_in_danger,
        on_own_bomb,
    ], dtype=np.float32)

    return map_feat, aux_feat


# ── Network: Dueling DQN (TorchScript-friendly) ──────────────────────────────
class DQNModel(nn.Module):
    def __init__(self, map_shape, aux_dim, num_actions):
        super().__init__()
        c, h, w = map_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 1), nn.ReLU(inplace=True),  # 1x1 channel squeeze
        )
        with torch.no_grad():
            flat = int(self.conv(torch.zeros(1, c, h, w)).reshape(1, -1).size(1))
        self.aux_net = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(nn.Linear(flat + 32, 256), nn.ReLU(inplace=True))
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True),
                                   nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True),
                                 nn.Linear(128, num_actions))

    def forward(self, map_x, aux_x):
        spatial = self.conv(map_x).flatten(1)
        aux = self.aux_net(aux_x)
        f = self.trunk(torch.cat([spatial, aux], dim=1))
        v = self.value(f)
        a = self.adv(f)
        return v + (a - a.mean(dim=1, keepdim=True))


# ── Safety planner (pure python/numpy, fast) ─────────────────────────────────
def _walkable(grid, x, y):
    H, W = grid.shape
    return 0 < x < H - 1 and 0 < y < W - 1 and grid[x, y] in (GRASS, ITEM_RADIUS, ITEM_CAPACITY)


def _survivable(start, k0, instants, grid, bomb_tiles, horizon):
    """
    Can an agent currently at `start` (having safely completed step k0) survive
    every remaining explosion? Time-expanded BFS over (position, step). Returns
    True if a sequence of moves/stops avoids all lethal instants.
    """
    if not instants:
        return True
    memo: dict = {}

    def rec(pos, k):
        if k > horizon:
            return True
        future = instants.get(pos)
        if future is None or max(future) <= k:
            return True   # no future explosion can reach this tile -> stay forever
        key = (pos, k)
        cached = memo.get(key)
        if cached is not None:
            return cached
        memo[key] = False  # guard against cycles
        nk = k + 1
        # candidate next tiles: stay, then the four moves
        cands = [pos]
        x, y = pos
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if _walkable(grid, nx, ny) and (nx, ny) not in bomb_tiles:
                cands.append((nx, ny))
        ok = False
        for npos in cands:
            fut = instants.get(npos)
            if fut is not None and nk in fut:
                continue   # would be standing in a blast at instant nk -> dead
            if rec(npos, nk):
                ok = True
                break
        memo[key] = ok
        return ok

    return rec(start, k0)


def safe_action(obs, agent_id, q_values):
    """
    Pick an action: greedily follow the network's Q-ranking but discard moves
    that are invalid or lead to unavoidable death. Returns int in [0, 5].

    `q_values` is a length-NUM_ACTIONS array (np or list).
    """
    grid = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    uid = int(agent_id)
    if uid >= len(players) or int(players[uid][2]) != 1:
        return 0

    mx, my = int(players[uid][0]), int(players[uid][1])
    my_bleft = int(players[uid][3])
    my_radius = 1 + int(players[uid][4])
    pos = (mx, my)

    bombs = _bombs_array(obs)
    instants, bomb_tiles = compute_danger(grid, bombs, players)
    horizon = max((max(s) for s in instants.values()), default=0)

    allowed = []
    fallback_scores = {}  # action -> latest lethal instant reached (for least-bad)

    for a in range(NUM_ACTIONS):
        if a == 5:  # PLACE_BOMB
            if my_bleft <= 0 or pos in bomb_tiles:
                continue
            inst2, btiles2 = compute_danger(
                grid, bombs, players, extra_bomb=(mx, my, my_radius))
            hz2 = max((max(s) for s in inst2.values()), default=0)
            if 1 not in inst2.get(pos, set()) and _survivable(pos, 1, inst2, grid, btiles2, hz2):
                allowed.append(a)
            continue

        dx, dy = ACTION_DELTA[a]
        npos = (mx + dx, my + dy)
        if a == 0:
            npos = pos
        else:
            if not _walkable(grid, npos[0], npos[1]) or npos in bomb_tiles:
                continue
        lethal_next = 1 in instants.get(npos, set())
        # least-bad bookkeeping: how long do we survive heading to npos?
        fut = instants.get(npos)
        fallback_scores[a] = (10**6 if fut is None else max(fut)) - (0 if not lethal_next else 100)
        if not lethal_next and _survivable(npos, 1, instants, grid, bomb_tiles, horizon):
            allowed.append(a)

    q = np.asarray(q_values, dtype=np.float64).reshape(-1)
    if allowed:
        return int(max(allowed, key=lambda a: q[a]))
    # Trapped: choose the move that dies latest (or STOP if nothing better).
    if fallback_scores:
        return int(max(fallback_scores, key=lambda a: fallback_scores[a]))
    return 0


def legal_safe_actions(obs, agent_id):
    """The set of non-suicidal, valid actions (used for safe exploration in training)."""
    grid = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    uid = int(agent_id)
    if uid >= len(players) or int(players[uid][2]) != 1:
        return [0]
    # Reuse safe_action's filtering by feeding flat Q so it returns the full allow-set.
    mx, my = int(players[uid][0]), int(players[uid][1])
    my_bleft = int(players[uid][3])
    my_radius = 1 + int(players[uid][4])
    pos = (mx, my)
    bombs = _bombs_array(obs)
    instants, bomb_tiles = compute_danger(grid, bombs, players)
    horizon = max((max(s) for s in instants.values()), default=0)

    out = []
    for a in range(NUM_ACTIONS):
        if a == 5:
            if my_bleft <= 0 or pos in bomb_tiles:
                continue
            inst2, btiles2 = compute_danger(grid, bombs, players, extra_bomb=(mx, my, my_radius))
            hz2 = max((max(s) for s in inst2.values()), default=0)
            if 1 not in inst2.get(pos, set()) and _survivable(pos, 1, inst2, grid, btiles2, hz2):
                out.append(a)
            continue
        dx, dy = ACTION_DELTA[a]
        npos = pos if a == 0 else (mx + dx, my + dy)
        if a != 0 and (not _walkable(grid, npos[0], npos[1]) or npos in bomb_tiles):
            continue
        if 1 not in instants.get(npos, set()) and _survivable(npos, 1, instants, grid, bomb_tiles, horizon):
            out.append(a)
    return out if out else [0]
