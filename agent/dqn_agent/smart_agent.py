"""
Search / heuristic SURVIVAL-FIRST agent for Bomberland.   (v2 "hunter")

WHY A SEARCH AGENT: the PPO+BC+shield lineage plateaued (~mu 114) and every
aggression fine-tune REGRESSED (the policy turned reckless and died ~step 365 vs
the proven ~440 -- those used a SOFT reward to push aggression). This game has a
KNOWN deterministic forward model and a 100ms/step budget, so a survival-FIRST
search agent that reasons about blasts/escape-room beats a reactive net -- and,
crucially, every aggressive choice it makes is HARD-GATED by a guaranteed escape,
so aggression can never make it die early the way the PPO experiments did.

v1 (this file's previous version) survived 49/50 vs the rule MIX but UNDER-CONVERTED
(wins ~36-43%): it reached a TIE instead of cornering+killing the last enemy, and a
strong opponent could bomb it mid-farm. v2 keeps the exact survival-first design and
the 49/50 floor and adds three upgrades that map 1:1 to the three failures the user
reported -- each still gated by the survival floor:

  A) ADVERSARIAL SURVIVAL  ("bị top-tier bom chết khi đang farm"):
       * rank a destination by ENEMY-PROOF escape room (room an enemy cannot deny by
         dropping a bomb right now), not just room vs the bombs already on the board;
       * plan the bomb-placement escape over the FULL 7-step fuse (was 6 -> it under-
         counted our own freshly-placed bomb) AND against the nearest enemy's
         counter-bomb -> we stop walking into squares a strong bot can trap us in.

  B) ACTIVE CORNERING     ("giữ 2-3 bom nhưng không ép / dồn địch vào góc"):
       * in HUNT phase, prefer the move from which a bomb would most SHRINK the target
         enemy's escape room (a 1-ply "get into a trapping position" search), so we
         herd it toward walls/corners instead of merely shadowing it;
       * STRIKE when the enemy's post-bomb escape room is tiny (near-certain), or apply
         pressure in the 1v1 endgame while holding a reserve bomb.

  C) NEVER IDLE / NEVER CIRCLE  ("farm xong không biết đi đâu, đứng đơ / đi quanh mình"):
       * always carry a direction -- the nearest enemy is the fallback objective, so
         when nothing's left to farm we PRESSURE instead of wandering in place;
       * keep farming boxes (safely) even once powered when no kill is on -> out-farms
         v1 (more radius -> bigger blasts -> easier kills + better step-500 tiebreak).

Pure numpy + model.py primitives (NO net, NO model.pt) -> ships flat as
agent.py + model.py + smart_agent.py. Every new behaviour is behind a keyword flag
(default ON) so test_smart_ab.py can flip them off to reproduce v1 for a paired A/B.
"""
from __future__ import annotations

from collections import deque

import numpy as np

try:                                  # flat submission (siblings already on sys.path)
    from model import (compute_danger, _blast_tiles, _bombs_array, _walkable,
                       _survivable, ACTION_DELTA, NUM_ACTIONS, BOMB_TIMER_MAX,
                       GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY)
except ImportError:
    try:                              # package (local dev: python -m agent.dqn_agent...)
        from .model import (compute_danger, _blast_tiles, _bombs_array, _walkable,
                            _survivable, ACTION_DELTA, NUM_ACTIONS, BOMB_TIMER_MAX,
                            GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY)
    except ImportError:               # flat but dir NOT on sys.path (grader precheck via
        import os as _os, sys as _sys  # file-path import) -> add our own dir, retry absolute
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from model import (compute_danger, _blast_tiles, _bombs_array, _walkable,
                           _survivable, ACTION_DELTA, NUM_ACTIONS, BOMB_TIMER_MAX,
                           GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY)

RADIUS_BONUS_MAX = 4                   # bonus in [0,4] -> radius up to 5
DEFAULT_HORIZON = 6                    # survival lookahead vs existing bombs (shield sweet spot)

# ── decision weights (survival dominates; tuned via test_smart_ab.py) ─────────
W_ROOM = 1.0          # per reachable safe tile at the destination (escape room vs live bombs)
W_SAFEROOM = 1.5      # per ENEMY-PROOF safe tile (room an enemy can't deny by bombing now)
W_THREAT = 8.0        # penalty for standing where an enemy could blast you this step
W_PROGRESS = 2.5      # per tile closer to the current objective (farm target / enemy)
W_STOP = 6.0          # penalty for STOP (anti-idle; only chosen if nothing better)
W_CORNER = 1.0        # HUNT: penalty per tile of the target enemy's room from this position
                      #       (minimising it = move into a trapping/cornering position)

HUNT_RADIUS_BONUS = 3  # bonus>=3 (radius>=4) -> powered, switch from farming to hunting
ENEMY_NEAR = 7         # Manhattan distance at which an enemy is "close" (gates farm-vs-hunt)
CORNER_RANGE = 5       # only engage the cornering pull within this Manhattan range of the
                       # target (far away -> just navigate; don't starve item-grabbing)
STRIKE_ROOM = 6        # bomb an enemy only if its post-bomb escape room <= this (a real trap,
                       # not a speculative pressure bomb -- speculative bombs waste the bomb,
                       # open the map, and LOWERED win-rate locally / regressed 3x on the board)
COUNTER_BOMB_NEAR = 5  # only harden a bomb's escape against an enemy counter-bomb when this close


# ── small geometry / danger helpers ──────────────────────────────────────────
def _imminent(instants, pos):
    s = instants.get(pos)
    return s is not None and min(s) <= 1


def _live_enemies(players, uid):
    """(x, y, radius, pid) for every live opponent."""
    return [(int(players[p][0]), int(players[p][1]), 1 + int(players[p][4]), p)
            for p in range(len(players))
            if p != uid and int(players[p][2]) == 1]


def _nearest_enemy(pos, enemies):
    """The live enemy closest by Manhattan distance (target for cornering)."""
    if not enemies:
        return None
    return min(enemies, key=lambda e: abs(e[0] - pos[0]) + abs(e[1] - pos[1]))


def _enemy_blast_tiles(grid, enemies):
    """Every tile some live enemy could cover if it dropped a bomb RIGHT NOW. Standing
    here means an enemy can threaten you next step -> a tile to avoid when we have a
    choice (this is what 'bị dí'/being herded into a trap looks like)."""
    tiles = set()
    for (ex, ey, er, _p) in enemies:
        tiles.update(_blast_tiles(grid, ex, ey, er))
    return tiles


def _escape_room(grid, start, instants, bomb_tiles, max_depth=12):
    """Flood-fill count of distinct walkable, non-bomb, non-imminent tiles reachable
    from `start` within `max_depth` steps. A higher number == more room to run ==
    much harder to corner. This is the core survival signal the rule baselines lack.
    Pass `bomb_tiles = live_bomb_tiles | enemy_blast_tiles` to get ENEMY-PROOF room."""
    q = deque([(start, 0)])
    seen = {start}
    n = 0
    while q:
        pos, d = q.popleft()
        n += 1
        if d >= max_depth:
            continue
        x, y = pos
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            np_ = (x + dx, y + dy)
            if np_ in seen or not _walkable(grid, np_[0], np_[1]) or np_ in bomb_tiles:
                continue
            if _imminent(instants, np_):
                continue
            seen.add(np_)
            q.append((np_, d + 1))
    return n


def _bfs_dist(grid, start, targets, bomb_tiles, avoid, max_depth=40):
    """Shortest #steps from `start` to the nearest tile in `targets`, over walkable
    tiles, avoiding bombs and `avoid` (imminent-danger tiles). Returns a big number if
    unreachable. `targets` may include tiles adjacent to (not on) a box/enemy."""
    if not targets:
        return 10 ** 6
    if start in targets:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        pos, d = q.popleft()
        if d >= max_depth:
            continue
        x, y = pos
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            np_ = (x + dx, y + dy)
            if np_ in seen:
                continue
            if np_ in targets:
                return d + 1
            if not _walkable(grid, np_[0], np_[1]) or np_ in bomb_tiles or np_ in avoid:
                continue
            seen.add(np_)
            q.append((np_, d + 1))
    return 10 ** 6


# ── objectives (phase: items -> boxes -> enemies) ─────────────────────────────
def _item_tiles(grid):
    H, W = grid.shape
    return {(x, y) for x in range(H) for y in range(W)
            if grid[x, y] == ITEM_RADIUS or grid[x, y] == ITEM_CAPACITY}


def _box_spot_tiles(grid):
    """Walkable tiles ADJACENT to a box (stand here to bomb the box)."""
    H, W = grid.shape
    spots = set()
    for x in range(H):
        for y in range(W):
            if grid[x, y] != BOX:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ax, ay = x + dx, y + dy
                if _walkable(grid, ax, ay):
                    spots.add((ax, ay))
    return spots


def _enemy_adj_tiles(grid, enemies):
    """Walkable tiles adjacent to any live enemy (stand here to threaten it)."""
    adj = set()
    for (ex, ey, _er, _p) in enemies:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ax, ay = ex + dx, ey + dy
            if _walkable(grid, ax, ay):
                adj.add((ax, ay))
    return adj


def _objective_targets(grid, players, uid, enemies, powered):
    """Goal tiles by phase. Items are always worth grabbing; then farm boxes while
    under-powered; once powered (or no boxes/items left) hunt enemies. The nearest
    enemy is ALWAYS the final fallback so the agent never runs out of direction
    (the 'farm xong đứng đơ / đi quanh mình' bug)."""
    items = _item_tiles(grid)
    if items:
        return items, "farm"
    box_spots = _box_spot_tiles(grid)
    if box_spots and not powered:
        return box_spots, "farm"
    enemy_adj = _enemy_adj_tiles(grid, enemies)
    if enemy_adj:
        return enemy_adj, "hunt"
    if box_spots:                       # powered but no enemy reachable -> keep farming
        return box_spots, "farm"
    # last resort: head onto the enemy tiles directly (keeps a direction)
    return {(ex, ey) for (ex, ey, _r, _p) in enemies}, "hunt"


# ── bombing decisions ─────────────────────────────────────────────────────────
def _bomb_hits(grid, enemies, bx, by, radius):
    """(hits_box, hits_enemy) for a bomb dropped at (bx,by)."""
    enemy_pos = {(ex, ey) for (ex, ey, _r, _p) in enemies}
    hits_box = hits_enemy = False
    for (tx, ty) in _blast_tiles(grid, bx, by, radius):
        if (tx, ty) in enemy_pos:
            hits_enemy = True
        if grid[tx, ty] == BOX:
            hits_box = True
    return hits_box, hits_enemy


def _bomb_escape_ok(grid, players, bombs, pos, radius, uid, counter_enemy=None):
    """Can we still survive after dropping a bomb here? Plans the escape over the FULL
    7-step fuse of our own fresh bomb (a 6-step horizon under-counted it), and -- when
    `counter_enemy` is given -- against that enemy ALSO dropping a bomb now, so we never
    commit a bomb whose only escape a nearby enemy can immediately cut off."""
    fuse = BOMB_TIMER_MAX
    aug = [list(b) for b in bombs]
    aug.append([int(pos[0]), int(pos[1]), fuse, int(uid)])
    if counter_enemy is not None:
        ex, ey, _er, epid = counter_enemy
        aug.append([int(ex), int(ey), fuse, int(epid)])
    inst2, bt2 = compute_danger(grid, aug, players)
    return (not _imminent(inst2, pos)) and _survivable(pos, 1, inst2, grid, bt2, fuse)


def _enemy_room_after_my_bomb(grid, players, bombs, bomb_pos, radius, enemy, max_depth=8):
    """The target enemy's escape room (reachable safe tiles) if I drop a bomb at
    `bomb_pos` now. Low == the enemy is boxed in == a good time to strike / approach."""
    inst2, bt2 = compute_danger(grid, bombs, players,
                                extra_bomb=(int(bomb_pos[0]), int(bomb_pos[1]), int(radius)))
    return _escape_room(grid, (enemy[0], enemy[1]), inst2, bt2, max_depth=max_depth)


# ── the policy ────────────────────────────────────────────────────────────────
def smart_action(obs, agent_id, horizon=DEFAULT_HORIZON, *,
                 adversarial=True, corner=True, farm_powered=True):
    grid = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    uid = int(agent_id)
    if uid >= len(players) or int(players[uid][2]) != 1:
        return 0

    mx, my = int(players[uid][0]), int(players[uid][1])
    pos = (mx, my)
    bonus = int(players[uid][4])
    radius = 1 + bonus
    bombs_left = int(players[uid][3])
    bombs = _bombs_array(obs)
    instants, bomb_tiles = compute_danger(grid, bombs, players)
    enemies = _live_enemies(players, uid)
    enemy_blast = _enemy_blast_tiles(grid, enemies) if adversarial else set()
    nearest = _nearest_enemy(pos, enemies)
    enemy_close = (nearest is not None
                   and abs(nearest[0] - mx) + abs(nearest[1] - my) <= ENEMY_NEAR)

    # ---- survival floor: physically-legal, survivable moves (incl. STOP) --------
    safe_moves = []                                   # list of (action, next_pos)
    for a in range(5):                                # 0 STOP, 1-4 moves
        if a == 0:
            npos = pos
        else:
            dx, dy = ACTION_DELTA[a]
            npos = (mx + dx, my + dy)
            if not _walkable(grid, npos[0], npos[1]) or npos in bomb_tiles:
                continue
        if _imminent(instants, npos):
            continue
        if not _survivable(npos, 1, instants, grid, bomb_tiles, horizon):
            continue
        safe_moves.append((a, npos))

    # harden a bomb's escape against the nearest enemy's counter-bomb when it's close
    counter = nearest if (adversarial and enemy_close) else None
    bomb_ok = (bombs_left > 0 and pos not in bomb_tiles
               and _bomb_escape_ok(grid, players, bombs, pos, radius, uid, counter))

    if not safe_moves and not bomb_ok:                # genuinely trapped -> die latest
        return _die_latest(grid, players, pos, instants, bombs_left, bomb_tiles)

    in_danger = _imminent(instants, pos) or pos in instants
    box_spots = _box_spot_tiles(grid)
    items_left = bool(_item_tiles(grid))
    powered = bonus >= HUNT_RADIUS_BONUS
    farmed_out = (not box_spots) and (not items_left)
    n_enemies = len(enemies)
    # HUNT when powered or nothing's left to farm; with cornering on, ALSO in the 1v1
    # endgame -- the case where a TIE is converted into a WIN, so stop farming and chase.
    hunt_mode = (nearest is not None) and (powered or farmed_out
                                           or (corner and n_enemies == 1))

    # ---- 1) SMART BOMB (only when not fleeing): strike an enemy or farm a box ----
    if bomb_ok and not _imminent(instants, pos):
        hits_box, hits_enemy = _bomb_hits(grid, enemies, mx, my, radius)
        # (a) STRIKE: a bomb here reaches an enemy AND boxes its escape room in (<=STRIKE_ROOM
        #     reachable safe tiles == a real trap, not a speculative pressure bomb). Already
        #     gated by our own guaranteed escape (bomb_ok), so a strike never risks us.
        if hits_enemy and enemies:
            if corner and nearest is not None:           # target the one we're cornering
                room_after = _enemy_room_after_my_bomb(grid, players, bombs, pos, radius,
                                                       nearest)
            else:                                        # v1: any enemy boxed in
                room_after = min(_enemy_room_after_my_bomb(grid, players, bombs, pos, radius, e)
                                 for e in enemies)
            if room_after <= STRIKE_ROOM:
                return 5
        # (b) FARM: under-powered always opens a box; powered keeps farming only when
        #     no enemy is close (don't burn the bomb we may want for a corner).
        if hits_box:
            if not powered:
                return 5
            if farm_powered and not enemy_close:
                return 5

    # ---- 2) MOVE: best safe move toward the objective, max (enemy-proof) room ----
    targets, phase = _objective_targets(grid, players, uid, enemies, powered)
    if hunt_mode and nearest is not None:
        # hunt TOWARD the enemy but keep items in the target set so we still grab power
        # on the way (chasing blindly bled ~0.8 items/game in the ablation).
        targets = _enemy_adj_tiles(grid, enemies) | _item_tiles(grid)
        phase = "hunt"
    danger_tiles = {t for t, s in instants.items() if min(s) <= 1}
    in_corner_range = (nearest is not None
                       and abs(nearest[0] - mx) + abs(nearest[1] - my) <= CORNER_RANGE)

    best_a, best_score = None, -1e18
    for (a, npos) in safe_moves:
        room = _escape_room(grid, npos, instants, bomb_tiles)
        if adversarial:
            safe_room = _escape_room(grid, npos, instants, bomb_tiles | enemy_blast)
        else:
            safe_room = room
        threat = 1.0 if npos in enemy_blast else 0.0
        dist = _bfs_dist(grid, npos, targets, bomb_tiles, danger_tiles)
        progress = -min(dist, 60)

        if in_danger:                                 # escaping a live blast: room is all
            score = (W_ROOM * room * 2.0 + W_SAFEROOM * safe_room
                     - W_THREAT * threat - (W_STOP if a == 0 else 0.0))
        else:
            score = (W_ROOM * room + W_SAFEROOM * safe_room - W_THREAT * threat
                     + W_PROGRESS * progress - (W_STOP if a == 0 else 0.0))
            # CORNERING: once CLOSE to the target, pull toward a tile from which a bomb
            # would most shrink its escape room (herd it into a corner). Range-gated so
            # long-range navigation still goes for items first.
            if (corner and phase == "hunt" and nearest is not None and bombs_left > 0
                    and in_corner_range):
                e_room = _enemy_room_after_my_bomb(grid, players, bombs, npos, radius,
                                                   nearest)
                score -= W_CORNER * min(e_room, 20)

        if score > best_score:
            best_score, best_a = score, a

    if best_a is not None:
        return int(best_a)
    return 5 if bomb_ok else 0


def _die_latest(grid, players, pos, instants, bombs_left, bomb_tiles):
    """Trapped: take the physically-legal action whose tile stays lethal-free the
    longest (gives bomb chains a chance to clear) instead of an arbitrary suicide."""
    mx, my = pos
    best_a, best = 0, -1.0
    for a in range(5):
        if a == 0:
            npos = pos
        else:
            dx, dy = ACTION_DELTA[a]
            npos = (mx + dx, my + dy)
            if not _walkable(grid, npos[0], npos[1]) or npos in bomb_tiles:
                continue
        s = instants.get(npos)
        latest = max(s) if s else 10 ** 6
        if latest > best:
            best, best_a = latest, a
    return int(best_a)


class SmartAgent:
    """Submission-compatible: SmartAgent(agent_id).act(obs) -> int in [0,5].
    No weights file needed -- pure search/heuristic. Flags expose the v2 upgrades for
    A/B testing (all default ON; set them False to reproduce v1)."""

    team_id = "SmartSearch"

    def __init__(self, agent_id: int, horizon: int = DEFAULT_HORIZON, *,
                 adversarial: bool = True, corner: bool = True, farm_powered: bool = True):
        self.agent_id = int(agent_id)
        self.horizon = int(horizon)
        self.adversarial = bool(adversarial)
        self.corner = bool(corner)
        self.farm_powered = bool(farm_powered)

    def reset(self):
        pass

    def act(self, obs) -> int:
        try:
            return int(smart_action(obs, self.agent_id, self.horizon,
                                    adversarial=self.adversarial, corner=self.corner,
                                    farm_powered=self.farm_powered))
        except Exception:
            return 0
