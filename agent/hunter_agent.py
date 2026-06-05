import random

from .genius_rule_agent import GeniusRuleAgent


class HunterAgent(GeniusRuleAgent):
    """
    Aggressive ENEMY-FIRST bot. Unlike GeniusRuleAgent (which farms items/boxes
    first and only pressures enemies as a last resort), the Hunter relentlessly
    beelines to the nearest opponent and bombs it, bombing through boxes that
    block the path.

    Purpose: a training sparring partner that PUNISHES camping. In 4-player FFA a
    learner can win cheaply by turtling in a corner while the other bots fight and
    die. Facing a Hunter that comes and corners it, the learner can no longer just
    survive passively -- it must move, control space and fight back. This is the
    pressure that pure reward-shaping cannot create.
    """

    team_id = "HunterAgent"

    def act(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)

        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]

        blocked = set(bomb_positions)
        blocked.discard(my_pos)
        danger_soon, _ = self._danger_tiles(grid, bombs, players)
        valid_actions = self._valid_actions(grid, my_pos, blocked)

        # 1) survive: always escape an incoming blast first
        if self.escape_mode or my_pos in danger_soon:
            escape = self._move_to_nearest_safe(
                grid, my_pos, blocked, danger_soon, search_depth=10)
            if escape is not None:
                if my_pos not in danger_soon:
                    self.escape_mode = False
                return escape

        # 2) if a bomb here would hit an enemy and we can flee -> bomb it
        if bombs_left > 0 and my_pos not in bomb_positions and enemies:
            if self._can_hit_enemy_with_bomb(grid, my_pos, enemies, bomb_radius):
                escape = self._move_to_nearest_safe(
                    grid, my_pos, blocked,
                    danger_soon | self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius),
                    search_depth=6)
                if escape is not None:
                    self.escape_mode = True
                    return 5

        # 3) hunt: move straight toward the nearest enemy
        if enemies:
            move = self._move_toward_targets(
                grid, my_pos, set(enemies), blocked, danger_soon)
            if move is not None:
                return move

            # path to the enemy is walled off by boxes -> blow one open
            if bombs_left > 0 and my_pos not in bomb_positions:
                adj_box = any(
                    self._in_bounds(grid, my_pos[0] + dx, my_pos[1] + dy)
                    and grid[my_pos[0] + dx, my_pos[1] + dy] == 2
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if adj_box:
                    escape = self._move_to_nearest_safe(
                        grid, my_pos, blocked,
                        danger_soon | self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius),
                        search_depth=6)
                    if escape is not None:
                        self.escape_mode = True
                        return 5

        # 4) fallback: any safe move
        safe_moves = [a for a in valid_actions
                      if self._next_pos(my_pos, a) not in danger_soon]
        return random.choice(safe_moves) if safe_moves else 0
