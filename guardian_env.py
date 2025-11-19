import numpy as np
from collections import deque

# Simple Gym-like environment for Guardian's Sanctum
# Actions: 0=up,1=right,2=down,3=left,4=fast_up,5=fast_right,6=fast_down,7=fast_left

class GuardianEnv:
    def __init__(self):
        # Map layout: S=start, K=key, D=door, T=treasure, .=floor, #=wall
        # Simple sample layout (7x7)
        self.raw_map = [
            "#######",
            "#S...K#",
            "#.#.#.#",
            "#..D..#",
            "#.#.#.#",
            "#...T.#",
            "#######",
        ]
        self.h = len(self.raw_map)
        self.w = len(self.raw_map[0])
        self._parse_map()
        # sentinel patrols: list of patrol routes (list of (x,y) positions)
        self.sentinels = [
            [(1,4),(2,4),(3,4),(2,4)],
            [(5,2),(5,3),(5,4),(5,3)],
        ]
        # internal state
        self.step_count = 0
        self.max_steps = 500

    def _parse_map(self):
        self.walls = set()
        self.start = None
        self.key_pos = None
        self.door_pos = None
        self.treasure_pos = None
        for y,row in enumerate(self.raw_map):
            for x,ch in enumerate(row):
                if ch == '#':
                    self.walls.add((x,y))
                elif ch == 'S':
                    self.start = (x,y)
                elif ch == 'K':
                    self.key_pos = (x,y)
                elif ch == 'D':
                    self.door_pos = (x,y)
                elif ch == 'T':
                    self.treasure_pos = (x,y)

    def reset(self):
        self.agent_pos = self.start
        self.has_key = False
        # sentinel indices along patrols (index into each route)
        self.sentinel_idxs = [0 for _ in self.sentinels]
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        # encode state as a tuple (x,y,has_key, phase)
        phase = self.step_count % self._patrol_cycle()
        return (self.agent_pos[0], self.agent_pos[1], int(self.has_key), phase)

    def _patrol_cycle(self):
        # naive lcm: use product for simplicity (small patrol lengths)
        prod = 1
        for route in self.sentinels:
            prod *= len(route)
        return prod

    def _in_bounds(self,pos):
        x,y = pos
        return 0 <= x < self.w and 0 <= y < self.h

    def _is_free(self,pos):
        # door blocks if not opened
        if pos in self.walls:
            return False
        if pos == self.door_pos and not self.has_key:
            return False
        return True

    def _move(self,pos,action):
        x,y = pos
        if action == 0 or action == 4:  # up
            return (x,y-1)
        if action == 1 or action == 5:  # right
            return (x+1,y)
        if action == 2 or action == 6:  # down
            return (x,y+1)
        if action == 3 or action == 7:  # left
            return (x-1,y)
        return pos

    def step(self, action):
        assert 0 <= action < 8
        fast = action >=4
        base_action = action % 4
        reward = -1.0  # small step penalty
        done = False
        info = {}

        # perform move (fast moves attempt two steps)
        target = self._move(self.agent_pos, action)
        if fast:
            # first intermediate step
            mid = self._move(self.agent_pos, base_action)
            if self._in_bounds(mid) and self._is_free(mid):
                self.agent_pos = mid
            # second step
            final = self._move(self.agent_pos, base_action)
            if self._in_bounds(final) and self._is_free(final):
                self.agent_pos = final
            noise_pos = self.agent_pos
        else:
            if self._in_bounds(target) and self._is_free(target):
                self.agent_pos = target
            noise_pos = None

        # check for key pickup
        if (self.agent_pos == self.key_pos) and (not self.has_key):
            self.has_key = True
            reward += 50.0

        # sentinels move after agent; compute their next positions
        new_positions = []
        for i,route in enumerate(self.sentinels):
            cur_idx = self.sentinel_idxs[i]
            cur_pos = route[cur_idx]
            next_idx = (cur_idx + 1) % len(route)
            # if noise and sentinel within radius 3, move one step toward the noise
            if noise_pos is not None and abs(cur_pos[0]-noise_pos[0]) + abs(cur_pos[1]-noise_pos[1]) <= 3:
                # move one step towards noise_pos
                dx = np.sign(noise_pos[0] - cur_pos[0])
                dy = np.sign(noise_pos[1] - cur_pos[1])
                candidate = (cur_pos[0] + dx, cur_pos[1] + dy)
                # if candidate is free (not wall), use it; else use patrol next
                if candidate not in self.walls:
                    new_positions.append(candidate)
                    # note: do not advance patrol idx when attracted
                    self.sentinel_idxs[i] = cur_idx  # freeze index
                    continue
            # default: follow patrol
            new_positions.append(route[next_idx])
            self.sentinel_idxs[i] = next_idx

        # check caught: if any sentinel position equals agent
        for s_pos in new_positions:
            if s_pos == self.agent_pos:
                reward -= 100.0
                done = True
                info['caught'] = True
                return self._get_state(), reward, done, info

        # update step count
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            info['timeout'] = True

        # check treasure
        if self.agent_pos == self.treasure_pos and self.has_key:
            reward += 200.0
            done = True
            info['win'] = True

        return self._get_state(), reward, done, info

    def render(self):
        # simple ASCII render
        grid = [list(r) for r in self.raw_map]
        ax,ay = self.agent_pos
        grid[ay][ax] = 'A'
        for i,route in enumerate(self.sentinels):
            si = self.sentinel_idxs[i]
            sx,sy = route[si]
            grid[sy][sx] = 'S'
        return '\n'.join(''.join(row) for row in grid)

if __name__ == '__main__':
    env = GuardianEnv()
    s = env.reset()
    print(env.render())
    for _ in range(5):
        s,r,d,info = env.step(1)
        print('\nStep',_,'reward',r,'done',d)
        print(env.render())
        if d:
            break