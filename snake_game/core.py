import math
import numpy as np
from .utils import *


class Snake:
    def __init__(
        self,
        fps=60,
        max_step=99999,
        init_length=4,
        food_reward=6.0,
        dist_reward=None,
        living_bonus=0.0,
        death_penalty=-10.0,
        no_food_penalty=-0.006,
        steps_since_food = 0,
        width=40,
        height=40,
        block_size=20,
        background_color=Color.orange,
        food_color=Color.red,
        head_color=Color.purple,
        body_color=Color.blue,
    ) -> None:

        self.episode = 0
        self.fps = fps
        self.max_step = max_step
        self.init_length = min(init_length, width//2)
        self.food_reward = food_reward
        self.dist_reward = (
            width+height)//4 if dist_reward is None else dist_reward
        self.living_bonus = living_bonus
        self.no_food_penalty = no_food_penalty
        self.food_limit = width*height-1
        self.death_penalty = death_penalty
        self.steps_since_food = steps_since_food
        self.blocks_x = width
        self.blocks_y = height
        self.food_color = food_color
        self.head_color = head_color
        self.body_color = body_color
        self.background_color = background_color
        self.food = Food(self.blocks_x, self.blocks_y, food_color)
        Block.size = block_size

        self.map = None
        self.screen = None
        self.clock = None
        self.human_playing = False

    def init(self):
        self.episode += 1
        self.score = 0
        self.direction = 3
        self.current_step = 0
        self.head = Block(self.blocks_x//2, self.blocks_y//2, self.head_color)
        self.body = [self.head.copy(i, 0, self.body_color)
                     for i in range(-self.init_length, 0)]
        self.blocks = [self.food.block, self.head, *self.body]
        self.food.new_food(self.blocks)

    def close(self):
        pygame.quit()
        pygame.display.quit()
        self.screen = None
        self.clock = None

    def render(self):
        if self.screen is None:
            self.screen, self.clock = game_start(
                self.blocks_x*Block.size, self.blocks_y*Block.size)
        self.clock.tick(self.fps)
        update_screen(self.screen, self)
        handle_input()

    def step(self, direction):
        if direction is None:
            direction = self.direction
        self.current_step += 1
        truncated = True if self.current_step == self.max_step else False
        (x, y) = (self.head.x, self.head.y)
        step = Direction.step(direction)
        if (direction == 0 or direction == 1) and (self.direction == 0 or self.direction == 1):
            step = Direction.step(self.direction)
        elif (direction == 2 or direction == 3) and (self.direction == 2 or self.direction == 3):
            step = Direction.step(self.direction)
        else:
            self.direction = direction
        self.head.x += step[0]
        self.head.y += step[1]
        self.steps_since_food += 1

        reward = self.living_bonus + self.calc_reward() + self.no_food_penalty*self.steps_since_food
        dead = False

        if self.head == self.food.block:
            self.score += 1
            self.grow(x, y)
            self.food.new_food(self.blocks)
            reward = self.food_reward
            self.steps_since_food = 0
        else:
            self.move(x, y)
            for block in self.body:
                if self.head == block:
                    dead = True
            if self.head.x >= self.blocks_x or self.head.x < 0 or self.head.y < 0 or self.head.y >= self.blocks_y:
                dead = True

        if not dead and self.steps_since_food > self.food_limit:
            truncated = True
            self.steps_since_food = 0

        if dead:
            reward = self.death_penalty
        return self.observation(dead), reward, dead, truncated

    def observation(self, dead=False):
        dx = self.head.x - self.food.block.x
        dy = self.head.y - self.food.block.y
        dx, dy = normalize(dx, dy)
        d0, d1, d2, d3 = self.calc_distance(dead)
        front, left, right = self._head_cell_status(dead)
        # chieu dai ran 
        body_len = len(self.body)/ max(1, self.blocks_x*self.blocks_y)
        return np.array([dx, dy, d0, d1, d2, d3, front, left, right, body_len], dtype=np.float32)

    def calc_distance(self, dead):
        if dead:
            return 0, 0, 0, 0
        self.map = np.zeros((self.blocks_x, self.blocks_y), dtype=int)
        for block in self.blocks:
            if block.x >= 0 and block.x < self.blocks_x and block.y >= 0 and block.y < self.blocks_y:
                self.map[block.x][block.y] = -1
        self.map[self.food.block.x][self.food.block.y] = 0
        d0, d1, d2, d3 = 0, 0, 0, 0,
        x, y = self.head.x, self.head.y - 1
        while y >= 0 and self.map[x][y] == 0:
            d0 += 1
            y -= 1
        x, y = self.head.x, self.head.y + 1
        while y < self.blocks_y and self.map[x][y] == 0:
            d1 += 1
            y += 1
        x, y = self.head.x - 1, self.head.y
        while x >= 0 and self.map[x][y] == 0:
            d2 += 1
            x -= 1
        x, y = self.head.x + 1, self.head.y
        while x < self.blocks_x and self.map[x][y] == 0:
            d3 += 1
            x += 1
        self.map[self.food.block.x][self.food.block.y] = 1
        return d0/self.blocks_y, d1/self.blocks_y, d2/self.blocks_x, d3/self.blocks_x

    def _cell_status(self, x: int, y: int) -> float:
        if x < 0 or x >= self.blocks_x or y < 0 or y >= self.blocks_y:
            return -1.0
        if (x, y) == (self.food.block.x, self.food.block.y):
            return 1.0
        for block in self.body:
            if (x, y) == (block.x, block.y):
                return -1.0
        return 0.0

    def _head_cell_status(self, dead: bool) -> tuple[float, float, float]:
        if dead: 
            return -1.0, -1.0, -1.0
        
        left = {0:2, 2:1, 1:3, 3:0}
        right = {0:3, 3:1, 1:2, 2:0}
        step = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0),
        }

        hx, hy = self.head.x, self.head.y
        fx, fy = step[self.direction]
        lx, ly = step[left[self.direction]]
        rx, ry = step[right[self.direction]]

        front = self._cell_status(hx+fx, hy+fy)
        left = self._cell_status(hx+lx, hy+ly)
        right = self._cell_status(hx+rx, hy+ry)
        return front, left, right
    
    def calc_reward(self):
        if self.dist_reward == 0.0:
            return 0
        # kc euclid
        x = self.head.x - self.food.block.x
        y = self.head.y - self.food.block.y
        d = math.sqrt(x*x + y*y)
        return (self.dist_reward-d)/self.dist_reward

    def grow(self, x, y):
        body = Block(x, y, Color.blue)
        self.blocks.append(body)
        self.body.append(body)

    def move(self, x, y):
        tail = self.body.pop(0)
        tail.move_to(x, y)
        self.body.append(tail)

    def info(self):
        return {
            'head': (self.head.x, self.head.y),
            'food': (self.food.block.x, self.food.block.y),
            'map': self.map.T,
            'score': self.score,
        }

    def play(self, fps=10, acceleration=True, step=1, frep=10):
        self.max_step = 99999
        self.fps = fps
        self.food_reward = 1
        self.living_bonus = 0
        self.dist_reward = 0
        self.death_penalty = 0
        self.human_playing = True
        self.init()
        screen, clock = game_start(
            self.blocks_x*Block.size, self.blocks_y*Block.size)
        total_r = 0
        while pygame.get_init():
            clock.tick(self.fps)
            _, r, d, _ = self.step(handle_input())
            total_r += r
            if acceleration and total_r == frep:
                self.fps += step
                total_r = 0
            if d:
                self.init()
                total_r = 0
                self.fps = fps
            update_screen(screen, self, True)
