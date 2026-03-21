import pygame
import math

from random import randint


class Food:
    def __init__(self, blocks_x, blocks_y, color) -> None:
        self.blocks_x = blocks_x
        self.blocks_y = blocks_y
        self.block = Block(0, 0, color)

    def new_food(self, blocks):
        (x, y) = (randint(0, self.blocks_x-1), randint(0, self.blocks_y-1))
        done = False

        while not done:
            done = True
            for block in blocks:
                if (x, y) == (block.x, block.y):
                    (x, y) = (randint(0, self.blocks_x-1),
                              randint(0, self.blocks_y-1))
                    done = False
                    break
        self.block.move_to(x, y)


class Block:
    size = 20

    def __init__(self, x, y, color) -> None:
        self.x = x
        self.y = y
        self.color = color

    def copy(self, dx, dy, color=None):
        if color is None:
            color = self.color
        return Block(self.x+dx, self.y+dy, color)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def move_to(self, x, y):
        self.x = x
        self.y = y

    @property
    def rect(self):
        return (self.x*Block.size, self.y*Block.size, Block.size, Block.size)


class Color:
    red = (255, 90, 90)
    orange = (255, 169, 89)
    blue = (89, 172, 255)
    green_blue = (89, 255, 172)
    purple = (197, 90, 255)


class Direction:
    up = (0, -1)
    down = (0, 1)
    left = (-1, 0)
    right = (1, 0)

    @staticmethod
    def step(dir):
        match dir:
            case 0:
                return Direction.up
            case 1:
                return Direction.down
            case 2:
                return Direction.left
            case 3:
                return Direction.right


def normalize(x, y):
    c = math.sqrt(x*x + y*y)
    return x/c, y/c


def handle_input():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.display.quit()
            pygame.quit()
            return None
    key_list = pygame.key.get_pressed()
    if key_list[pygame.K_ESCAPE]:
        pygame.display.quit()
        pygame.quit()
        return None
    key_list = pygame.key.get_pressed()
    u, d, l, r = key_list[pygame.K_UP] or key_list[pygame.K_w], \
        key_list[pygame.K_DOWN] or key_list[pygame.K_s], \
        key_list[pygame.K_LEFT] or key_list[pygame.K_a], \
        key_list[pygame.K_RIGHT] or key_list[pygame.K_d]
    if u:
        return 0
    if d:
        return 1
    if l:
        return 2
    if r:
        return 3
    return None


def update_screen(screen, snake, human_playing=False):
    if not pygame.display.get_init():
        return
    width = screen.get_width()
    height = screen.get_height() - 40
    font = pygame.font.SysFont('microsoft Yahei', 30, True)
    score = font.render('Scores: '+str(snake.score), False, Color.purple)
    episode = font.render('Episodes: '+str(snake.episode), False, Color.purple)
    step_remain = font.render(
        'Steps Remain: '+str(snake.max_step-snake.current_step), False, Color.purple)
    screen.fill(Color.orange)
    screen.blit(score, (20, height))
    if human_playing:
        fps = font.render('Speed: '+str(snake.fps), False, Color.purple)
        screen.blit(fps, (200, height))
    else:
        screen.blit(episode, (200, height))
        screen.blit(step_remain, (500, height))

    for block in snake.blocks:
        pygame.draw.rect(screen, block.color, pygame.Rect(block.rect))
    pygame.draw.line(screen, Color.purple, (0, height),
                     (width, height), 2)
    pygame.display.flip()


def game_start(width, height, score_board=40):
    pygame.init()
    pygame.display.set_caption('Snake')
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((width, height + score_board))
    return screen, clock


def calc_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return advantage_list
