import os
import cv2
import math
import pygame
import random
import numpy as np
from os import listdir
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from os.path import join, isfile
from stable_baselines3 import PPO
from gymnasium.wrappers import frame_stack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


pygame.init()
pygame.display.set_caption("platformer game")

# constants
WIDTH,HEIGHT = 1056, 720
FPS = 30
PLAYER_VEL = 5
REGION_SIZE = 146
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))


def flip(sprites):
    return [pygame.transform.flip(sprite, True, False) for sprite in sprites]


def load_sprite_sheets(dir1, dir2, width, height, direction=False):
    path = join("Python-Platformer-main\Python-Platformer-main", "assets", dir1, dir2)
    images = [f for f in listdir(path) if isfile(join(path, f))]

    all_sprites = {}

    for image in images:
        sprite_sheet = pygame.image.load(join(path, image)).convert_alpha()

        sprites = []
        for i in range(sprite_sheet.get_width() // width):
            surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
            rect = pygame.Rect(i * width, 0, width, height)
            surface.blit(sprite_sheet, (0, 0), rect)
            sprites.append(pygame.transform.scale2x(surface))

        if direction:
            all_sprites[image.replace(".png", "") + "_right"] = sprites
            all_sprites[image.replace(".png", "") + "_left"] = flip(sprites)
        else:
            all_sprites[image.replace(".png", "")] = sprites

    return all_sprites


def get_block(size):
    path = join("Python-Platformer-main\Python-Platformer-main", "assets", "Terrain", "Terrain.png")
    image = pygame.image.load(path).convert_alpha()
    surface = pygame.Surface((size, size), pygame.SRCALPHA, 32)
    rect = pygame.Rect(96, 0, size, size)
    surface.blit(image, (0, 0), rect)
    return pygame.transform.scale2x(surface)


class Player(pygame.sprite.Sprite):
    COLOR = (255, 0, 0)
    GRAVITY = 1
    SPRITES = load_sprite_sheets("MainCharacters", "MaskDude", 32, 32, True)
    ANIMATION_DELAY = 3

    def __init__(self, x, y, width, height):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.x_vel = 0
        self.y_vel = 0
        self.mask = None
        self.direction = "left"
        self.animation_count = 0
        self.fall_count = 0
        self.jump_count = 0
        self.hit = False
        self.hit_count = 0
        self.sprite = self.SPRITES["idle_left"][0] 
        self.sprite_sheet = "idle"
        self.is_jumping = False

    def jump(self):
        self.y_vel = -self.GRAVITY * 8
        self.is_jumping = True
        self.animation_count = 0
        self.jump_count += 1
        if self.jump_count == 1:
            self.fall_count = 0

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

    def make_hit(self):
        self.hit = True

    def move_left(self, vel):
        self.x_vel = -vel
        if self.direction != "left":
            self.direction = "left"
            self.animation_count = 0

    def move_right(self, vel):
        self.x_vel = vel
        # print(self.x_vel, 'hijf')
        if self.direction != "right":
            self.direction = "right"
            self.animation_count = 0

    def jump_and_move_right(self, vel):
        self.y_vel = -self.GRAVITY * 8
        self.x_vel += vel
        self.animation_count = 0
        self.jump_count += 1
        if self.jump_count == 1:
            self.fall_count = 0
        # print(self.x_vel, 'hefjd')

    def loop(self, fps):
        self.y_vel += min(5, (self.fall_count / fps) * self.GRAVITY)
        self.move(self.x_vel, self.y_vel)

        if self.hit:
            self.hit_count += 1
        if self.hit_count > fps * 2:
            self.hit = False
            self.hit_count = 0

        self.fall_count += 1
        self.update_sprite()

    def landed(self):
        self.fall_count = 0
        self.y_vel = 0
        self.jump_count = 0
        self.is_jumping = False

    def hit_head(self):
        self.count = 0
        self.y_vel *= -1

    def update_sprite(self):
        self.sprite_sheet = "idle"
        if self.hit:
            self.sprite_sheet = "hit"
        elif self.y_vel < 0:
            if self.jump_count == 1:
                self.sprite_sheet = "jump"
            elif self.jump_count == 2:
                self.sprite_sheet = "double_jump"
        elif self.y_vel > self.GRAVITY * 2:
            self.sprite_sheet = "fall"
        elif self.x_vel != 0:
            self.sprite_sheet = "run"

        sprite_sheet_name = self.sprite_sheet + "_" + self.direction
        sprites = self.SPRITES[sprite_sheet_name]
        sprite_index = (self.animation_count //
                        self.ANIMATION_DELAY) % len(sprites)
        self.sprite = sprites[sprite_index]
        self.animation_count += 1
        self.update()

    def update(self):
        self.rect = self.sprite.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.sprite)

    def draw(self, win, offset_x):
        win.blit(self.sprite, (self.rect.x - offset_x, self.rect.y))


class Object(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, name=None):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        self.width = width
        self.height = height
        self.name = name

    def draw(self, win, offset_x):
        win.blit(self.image, (self.rect.x - offset_x, self.rect.y))


class Block(Object):
    def __init__(self, x, y, size):
        super().__init__(x, y, size, size)
        block = get_block(size)
        self.image.blit(block, (0, 0))
        self.mask = pygame.mask.from_surface(self.image)


class Fire(Object):
    ANIMATION_DELAY = 3

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "fire")
        self.fire = load_sprite_sheets("Traps", "Fire", width, height)
        self.image = self.fire["off"][0]
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0
        self.animation_name = "off"

    def on(self):
        self.animation_name = "on"

    def off(self):
        self.animation_name = "off"

    def loop(self):
        sprites = self.fire[self.animation_name]
        sprite_index = (self.animation_count //
                        self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_index]
        self.animation_count += 1

        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)

        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

class Checkpoint(Object):
    ANIMATION_DELAY = 4
    def __init__(self, x, y, width=64, height=64):
        super().__init__(x, y,width, height, name="flag")
        self.x = x
        self.y = y
        self.flag = load_sprite_sheets("Items\Checkpoints", "Checkpoint", width, height)
        self.image = self.flag["idle"][0]
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0
        self.animation_name = "idle"

    def loop(self):
        sprites = self.flag[self.animation_name]
        sprite_index = (self.animation_count //
                        self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_index]
        self.animation_count += 1

        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)

        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

class Spikes(Object):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "spikes")
        self.spikes = load_sprite_sheets("Traps", "Spikes", width, height)
        self.image = self.spikes["Idle"][0]
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0
        self.animation_name = "Idle"

class Flying_enemy(Object):
    ANIMATION_DELAY = 4
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "flying_box")
        self.flying_box = load_sprite_sheets("Traps", "flying_enemy", width, height)
        self.image = self.flying_box["Idle"][0]
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0
        self.animation_name = "Idle"
    
    def fly(self):
        self.animation_name = "Fly"
    
    def attack(self):
        self.animation_name = "Attack"

    def hit(self):
        self.animation_name = "Hit"

    def loop(self):
        sprites = self.flying_box[self.animation_name]
        sprite_index = (self.animation_count //
                        self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_index]
        self.animation_count += 1

        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)
        if self.animation_count // self.ANIMATION_DELAY == len(sprites) // 2:
            self.attack()
        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

class Green_enemy(Object):
    ANIMATION_DELAY = 4
    THRESHOLD = 100
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "green_enemy")
        self.x = x
        self.y = y
        self.x_vel = 0
        self.steps = 0
        self.green_enemy = load_sprite_sheets("Traps", "green_enemy", width, height, direction=True)
        self.image = self.green_enemy["Idle_left"][0]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.animation_count = 0
        self.direction = "left"
        self.animation_name = "Idle"
        self.hit = False
        self.hit_count = 0

    def move(self, dx):
        self.rect.x += dx

    def make_hit(self):
        self.hit = True

    def move_left(self, vel):
        self.x_vel = -vel
        if self.direction != "right":
            self.direction = "right"
            self.animation_count = 0

    def move_right(self, vel):
        self.x_vel = vel
        if self.direction != "left":
            self.direction = "left"
            self.animation_count = 0

    def loop(self, fps):
        self.move(self.x_vel)

        if self.hit:
            self.hit_count += 1
        if self.hit_count > 10:
            self.hit = False
            self.hit_count = 0

        self.update_sprite()


    def update_sprite(self):
        if self.hit:
            self.animation_name = "Hit"

        sprite_name = self.animation_name + "_" + self.direction
        sprites = self.green_enemy[sprite_name]
        sprite_index = (self.animation_count //
                        self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_index]
        self.animation_count += 1
        self.steps += 1
        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)
        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0
        self.update()

    def update(self):
        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)

    def run(self):
        self.animation_name = "Run"
        self.x_vel = 3

    def run_attack(self):
        self.animation_name = "Run_Attack"

    def movement(self, player): 
        if self.x_vel == 0:
            self.run()

        if player.rect.x <= self.rect.x + self.THRESHOLD:
            self.run_attack()

        if 0 <= self.steps % (self.ANIMATION_DELAY * 40) <= self.ANIMATION_DELAY * 20:
            self.move_left(PLAYER_VEL)
        else:
            self.move_right(PLAYER_VEL)


def get_background(name):
    image = pygame.image.load(join("Python-Platformer-main\Python-Platformer-main", "assets", "Background", name))
    _, _, width, height = image.get_rect()
    tiles = []

    for i in range(WIDTH // width + 1):
        for j in range(HEIGHT // height + 1):
            pos = (i * width, j * height)
            tiles.append(pos)

    return tiles, image


def draw(window, background, bg_image, player, objects, offset_x):
    for tile in background:
        window.blit(bg_image, tile)

    for obj in objects:
        obj.draw(window, offset_x)

    player.draw(window, offset_x)
    # clock.tick(FPS)
    pygame.display.update()


def handle_vertical_collision(player, objects, dy):
    collided_objects = []
    for obj in objects:
        if pygame.sprite.collide_mask(player, obj):
            if dy > 0:
                if obj.name == "flying_box":
                    if player.rect.bottom >= HEIGHT - 100:
                        player.hit_head()
                        continue
                player.rect.bottom = obj.rect.top
                player.landed()
            elif dy < 0 and player.rect.bottom < HEIGHT - 100:
                player.rect.top = obj.rect.bottom
                print("collision ocured")
                player.hit_head()

            collided_objects.append(obj)

    return collided_objects


def collide(player, objects, dx):
    player.move(dx, 0)
    player.update()
    collided_object = None
    for obj in objects:
        if pygame.sprite.collide_mask(player, obj):
            if (obj.name == "green_enemy" 
                or obj.name == "fire" 
                or obj.name == "spikes"):
                player.make_hit()
            else:
                collided_object = obj
            break

    player.move(-dx, 0)
    player.update()
    return collided_object


def handle_move(player, objects):
    keys = pygame.key.get_pressed()

    player.x_vel = 0
    collide_left = collide(player, objects, -PLAYER_VEL * 2)
    collide_right = collide(player, objects, PLAYER_VEL * 2)

    if keys[pygame.K_LEFT] and not collide_left:
        player.move_left(PLAYER_VEL)
    if keys[pygame.K_RIGHT] and not collide_right:
        player.move_right(PLAYER_VEL)

    vertical_collide = handle_vertical_collision(player, objects, player.y_vel)
    to_check = [*vertical_collide]

    for obj in to_check:
        if obj and (obj.name == "fire" or obj.name == "spikes"):
            player.make_hit()
        
        if obj and obj.name == "flying_box":
            if player.rect.bottom < HEIGHT - 100:
                obj.hit()
            else:
                player.make_hit()

        if obj and obj.name == "green_enemy":
            obj.make_hit()

def movement(player, objects, action):
    player.x_vel = 0
    collide_left = collide(player, objects, -PLAYER_VEL * 2)
    collide_right = collide(player, objects, PLAYER_VEL * 2)

    if action == 0 and not collide_right:
        player.move_right(PLAYER_VEL)
    elif action == 1 and player.jump_count < 2:
        if player.is_jumping:
            player.move_right(PLAYER_VEL)
        player.jump()
    elif action == 2 and not collide_right and player.jump_count < 2:
        player.jump()
        player.move_right(PLAYER_VEL)
        player.is_jumping = True
    else:
        print('warning!!')

    vertical_collide = handle_vertical_collision(player, objects, player.y_vel)
    to_check = [*vertical_collide]

    for obj in to_check:
        if obj and (obj.name == "fire" or obj.name == "spikes"):
            player.make_hit()
        
        if obj and obj.name == "flying_box":
            if player.rect.bottom >= obj.rect.top:
                obj.hit()
            else:
                player.make_hit()
        
        if obj and obj.name == "green_enemy":
            obj.make_hit()
            



def main(window):
    clock = pygame.time.Clock()
    background, bg_image = get_background("Blue.png")

    block_size = 96
    spike_size = 16
    enemy_size = 48
    green_enemy_size = 72

    player = Player(146,HEIGHT - block_size - 100,50,50)
    fire = Fire(block_size * 7.5, HEIGHT - block_size * 2 - 64, 16, 32)
    flag = Checkpoint(block_size * 17, HEIGHT - block_size - 128)
    attacking_block = Flying_enemy(block_size * 9, HEIGHT - block_size - enemy_size * 3, enemy_size, enemy_size)
    green_enemy = Green_enemy(block_size * 15, HEIGHT - block_size - green_enemy_size - spike_size, green_enemy_size, enemy_size)
    fire.on()
    attacking_block.fly()
    green_enemy.movement(player)
    floor = [Block(i * block_size, HEIGHT - block_size, block_size)
             for i in range(-WIDTH // block_size, (WIDTH * 2) // block_size)]
    objects = [*floor, 
               Block(block_size * -5, HEIGHT - block_size * 2, block_size),
               Block(block_size * -5, HEIGHT - block_size * 3, block_size),
               Block(block_size * 18, HEIGHT - block_size * 2, block_size),
               Block(block_size * 18, HEIGHT - block_size * 3, block_size),
               Block(block_size * 3, HEIGHT - block_size * 4, block_size),
               Block(block_size * 4, HEIGHT - block_size * 4, block_size),
               Block(block_size * 6, HEIGHT - block_size * 2, block_size),
               Block(block_size * 7, HEIGHT - block_size * 2, block_size),
               Spikes(block_size * 5 + spike_size * 2, HEIGHT - block_size - spike_size *2, spike_size, spike_size),
               Spikes(block_size * 5 + spike_size * 4, HEIGHT - block_size - spike_size *2, spike_size, spike_size),
               flag,
               green_enemy,
               attacking_block,
               fire
               ]

    offset_x = 0
    scroll_area_width = 200
    i = 0
    run = True
    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and player.jump_count < 2:
                    player.jump()

        player.loop(FPS)
        fire.loop()
        flag.loop()
        attacking_block.loop()
        green_enemy.loop(FPS)
        actions = [0,2,1,1,1,0,0,0,0,0]
        # movement(player, objects, actions[i % len(actions)])
        handle_move(player, objects)
        i += 1
        green_enemy.movement(player)
        draw(window, background, bg_image, player, objects, offset_x)

        if ((player.rect.right - offset_x >= WIDTH - scroll_area_width) and player.x_vel > 0) or (
                (player.rect.left - offset_x <= scroll_area_width) and player.x_vel < 0):
            offset_x += player.x_vel

    pygame.quit()
    quit()

class Pls_learn(gym.Env):
    def __init__(self):
        self._screen =  pygame.display.set_mode((WIDTH,HEIGHT))
        self._block_size = 96
        self._enemy_size = 48
        self._green_enemy_size = 72
        self._spike_size = 16
        self._agent = Player(146,HEIGHT - self._block_size - 100,50,50)
        self._x_offset = 0
        self._steps = 0
        self.truncated = False
        self.terminated = False
        self._background = None
        self._bg_image = None
        self._x_offset = 0
        self._epsilon = 1
        self._screen_scroll_width = 146
        self.action_space = spaces.Discrete(4) # 0: Left, 1: Right, 2: jump/double jump, 3: do nothing
        self.observation_space = spaces.Box(low=0.0, high=255.0, 
                                            shape=((REGION_SIZE*2 * REGION_SIZE*2),), 
                                            dtype=np.float32)
        
        self._fire1 = Fire(self._block_size * 19.5, HEIGHT - self._block_size * 1 - 64, 16, 32)
        self._fire2 = Fire(self._block_size * 34.25, HEIGHT - self._block_size * 2 - 64, 16, 32)
        self._fire3 = Fire(self._block_size * 44.5, HEIGHT - self._block_size * 1 - 64, 16, 32)

        self._flag = Checkpoint(self._block_size * 60, HEIGHT - self._block_size - 128)

        self._fire1.on()
        self._fire2.on()
        self._fire3.on()

        self._attacking_block1 = Flying_enemy(self._block_size * 10, HEIGHT - self._block_size * 3.5 - self._enemy_size, self._enemy_size, self._enemy_size)
        self._attacking_block2 = Flying_enemy(self._block_size * 24, HEIGHT - self._block_size * 2 - self._enemy_size, self._enemy_size, self._enemy_size)

        self.green_enemy1 = Green_enemy(self._block_size * 17, HEIGHT - self._block_size - self._green_enemy_size - self._spike_size, self._green_enemy_size, self._enemy_size)
        self.green_enemy2 = Green_enemy(self._block_size * 42, HEIGHT - self._block_size - self._green_enemy_size - self._spike_size, self._green_enemy_size, self._enemy_size)

        self._fire1.on()
        self._fire2.on()
        self._fire3.on()

        self._attacking_block1.fly()
        self._attacking_block2.fly()

        self.green_enemy1.movement(self._agent)
        self.green_enemy2.movement(self._agent)

        floor = [Block(i * self._block_size, HEIGHT - self._block_size, self._block_size)
             for i in range(0, (WIDTH * 6) // self._block_size)]
        self._objects = [*floor, 
           Block(self._block_size * 3, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 6, HEIGHT - self._block_size * 2, self._block_size), 
           Block(self._block_size * 7, HEIGHT - self._block_size * 2, self._block_size),
           # fire1 is there on these blocks
           Block(self._block_size * 8, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 9, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 11, HEIGHT - self._block_size * 3, self._block_size), 
        #    Block(self._block_size * 13, HEIGHT - self._block_size * 3, self._block_size),
        #    Block(self._block_size * 15, HEIGHT - self._block_size * 3, self._block_size),
           Block(self._block_size * 18, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 20, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 27, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 29, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 33, HEIGHT - self._block_size * 2, self._block_size),
           Block(self._block_size * 34, HEIGHT - self._block_size * 2, self._block_size),
           Block(self._block_size * 35, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 36, HEIGHT - self._block_size * 3, self._block_size),   
           Block(self._block_size * 47, HEIGHT - self._block_size * 2, self._block_size),  

           # for the steps block
           Block(self._block_size * 51, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 52, HEIGHT - self._block_size * 2, self._block_size),
           Block(self._block_size * 52, HEIGHT - self._block_size * 3, self._block_size), 
           Block(self._block_size * 53, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 53, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 53, HEIGHT - self._block_size * 4, self._block_size),  
           Block(self._block_size * 54, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 54, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 54, HEIGHT - self._block_size * 4, self._block_size),  
           Block(self._block_size * 54, HEIGHT - self._block_size * 5, self._block_size),
           Block(self._block_size * 55, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 55, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 55, HEIGHT - self._block_size * 4, self._block_size),  
           Block(self._block_size * 55, HEIGHT - self._block_size * 5, self._block_size),  

        #    Spikes(self._block_size * 5, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),
           Spikes(self._block_size * 5 + self._spike_size * 2, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),
           Spikes(self._block_size * 5 + self._spike_size * 4, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),

           Spikes(self._block_size * 28, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),
           Spikes(self._block_size * 28 + self._spike_size * 2, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),
           Spikes(self._block_size * 28 + self._spike_size * 4, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size), 

           self._flag,
           self.green_enemy1,
           self.green_enemy2,
           self._attacking_block1,
           self._attacking_block2,
           self._fire1,
           self._fire2,
           self._fire3
           ]

    
    def reset(self, seed=None):
        self.terminated = False
        self.truncated = False
        self._steps = 0
        self._x_offset = 0
        self._agent = Player(146,HEIGHT - self._block_size - 100,50,50)
        self._background, self._bg_image = get_background("Blue.png")        

        self._fire1 = Fire(self._block_size * 19.5, HEIGHT - self._block_size * 1 - 64, 16, 32)
        self._fire2 = Fire(self._block_size * 34.25, HEIGHT - self._block_size * 2 - 64, 16, 32)
        self._fire3 = Fire(self._block_size * 44.5, HEIGHT - self._block_size * 1 - 64, 16, 32)

        self._flag = Checkpoint(self._block_size * 60, HEIGHT - self._block_size - 128)

        self._fire1.on()
        self._fire2.on()
        self._fire3.on()

        self._attacking_block1 = Flying_enemy(self._block_size * 10, HEIGHT - self._block_size * 3.5 - self._enemy_size, self._enemy_size, self._enemy_size)
        self._attacking_block2 = Flying_enemy(self._block_size * 24, HEIGHT - self._block_size * 2 - self._enemy_size, self._enemy_size, self._enemy_size)

        self.green_enemy1 = Green_enemy(self._block_size * 17, HEIGHT - self._block_size - self._green_enemy_size - self._spike_size, self._green_enemy_size, self._enemy_size)
        self.green_enemy2 = Green_enemy(self._block_size * 42, HEIGHT - self._block_size - self._green_enemy_size - self._spike_size, self._green_enemy_size, self._enemy_size)

        self._fire1.on()
        self._fire2.on()
        self._fire3.on()

        self._attacking_block1.fly()
        self._attacking_block2.fly()

        self.green_enemy1.movement(self._agent)
        self.green_enemy2.movement(self._agent)

        floor = [Block(i * self._block_size, HEIGHT - self._block_size, self._block_size)
             for i in range(0, (WIDTH * 6) // self._block_size)]
        self._objects = [*floor, 
           Block(self._block_size * 3, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 6, HEIGHT - self._block_size * 2, self._block_size), 
           Block(self._block_size * 7, HEIGHT - self._block_size * 2, self._block_size),
           # fire1 is there on these blocks
           Block(self._block_size * 8, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 9, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 11, HEIGHT - self._block_size * 3, self._block_size), 
        #    Block(self._block_size * 13, HEIGHT - self._block_size * 3, self._block_size),
        #    Block(self._block_size * 15, HEIGHT - self._block_size * 3, self._block_size),
           Block(self._block_size * 18, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 20, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 27, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 29, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 33, HEIGHT - self._block_size * 2, self._block_size),
           Block(self._block_size * 34, HEIGHT - self._block_size * 2, self._block_size),
           Block(self._block_size * 35, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 36, HEIGHT - self._block_size * 3, self._block_size),   
           Block(self._block_size * 47, HEIGHT - self._block_size * 2, self._block_size),  

           # for the steps block
           Block(self._block_size * 51, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 52, HEIGHT - self._block_size * 2, self._block_size),
           Block(self._block_size * 52, HEIGHT - self._block_size * 3, self._block_size), 
           Block(self._block_size * 53, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 53, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 53, HEIGHT - self._block_size * 4, self._block_size),  
           Block(self._block_size * 54, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 54, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 54, HEIGHT - self._block_size * 4, self._block_size),  
           Block(self._block_size * 54, HEIGHT - self._block_size * 5, self._block_size),
           Block(self._block_size * 55, HEIGHT - self._block_size * 2, self._block_size),  
           Block(self._block_size * 55, HEIGHT - self._block_size * 3, self._block_size),  
           Block(self._block_size * 55, HEIGHT - self._block_size * 4, self._block_size),  
           Block(self._block_size * 55, HEIGHT - self._block_size * 5, self._block_size),  

        #    Spikes(self._block_size * 5, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),
           Spikes(self._block_size * 5 + self._spike_size * 2, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),
           Spikes(self._block_size * 5 + self._spike_size * 4, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),

           Spikes(self._block_size * 28, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),
           Spikes(self._block_size * 28 + self._spike_size * 2, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size),
           Spikes(self._block_size * 28 + self._spike_size * 4, HEIGHT - self._block_size - self._spike_size * 2, self._spike_size, self._spike_size), 

           self._flag,
           self.green_enemy1,
           self.green_enemy2,
           self._attacking_block1,
           self._attacking_block2,
           self._fire1,
           self._fire2,
           self._fire3
           ]

        return self.get_obs(), {}

    def step(self, action):
        # clock.tick(FPS)
        self.truncated = False
        self.terminated = False
        reward = 0
        distance = 0
        penalty = 0
        time_penalty = -2
        self._agent.loop(FPS)
        self._flag.loop()

        self._fire1.loop()
        self._fire2.loop()
        self._fire3.loop()

        self._attacking_block1.loop()
        self._attacking_block2.loop()

        self.green_enemy1.loop(FPS)
        self.green_enemy2.loop(FPS)

        movement(self._agent, self._objects, action)
        # if player went below the floor
        if self._agent.rect.y >= HEIGHT - self._block_size:
            print(self._agent.rect.y)
        
        self.green_enemy1.movement(self._agent)
        self.green_enemy2.movement(self._agent)

        if ((self._agent.rect.right - self._x_offset >= WIDTH - self._screen_scroll_width and self._agent.x_vel > 0) 
            or (self._agent.rect.left - self._x_offset <= self._screen_scroll_width and self._agent.x_vel < 0)):
            self._x_offset += self._agent.x_vel

        self._steps += 1

        obs = self.get_obs()

        distance += 10 / (math.sqrt(((self._flag.x - self._agent.rect.x)**2 
                                    + (self._flag.y - self._agent.rect.y)**2)) 
                                    + self._epsilon)
        
        # time_penalty += self._steps * -1
        
        if self._agent.sprite_sheet == "hit":
            penalty -= 10
            reward = penalty + distance
            self.terminated = True
            return obs, reward, self.terminated, self.truncated, {}
        
        reward = distance + time_penalty

        # print(self._agent.x_vel,'ehejfkn',self._agent.y_vel)

        if (math.sqrt(((self._flag.x - self._agent.rect.x)**2 
                    + (self._flag.y - self._agent.rect.y)**2))) <= 3:
            self.terminated = True
            return obs, reward, self.terminated, self.truncated, {}
        
        if self._steps >= 5000:
            self.truncated = True

        draw(self._screen, self._background, self._bg_image, self._agent, self._objects, self._x_offset)
        return obs, reward, self.terminated, self.truncated, {}

    def get_obs(self):
        """
        Returns the observation as a region around the player. It takes a 2D region
        centered around the player, converts to grayscale if needed, and returns
        the cropped observation.
        """
        # Get screen surface and crop around the agent
        surface = pygame.surfarray.array3d(self._screen)  # Get the RGB array of the screen
        x, y = self._agent.rect.center  # Get the player's center coordinates

        # Ensure that the cropping does not go out of bounds, constraints
        min_x = max(0, x - REGION_SIZE)
        max_x = min(surface.shape[0], x + REGION_SIZE)
        min_y = max(0, y - REGION_SIZE)
        max_y = min(surface.shape[1], y + REGION_SIZE)

        cropped = surface[min_x:max_x, min_y:max_y].astype(np.float32)

        # If the cropped region is smaller than expected, resize it safely
        if cropped.size == 0:
            return np.zeros((REGION_SIZE*2* REGION_SIZE*2)) / 255.0  # Return a black region

        # resize and normalize
        resized = cv2.resize(cropped, (REGION_SIZE * 2, REGION_SIZE * 2))
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        flattened = resized.flatten()

        return flattened

    def close(self):
        pygame.quit()

def sampling():
    env = Pls_learn()
    obs,info = env.reset()

    # check_env(env)

    for i in range(10):
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    return
        
        obs, info = env.reset()

    env.close()

def train(env, steps):
    env = DummyVecEnv([lambda: Pls_learn()])  

    # log_dir = "logs/"
    # os.makedirs(log_dir, exist_ok=True)
   
    # model_params = {
    #     "policy": "MlpPolicy",  
    #     "env": env,             
    #     "learning_rate": 0.0003,  
    #     "n_steps": 2048,        
    #     "batch_size": 64,       
    #     "n_epochs": 10,         
    #     "gamma": 0.99,          
    #     "gae_lambda": 0.95,     
    #     "clip_range": 0.2,      
    #     "verbose": 1,          
    #     "tensorboard_log": log_dir 
    # }

    model = PPO("MlpPolicy", env, learning_rate=0.0002, n_steps=1024, tensorboard_log="platformer_board", verbose=1)

    # eval_callback = EvalCallback(env, best_model_save_path='logs/', log_path='logs/', eval_freq=1000, deterministic=True, render=False)

    # check_env(env)

    model = PPO.load("platformer_agent", env)

    model.learn(total_timesteps = steps)
    model.save("platformer-load250k-10k")


def test(env):
    # env = DummyVecEnv([lambda: Pls_learn()])  
    # model = PPO("MlpPolicy", env, verbose=1) 
    model = PPO.load("platformer-load50k-50k.zip", env=env)
    obs, info = env.reset()
    total_reward = 0
    for _ in range(10):
        terminated = False
        truncated = False
        obs = np.array(obs)
        while not terminated and not truncated:
            # print(len(obs[0]), len(obs[1]))
            action, _ = model.predict(obs)
            # print(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(reward,'rew')
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    return
        print(total_reward)
        obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    # main(screen)
    env = Pls_learn()
    # check_env(env)
    # sampling()
    # train(env, steps=10000)
    test(env)