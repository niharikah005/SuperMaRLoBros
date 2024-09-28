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


pygame.init()
pygame.display.set_caption("platformer game")

# constants
WIDTH,HEIGHT = 1056, 720
FPS = 30
PLAYER_VEL = 5
REGION_SIZE = 146
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("platformer game")


# file loading utils
def flip(sprites):
    return [pygame.transform.flip(sprite, True, False) for sprite in sprites]

def load_sprites(dir1,dir2,width,height,direction=False):
    path = join("Python-Platformer-main\Python-Platformer-main","assets", dir1, dir2)
    images = [f for f in listdir(path) if isfile(join(path,f))]
    all_sprites = {}
    for image in images:
        sprite_sheet = pygame.image.load(join(path, image)).convert_alpha()
        sprites = []
        for i in range(sprite_sheet.get_width()//width):
            surface = pygame.Surface((width,height), pygame.SRCALPHA, 32)
            rect = pygame.Rect(i*width, 0, width, height)
            surface.blit(sprite_sheet, (0,0), rect)
            sprites.append(pygame.transform.scale2x(surface))

        if direction:
            all_sprites[image.replace(".png","") + "_right"] = sprites
            all_sprites[image.replace(".png","") + "_left"] = flip(sprites)
        else:
            all_sprites[image.replace(".png","")] = sprites

    return all_sprites

# env load functions
def load_block(size):
    path = join("Python-Platformer-main\Python-Platformer-main","assets", "Terrain", "Terrain.png")
    image = pygame.image.load(path)
    surface = pygame.Surface((size,size), pygame.SRCALPHA, 32)
    rect = pygame.Rect(96,128,size,size)
    surface.blit(image, (0,0), rect)
    return pygame.transform.scale2x(surface)

def get_background(name):
    image = pygame.image.load(join("Python-Platformer-main\Python-Platformer-main","assets", "Background", name))
    _,_,width,height = image.get_rect()
    tiles = []
    for i in range(WIDTH//width + 1):
        for j in range(HEIGHT//height + 1):
            pos = (i*height, j*width)
            tiles.append(pos)
    return tiles, image

# general utils:
def draw(window, background, image, player, objects, x_offset):
    for tile in background:
        window.blit(image, tile)
    player.draw(window, x_offset)
    for object in objects:
        object.draw(window, x_offset)
    # clock.tick(FPS)
    pygame.display.update()

def handle_vertical_collision(player, objects, dy):
    colliding_objects = []
    for obj in objects:
        if pygame.sprite.collide_mask(player, obj):
            if dy < 0:
                player.rect.top = obj.rect.bottom
                player.hit_head()
            if dy > 0:
                player.rect.bottom = obj.rect.top
                player.landed()
        colliding_objects.append(obj)
    return colliding_objects

def handle_horizontal_collision(player, objects, dx):
    player.move(dx, 0) # anticipate the collision
    player.update() # to check for collisions
    collided_object = None # buffer
    for obj in objects:
        if pygame.sprite.collide_mask(player, obj):
            collided_object = obj
            break
    player.move(-dx, 0) # get the player back to where it would be
    player.update()
    return collided_object

def movement(player, objects, action):
    keys = pygame.key.get_pressed()
    # the x2 in PLAYER_VEL is to avoid glitching due to animations
    collide_left = handle_horizontal_collision(player, objects, -PLAYER_VEL*2) 
    collide_right = handle_horizontal_collision(player, objects, PLAYER_VEL*2)

    player.x_vel = 0 # to reset the velocity.
    if action == 0 and not collide_left:
        player.move_left(PLAYER_VEL)
    if action == 1 and not collide_right:
        player.move_right(PLAYER_VEL)
    if action == 2 and player.jump_count < 2:
        player.jump()

    handle_vertical_collision(player, objects, player.y_vel)


# agent controls
class Player(pygame.sprite.Sprite): # self.sprite come from here
    GRAVITY = 1
    COLOR = (255,0,0)
    SPRITES = load_sprites("MainCharacters", "MaskDude", 32, 32, True)
    ANIMATION_DELAY = 4
    def __init__(self, x,y,width,height):
        super().__init__()
        self.rect = pygame.Rect(x,y,width,height)
        self.x_vel = 0
        self.y_vel = 0
        self.mask = None
        self.direction = "left"
        self.animation_count = 0
        self.fall_count = 0
        self.jump_count = 0
        self.count = 0
        self.hit = False
        self.sprite = self.SPRITES["idle_left"][0] 

    def jump(self):
        self.y_vel = -self.GRAVITY * 8
        self.animation_count = 0
        self.jump_count += 1
        if self.jump_count == 1:
            self.fall_count = 0 # removes extra gravity

    def move(self,dx,dy):
        self.rect.x += dx
        self.rect.y += dy

    def move_left(self,vel):
        self.x_vel = -vel
        if self.direction != "left":
            self.direction = "left"
            self.animation_count = 0 # refresh the counter

    def move_right(self,vel):
        self.x_vel = vel
        if self.direction != "right":
            self.direction = "right"
            self.animation_count = 0

    def landed(self):
        self.fall_count = 0
        self.y_vel = 0
        self.jump_count = 0

    def hit_head(self):
        self.y_vel *= -1
        self.count = 0

    def got_hit(self):
        self.hit = True

    def loop(self, fps):
        # will be used to move everything
        self.y_vel += min(1, self.GRAVITY*(self.fall_count/fps))
        self.move(self.x_vel,self.y_vel) # dx = self.x_vel, dy = self.y_vel

        self.fall_count += 1 # put this back to zero when jumping or landing to avoid extreme values of gravity on player.
        self.update_sprite()

    def update_sprite(self):
        sprite_sheet = "idle"
        if self.y_vel < 0:
            if self.jump_count == 1:
                sprite_sheet = "jump"
            elif self.jump_count == 2:
                sprite_sheet = "double_jump"
        elif self.y_vel > self.GRAVITY * 2: # max y_vel limiter
            sprite_sheet = "fall"
        elif self.x_vel != 0:
            sprite_sheet = "run" # to choose the sprite file

        sprite_sheet_name = sprite_sheet + "_" + self.direction # naming the file
        sprite = self.SPRITES[sprite_sheet_name] # creating a list of the sprites in that file 
        sprite_index = (self.animation_count 
                        // self.ANIMATION_DELAY) % len(sprite) # adding some delay in changing the sprite
        self.sprite = sprite[sprite_index] 
        self.animation_count += 1 
        self.update()

    def update(self):
        self.rect = self.sprite.get_rect(topleft=(self.rect.x,self.rect.y)) # adj the rect-size depending on the sprite.
        self.mask = pygame.mask.from_surface(self.sprite) # for pixel perfect collision

    def draw(self, window, offset):
        window.blit(self.sprite, (self.rect.x - offset,self.rect.y))


# object ADT:
class Object(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, name=None):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.image = pygame.Surface((width,height), pygame.SRCALPHA)
        self.width = width
        self.height = height
        self.name = name

    def draw(self, window, offset):
        window.blit(self.image, (self.rect.x - offset, self.rect.y))

# blocks
class Block(Object):
    def __init__(self, x, y, size): # square
        super().__init__(x, y, size, size)
        block = load_block(size)
        self.image.blit(block, (0,0))
        self.mask = pygame.mask.from_surface(self.image) 

# enemies classes
class Fire(Object):
    ANIMATION_DELAY = 4
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "fire")
        self.fire = load_sprites("Traps", "Fire", width, height)
        print(len(self.fire["off"]))
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


# SB3 environment
class Pls_learn(gym.Env):
    def __init__(self):
        self._screen =  pygame.display.set_mode((WIDTH,HEIGHT))
        self._block_size = 96
        self._trap_size = 48
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
        # -WIDTH // block_size will put it to # the left of the screen width and WIDTH * 2// block_size will put it to the right full width
        self._floor = [Block(i * self._block_size, HEIGHT - self._block_size, self._block_size) 
                for i in range(-WIDTH // self._block_size, (WIDTH*2) // self._block_size)] 
        self._objects = [*self._floor, Block(self._block_size*4, HEIGHT - self._block_size*2, self._block_size), 
               Block(self._block_size*7, HEIGHT - self._block_size*4, self._block_size)]
        
        self.action_space = spaces.Discrete(2) # 0: Left, 1: Right, 2: jump/double jump
        self.observation_space = spaces.Box(low=0.0, high=255.0, 
                                            shape=((REGION_SIZE*2), (REGION_SIZE*2),3), 
                                            dtype=np.float32)
    
    def reset(self, seed=None):
        self.terminated = False
        self.truncated = False
        self._steps = 0
        self._agent = Player(146,HEIGHT - self._block_size - 100,50,50)

        self._background, self._bg_image = get_background("Blue.png") 
        return self.get_obs(), {}

    def step(self, action):
        self.truncated = False
        self.terminated = False
        reward = 0
        movement(self._agent, self._objects, action)
        self._agent.loop(FPS)
        if ((self._agent.rect.right - self._x_offset >= WIDTH - self._screen_scroll_width and self._agent.x_vel > 0) 
            or (self._agent.rect.left - self._x_offset <= self._screen_scroll_width and self._agent.x_vel < 0)):
            self._x_offset += self._agent.x_vel
        self._steps += 1

        obs = self.get_obs()
        reward += 100 / (math.sqrt(((self._block_size*7 - self._agent.rect.x)**2 + ((HEIGHT - self._block_size*4 - 50) - self._agent.rect.y))) + self._epsilon)
        if reward == 100:
            self.terminated = True
            return obs, reward, self.terminated, self.truncated, {}
        if self._steps >= 1000:
            self.truncated = True

        draw(self._screen, self._background, self._bg_image, self._agent, self._objects, self._x_offset)
        return obs, reward, self.terminated, self.truncated, {}

    def get_obs(self):
        x = max(REGION_SIZE, self._agent.rect.x)
        y = self._agent.rect.y
        region = pygame.surfarray.array3d(self._screen).transpose(1, 0, 2)
        cropped_region = region[y-REGION_SIZE:y+REGION_SIZE, x-REGION_SIZE:x+REGION_SIZE,:].astype(np.float32)
        return cropped_region

    def close(self):
        pygame.quit()

def main():
    env = Pls_learn()
    obs,info = env.reset()

    check_env(env)

    for i in range(10):
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action = env.action_space.sample()
            print(f"Sampled action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            # env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    return
        
        obs, info = env.reset()

    env.close()

def train_agent():
    env = DummyVecEnv([lambda: Pls_learn()])  

   
    model_params = {
        "policy": "MlpPolicy",  
        "env": env,             
        "learning_rate": 0.0003,  
        "n_steps": 2048,        
        "batch_size": 64,       
        "n_epochs": 10,         
        "gamma": 0.99,          
        "gae_lambda": 0.95,     
        "clip_range": 0.2,      
        "verbose": 1,          
        "tensorboard_log": "./ppo_tensorboard" 
    }

    model = PPO(**model_params)
    model.learn(total_timesteps=10000)
    model.save("ppo_platformer")

    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    train_agent()

