import pygame
from sys import exit
from random import randint, choice
import os

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.transform.rotozoom(pygame.image.load(os.path.join('Mini-Project', 'assets', 'images', 'cat.png')).convert_alpha(), 0, 3)
        self.rect = self.image.get_rect(midbottom = (100, 310))
        self.gravity = 0

    def player_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and self.rect.bottom >= 300: # to only jump when on ground
            self.gravity = -20
    
    def apply_gravity(self): 
        self.gravity += 1 
        self.rect.y += self.gravity

        if self.rect.bottom >= 330: # illusion of keeping player on ground
            self.rect.bottom = 330
    
    def update(self):
        self.player_input()
        self.apply_gravity()

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, type):
        super().__init__()
        if type == 'fly': 
            fly1 = pygame.image.load(os.path.join('Mini-Project', 'assets', 'images', 'fly1.png')).convert_alpha()
            fly2 = pygame.image.load(os.path.join('Mini-Project', 'assets', 'images', 'fly2.png')).convert_alpha()
            self.frames = [fly1, fly2]
            y_pos = 210
        else:
            banana = pygame.transform.scale2x(pygame.image.load(os.path.join('Mini-Project', 'assets', 'images', 'banana.png')).convert_alpha())
            self.frames = [banana]
            y_pos = 325

        self.animation_index = 0
        self.image = self.frames[self.animation_index]
        self.rect = self.image.get_rect(bottomright = (randint(900, 1100), y_pos))

    def animation(self):
        self.animation_index += 0.1 # increment slowly so animation is smooth and not too fast
        if self.animation_index >= len(self.frames):
            self.animation_index = 0
        self.image = self.frames[int(self.animation_index)]

    def update(self):
        self.animation()
        self.rect.x -= 5
        self.destroy()

    def destroy(self):
        if self.rect.x <= -100: # obstacle went outside borders
            self.kill()

class Reward(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load(os.path.join('Mini-Project', 'assets', 'images', 'fish.png')).convert_alpha()
        self.rect = self.image.get_rect(midbottom = (randint(900, 1200), 320))
    
    def update(self):
        self.rect.x -= 5
        self.destroy()

    def destroy(self):
        if self.rect.x <= -100:
            self.kill()

def display_score():
    current_time = pygame.time.get_ticks() - start_time
    score_surf = test_font.render(f'Score: {current_time // 1000}', False, 'Black')
    score_rect = score_surf.get_rect(center = (300, 50))
    screen.blit(score_surf, score_rect)
    return current_time

def collisions():
    if pygame.sprite.spritecollide(player.sprite, obstacle_grp, False):
        obstacle_grp.empty() # remove all obstacles from grp cuz game over
        return False
    return True

def reward_collide():
    for reward in reward_grp:
        if reward.rect.colliderect(player.sprite.rect):
            reward.kill() # remove that collided reward from grp
            return 1
    return 0

pygame.init()

screen = pygame.display.set_mode((800, 400))
pygame.display.set_caption("Banana Cat")

clock = pygame.time.Clock()
test_font = pygame.font.Font(os.path.join('Mini-Project', 'assets', 'font', 'Pixeltype.ttf'), 24)

game_active = False
start_time = 0
score = 0
reward_score = 0
fps = 60

#groups
player = pygame.sprite.GroupSingle()
player.add(Player())
obstacle_grp = pygame.sprite.Group()
reward_grp = pygame.sprite.Group()

sky_surf = pygame.image.load(os.path.join('Mini-Project', 'assets', 'images', 'sky.png')).convert()
ground_surf = pygame.image.load(os.path.join('Mini-Project', 'assets', 'images', 'ground.png')).convert()

cat_surf = pygame.transform.rotozoom(pygame.image.load(os.path.join('Mini-Project', 'assets', 'images', 'cat.png')).convert_alpha(), 0, 6)
cat_gamescreen_rect = cat_surf.get_rect(center=(400, 200))

game_name = test_font.render('Banana Cat', False, 'Black')
game_name_rect = game_name.get_rect(center=(400, 100))

start_message = test_font.render('Press space to play', False, 'Black')
start_message_rect = start_message.get_rect(center=(400, 300))

game_message = test_font.render('You lost! Press Space to restart', False, 'Black')
game_message_rect = game_message.get_rect(center=(400, 300))

obstacle_timer = pygame.USEREVENT + 1 # custom event for obstacle spawning
pygame.time.set_timer(obstacle_timer, 1000)

reward_timer = pygame.USEREVENT + 2  # custom event for reward spawning
pygame.time.set_timer(reward_timer, 2000)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            pygame.quit()
            exit()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if not game_active: # move on from game state and begin
                game_active = True
                start_time = pygame.time.get_ticks()
            else: # jump
                if player.sprite.rect.bottom >= 300:
                    player.sprite.gravity = -20
                
        if game_active and event.type == obstacle_timer: # obstacle spawn
            obstacle_grp.add(Obstacle(choice(['fly', 'banana'])))

        if game_active and event.type == reward_timer: # reward spawn
            reward_grp.add(Reward())

    if game_active:
        screen.blit(sky_surf, (0, 0))
        screen.blit(ground_surf, (0, 300))
        score = display_score()

        player.draw(screen)
        player.update()
        pygame.draw.rect(screen, 'Red', player.sprite.rect, 2)  # bounding box

        obstacle_grp.draw(screen)
        obstacle_grp.update()
        for obstacle in obstacle_grp:  # bounding box
            pygame.draw.rect(screen, 'Red', obstacle.rect, 2)  

        reward_grp.draw(screen)
        reward_grp.update()
        for reward in reward_grp:  # bounding box
            pygame.draw.rect(screen, 'Red', reward.rect, 2)

        reward_score += reward_collide()
        reward_surf = test_font.render(f'Fish: {reward_score}', False, 'Black')
        reward_rect = reward_surf.get_rect(center = (500, 50))
        screen.blit(reward_surf, reward_rect)

        game_active = collisions() # to set game_active

    else:
        screen.fill((94, 129, 162))
        screen.blit(cat_surf, cat_gamescreen_rect)
        screen.blit(game_name, game_name_rect)

        if score == 0: # not started
            screen.blit(start_message, start_message_rect)
        else: # game over
            screen.blit(game_message, game_message_rect)
            reward_score = 0
            reward_grp.empty()

    pygame.display.update()
    clock.tick(fps)

