import pygame
from sys import exit
from random import randint
import os

def display_score():
    current_time = pygame.time.get_ticks() - start_time
    score_surf = test_font.render(f'Score: {current_time // 1000}', False, 'Black')
    score_rect = score_surf.get_rect(center = (400, 50))
    screen.blit(score_surf, score_rect)
    return current_time

def obstacle_movement(obstacle_list):
    if obstacle_list:
        for obstacle_rect in obstacle_list:
            obstacle_rect.x -= 5
            if obstacle_rect.bottom == 210:
                screen.blit(fly_surf, obstacle_rect)
            else:
                screen.blit(banana_surf, obstacle_rect)

        obstacle_list = [obstacle for obstacle in obstacle_list if obstacle.x > 70]

    return obstacle_list

def collision(player, obstacles):
    if obstacles:
        for obstacle in obstacles:
            if player.colliderect(obstacle): 
                return False
    return True

pygame.init()

screen = pygame.display.set_mode((800, 400))
pygame.display.set_caption("Banana Cat")

clock = pygame.time.Clock()
base_path = 'Mini-Project'
test_font = pygame.font.Font(os.path.join(base_path, 'assets', 'font', 'Pixeltype.ttf'), 24)

game_active = False
start_time = 0
score = 0

sky_surf = pygame.image.load(os.path.join(base_path, 'assets', 'images', 'sky.png')).convert()
ground_surf = pygame.image.load(os.path.join(base_path, 'assets', 'images', 'ground.png')).convert()

banana_surf = pygame.transform.scale2x(pygame.image.load(os.path.join(base_path, 'assets', 'images', 'banana.png')).convert_alpha())
banana_rect = banana_surf.get_rect(midbottom = (800, 325))
fly_surf = pygame.image.load(os.path.join(base_path, 'assets', 'images', 'fly.png')).convert_alpha()

obstacle_rect_list = []

cat_surf = pygame.transform.rotozoom(pygame.image.load(os.path.join(base_path, 'assets', 'images', 'cat.png')).convert_alpha(), 0, 3)
cat_rect = cat_surf.get_rect(midbottom = (500, 330))
cat_gravity = 0

scale = 3
cat_gamescreen = pygame.transform.scale(cat_surf, (cat_surf.width * scale , cat_surf.height * scale))
cat_gamescreen_rect = cat_gamescreen.get_rect(center = (400, 200))

game_name = test_font.render('Banana Cat', False, 'Black')
game_name_rect = game_name.get_rect(center = (400, 100))

start_message = test_font.render('Press space to play', False, 'Black')
start_message_rect = start_message.get_rect(center = (400, 300))

game_message = test_font.render('You lost!  Press Space to restart', False, 'Black')
game_message_rect = game_message.get_rect(center = (400, 300))

obstacle_timer = pygame.USEREVENT + 1
pygame.time.set_timer(obstacle_timer, 1000)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if game_active:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and cat_rect.bottom >= 300:
                    cat_gravity = -20

                if event.key == pygame.K_BACKSPACE:
                    game_active = True
        else:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                game_active = True
                banana_rect.left = 800
                start_time = pygame.time.get_ticks()

        if event.type == obstacle_timer and game_active:
            if randint(0, 2):
                obstacle_rect_list.append(banana_surf.get_rect(bottomright = (randint(900, 1100), 325)))
            else:
                obstacle_rect_list.append(fly_surf.get_rect(bottomright = (randint(900, 1100), 210)))


    if game_active:
        screen.blit(sky_surf, (0, 0))
        screen.blit(ground_surf, (0, 300))
        score = display_score()
        screen.blit(banana_surf, banana_rect)

        cat_gravity += 1
        cat_rect.y += cat_gravity

        if cat_rect.bottom >= 330:
            cat_rect.bottom = 330

        screen.blit(cat_surf, cat_rect)

        obstacle_rect_list = obstacle_movement(obstacle_rect_list)

        game_active = collision(cat_rect, obstacle_rect_list)

    else:
        screen.fill((94, 129, 162))
        screen.blit(cat_gamescreen, cat_gamescreen_rect)
        screen.blit(game_name, game_name_rect)

        obstacle_rect_list.clear()
        cat_rect.midbottom = (80, 300)
        cat_gravity = 0

        if score == 0:
            screen.blit(start_message, start_message_rect)
        else:
            screen.blit(game_message, game_message_rect)


    pygame.display.update()
    clock.tick(60)

