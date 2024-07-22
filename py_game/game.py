import os
import pygame
import time
import random
pygame.init()
pygame.font.init()
# Load assets:
DINO_IMG = pygame.transform.scale(pygame.image.load(os.path.join("dino.png")), (100,200))
CROUCH_IMG = pygame.transform.scale(pygame.image.load(os.path.join("crouch.png")), (100,200))
ENEMY_IMG = pygame.transform.scale(pygame.image.load(os.path.join("tree.png")),(100,170))
# game window
WIDTH,HEIGHT = 750,750
WIN = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('random game')
BG = pygame.Color(0,155,255)


# creating a class:
class Box:
    def __init__(self,x, y):
        self.x = x
        self.y = y
        self.obj_img = None

    def draw(self, window):
        window.blit(self.obj_img, (self.x,self.y))

    def width(self):
        return self.ship_img.get_width()
    def height(self):
        return self.ship_img.get_height()

class PlayerBox(Box):
    def __init__(self,x,y):
        super().__init__(x,y)
        self.obj_img = DINO_IMG
        self.mask = pygame.mask.from_surface(self.obj_img)

class Enemy(Box):
    def __init__(self,x,y):
        super().__init__(x,y)
        self.obj_img = ENEMY_IMG
        self.mask = pygame.mask.from_surface(self.obj_img)

    def move(self, vel):
        self.x -= vel

     

# game loop
def main():
    run = True
    FPS = 60
    score = 0
    font = pygame.font.SysFont("sans", 45)
    clock = pygame.time.Clock()
    player = PlayerBox(50,HEIGHT-250)
    enemies = []
    wave_length = 3
    enemy_vel = 6
    crouch = False
    count = 0
    lost_count = 0
    lost = False
    # redraw window:
    def redraw(score):
        # background
        score = font.render(f"score: {score}", 1, (255,255,255))
        WIN.fill(BG)
        pygame.draw.rect(WIN, (150,75,10), (0, HEIGHT-100, WIDTH, 100), 0)
        pygame.draw.rect(WIN, (80,175,42), (0,HEIGHT-120,WIDTH,20), 0)
        for enemy in enemies:
            enemy.draw(WIN)

        player.draw(WIN)

        if lost:
            lost_label = font.render("You lost!!", 1, (255,255,255))
            WIN.fill((0,0,0))
            WIN.blit(lost_label, (WIDTH/2-lost_label.get_width()/2, 350))
        WIN.blit(score, (WIDTH-score.get_width()-10,10))
        # to refresh the screen
        pygame.display.update()

    def down():
        air_time = 0
        if player.y < HEIGHT-250:
            player.y += 10
        if player.y < 420:
            player.y = 420
            air_time += 1
        if air_time > 20:
            player.y = HEIGHT-250

    def up():
        if player.y > HEIGHT-250:
            player.y = HEIGHT-250
        if player.height != 95:
            player.height = 95

    def collision(enemy):
        offset_x = player.x - enemy.x
        offset_y = player.y - enemy.y
        collision_point = player.mask.overlap(enemy.mask, (offset_x,offset_y))
        if collision_point:
            return True

    while run:
        clock.tick(FPS)
        count += 1
        down()
        up()
        redraw(score)

        for enemy in enemies:
            if collision(enemy):
                lost = True
                lost_count += 1

        if lost:
            if lost_count > FPS * 2:
                run = False
            else:
                continue
        
        if count%30 == 0:
            score += 1

        if len(enemies) == 0:
            wave_length += 1
            if enemy_vel >= 15:
                enemy_vel = 15
            for i in range(wave_length):
                enemy = Enemy(random.randrange(800, 2000), HEIGHT-235)
                enemies.append(enemy)
        i = 0
        while i < len(enemies) - 1:
            if enemies[i].x >= enemies[i+1].x - 150:
                enemies.pop(i)
            else:
                i += 1


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            player.y -= 20
        if keys[pygame.K_DOWN]:
            crouch = True

        if crouch == True:
            player.obj_img = CROUCH_IMG
        else:
            player.obj_img = DINO_IMG

        for enemy in enemies[:]:
            enemy.move(enemy_vel)
            if enemy.x < 0:
                enemies.remove(enemy)

        crouch = False
        
    
    pygame.quit()

main()