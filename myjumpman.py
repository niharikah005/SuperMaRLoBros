import pygame
import random

pygame.init()

# dimensions
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My JumpMan")

BG = (201, 0,  151)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

GRAVITY = 0.67
JUMP_STRENGTH = -21
PLAYER_SIZE = 50
POINT_SIZE = 20
POINT_HEIGHT_VARIATION = 300  
POINT_SPEED = 5  
GROUND_LEVEL = HEIGHT - PLAYER_SIZE  

PLAYER_IMAGE = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE))
PLAYER_IMAGE.fill(BLACK)
POINT_IMAGE = pygame.Surface((POINT_SIZE, POINT_SIZE))
POINT_IMAGE.fill(GREEN)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = PLAYER_IMAGE
        self.rect = self.image.get_rect(center=(WIDTH // 4, HEIGHT - PLAYER_SIZE))
        self.speed = 5
        self.velocity_y = 0
        self.jumping = False

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
        if keys[pygame.K_SPACE] and not self.jumping:
            self.jumping = True
            self.velocity_y = JUMP_STRENGTH

        self.velocity_y += GRAVITY
        self.rect.y += self.velocity_y

        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
            self.jumping = False

class Point(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = POINT_IMAGE
        self.rect = self.image.get_rect()
        # Randomly choose point can be up or down
        if random.choice([True, False]):
            self.rect.y = HEIGHT - POINT_SIZE  # down
        else:
            self.rect.y = HEIGHT - POINT_SIZE - random.randint(100, POINT_HEIGHT_VARIATION)  # upp
        self.rect.x = WIDTH

    def update(self):
        self.rect.x -= POINT_SPEED
        if self.rect.right < 0:
            self.kill()


all_sprites = pygame.sprite.Group()
points = pygame.sprite.Group()
player = Player()
all_sprites.add(player)

clock = pygame.time.Clock()
score = 0
current_point = None
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    all_sprites.update()

    if not current_point or current_point.rect.right < 0:
        if current_point:
            current_point.kill()
        current_point = Point()
        all_sprites.add(current_point)
        points.add(current_point)

    if current_point and pygame.sprite.collide_rect(player, current_point):
        score += 2
        current_point.kill()
        current_point = None 

    if current_point and current_point.rect.right < player.rect.left:
        score -= 1
        current_point.kill()
        current_point = None

    SCREEN.fill(BG)
    all_sprites.draw(SCREEN)

    # score
    font = pygame.font.SysFont(None, 60)
    score_text = font.render(f'Score: {score}', True, BLACK)
    SCREEN.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(45)

pygame.quit()
