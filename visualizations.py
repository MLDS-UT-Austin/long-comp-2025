import pygame
from pygame.locals import *
import numpy as np

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Colors
WHITE = (255,255,255)
BLACK = (0,0,0)

# Font
font = pygame.font.Font(None, 36)  # None for default font, 36 for font size

# Dialogue data
dialogue = [
    "Hello!",
    "Hi!"
]
dialogue_index = 0

# Function to render text
def render_text(text, text_pos):
    text_surface = font.render(text, True, BLACK)
    screen.blit(text_surface, text_pos)

# Position players on screen
agent1_pos = pygame.Vector2(WIDTH / 4, HEIGHT / 4)
agent2_pos = pygame.Vector2(WIDTH / 4, HEIGHT * 3/4)
agent3_pos = pygame.Vector2(WIDTH * 3/4, HEIGHT * 3/4)
agent4_pos = pygame.Vector2(WIDTH * 3/4, HEIGHT / 4)

# Position dialogue on screen
text1_pos = pygame.Vector2(WIDTH / 4, HEIGHT / 4 + 100)
text2_pos = pygame.Vector2(WIDTH / 4, HEIGHT * 3/4 + 100)
text3_pos = pygame.Vector2(WIDTH * 3/4, HEIGHT * 3/4 + 100)
text4_pos = pygame.Vector2(WIDTH * 3/4, HEIGHT / 4 + 100)

# Load in images of players and spy
spy_img = pygame.image.load("media/spy.png")
spy_img = pygame.transform.scale(spy_img, (100, 100))
player_img = pygame.image.load("media/player.jpg")
player_img = pygame.transform.scale(player_img, (100, 100))

# Main game loop  
running = True        
while running:
    screen.fill(WHITE)

    # Display spy and player images on screen
    screen.blit(spy_img, agent1_pos)
    screen.blit(player_img, agent2_pos)
    screen.blit(player_img, agent3_pos)
    screen.blit(player_img, agent4_pos)

    # Display the current dialogue
    render_text(dialogue[dialogue_index], text1_pos)
    render_text(dialogue[dialogue_index], text2_pos)
    render_text(dialogue[dialogue_index], text3_pos)
    render_text(dialogue[dialogue_index], text4_pos)

    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # pygame.QUIT event means the user clicked X to close your window
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Move to the next dialogue or exit if it's the end
                dialogue_index += 1
                if dialogue_index >= len(dialogue):
                    running = False

    # flip() the display to put your work on screen
    pygame.display.flip()

pygame.quit()
