import numpy as np
import pandas as pd
import pygame
from pygame.locals import *

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Font
font = pygame.font.Font(None, 36)  # None for default font, 36 for font size

# Load in images of players and spy
spy_img = pygame.image.load("media/spy.png")
spy_img = pygame.transform.scale(spy_img, (100, 100))
player_img = pygame.image.load("media/player.jpg")
player_img = pygame.transform.scale(player_img, (100, 100))

# Position players on screen
player1_pos = pygame.Vector2(
    WIDTH / 4 - spy_img.get_width() / 2, HEIGHT / 4 - spy_img.get_height() / 2
)
player3_pos = pygame.Vector2(
    WIDTH / 4 - spy_img.get_width() / 2, HEIGHT * 3 / 4 - spy_img.get_height() / 2
)
player4_pos = pygame.Vector2(
    WIDTH * 3 / 4 - spy_img.get_width() / 2, HEIGHT * 3 / 4 - spy_img.get_height() / 2
)
player2_pos = pygame.Vector2(
    WIDTH * 3 / 4 - spy_img.get_width() / 2, HEIGHT / 4 - spy_img.get_height() / 2
)

# Position player labels on screen
label1_pos = text_pos = pygame.Vector2(
    WIDTH / 4 - spy_img.get_width() / 2, HEIGHT / 4 - 50 - spy_img.get_height() / 2
)
label3_pos = pygame.Vector2(
    WIDTH / 4 - spy_img.get_width() / 2, HEIGHT * 3 / 4 - 50 - spy_img.get_height() / 2
)
label4_pos = pygame.Vector2(
    WIDTH * 3 / 4 - spy_img.get_width() / 2,
    HEIGHT * 3 / 4 - 50 - spy_img.get_height() / 2,
)
label2_pos = pygame.Vector2(
    WIDTH * 3 / 4 - spy_img.get_width() / 2, HEIGHT / 4 - 50 - spy_img.get_height() / 2
)

# Get conversation from conversation.csv
conv = pd.read_csv("conversation.csv", index_col=None)
conv_lst = conv.values.tolist()
conv_idx = 0


# Function to wrap text
def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        # Check if adding the next word exceeds the max width
        test_line = f"{current_line} {word}".strip()
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word  # Start a new line with the current word
    lines.append(current_line)  # Add the last line
    return lines


# Function to render wrapped text using Vector2 for position
def render_wrapped_text(text, font, color, position, max_width, line_spacing=5):
    """Render text wrapped to fit within a specific width using a Vector2 for position."""
    lines = wrap_text(text, font, max_width)
    for i, line in enumerate(lines):
        text_surface = font.render(line, True, color)
        line_position = position + pygame.Vector2(
            0, i * (font.get_linesize() + line_spacing)
        )
        screen.blit(text_surface, line_position)


# Main game loop
running = True
while running:
    screen.fill(WHITE)

    # Display spy and player images on screen
    screen.blit(spy_img, player1_pos)
    screen.blit(player_img, player2_pos)
    screen.blit(player_img, player3_pos)
    screen.blit(player_img, player4_pos)

    # From player number, get text position
    player_idx = conv_lst[conv_idx][0]
    if player_idx == 1:
        text_pos = pygame.Vector2(
            WIDTH / 4 - spy_img.get_width() / 2,
            HEIGHT / 4 + 100 - spy_img.get_height() / 2,
        )
    elif player_idx == 2:
        text_pos = pygame.Vector2(
            WIDTH * 3 / 4 - spy_img.get_width() / 2,
            HEIGHT / 4 + 100 - spy_img.get_height() / 2,
        )
    elif player_idx == 3:
        text_pos = pygame.Vector2(
            WIDTH / 4 - spy_img.get_width() / 2,
            HEIGHT * 3 / 4 + 100 - spy_img.get_height() / 2,
        )
    else:
        text_pos = pygame.Vector2(
            WIDTH * 3 / 4 - spy_img.get_width() / 2,
            HEIGHT * 3 / 4 + 100 - spy_img.get_height() / 2,
        )

    # Display player labels
    render_wrapped_text(f"Player 1", font, BLACK, label1_pos, WIDTH / 4)
    render_wrapped_text(f"Player 2", font, BLACK, label2_pos, WIDTH / 4)
    render_wrapped_text(f"Player 3", font, BLACK, label3_pos, WIDTH / 4)
    render_wrapped_text(f"Player 4", font, BLACK, label4_pos, WIDTH / 4)

    # Display the current dialogue
    render_wrapped_text(conv_lst[conv_idx][1], font, BLACK, text_pos, WIDTH / 4)

    for event in pygame.event.get():
        if (
            event.type == pygame.QUIT
        ):  # pygame.QUIT event means click X to close the window
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Move to the next dialogue or exit if it's the end
                conv_idx += 1
                if conv_idx >= len(conv_lst):
                    running = False

    # flip() the display to put your work on screen
    pygame.display.flip()

pygame.quit()
