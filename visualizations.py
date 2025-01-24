import pygame
from pygame.locals import *

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

WIDTH, HEIGHT = 1280, 720

# Load in images of players and spy
SPY_IMG = pygame.image.load("media/spy.png")
SPY_IMG = pygame.transform.scale(SPY_IMG, (100, 100))
PLAYER_IMG = pygame.image.load("media/player.jpg")
PLAYER_IMG = pygame.transform.scale(PLAYER_IMG, (100, 100))

# Position players on self.screen
PLAYER4_POS = pygame.Vector2(
    WIDTH / 4 - SPY_IMG.get_width() / 2, HEIGHT / 4 - SPY_IMG.get_height() / 2
)
PLAYER2_POS = pygame.Vector2(
    WIDTH / 4 - SPY_IMG.get_width() / 2, HEIGHT * 3 / 4 - SPY_IMG.get_height() / 2
)
PLAYER3_POS = pygame.Vector2(
    WIDTH * 3 / 4 - SPY_IMG.get_width() / 2, HEIGHT * 3 / 4 - SPY_IMG.get_height() / 2
)
PLAYER1_POS= pygame.Vector2(
    WIDTH * 3 / 4 - SPY_IMG.get_width() / 2, HEIGHT / 4 - SPY_IMG.get_height() / 2
)

# Position player labels on self.screen
LABEL4_POS = text_pos = pygame.Vector2(
    WIDTH / 4 - SPY_IMG.get_width() / 2, HEIGHT / 4 - 50 - SPY_IMG.get_height() / 2
)
LABEL2_POS= pygame.Vector2(
    WIDTH / 4 - SPY_IMG.get_width() / 2, HEIGHT * 3 / 4 - 50 - SPY_IMG.get_height() / 2
)
LABEL3_POS= pygame.Vector2(
    WIDTH * 3 / 4 - SPY_IMG.get_width() / 2,
    HEIGHT * 3 / 4 - 50 - SPY_IMG.get_height() / 2,
)
LABEL1_POS= pygame.Vector2(
    WIDTH * 3 / 4 - SPY_IMG.get_width() / 2, HEIGHT / 4 - 50 - SPY_IMG.get_height() / 2
)


class Visualization:
    def __init__(self, player_names: list[str], spy_index: int):
        self.player_names = player_names
        self.spy_index = spy_index

        # Initialize pygame
        pygame.init()

        # Screen dimensions
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        # Font
        self.font = pygame.font.Font(
            None, 36
        )  # None for default font, 36 for font size

    def __del__(self):
        pygame.quit()

    # Function to wrap text
    def _wrap_text(self, text, font, max_width):
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
    def _render_wrapped_text(
        self, text, font, color, position, max_width, line_spacing=5
    ):
        """Render text wrapped to fit within a specific width using a Vector2 for position."""
        lines = self._wrap_text(text, font, max_width)
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, color)
            line_position = position + pygame.Vector2(
                0, i * (font.get_linesize() + line_spacing)
            )
            self.screen.blit(text_surface, line_position)

    def render_text(self, player_idx: int, msg: str):
        self.screen.fill(WHITE)

        # Display spy and player images on self.screen
        for i, pos in enumerate([PLAYER1_POS, PLAYER2_POS, PLAYER3_POS, PLAYER4_POS]):
            img = SPY_IMG if i == self.spy_index else PLAYER_IMG
            self.screen.blit(img, pos)

        # From player number, get text position
        if player_idx == 1:
            text_pos = pygame.Vector2(
                WIDTH / 4 - SPY_IMG.get_width() / 2,
                HEIGHT / 4 + 100 - SPY_IMG.get_height() / 2,
            )
        elif player_idx == 2:
            text_pos = pygame.Vector2(
                WIDTH * 3 / 4 - SPY_IMG.get_width() / 2,
                HEIGHT / 4 + 100 - SPY_IMG.get_height() / 2,
            )
        elif player_idx == 3:
            text_pos = pygame.Vector2(
                WIDTH / 4 - SPY_IMG.get_width() / 2,
                HEIGHT * 3 / 4 + 100 - SPY_IMG.get_height() / 2,
            )
        else:
            text_pos = pygame.Vector2(
                WIDTH * 3 / 4 - SPY_IMG.get_width() / 2,
                HEIGHT * 3 / 4 + 100 - SPY_IMG.get_height() / 2,
            )

        # Display player labels
        for i, pos in enumerate([LABEL1_POS, LABEL2_POS, LABEL3_POS, LABEL4_POS]):
            self._render_wrapped_text(
                self.player_names[i], self.font, BLACK, pos, WIDTH / 4
            )

        # Display the current dialogue
        self._render_wrapped_text(msg, self.font, BLACK, text_pos, WIDTH / 4)

        # flip() the display to put your work on self.screen
        pygame.display.flip()
