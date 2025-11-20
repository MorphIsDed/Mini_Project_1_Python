import pygame
import random
from enum import Enum
from collections import deque

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class SnakeGame:
    def __init__(self, width=800, height=600, block_size=20):
        pygame.init()
        self.width = width
        self.height = height
        self.block_size = block_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        
        self.reset_game()
        self.show_loading_screen()
    
    def show_loading_screen(self):
        loading = True
        while loading:
            self.screen.fill((0, 0, 0))
            title_text = self.font.render("Snake Game", True, (255, 255, 255))
            play_text = self.font.render("Press ENTER to Play", True, (255, 255, 255))
            self.screen.blit(title_text, (self.width // 2 - title_text.get_width() // 2, self.height // 2 - 50))
            self.screen.blit(play_text, (self.width // 2 - play_text.get_width() // 2, self.height // 2 + 10))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    loading = False
    
    def reset_game(self):
        start_x = self.width // (2 * self.block_size)
        start_y = self.height // (2 * self.block_size)
        self.snake = deque([(start_x, start_y)])
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False
    
    def spawn_food(self):
        while True:
            x = random.randint(0, (self.width // self.block_size) - 1)
            y = random.randint(0, (self.height // self.block_size) - 1)
            if (x, y) not in self.snake:
                return (x, y)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.next_direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.next_direction = Direction.DOWN
                elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.next_direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.next_direction = Direction.RIGHT
                elif event.key == pygame.K_SPACE and self.game_over:
                    self.reset_game()
                elif event.key == pygame.K_RETURN and self.game_over:
                    self.reset_game()
        return True
    
    def update(self):
        if self.game_over:
            return
        
        self.direction = self.next_direction
        dx, dy = self.direction.value
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)
        
        if (new_head in self.snake or 
            new_head[0] < 0 or new_head[0] >= self.width // self.block_size or
            new_head[1] < 0 or new_head[1] >= self.height // self.block_size):
            self.game_over = True
            return
        
        self.snake.appendleft(new_head)
        
        if new_head == self.food:
            self.score += 10
            self.food = self.spawn_food()
        else:
            self.snake.pop()
    
    def draw(self):
        self.screen.fill((30, 30, 30))
        
        for segment in self.snake:
            rect = pygame.Rect(segment[0] * self.block_size, segment[1] * self.block_size,
                             self.block_size, self.block_size)
            pygame.draw.rect(self.screen, (0, 255, 0), rect)  
        
        food_rect = pygame.Rect(self.food[0] * self.block_size, self.food[1] * self.block_size,
                               self.block_size, self.block_size)
        pygame.draw.rect(self.screen, (255, 0, 0), food_rect)
        
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        if self.game_over:
            game_over_text = self.font.render("GAME OVER!", True, (255, 0, 0))
            restart_text = self.font.render("Press ENTER to restart", True, (255, 255, 255))
            text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 20))
            restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 20))
            self.screen.blit(game_over_text, text_rect)
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(10)
        
        pygame.quit()

if __name__ == "__main__":
    game = SnakeGame()
    game.run()