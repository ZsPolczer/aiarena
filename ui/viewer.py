# evo_arena/ui/viewer.py
import pygame

class Viewer:
    def __init__(self, width, height, title="Evo Arena"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)


    def run_manual_loop(self, arena, manual_agent, fps):
        running = True
        dt = 1.0 / fps

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if not manual_agent.is_dummy:
                manual_agent.manual_control(keys)

            # Update game state
            arena.update(dt) # This will call agent.update() and handle collisions

            # Drawing
            self.screen.fill((0, 0, 0))  # Black background
            arena.draw_bounds(self.screen)
            for agent_to_draw in arena.agents:
                agent_to_draw.draw(self.screen)
            
            # Display FPS (optional)
            # current_fps = self.clock.get_fps()
            # fps_text = self.font.render(f"FPS: {current_fps:.2f}", True, (255, 255, 255))
            # self.screen.blit(fps_text, (10, 10))


            pygame.display.flip()
            self.clock.tick(fps)

        pygame.quit()