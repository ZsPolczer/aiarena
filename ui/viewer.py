# evo_arena/ui/viewer.py
import pygame
import math # For radians

class Viewer:
    def __init__(self, width, height, title="Evo Arena"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30) # Slightly larger font
        self.info_font = pygame.font.SysFont(None, 24)


    def draw_firing_cone(self, screen, agent):
        if not agent.is_alive() or not agent.is_firing_command : # or agent.weapon_cooldown_timer < agent.weapon_cooldown_time * 0.9: # Show cone briefly after firing
            # Only draw if actively trying to fire *this tick* or very recently fired.
            # The agent.is_firing_command is set if cooldown allows and input commands fire.
            # So this condition should be fine.
            return

        cone_color = (255, 255, 0, 100)  # Yellow, semi-transparent
        
        # Points for the cone polygon
        # Point 1: Agent's position
        p1 = (int(agent.x), int(agent.y))
        
        # Point 2: End of cone, left edge
        angle_left_rad = math.radians(agent.angle_deg - agent.weapon_arc_deg / 2.0)
        p2_x = agent.x + agent.weapon_range * math.cos(angle_left_rad)
        p2_y = agent.y + agent.weapon_range * math.sin(angle_left_rad)
        p2 = (int(p2_x), int(p2_y))

        # Point 3: End of cone, right edge
        angle_right_rad = math.radians(agent.angle_deg + agent.weapon_arc_deg / 2.0)
        p3_x = agent.x + agent.weapon_range * math.cos(angle_right_rad)
        p3_y = agent.y + agent.weapon_range * math.sin(angle_right_rad)
        p3 = (int(p3_x), int(p3_y))

        # Create a surface for transparency
        cone_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.polygon(cone_surface, cone_color, [p1, p2, p3])
        screen.blit(cone_surface, (0,0))


    def run_manual_loop(self, arena, manual_agent, fps): # `manual_agent` ref might not be needed if arena handles all
        running = True
        dt = 1.0 / fps
        match_message = ""
        game_over = False

        # Store initial agent states for reset if needed (simple way)
        for ag in arena.agents:
            ag.initial_x = ag.x
            ag.initial_y = ag.y
            ag.initial_angle_deg = ag.angle_deg
            ag.max_hp_initial = ag.max_hp # Store original max_hp

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and game_over: # Reset key
                        arena.reset() # Resets game time, agent HP/pos
                        for ag in arena.agents: # Full agent state reset
                            ag.x = ag.initial_x
                            ag.y = ag.initial_y
                            ag.angle_deg = ag.initial_angle_deg
                            ag.hp = ag.max_hp_initial # Use stored initial max_hp
                            ag.weapon_cooldown_timer = 0.0
                            ag.vx = 0.0
                            ag.vy = 0.0
                            ag.is_firing_command = False
                        game_over = False
                        match_message = ""


            if not game_over:
                keys = pygame.key.get_pressed()
                # Find the manual agent in the arena list if not passed directly
                # This assumes only one manual agent or clear identification
                for agent_in_arena in arena.agents:
                    if not agent_in_arena.is_dummy and agent_in_arena.brain is None: # Assuming this is our manual agent
                        agent_in_arena.manual_control(keys)
                        break # Process only one manual agent

                # Update game state
                arena.update(dt) 

                # Check for match end
                is_over, winner_team, message = arena.check_match_end_conditions()
                if is_over:
                    game_over = True
                    match_message = message
                    if winner_team is not None:
                        match_message += f" Winner: Team {winner_team}"
                    print(match_message)


            # Drawing
            self.screen.fill((30, 30, 30))  # Dark grey background
            arena.draw_bounds(self.screen)
            
            for agent_to_draw in arena.agents:
                agent_to_draw.draw(self.screen) # Agent draws itself (body, HP bar, etc.)
                if agent_to_draw.is_firing_command and agent_to_draw.is_alive(): # Draw cone if firing this tick
                    self.draw_firing_cone(self.screen, agent_to_draw)
            
            # Display Game Over Message
            if game_over:
                msg_surf = self.font.render(match_message, True, (255, 255, 0))
                msg_rect = msg_surf.get_rect(center=(self.width / 2, self.height / 2))
                self.screen.blit(msg_surf, msg_rect)
                
                reset_surf = self.info_font.render("Press 'R' to reset", True, (200, 200, 200))
                reset_rect = reset_surf.get_rect(center=(self.width / 2, self.height / 2 + 40))
                self.screen.blit(reset_surf, reset_rect)

            # Display FPS and Game Time (optional)
            # current_fps = self.clock.get_fps()
            # fps_text = self.info_font.render(f"FPS: {current_fps:.1f}", True, (220, 220, 220))
            # self.screen.blit(fps_text, (10, 10))
            time_text = self.info_font.render(f"Time: {arena.game_time:.1f}s", True, (220, 220, 220))
            self.screen.blit(time_text, (10, 10))


            pygame.display.flip()
            self.clock.tick(fps)

        pygame.quit()