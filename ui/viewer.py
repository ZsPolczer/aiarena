# evo_arena/ui/viewer.py
import pygame
import math # For radians

class Viewer:
    def __init__(self, width, height, arena, title="Evo Arena"): # MODIFIED signature
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.arena = arena  # ADDED: Store arena instance
        pygame.display.set_caption(title) # CORRECTED: Uses the string 'title'
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30) 
        self.info_font = pygame.font.SysFont(None, 24)


    def draw_firing_cone(self, screen, agent):
        if not agent.is_alive() or not agent.is_firing_command :
            return

        cone_color = (255, 255, 0, 100)  # Yellow, semi-transparent
        
        p1 = (int(agent.x), int(agent.y))
        
        angle_left_rad = math.radians(agent.angle_deg - agent.weapon_arc_deg / 2.0)
        p2_x = agent.x + agent.weapon_range * math.cos(angle_left_rad)
        p2_y = agent.y + agent.weapon_range * math.sin(angle_left_rad)
        p2 = (int(p2_x), int(p2_y))

        angle_right_rad = math.radians(agent.angle_deg + agent.weapon_arc_deg / 2.0)
        p3_x = agent.x + agent.weapon_range * math.cos(angle_right_rad)
        p3_y = agent.y + agent.weapon_range * math.sin(angle_right_rad)
        p3 = (int(p3_x), int(p3_y))

        cone_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.polygon(cone_surface, cone_color, [p1, p2, p3])
        screen.blit(cone_surface, (0,0))

    # RENAMED from run_manual_loop and MODIFIED parameters
    def run_simulation_loop(self, fps, manual_agent_id=None):
        running = True
        dt = 1.0 / fps
        match_message = ""
        game_over = False

        # Store initial agent states for reset if needed
        for ag in self.arena.agents: # MODIFIED: use self.arena
            ag.initial_x = ag.x
            ag.initial_y = ag.y
            ag.initial_angle_deg = ag.angle_deg
            # Ensure max_hp_initial exists or is correctly initialized
            if not hasattr(ag, 'max_hp_initial'): # Defensive check
                 ag.max_hp_initial = ag.max_hp

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and game_over: 
                        self.arena.reset() # MODIFIED: use self.arena
                        for ag in self.arena.agents: # MODIFIED: use self.arena
                            ag.x = ag.initial_x
                            ag.y = ag.initial_y
                            ag.angle_deg = ag.initial_angle_deg
                            ag.hp = ag.max_hp_initial 
                            ag.weapon_cooldown_timer = 0.0
                            ag.vx = 0.0
                            ag.vy = 0.0
                            ag.is_firing_command = False
                        game_over = False
                        match_message = ""


            if not game_over:
                keys = pygame.key.get_pressed()
                
                # MODIFIED: Control agent based on manual_agent_id
                if manual_agent_id:
                    for agent_in_arena in self.arena.agents: # MODIFIED: use self.arena
                        if agent_in_arena.agent_id == manual_agent_id:
                            # Make sure it's controllable (not dummy, no brain)
                            if not agent_in_arena.is_dummy and agent_in_arena.brain is None:
                                agent_in_arena.manual_control(keys)
                            break 
                
                self.arena.update(dt) # MODIFIED: use self.arena

                is_over, winner_team, message = self.arena.check_match_end_conditions() # MODIFIED: use self.arena
                if is_over:
                    game_over = True
                    match_message = message
                    if winner_team is not None:
                        match_message += f" Winner: Team {winner_team}"
                    else: # Handle draw case message more explicitly if needed
                        match_message += " (Draw)" 
                    print(match_message)


            # Drawing
            self.screen.fill((30, 30, 30))
            self.arena.draw_bounds(self.screen) # MODIFIED: use self.arena
            
            for agent_to_draw in self.arena.agents: # MODIFIED: use self.arena
                agent_to_draw.draw(self.screen) 
                if agent_to_draw.is_firing_command and agent_to_draw.is_alive(): 
                    self.draw_firing_cone(self.screen, agent_to_draw)
            
            if game_over:
                msg_surf = self.font.render(match_message, True, (255, 255, 0))
                msg_rect = msg_surf.get_rect(center=(self.width / 2, self.height / 2))
                self.screen.blit(msg_surf, msg_rect)
                
                reset_surf = self.info_font.render("Press 'R' to reset", True, (200, 200, 200))
                reset_rect = reset_surf.get_rect(center=(self.width / 2, self.height / 2 + 40))
                self.screen.blit(reset_surf, reset_rect)

            time_text = self.info_font.render(f"Time: {self.arena.game_time:.1f}s", True, (220, 220, 220)) # MODIFIED: use self.arena
            self.screen.blit(time_text, (10, 10))


            pygame.display.flip()
            self.clock.tick(fps)

        pygame.quit()