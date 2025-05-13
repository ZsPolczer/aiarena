# evo_arena/arena/arena.py
import pygame # <--- ADD THIS LINE

class Arena:
    def __init__(self, width, height, wall_bounce_loss_factor=0.9):
        self.width = width
        self.height = height
        self.agents = []
        self.wall_bounce_loss_factor = wall_bounce_loss_factor

    def add_agent(self, agent):
        self.agents.append(agent)

    def update(self, dt):
        for agent in self.agents:
            if agent.is_dummy: # Let agent update its internal state if any (none for dummy now)
                agent.update(dt) # But dummy won't move
                continue

            # Agent decides its velocity (already done in agent.update for manual)
            # Then agent updates its own position based on its velocity
            agent.update(dt) # This moves the agent

            # Wall collision
            # Left wall
            if agent.x - agent.radius < 0:
                agent.x = agent.radius
                agent.vx *= -self.wall_bounce_loss_factor
                # To make speed loss more persistent, we might need to affect agent.current_speed or agent.base_speed
                # For now, this impacts velocity for the current step only.
                if hasattr(agent, 'current_speed'): # Check if agent has current_speed
                    agent.current_speed *= self.wall_bounce_loss_factor


            # Right wall
            elif agent.x + agent.radius > self.width:
                agent.x = self.width - agent.radius
                agent.vx *= -self.wall_bounce_loss_factor
                if hasattr(agent, 'current_speed'):
                    agent.current_speed *= self.wall_bounce_loss_factor

            # Top wall
            if agent.y - agent.radius < 0:
                agent.y = agent.radius
                agent.vy *= -self.wall_bounce_loss_factor
                if hasattr(agent, 'current_speed'):
                    agent.current_speed *= self.wall_bounce_loss_factor

            # Bottom wall
            elif agent.y + agent.radius > self.height:
                agent.y = self.height - agent.radius
                agent.vy *= -self.wall_bounce_loss_factor
                if hasattr(agent, 'current_speed'):
                    agent.current_speed *= self.wall_bounce_loss_factor
                
    def draw_bounds(self, screen):
        # Draw the arena boundary rectangle
        pygame.draw.rect(screen, (50, 50, 50), (0, 0, self.width, self.height), 2) # 2 is thickness