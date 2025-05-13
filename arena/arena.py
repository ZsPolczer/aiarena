# evo_arena/arena/arena.py

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
            # Right wall
            elif agent.x + agent.radius > self.width:
                agent.x = self.width - agent.radius
                agent.vx *= -self.wall_bounce_loss_factor
            
            # Top wall
            if agent.y - agent.radius < 0:
                agent.y = agent.radius
                agent.vy *= -self.wall_bounce_loss_factor
            # Bottom wall
            elif agent.y + agent.radius > self.height:
                agent.y = self.height - agent.radius
                agent.vy *= -self.wall_bounce_loss_factor
                
            # After bounce, ensure velocity is updated if physics model implies it affects agent.current_speed
            # For our simple teleport step, vx/vy are recalculated each frame from current_speed and angle
            # So, for the wall bounce to have a lasting effect on speed, we should reduce current_speed or base_speed.
            # The spec says "10% speed loss". This should probably apply to the magnitude of velocity.
            # Let's adjust agent.base_speed or just vx, vy directly after collision and let it propagate.
            # The current agent.update() recalculates vx, vy from scratch based on thrust input.
            # So a direct modification of vx, vy here is only for the bounce itself.
            # To achieve persistent speed loss, the agent's internal `base_speed` would need to be affected,
            # or the `current_speed` should not be reset to `base_speed` if a bounce just occurred.
            # For now, the current implementation makes the agent lose speed *during* the bounce frame,
            # but can immediately accelerate back to full speed if thrust is applied.
            # The spec's "10% speed loss" for walls is more impactful if it reduces the agent's max speed for a short while or similar.
            # Given "teleport-step position = position + velocity·Δt", and AI will output velocities/thrusts,
            # it's simpler if the wall directly modifies the vx, vy that was used for that step.
            # For Phase 1, the simple vx *= -loss_factor is fine.

    def draw_bounds(self, screen):
        pygame.draw.rect(screen, (50, 50, 50), (0, 0, self.width, self.height), 2)