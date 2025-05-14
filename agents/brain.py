# evo_arena/agents/brain.py
import numpy as np

class TinyNet:
    def __init__(self, w_in=None, w_out=None, input_size=14, hidden_size=50, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if w_in is not None:
            self.w_in = np.array(w_in, dtype=np.float64)
        else:
            # Ensure random initialization is also float64
            self.w_in = np.random.uniform(-1, 1, (self.hidden_size, self.input_size)).astype(np.float64)
            
        if w_out is not None:
            self.w_out = np.array(w_out, dtype=np.float64)
        else:
            # Ensure random initialization is also float64
            self.w_out = np.random.uniform(-1, 1, (self.output_size, self.hidden_size)).astype(np.float64)

        self.fitness = 0.0

    def __call__(self, x):
        """Standard forward pass for evaluation/evolution."""
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)
        
        if x.shape[0] != self.input_size:
            # Basic padding/truncating for mismatched input during general call
            if x.shape[0] < self.input_size:
                x_padded = np.zeros(self.input_size, dtype=np.float64)
                x_padded[:x.shape[0]] = x
                x = x_padded
            elif x.shape[0] > self.input_size:
                x = x[:self.input_size]

        h_pre_activation = self.w_in @ x
        h_activated = np.tanh(h_pre_activation)
        
        y_pre_activation = self.w_out @ h_activated
        y_activated = np.tanh(y_pre_activation)
        return y_activated

    def forward_pass_for_gd(self, x):
        """Forward pass that returns intermediate activations needed for GD/RL."""
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)

        if x.shape[0] != self.input_size:
            # Consistent input handling with __call__
            if x.shape[0] < self.input_size:
                x_padded = np.zeros(self.input_size, dtype=np.float64)
                x_padded[:x.shape[0]] = x
                x = x_padded
            elif x.shape[0] > self.input_size:
                x = x[:self.input_size]

        h_pre_activation = self.w_in @ x
        h_activated = np.tanh(h_pre_activation)
        
        y_pre_activation = self.w_out @ h_activated
        y_activated = np.tanh(y_pre_activation)
        
        return x, h_pre_activation, h_activated, y_pre_activation, y_activated

    def backward_pass(self, x_input, h_activated, y_activated, target_outputs):
        """
        Performs backpropagation to compute gradients for SUPERVISED learning.
        x_input: The original input vector to the network.
        h_activated: Activations of the hidden layer.
        y_activated: Activations of the output layer (network's prediction).
        target_outputs: The desired output values for supervised learning.
        """
        if not isinstance(target_outputs, np.ndarray):
            target_outputs = np.array(target_outputs, dtype=np.float64)

        error_output_layer = y_activated - target_outputs
        delta_output_layer = error_output_layer * (1 - y_activated**2)
        dW_out = np.outer(delta_output_layer, h_activated)
        error_hidden_layer = self.w_out.T @ delta_output_layer
        delta_hidden_layer = error_hidden_layer * (1 - h_activated**2)
        dW_in = np.outer(delta_hidden_layer, x_input)
        
        return dW_in, dW_out

    def get_policy_gradient_for_action(self, x_input, h_activated, y_activated_actions, match_reward):
        """
        Calculates a HEURISTIC pseudo-gradient for a REINFORCE-like update.
        This version aims to push actions towards extremes based on reward.

        x_input: The original input vector to the network for this state.
        h_activated: Activations of the hidden layer for this state.
        y_activated_actions: The network's tanh outputs (actions taken) for this state.
        match_reward: Scalar reward for the entire episode (e.g., +1 for win, -1 for loss, 0 for draw).
        
        Returns: dW_in, dW_out gradients scaled by reward.
        """

        # Heuristic: "target" for y_activated is to push it towards its sign if reward is positive,
        # and away from its sign (towards opposite or zero) if reward is negative.
        # Error signal for output activations: y_activated - target_sign
        # If R > 0, we want y_activated to move towards sign(y_activated). Target error = y_activated - sign(y_activated)
        # If R < 0, we want y_activated to move away from sign(y_activated). Target error = y_activated + sign(y_activated)
        # This can be combined: error_for_y_activations = y_activated - (match_reward * np.sign(y_activated_actions))
        # No, simpler: the "effective error" that gets multiplied by (1-y^2) should scale with reward.
        
        # The error signal for the output layer's PRE-ACTIVATION (before tanh)
        # This signal will be multiplied by h_activated to get dW_out.
        # And then backpropagated.
        # A common REINFORCE gradient for action `a` from policy `pi(a|s, theta)` is `grad_theta(log pi(a|s, theta))`.
        # For deterministic policy `a = f(s, theta)`, this is more complex.
        #
        # Heuristic: We want to make the chosen action `y_activated_actions` more likely if `match_reward` is positive.
        # "more likely" means pushing the pre-activations that led to `y_activated_actions` further in that direction.
        # The gradient of `tanh(z)` is `1 - tanh(z)^2`.
        # The "error" term (often denoted delta) for the output layer pre-activations:
        # delta_output_pre_activation = "some_gradient_signal"
        #
        # If we consider Loss = -Reward * sum(y_activated_actions) (naive, but a starting point)
        # dLoss/dy_pre = -Reward * (1 - y_activated_actions**2)
        # This would reinforce positive actions if R is positive, and negative actions if R is positive.

        # Let's define an "advantage" or "target direction" for each action component.
        # If R > 0, we want action `y_i` to be more `y_i`.
        # If R < 0, we want action `y_i` to be less `y_i` (move towards 0 or opposite).
        # The error signal for y_j (output of tanh) could be `match_reward * y_j`.
        # If R=1, y_j=0.8 -> error=0.8.  If y_j=-0.5 -> error=-0.5.
        # This signal `e_j` is then used: delta_pre_activation_j = e_j * (1 - y_j^2)
        
        error_signal_on_output_activations = match_reward * y_activated_actions
        
        # Gradient for the output layer's pre-activation
        delta_output_layer_pre_activation = error_signal_on_output_activations * (1 - y_activated_actions**2)

        # Gradient of Loss w.r.t. w_out
        dW_out = np.outer(delta_output_layer_pre_activation, h_activated)

        # Propagate error to hidden layer
        error_hidden_layer_activation = self.w_out.T @ delta_output_layer_pre_activation

        # Gradient for the hidden layer's pre-activation
        delta_hidden_layer_pre_activation = error_hidden_layer_activation * (1 - h_activated**2)

        # Gradient of Loss w.r.t. w_in
        dW_in = np.outer(delta_hidden_layer_pre_activation, x_input)
        
        return dW_in, dW_out


    def update_weights(self, dW_in, dW_out, learning_rate):
        """Updates weights using the calculated gradients."""
        self.w_in -= learning_rate * dW_in
        self.w_out -= learning_rate * dW_out

    def mutate(self, sigma=0.2, mutation_rate_weights=1.0, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        w_in_mutated = self.w_in.copy()
        w_out_mutated = self.w_out.copy()

        if rng.random() < mutation_rate_weights:
            noise_in = rng.normal(0, sigma, self.w_in.shape).astype(np.float64)
            w_in_mutated += noise_in
        
        if rng.random() < mutation_rate_weights:
            noise_out = rng.normal(0, sigma, self.w_out.shape).astype(np.float64)
            w_out_mutated += noise_out
        
        return TinyNet(w_in_mutated, w_out_mutated, self.input_size, self.hidden_size, self.output_size)

    @classmethod
    def crossover(cls, parent1, parent2, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        mask_in = rng.random(parent1.w_in.shape) < 0.5
        w_in_child = np.where(mask_in, parent1.w_in, parent2.w_in)
        
        mask_out = rng.random(parent1.w_out.shape) < 0.5
        w_out_child = np.where(mask_out, parent1.w_out, parent2.w_out)
        
        return cls(w_in_child, w_out_child, parent1.input_size, parent1.hidden_size, parent1.output_size)

    def get_genome_params(self):
        return self.w_in, self.w_out