# evo_arena/agents/brain.py
import numpy as np

class TinyNet:
    def __init__(self, w_in=None, w_out=None, input_size=14, hidden_size=16, output_size=4):
        """
        A simple two-layer neural network.
        - input_size: Number of input neurons (14 for this project)
        - hidden_size: Number of neurons in the hidden layer (16 for this project)
        - output_size: Number of output neurons (4 for this project)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        # Use a global RNG for reproducibility if desired, or np.random for simplicity now
        # For actual evolution, you'll want to manage the seed carefully.
        # self.rng = np.random.default_rng(seed) 
        
        if w_in is not None:
            self.w_in = np.array(w_in)
        else:
            self.w_in = np.random.uniform(-1, 1, (self.hidden_size, self.input_size))
            
        if w_out is not None:
            self.w_out = np.array(w_out)
        else:
            self.w_out = np.random.uniform(-1, 1, (self.output_size, self.hidden_size))

        self.fitness = 0.0 # To store fitness during evolution

    def __call__(self, x):
        """
        Forward pass through the network.
        x: Input vector (numpy array of shape (input_size,))
        Returns: Output vector (numpy array of shape (output_size,))
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)
        
        if x.shape[0] != self.input_size:
            raise ValueError(f"Input vector size {x.shape[0]} does not match expected size {self.input_size}")

        # Hidden layer: h = tanh(W_in @ x)
        h = np.tanh(self.w_in @ x)
        
        # Output layer: y = tanh(W_out @ h)
        y = np.tanh(self.w_out @ h) # Outputs are in range (-1, 1)
        return y

    def mutate(self, sigma=0.2, mutation_rate_weights=1.0, rng=None):
        """
        Creates a new mutated TinyNet.
        sigma: Standard deviation for Gaussian noise.
        mutation_rate_weights: Probability that any given weight matrix (w_in, w_out) gets mutated.
        rng: Optional numpy.random.Generator instance for reproducible randomness.
        """
        if rng is None:
            rng = np.random.default_rng()

        w_in_mutated = self.w_in.copy()
        w_out_mutated = self.w_out.copy()

        # Mutate input weights
        if rng.random() < mutation_rate_weights: # Mutate the whole matrix with some probability
            noise_in = rng.normal(0, sigma, self.w_in.shape)
            w_in_mutated += noise_in
        
        # Mutate output weights
        if rng.random() < mutation_rate_weights: # Mutate the whole matrix
            noise_out = rng.normal(0, sigma, self.w_out.shape)
            w_out_mutated += noise_out
        
        # Optionally clip weights if they grow too large, though tanh helps manage activation scale
        # w_in_mutated = np.clip(w_in_mutated, -max_weight, max_weight)
        # w_out_mutated = np.clip(w_out_mutated, -max_weight, max_weight)

        return TinyNet(w_in_mutated, w_out_mutated, self.input_size, self.hidden_size, self.output_size)

    @classmethod
    def crossover(cls, parent1, parent2, rng=None):
        """
        Performs uniform crossover between two parent TinyNets.
        rng: Optional numpy.random.Generator instance for reproducible randomness.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Ensure parents have compatible shapes (could add assertions here)
        # For simplicity, assuming they are compatible.

        # Crossover for w_in
        mask_in = rng.random(parent1.w_in.shape) < 0.5
        w_in_child = np.where(mask_in, parent1.w_in, parent2.w_in)
        
        # Crossover for w_out
        mask_out = rng.random(parent1.w_out.shape) < 0.5
        w_out_child = np.where(mask_out, parent1.w_out, parent2.w_out)
        
        return cls(w_in_child, w_out_child, parent1.input_size, parent1.hidden_size, parent1.output_size)

    def get_genome_params(self):
        """Returns the weights as a tuple, suitable for saving."""
        return self.w_in, self.w_out

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # Test TinyNet
    input_vector = np.random.rand(14)
    net = TinyNet()
    output_vector = net(input_vector)
    print("Input:", input_vector)
    print("Output:", output_vector)
    print("w_in shape:", net.w_in.shape)   # Expected: (16, 14)
    print("w_out shape:", net.w_out.shape) # Expected: (4, 16)

    # Test mutation
    mutated_net = net.mutate(sigma=0.1)
    print("Original w_in[0,0]:", net.w_in[0,0])
    print("Mutated w_in[0,0]:", mutated_net.w_in[0,0])
    
    # Test crossover
    net2 = TinyNet()
    child_net = TinyNet.crossover(net, net2)
    print("Parent1 w_in[0,0]:", net.w_in[0,0])
    print("Parent2 w_in[0,0]:", net2.w_in[0,0])
    print("Child w_in[0,0]:", child_net.w_in[0,0]) # Should be one of the parent values