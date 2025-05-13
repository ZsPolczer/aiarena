# evo_arena/storage/persist.py
import numpy as np
import os
import json # For match replays later

from agents.brain import TinyNet # To reconstruct TinyNet objects

def save_genome(genome_brain, filename_prefix="genome", directory="storage/genomes", generation=None, fitness=None):
    """
    Saves the genome (weights of a TinyNet) to a .npz file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    w_in, w_out = genome_brain.get_genome_params()
    
    # Construct filename
    if generation is not None:
        filename = f"{filename_prefix}_g{generation:05d}"
    else:
        filename = filename_prefix
    
    if fitness is not None:
        filename += f"_fit{fitness:.3f}"
    filename += ".npz"
    
    filepath = os.path.join(directory, filename)
    
    save_data = {'w_in': w_in, 'w_out': w_out}
    if hasattr(genome_brain, 'fitness'):
      save_data['fitness'] = genome_brain.fitness
    elif fitness is not None:
      save_data['fitness'] = fitness


    np.savez(filepath, **save_data)
    print(f"Saved genome to {filepath}")
    return filepath

def load_genome(filepath, input_size=14, hidden_size=16, output_size=4):
    """
    Loads a genome from a .npz file and reconstructs a TinyNet.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Genome file not found: {filepath}")
        
    data = np.load(filepath)
    w_in = data['w_in']
    w_out = data['w_out']
    
    loaded_brain = TinyNet(w_in, w_out, input_size, hidden_size, output_size)
    if 'fitness' in data:
        loaded_brain.fitness = float(data['fitness'])
        
    print(f"Loaded genome from {filepath}")
    return loaded_brain

# --- Match Replay Functions (for later, as per spec section 6) ---
REPLAY_DIR = "storage/replays"

def start_match_replay(filename_prefix="match_replay"):
    if not os.path.exists(REPLAY_DIR):
        os.makedirs(REPLAY_DIR)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{filename_prefix}_{timestamp}.jsonl"
    filepath = os.path.join(REPLAY_DIR, filename)
    return open(filepath, 'w'), filepath # Return file handle and path

def record_arena_snapshot(replay_file_handle, game_time, agents_data):
    """
    Records a snapshot of the arena state to the replay file.
    agents_data: A list of dicts, each representing an agent's state.
    """
    snapshot = {
        't': round(game_time, 3),
        'state': agents_data 
    }
    replay_file_handle.write(json.dumps(snapshot) + '\n')

def close_match_replay(replay_file_handle):
    if replay_file_handle:
        replay_file_handle.close()

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # Test saving and loading
    test_brain = TinyNet()
    test_brain.fitness = 123.456
    
    # Create dummy directories if they don't exist
    if not os.path.exists("storage/genomes"):
        os.makedirs("storage/genomes")

    filepath = save_genome(test_brain, filename_prefix="test_dummy", generation=0, fitness=test_brain.fitness)
    loaded_brain = load_genome(filepath)
    
    assert np.array_equal(test_brain.w_in, loaded_brain.w_in)
    assert np.array_equal(test_brain.w_out, loaded_brain.w_out)
    assert hasattr(loaded_brain, 'fitness') and loaded_brain.fitness == test_brain.fitness
    print("Save/Load test successful.")

    # Test replay (rudimentary)
    # file_handle, replay_path = start_match_replay()
    # print(f"Replay file started: {replay_path}")
    # record_arena_snapshot(file_handle, 0.02, [{'id': 'a1', 'x': 10, 'y': 20, 'hp': 100}])
    # record_arena_snapshot(file_handle, 0.04, [{'id': 'a1', 'x': 12, 'y': 22, 'hp': 90}])
    # close_match_replay(file_handle)
    # print("Replay test successful.")