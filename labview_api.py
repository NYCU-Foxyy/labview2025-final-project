"""
LabVIEW-Compatible General.io AI Agent API
===========================================
This module provides a simple interface for LabVIEW to control an AI agent.

LabVIEW Integration:
1. Use Python Node in LabVIEW
2. Call initialize_agent() once at startup
3. Call forward(owner_map, armies_map) each turn to get next action (0-4)
4. Call reset_agent() when starting a new game

Map encoding:
- owner_map: 2D array with values:
  0: Plains, -1: Mountains, -2: Fog,
  1-N: player ownership,
  100: Uncaptured city, 100+player: Captured cities,
  200+player: Generals
- armies_map: 2D array with army counts per tile

Output format:
- Returns tuple (x, y, direction)
- x, y: coordinates of tile to move from
- direction: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
from typing import List, Union

# Import fog predictor utilities from models.py
from models import FogPredictor, ints_to_onehot, mask_fog_realistic, constrain_predictions

# ---------------------------
# Global agent instance
# ---------------------------
_global_agent = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Neural Network (same as models.py)
# ---------------------------
class TinyQNet(nn.Module):
    def __init__(self, H, W, num_actions=5):
        super().__init__()
        self.H = H
        self.W = W
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, num_actions, 1),  # Output: (B, num_actions, H, W)
        )

    def forward(self, x):
        # x shape (B,2,H,W)
        # output shape (B, num_actions, H, W) - Q-values for each tile and action
        return self.conv(x)


# ---------------------------
# Agent wrapper for LabVIEW
# ---------------------------
class LabVIEWAgent:
    """AI Agent wrapper optimized for LabVIEW calls"""
    
    def __init__(self, model_path: str = None, fog_predictor_path: str = None, map_size: int = 20):
        """
        Initialize the AI agent
        
        Args:
            model_path: Path to saved DQN model weights (.pth file). If None, uses random policy.
            fog_predictor_path: Path to saved fog predictor model. If None, won't predict fog.
            map_size: Size of the game map (default 20x20)
        """
        self.map_size = map_size
        self.device = _device
        self.model = TinyQNet(map_size, map_size).to(self.device)
        
        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded DQN model from {model_path}")
        else:
            print("Using untrained DQN model (random initialization)")
        
        self.model.eval()  # Set to evaluation mode
        
        # Initialize fog predictor
        self.fog_predictor = None
        self.fog_classes = [-2, -1, 0, 1, 2, 100, 101, 102, 201, 202]
        if fog_predictor_path and os.path.exists(fog_predictor_path):
            self.fog_predictor = FogPredictor(map_size, map_size, self.fog_classes).to(self.device)
            self.fog_predictor.load_state_dict(torch.load(fog_predictor_path, map_location=self.device))
            self.fog_predictor.eval()
            print(f"Loaded fog predictor from {fog_predictor_path}")
        else:
            print("Fog predictor not loaded - will use partial observations directly")
        
        self.turn_count = 0
        self.fog_token = -2
        self.seq_len = 3
        self.history = []  # Store recent observations for sequence input
    
    def predict(self, owner_map: np.ndarray, armies_map: np.ndarray, player_id: int = 1):
        """
        Predict next action given current game state
        
        Args:
            owner_map: 2D numpy array (H x W) with ownership values (may contain fog=-2)
            armies_map: 2D numpy array (H x W) with army counts
            player_id: Which player (1 or 2) to generate action for
            
        Returns:
            Tuple (x, y, direction):
            - x, y: coordinates of the tile to move from
            - direction: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        """
        self.turn_count += 1
        
        # Convert to numpy arrays if needed
        owner_map = np.array(owner_map, dtype=np.int32)
        armies_map = np.array(armies_map, dtype=np.float32)
        
        # Use fog predictor if available and fog is present
        if self.fog_predictor is not None and (owner_map == self.fog_token).any():
            owner_map = self._predict_fog(owner_map, player_id)
        
        # Normalize inputs
        owner_norm = owner_map.astype(np.float32)
        armies_norm = armies_map / 10.0  # Simple normalization
        
        # Stack channels: [owner, armies]
        stacked = np.stack([owner_norm, armies_norm], axis=0)  # Shape: (2, H, W)
        
        # Convert to tensor
        x = torch.tensor(stacked, dtype=torch.float32, device=self.device).unsqueeze(0)  # Add batch dim
        
        # Forward pass
        with torch.no_grad():
            q_values = self.model(x)  # Shape: (1, num_actions, H, W)
            q_values = q_values.squeeze(0)  # Shape: (num_actions, H, W)
            
            # Mask out tiles not owned by player
            # Only consider tiles owned by the player (including cities and generals)
            owned_mask = (
                (owner_map == player_id) | 
                (owner_map == player_id + 100) |  # Cities
                (owner_map == player_id + 200)     # General
            )
            
            # Set Q-values to very negative for non-owned tiles
            for action in range(q_values.shape[0]):
                q_values[action][~owned_mask] = -1e9
            
            # Find best tile and action
            # Flatten and find argmax
            q_flat = q_values.reshape(-1)
            best_idx = int(q_flat.argmax().item())
            
            # Convert flat index back to (action, y, x)
            num_actions, H, W = q_values.shape
            action = best_idx // (H * W)
            remainder = best_idx % (H * W)
            y = remainder // W
            x = remainder % W
        
        return (int(x), int(y), int(action))
    
    def _predict_fog(self, partial_map: np.ndarray, player_id: int) -> np.ndarray:
        """
        Use fog predictor to fill in fogged tiles
        
        Args:
            partial_map: Map with fog tokens (-2) for hidden areas
            player_id: Current player ID
            
        Returns:
            Predicted full map with fog tiles filled in
        """
        # Add current observation to history
        self.history.append(partial_map.copy())
        if len(self.history) > self.seq_len:
            self.history.pop(0)
        
        # Create sequence (repeat if not enough history)
        seq = self.history.copy()
        while len(seq) < self.seq_len:
            seq.insert(0, partial_map.copy())
        
        # Take last seq_len observations
        seq = seq[-self.seq_len:]
        seq_array = np.stack(seq, axis=0)[None]  # (1, T, H, W)
        
        # Convert to one-hot
        x = ints_to_onehot(seq_array, self.fog_classes)  # (1, T, C, H, W)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.fog_predictor(x_t)  # (1, C, H, W)
            pred_idx = logits.argmax(1).cpu().numpy()[0]  # (H, W)
        
        # Convert indices back to class values
        idx2cls = {i: c for i, c in enumerate(self.fog_classes)}
        pred_map = np.vectorize(lambda k: idx2cls[k])(pred_idx)
        
        # Apply constraints to respect game rules
        pred_map = constrain_predictions(pred_map, partial_map, fog_token=self.fog_token)
        
        return pred_map
    
    def reset(self):
        """Reset agent state (call when starting new game)"""
        self.turn_count = 0
        self.history = []


# ---------------------------
# LabVIEW-friendly global functions
# ---------------------------

def initialize_agent(model_path: str = "models/dqn_agent_20x20.pth", 
                     fog_predictor_path: str = "models/fog_predictor_20x20.pth",
                     map_size: int = 20):
    """
    Initialize the AI agent (call once at startup)
    
    Args:
        model_path: Path to trained DQN model file
        fog_predictor_path: Path to trained fog predictor model file
        map_size: Map dimensions (default 20x20)
    
    Returns:
        0 if successful, -1 if failed
    """
    global _global_agent
    try:
        _global_agent = LabVIEWAgent(model_path, fog_predictor_path, map_size)
        print(f"Agent initialized successfully on device: {_device}")
        return 0
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return -1


def forward(owner_map: Union[List, np.ndarray], armies_map: Union[List, np.ndarray], player_id: int = 1):
    """
    Get next action from AI agent (main function called by LabVIEW each turn)
    
    Args:
        owner_map: 2D array (list or numpy) showing tile ownership
        armies_map: 2D array (list or numpy) showing army counts
        player_id: Which player to generate action for (default: 1)
    
    Returns:
        Tuple (x, y, direction) where:
        - x, y: coordinates of the tile to move from
        - direction: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        Returns (-1, -1, -1) on error
    
    Example:
        >>> x, y, direction = forward(owner_map, armies_map, player_id=1)
        >>> # Move from tile (x, y) in the given direction
        >>> # e.g., (2, 2, 1) means move from (2,2) upward
    """
    global _global_agent
    
    # Auto-initialize if not done
    if _global_agent is None:
        print("Agent not initialized, initializing with defaults...")
        initialize_agent()
    
    try:
        # Convert lists to numpy arrays
        owner = np.array(owner_map, dtype=np.float32)
        armies = np.array(armies_map, dtype=np.float32)
        
        # Get prediction
        x, y, action = _global_agent.predict(owner, armies, player_id)
        return (x, y, action)
    
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return (-1, -1, -1)


def reset_agent():
    """
    Reset agent state for new game (call when starting new episode)
    
    Returns:
        0 if successful
    """
    global _global_agent
    if _global_agent:
        _global_agent.reset()
    return 0


def get_action_name(action: int) -> str:
    """
    Convert action number to human-readable name
    
    Args:
        action: Integer 0-4
    
    Returns:
        Action name string
    """
    action_names = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
    return action_names.get(action, "UNKNOWN")


# ---------------------------
# Testing / Example usage
# ---------------------------
if __name__ == "__main__":
    print("=== Testing LabVIEW AI Agent API ===\n")
    
    # Initialize agent
    result = initialize_agent("models/dqn_agent_20x20.pth", "models/fog_predictor_20x20.pth", map_size=20)
    print(f"Initialization result: {result}\n")
    
    # Create test map (20x20)
    owner_test = np.zeros((20, 20), dtype=np.int32)
    armies_test = np.zeros((20, 20), dtype=np.int32)
    
    # Set up initial positions
    owner_test[0, 0] = 201  # Player 1 general
    armies_test[0, 0] = 5
    owner_test[1, 0] = 1    # Player 1 territory
    armies_test[1, 0] = 3
    owner_test[2, 0] = 1
    armies_test[2, 0] = 2
    
    # Add some fog (player 2 area hidden)
    owner_test[18:20, 18:20] = -2  # Fog in bottom-right
    
    # Mountains
    owner_test[10, 10] = -1
    owner_test[10, 11] = -1
    
    print("Test map created (20x20)")
    print(f"Player 1 general at (0,0), territory at (1,0) and (2,0)")
    print(f"Fog at (18-19, 18-19)\n")
    
    # Test forward function multiple times
    print("Testing forward() function:")
    for turn in range(5):
        x, y, direction = forward(owner_test, armies_test, player_id=1)
        action_name = get_action_name(direction)
        print(f"Turn {turn+1}: Tile=({x},{y}), Direction={direction} ({action_name})")
    
    print("\n=== API Ready for LabVIEW Integration ===")
    print("\nLabVIEW Usage:")
    print("1. Call: initialize_agent('models/dqn_agent_20x20.pth', 'models/fog_predictor_20x20.pth', 20)")
    print("2. Each turn: x, y, direction = forward(owner_map, armies_map, player_id)")
    print("3. New game: reset_agent()")
    print("\nOutput: (x, y, direction) - move from tile (x,y) in given direction")
    print("Direction mapping: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT")
    print("\nNote: Fog predictor automatically fills in fogged tiles (value=-2) before decision making")
