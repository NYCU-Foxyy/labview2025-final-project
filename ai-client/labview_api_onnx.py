"""
LabVIEW-Compatible General.io AI Agent API (ONNX Version)
=========================================================
This module provides a 32-bit compatible interface for LabVIEW using ONNX Runtime.

INSTALLATION (32-bit Python):
pip install onnxruntime numpy

CONVERSION REQUIRED (run once in 64-bit Python):
Use convert_to_onnx.py to convert PyTorch models to ONNX format.

LabVIEW Integration:
1. Use Python Node in LabVIEW
2. Call initialize_agent() once at startup
3. Call forward(owner_map, armies_map) each turn to get next action (0-4)
4. Call reset_agent() when starting a new game
"""

import numpy as np
import onnxruntime as ort
from typing import List, Union, Tuple
import os

# ---------------------------
# Global agent instance
# ---------------------------
_global_agent = None

# ---------------------------
# Helper functions for fog predictor
# ---------------------------
def ints_to_onehot(arr: np.ndarray, classes: list) -> np.ndarray:
    """
    Convert integer array to one-hot encoding
    
    Args:
        arr: Array of shape (B, T, H, W) with integer values
        classes: List of valid class values
    
    Returns:
        One-hot array of shape (B, T, C, H, W)
    """
    cls2idx = {c: i for i, c in enumerate(classes)}
    B, T, H, W = arr.shape
    C = len(classes)
    onehot = np.zeros((B, T, C, H, W), dtype=np.float32)
    
    for b in range(B):
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    val = arr[b, t, h, w]
                    if val in cls2idx:
                        idx = cls2idx[val]
                        onehot[b, t, idx, h, w] = 1.0
    return onehot


def constrain_predictions(pred_map: np.ndarray, partial_map: np.ndarray, fog_token: int = -2) -> np.ndarray:
    """
    Apply constraints to predicted map based on partial observations
    
    Args:
        pred_map: Predicted full map
        partial_map: Partial observation (with fog)
        fog_token: Value representing fog
    
    Returns:
        Constrained prediction map
    """
    result = pred_map.copy()
    # Keep all non-fog tiles from partial observation
    mask = partial_map != fog_token
    result[mask] = partial_map[mask]
    return result


# ---------------------------
# Agent wrapper for LabVIEW (ONNX)
# ---------------------------
class LabVIEWAgentONNX:
    """AI Agent wrapper using ONNX Runtime (32-bit compatible)"""
    
    def __init__(self, model_path: str = None, fog_predictor_path: str = None, map_size: int = 20):
        """
        Initialize the AI agent with ONNX models
        
        Args:
            model_path: Path to ONNX DQN model (.onnx file)
            fog_predictor_path: Path to ONNX fog predictor model
            map_size: Size of the game map (default 20x20)
        """
        self.map_size = map_size
        self.model_session = None
        self.fog_predictor_session = None
        
        # Load DQN model
        if model_path and os.path.exists(model_path):
            self.model_session = ort.InferenceSession(model_path)
            print(f"Loaded DQN model from {model_path}")
        else:
            print(f"Warning: DQN model not found at {model_path}")
            print("Agent will use random policy")
        
        # Load fog predictor
        self.fog_classes = [-2, -1, 0, 1, 2, 100, 101, 102, 201, 202]
        if fog_predictor_path and os.path.exists(fog_predictor_path):
            self.fog_predictor_session = ort.InferenceSession(fog_predictor_path)
            print(f"Loaded fog predictor from {fog_predictor_path}")
        else:
            print("Fog predictor not loaded - will use partial observations directly")
        
        self.turn_count = 0
        self.fog_token = -2
        self.seq_len = 3
        self.history = []  # Store recent observations for sequence input
    
    def predict(self, owner_map: np.ndarray, armies_map: np.ndarray, player_id: int = 1) -> Tuple[int, int, int]:
        """
        Predict next action given current game state
        
        Args:
            owner_map: 2D numpy array (H x W) with ownership values
            armies_map: 2D numpy array (H x W) with army counts
            player_id: Which player to generate action for
            
        Returns:
            Tuple (x, y, direction)
        """
        self.turn_count += 1
        
        # Convert to numpy arrays
        owner_map = np.array(owner_map, dtype=np.int32)
        armies_map = np.array(armies_map, dtype=np.float32)
        
        # Use fog predictor if available and fog is present
        if self.fog_predictor_session is not None and (owner_map == self.fog_token).any():
            owner_map = self._predict_fog(owner_map, player_id)
        
        # Normalize inputs
        owner_norm = owner_map.astype(np.float32)
        armies_norm = armies_map / 10.0
        
        # Stack channels: [owner, armies]
        stacked = np.stack([owner_norm, armies_norm], axis=0)  # (2, H, W)
        x = stacked[np.newaxis, ...]  # Add batch dim: (1, 2, H, W)
        
        # Run inference
        if self.model_session is None:
            # Random policy if no model
            action = np.random.randint(0, 5)
            owned_tiles = np.argwhere((owner_map == player_id) | 
                                     (owner_map == player_id + 100) | 
                                     (owner_map == player_id + 200))
            if len(owned_tiles) > 0:
                tile = owned_tiles[np.random.randint(len(owned_tiles))]
                return (int(tile[1]), int(tile[0]), int(action))
            else:
                return (0, 0, 0)
        
        # Get input name from ONNX model
        input_name = self.model_session.get_inputs()[0].name
        
        # Run inference
        q_values = self.model_session.run(None, {input_name: x})[0]  # (1, num_actions, H, W)
        q_values = q_values.squeeze(0)  # (num_actions, H, W)
        
        # Mask out tiles not owned by player
        owned_mask = (
            (owner_map == player_id) | 
            (owner_map == player_id + 100) |  # Cities
            (owner_map == player_id + 200)     # General
        )
        
        # Set Q-values to very negative for non-owned tiles
        for action in range(q_values.shape[0]):
            q_values[action][~owned_mask] = -1e9
        
        # Find best tile and action
        q_flat = q_values.reshape(-1)
        best_idx = int(q_flat.argmax())
        
        # Convert flat index back to (action, y, x)
        num_actions, H, W = q_values.shape
        action = best_idx // (H * W)
        remainder = best_idx % (H * W)
        y = remainder // W
        x = remainder % W
        
        return (int(x), int(y), int(action))
    
    def _predict_fog(self, partial_map: np.ndarray, player_id: int) -> np.ndarray:
        """Use fog predictor to fill in fogged tiles"""
        # Add current observation to history
        self.history.append(partial_map.copy())
        if len(self.history) > self.seq_len:
            self.history.pop(0)
        
        # Create sequence
        seq = self.history.copy()
        while len(seq) < self.seq_len:
            seq.insert(0, partial_map.copy())
        
        seq = seq[-self.seq_len:]
        seq_array = np.stack(seq, axis=0)[None]  # (1, T, H, W)
        
        # Convert to one-hot
        x = ints_to_onehot(seq_array, self.fog_classes)  # (1, T, C, H, W)
        
        # Get input name from ONNX model
        input_name = self.fog_predictor_session.get_inputs()[0].name
        
        # Run inference
        logits = self.fog_predictor_session.run(None, {input_name: x})[0]  # (1, C, H, W)
        pred_idx = logits.argmax(1)[0]  # (H, W)
        
        # Convert indices back to class values
        idx2cls = {i: c for i, c in enumerate(self.fog_classes)}
        pred_map = np.vectorize(lambda k: idx2cls[k])(pred_idx)
        
        # Apply constraints
        pred_map = constrain_predictions(pred_map, partial_map, fog_token=self.fog_token)
        
        return pred_map
    
    def reset(self):
        """Reset agent state"""
        self.turn_count = 0
        self.history = []


# ---------------------------
# LabVIEW-friendly global functions
# ---------------------------

def initialize_agent(model_path: str = "models/dqn_agent_20x20.onnx", 
                     fog_predictor_path: str = "models/fog_predictor_20x20.onnx",
                     map_size: int = 20) -> int:
    """
    Initialize the AI agent (call once at startup)
    
    Args:
        model_path: Path to ONNX DQN model file
        fog_predictor_path: Path to ONNX fog predictor model file
        map_size: Map dimensions (default 20x20)
    
    Returns:
        0 if successful, -1 if failed
    """
    global _global_agent
    try:
        _global_agent = LabVIEWAgentONNX(model_path, fog_predictor_path, map_size)
        print(f"Agent initialized successfully with ONNX Runtime")
        return 0
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return -1


def forward(owner_map: Union[List, np.ndarray], 
                  armies_map: Union[List, np.ndarray], 
                  player_id: int = 1) -> List[int]:
    """
    LabVIEW compatible version - returns array of 3 integers
    
    Args:
        owner_map: 2D array showing tile ownership
        armies_map: 2D array showing army counts
        player_id: Which player to generate action for
    
    Returns:
        List [x, y, direction]
    """
    global _global_agent
    
    if _global_agent is None:
        print("Agent not initialized, initializing with defaults...")
        initialize_agent()
    
    try:
        owner = np.array(owner_map, dtype=np.float32)
        armies = np.array(armies_map, dtype=np.float32)
        
        x, y, action = _global_agent.predict(owner, armies, player_id)
        
        # Return as list/array
        #return [1, 2, 3]
        return [int(x), int(y), int(action)]
    
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return [-1, -1, -1]


def reset_agent() -> int:
    """Reset agent state for new game"""
    global _global_agent
    if _global_agent:
        _global_agent.reset()
    return 0


def get_action_name(action: int) -> str:
    """Convert action number to human-readable name"""
    action_names = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
    return action_names.get(action, "UNKNOWN")


# ---------------------------
# Testing
# ---------------------------
if __name__ == "__main__":
    print("=== Testing LabVIEW AI Agent API (ONNX) ===\n")
    
    # Initialize agent
    result = initialize_agent("models/dqn_agent_20x20.onnx", 
                             "models/fog_predictor_20x20.onnx", 
                             map_size=20)
    print(f"Initialization result: {result}\n")
    
    # Create test map
    owner_test = np.zeros((20, 20), dtype=np.int32)
    armies_test = np.zeros((20, 20), dtype=np.int32)
    
    owner_test[0, 0] = 201
    armies_test[0, 0] = 5
    owner_test[1, 0] = 1
    armies_test[1, 0] = 3
    
    print("Test map created (20x20)")
    
    # Test forward function
    print("\nTesting forward() function:")
    for turn in range(5):
        ret = forward(owner_test, armies_test, player_id=1)
        x, y, direction = ret[0], ret[1], ret[2]
        action_name = get_action_name(direction)
        print(f"Turn {turn+1}: Tile=({x},{y}), Direction={direction} ({action_name})")
    
    print("\n=== ONNX Agent Ready for LabVIEW (32-bit compatible) ===")