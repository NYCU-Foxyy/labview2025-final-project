# dqn_selfplay_full.py
"""
Integrated DQN + Snapshot Self-Play + Terrain/Bases/Multi-unit + Fog Predictor Demo
- Trains DQN on full observations
- Snapshot self-play opponents every 20 episodes; sample opponents 80% from pool / 20% current
- Map: owner_grid (ints) + armies array (ints)
- Bases encoded as owner + 10 (11,12,...)
- Reward: general.io style (army change + base capture bonus)
- Minimal TensorBoard logging (reward, epsilon, loss)
- After DQN train: collect maps => train fog predictor (Conv->LSTM->Conv) => evaluate under fog
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random, copy, time
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ---------------------------
# Config
# ---------------------------
MAP_SIZE = 20
NUM_PLAYERS = 2
EPISODES = 400
SNAPSHOT_INTERVAL = 10
OPP_SAMPLE_FROM_SNAPSHOTS_PCT = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Map encoding (State Matrix - Owner/Terrain):
# 0  : Plains (uncaptured/neutral terrain)
# -1 : Mountains (impassable)
# -2 : Fog (hidden tiles, only for predictor inference)
# 1..N : player-owned plains
# 100 : Uncaptured city
# 100+player : Captured city (101=player1 city, 102=player2 city, etc.)
# 200+player : General/Castle (201=player1 general, 202=player2 general, etc.)
# Army Matrix: separate array with troop counts for each cell

GENERAL_OFFSET = 200  # General (main base) offset
CITY_OFFSET = 100     # City offset
UNCAPTURED_CITY = 100 # Uncaptured city value

# ---------------------------
# Helper: directions
# ---------------------------
DIRS = {
    0: (0,0),   # stay
    1: (-1,0),  # up
    2: (1,0),   # down
    3: (0,-1),  # left
    4: (0,1),   # right
}
NUM_ACTIONS = len(DIRS)

# ---------------------------
# Environment
# ---------------------------
class GeneralIOEnv:
    def __init__(self, map_size=MAP_SIZE, max_steps=200, num_players=NUM_PLAYERS):
        self.map_size = map_size
        self.max_steps = max_steps
        self.num_players = num_players
        self.reset()

    def reset(self):
        H = self.map_size
        W = self.map_size
        # owner grid (ints)
        self.owner = np.zeros((H,W), dtype=np.int32)  # -1 mountains, 0 plains, 1.. players, 100 cities, 200+ generals
        self.armies = np.zeros((H,W), dtype=np.int32) # army counts
        self.turn = 0

        # place mountains randomly (sparse)
        num_mnts = (H*W)//20
        for _ in range(num_mnts):
            y,x = np.random.randint(0,H), np.random.randint(0,W)
            self.owner[y,x] = -1

        # place castles (main bases) randomly with minimum distance constraint
        min_distance = max(H, W) * 0.4  # at least 40% of map dimension apart
        castle_positions = []
        
        for p in range(1, self.num_players+1):
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                y, x = np.random.randint(0, H), np.random.randint(0, W)
                
                # check if position is valid (not mountain)
                if self.owner[y,x] == -1:
                    continue
                
                # check minimum distance from other castles
                valid_position = True
                for (py, px) in castle_positions:
                    dist = np.sqrt((y - py)**2 + (x - px)**2)
                    if dist < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    self.owner[y,x] = p + GENERAL_OFFSET  # General (main base)
                    self.armies[y,x] = 5  # starting garrison
                    castle_positions.append((y, x))
                    placed = True
                    break
            
            # fallback: if couldn't place randomly, use corners
            if not placed:
                corners = [(0,0),(0,W-1),(H-1,0),(H-1,W-1)]
                y, x = corners[p-1]
                if self.owner[y,x] == -1:
                    self.owner[y,x] = 0
                self.owner[y,x] = p + GENERAL_OFFSET
                self.armies[y,x] = 5
                castle_positions.append((y, x))

        # place neutral cities (uncaptured)
        num_cities = max(2, (H * W) // 80)  # roughly 5 cities on a 20x20 map
        for _ in range(num_cities):
            max_attempts = 50
            for attempt in range(max_attempts):
                y, x = np.random.randint(0, H), np.random.randint(0, W)
                # place on empty plains tiles only
                if self.owner[y, x] == 0:
                    self.owner[y, x] = 100  # neutral city
                    self.armies[y, x] = np.random.randint(30, 50)  # cities have strong garrison
                    break

        # scatter some neutral troops on neutral tiles
        for _ in range((H*W)//15):
            y,x = np.random.randint(0,H), np.random.randint(0,W)
            if self.owner[y,x] == 0:
                self.armies[y,x] = np.random.randint(0,3)

        # initial player placement: give each player a nearby tile
        # also place small army near castle
        for p in range(1, self.num_players+1):
            by, bx = castle_positions[p-1]
            # try adjacent tiles
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny = np.clip(by + dy, 0, H-1)
                nx = np.clip(bx + dx, 0, W-1)
                if self.owner[ny,nx] == 0:
                    self.owner[ny,nx] = p
                    self.armies[ny,nx] = 3
                    break

        self.last_my_army_total = None
        self.last_owned_bases = None
        return self._get_obs(player_id=1)

    def _get_obs(self, player_id=1):
        # return full observation (owner grid and armies)
        # Here we encode observation as two-channel float arrays: owner_map, normalized_armies
        owner_copy = self.owner.copy().astype(np.int32)
        armies_copy = self.armies.copy().astype(np.float32)
        # for network input we'll convert owner map to ints but DQN will expect a single 2D int map in example pipeline
        return {"owner": owner_copy, "armies": armies_copy, "player": player_id}

    def step(self, player_action, player_id=1, opponent_policy=None):
        """
        player_action: tuple (tile_y, tile_x, direction) for per-tile action
                      OR integer 0..4 for backward compatibility with global actions
        opponent_policy: callable(obs)->action for the opponent
        Returns: next_obs, reward, done, info
        """
        self.turn += 1
        H,W = self.map_size, self.map_size

        # store prev totals for reward
        prev_total_army = self.armies[self.owner == player_id].sum()
        prev_bases = (self.owner == (player_id + GENERAL_OFFSET)).sum()

        # Apply player's movement
        if isinstance(player_action, tuple) and len(player_action) == 3:
            # Per-tile action: (y, x, direction)
            self._apply_tile_move(player_id, player_action)
        else:
            # Global action for backward compatibility
            self._apply_global_move(player_id, player_action)

        # Opponent(s) moves: for demo we handle single opponent player 2
        if self.num_players >= 2:
            if opponent_policy is None:
                # Random per-tile action
                owned_mask = (
                    (self.owner == 2) | 
                    (self.owner == 2 + CITY_OFFSET) | 
                    (self.owner == 2 + GENERAL_OFFSET)
                )
                owned_positions = np.argwhere(owned_mask)
                if len(owned_positions) > 0:
                    idx = np.random.randint(0, len(owned_positions))
                    tile_y, tile_x = owned_positions[idx]
                    direction = random.randint(0, NUM_ACTIONS-1)
                    opp_action = (int(tile_y), int(tile_x), direction)
                else:
                    opp_action = (0, 0, 0)
            else:
                obs_for_opp = self._get_obs(player_id=2)
                opp_action = opponent_policy(obs_for_opp)
            
            # Apply opponent move (handle both tuple and int formats)
            if isinstance(opp_action, tuple) and len(opp_action) == 3:
                self._apply_tile_move(2, opp_action)
            else:
                self._apply_global_move(2, opp_action)

        # produce at generals and cities: each general/city adds 1 army
        general_positions = np.argwhere(self.owner >= GENERAL_OFFSET)
        city_positions = np.argwhere((self.owner >= CITY_OFFSET) & (self.owner < GENERAL_OFFSET))
        for (y,x) in general_positions:
            self.armies[y,x] += 1
        for (y,x) in city_positions:
            self.armies[y,x] += 1

        # clamp armies
        self.armies = np.clip(self.armies, 0, 1000)

        # compute reward for player 1
        my_total_army = self.armies[self.owner == player_id].sum()
        delta_army = int(my_total_army - prev_total_army)

        my_general = (self.owner == (player_id + GENERAL_OFFSET)).sum()
        my_cities = ((self.owner >= (player_id + CITY_OFFSET)) & (self.owner < (player_id + GENERAL_OFFSET))).sum()
        total_bases = my_general + my_cities
        base_gain = int(total_bases - prev_bases)
        reward = float(delta_army) + 5.0 * float(base_gain)

        # done conditions
        done = False
        if self.turn >= self.max_steps:
            done = True
        # if captured opponent's general -> win
        opponent_general = (self.owner == (2 + GENERAL_OFFSET)).sum()
        if opponent_general == 0:
            done = True
            reward += 10.0

        info = {}
        return self._get_obs(player_id=player_id), reward, done, info

    def _apply_tile_move(self, player_id, action_tuple):
        """
        Move armies from a specific tile in a specific direction.
        action_tuple: (tile_y, tile_x, direction)
        """
        tile_y, tile_x, direction = action_tuple
        H, W = self.map_size, self.map_size
        dy, dx = DIRS[direction]
        move_frac = 0.5  # move 50% of armies from the selected tile
        
        # Check if tile is owned by player
        is_owned = (
            (self.owner[tile_y, tile_x] == player_id) or
            (self.owner[tile_y, tile_x] == player_id + CITY_OFFSET) or
            (self.owner[tile_y, tile_x] == player_id + GENERAL_OFFSET)
        )
        
        if not is_owned:
            return  # Can't move from unowned tile
        
        # Calculate movement
        avail = int(self.armies[tile_y, tile_x])
        moving = max(1, int(avail * move_frac))  # Move at least 1 if available
        if moving <= 0 or avail < 2:  # Need at least 2 armies to move (leave 1 behind)
            return
        
        # Calculate destination
        ny = np.clip(tile_y + dy, 0, H-1)
        nx = np.clip(tile_x + dx, 0, W-1)
        
        # If trying to move into mountain or out of bounds, do nothing
        if self.owner[ny, nx] == -1:
            return
        
        # Reduce armies at source
        self.armies[tile_y, tile_x] = max(0, self.armies[tile_y, tile_x] - moving)
        
        # Check if destination is owned by player
        dest_is_mine = (
            (self.owner[ny, nx] == player_id) or
            (self.owner[ny, nx] == player_id + CITY_OFFSET) or
            (self.owner[ny, nx] == player_id + GENERAL_OFFSET)
        )
        
        if self.owner[ny, nx] == 0 or dest_is_mine:
            # Empty or friendly: just add armies
            self.armies[ny, nx] += moving
            if self.owner[ny, nx] == 0:
                self.owner[ny, nx] = player_id
        else:
            # Combat with enemy/neutral
            defender_owner = self.owner[ny, nx]
            defender_army = int(self.armies[ny, nx])
            
            if moving > defender_army:
                # Capture
                remaining = moving - defender_army
                if defender_owner >= GENERAL_OFFSET:
                    self.owner[ny, nx] = player_id + GENERAL_OFFSET
                elif defender_owner >= CITY_OFFSET or defender_owner == 100:
                    self.owner[ny, nx] = player_id + CITY_OFFSET
                else:
                    self.owner[ny, nx] = player_id
                self.armies[ny, nx] = remaining
            else:
                # Defender holds
                self.armies[ny, nx] = defender_army - moving

    def _apply_global_move(self, player_id, action):
        """
        For simplicity, the agent chooses ONE global direction and a fixed fraction of troops
        from ALL owned tiles move to that direction each step.
        This is a simplification of per-tile orders but preserves multi-unit dynamics.
        """
        dy,dx = DIRS[action]
        move_frac = 0.25  # fraction to move
        H,W = self.map_size, self.map_size

        # we'll accumulate incoming troops into a temp grid
        incoming = np.zeros_like(self.armies, dtype=np.int32)

        # iterate all tiles owned by player (including generals, cities, and regular owned tiles)
        owned_mask = ( (self.owner == player_id) | 
                      (self.owner == player_id + CITY_OFFSET) | 
                      (self.owner == player_id + GENERAL_OFFSET) )
        ys, xs = np.where(owned_mask)
        for y,x in zip(ys,xs):
            # compute how many move
            avail = int(self.armies[y,x])
            moving = int(avail * move_frac)
            if moving <= 0:
                continue
            # reduce from origin
            self.armies[y,x] = max(0, self.armies[y,x] - moving)
            ny = np.clip(y + dy, 0, H-1)
            nx = np.clip(x + dx, 0, W-1)
            # if mountain, troops can't move; return to origin
            if self.owner[ny,nx] == -1:
                self.armies[y,x] += moving
                continue
            # if empty or same owner -> add to incoming
            is_mine = (self.owner[ny,nx] == player_id) or \
                      (self.owner[ny,nx] == player_id + CITY_OFFSET) or \
                      (self.owner[ny,nx] == player_id + GENERAL_OFFSET)
            if (self.owner[ny,nx] == 0) or is_mine:
                incoming[ny,nx] += moving
            else:
                # combat with enemy/neutral troops: resolve immediately
                defender_owner = self.owner[ny,nx]
                defender_army = int(self.armies[ny,nx])
                if moving > defender_army:
                    # capture tile: remaining armies occupy, owner switches
                    remaining = moving - defender_army
                    # if defender was a general, convert to your general
                    if defender_owner >= GENERAL_OFFSET:
                        # capturing enemy general -> become your general
                        self.owner[ny,nx] = player_id + GENERAL_OFFSET
                    elif defender_owner >= CITY_OFFSET or defender_owner == 100:
                        # capturing city (enemy, neutral, or uncaptured) -> become your city
                        self.owner[ny,nx] = player_id + CITY_OFFSET
                    else:
                        self.owner[ny,nx] = player_id
                    self.armies[ny,nx] = remaining
                else:
                    # defender survives, reduce defender armies
                    self.armies[ny,nx] = defender_army - moving

        # apply incoming reinforcement
        self.armies += incoming
        # if incoming captured neutral tiles: assign ownership
        # for tiles that were neutral (owner==0) with incoming >0, set owner to player_id
        neutral_mask = (self.owner == 0) & (incoming > 0)
        self.owner[neutral_mask] = player_id
        # clamp
        self.armies = np.clip(self.armies, 0, 1000)

    def render_text(self):
        # simple debug print of owners and armies
        print("Owners:")
        print(self.owner)
        print("Armies:")
        print(self.armies)

# ---------------------------
# Tiny CNN Q network (operates on combined map)
# ---------------------------
class TinyQNet(nn.Module):
    def __init__(self, H, W, num_actions=NUM_ACTIONS):
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
# Replay Buffer & Agent
# ---------------------------
import collections
Transition = collections.namedtuple('Transition', ['s','a','r','s2','done'])

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, H, W, lr=1e-3, gamma=0.99):
        self.device = DEVICE
        self.qnet = TinyQNet(H, W).to(self.device)
        self.target = TinyQNet(H, W).to(self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.opt = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = gamma
        self.step_count = 0
        self.target_update_steps = 400
        # epsilon schedule
        self.eps_start, self.eps_end, self.eps_decay = 1.0, 0.05, 5000

    def act(self, obs_dict, eval_mode=False):
        # build input: 2 channels -> owner (scaled), armies (scaled)
        inp = self._obs_to_tensor(obs_dict)
        player_id = obs_dict.get("player", 1)
        owner = obs_dict["owner"]
        
        if eval_mode or random.random() > self.epsilon():
            with torch.no_grad():
                q = self.qnet(inp)  # Shape: (1, num_actions, H, W)
                q = q.squeeze(0)    # Shape: (num_actions, H, W)
                
                # Mask out non-owned tiles
                owned_mask = (
                    (owner == player_id) | 
                    (owner == player_id + CITY_OFFSET) | 
                    (owner == player_id + GENERAL_OFFSET)
                )
                
                # Set Q-values to very negative for non-owned tiles
                for action in range(q.shape[0]):
                    q[action][~owned_mask] = -1e9
                
                # Find best tile and action
                q_flat = q.reshape(-1)
                best_idx = int(q_flat.argmax().item())
                
                # Convert flat index back to (action, y, x)
                num_actions, H, W = q.shape
                action = best_idx // (H * W)
                tile_idx = best_idx % (H * W)
                tile_y = tile_idx // W
                tile_x = tile_idx % W
                
                return (int(tile_y), int(tile_x), int(action))
        else:
            # Random action from a random owned tile
            H, W = owner.shape
            owned_mask = (
                (owner == player_id) | 
                (owner == player_id + CITY_OFFSET) | 
                (owner == player_id + GENERAL_OFFSET)
            )
            owned_positions = np.argwhere(owned_mask)
            if len(owned_positions) > 0:
                idx = random.randint(0, len(owned_positions) - 1)
                tile_y, tile_x = owned_positions[idx]
                action = random.randint(0, NUM_ACTIONS - 1)
                return (int(tile_y), int(tile_x), int(action))
            else:
                # Fallback if no owned tiles
                return (0, 0, 0)

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * max(0, (self.eps_decay - self.step_count))/self.eps_decay

    def _obs_to_tensor(self, obs_dict):
        owner = obs_dict["owner"].astype(np.float32)  # shape H,W
        armies = obs_dict["armies"].astype(np.float32)
        # normalize owners / armies for net: map owner values to limited range
        # convert owner to -2..(base) range scaled
        owner_norm = owner.copy()
        # simple normalization: divide armies by 10
        armies_norm = armies / 10.0
        stacked = np.stack([owner_norm, armies_norm], axis=0)
        t = torch.tensor(stacked, dtype=torch.float32, device=self.device).unsqueeze(0)
        return t

    def store(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        self.step_count += 1
        batch = self.replay.sample(self.batch_size)
        s_batch = torch.cat([self._obs_to_tensor(s) for s in batch.s], dim=0)
        s2_batch = torch.cat([self._obs_to_tensor(s2) for s2 in batch.s2], dim=0)
        
        # Convert tuple actions (y, x, direction) to flat indices
        flat_actions = []
        for action in batch.a:
            if isinstance(action, tuple) and len(action) == 3:
                y, x, direction = action
                # Flat index: direction * (H*W) + y * W + x
                H, W = s_batch.shape[2], s_batch.shape[3]
                flat_idx = direction * (H * W) + y * W + x
                flat_actions.append(flat_idx)
            else:
                # Old integer action format - shouldn't happen but handle it
                flat_actions.append(action)
        
        a_batch = torch.tensor(flat_actions, dtype=torch.int64, device=self.device)
        r_batch = torch.tensor(batch.r, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        q_vals = self.qnet(s_batch)  # Shape: (B, num_actions, H, W)
        B, num_actions, H, W = q_vals.shape
        
        # Flatten Q-values to (B, num_actions * H * W) for indexing
        q_vals_flat = q_vals.view(B, -1)  # (B, num_actions * H * W)
        q_a = q_vals_flat.gather(1, a_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target(s2_batch)  # (B, num_actions, H, W)
            q_next_flat = q_next.view(B, -1)  # (B, num_actions * H * W)
            q_next_max = q_next_flat.max(dim=1)[0]
            target = r_batch + self.gamma * q_next_max * (1.0 - done_batch)

        loss = F.mse_loss(q_a, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.step_count % self.target_update_steps == 0:
            self.target.load_state_dict(self.qnet.state_dict())
        return float(loss.item())

    def snapshot(self):
        """Return a copy of current policy weights as an opponent wrapper"""
        state = copy.deepcopy(self.qnet.state_dict())
        class Opponent:
            def __init__(self, state, H, W):
                self.state = state
                self.device = "cpu"
                self.net = TinyQNet(H,W).to(self.device)
                self.net.load_state_dict(state)
                self.net.eval()
            def __call__(self, obs):
                # obs is dict; convert to same normalized tensor on cpu
                owner = obs["owner"].astype(np.float32)
                armies = obs["armies"].astype(np.float32)
                armies = armies/10.0
                stacked = np.stack([owner, armies], axis=0)
                t = torch.tensor(stacked, dtype=torch.float32, device="cpu").unsqueeze(0)
                
                player_id = obs.get("player", 1)
                with torch.no_grad():
                    q = self.net(t)  # (1, num_actions, H, W)
                    q = q.squeeze(0)  # (num_actions, H, W)
                    
                    # Mask non-owned tiles
                    owned_mask = (
                        (owner == player_id) | 
                        (owner == player_id + 100) | 
                        (owner == player_id + 200)
                    )
                    for action in range(q.shape[0]):
                        q[action][~owned_mask] = -1e9
                    
                    # Get best action
                    q_flat = q.reshape(-1)
                    best_idx = int(q_flat.argmax().item())
                    num_actions, H, W = q.shape
                    action = best_idx // (H * W)
                    return int(action)
        return Opponent(state, MAP_SIZE, MAP_SIZE)

# ---------------------------
# Fog Predictor Model (Conv->LSTM->Conv)
# ---------------------------
class FogPredictor(nn.Module):
    def __init__(self, H, W, classes: List[int], hidden=128):
        super().__init__()
        self.H = H; self.W = W; self.classes = classes; self.C = len(classes)
        self.enc = nn.Sequential(
            nn.Conv2d(self.C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.flat = 64 * H * W
        self.reduce = nn.Linear(self.flat, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.expand = nn.Linear(hidden, self.flat)
        self.dec = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, self.C, 1)
        )
    def forward(self, x): # x (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        enc = self.enc(x)  # (B*T,64,H,W)
        enc_flat = enc.view(B, T, -1)  # (B,T,flat)
        red = self.reduce(enc_flat)
        out, _ = self.lstm(red)
        last = out[:, -1, :]
        exp = self.expand(last).view(B, 64, H, W)
        logits = self.dec(exp)
        return logits

# ---------------------------
# Utilities for predictor training
# ---------------------------
def ints_to_onehot(batch: np.ndarray, classes: List[int]):
    # batch shape (B,T,H,W) ints -> returns (B,T,C,H,W)
    B,T,H,W = batch.shape
    C = len(classes)
    mapping = {v:i for i,v in enumerate(classes)}
    out = np.zeros((B,T,C,H,W), dtype=np.float32)
    for b in range(B):
        for t in range(T):
            for v,ch in mapping.items():
                mask = (batch[b,t] == v)
                out[b,t,ch][mask] = 1.0
    return out

def mask_fog_once(full_maps: np.ndarray, fog_token=-1, prob=0.4):
    # full_maps (B,H,W) -> partial maps (B,H,W) with some tiles set to fog_token
    B,H,W = full_maps.shape
    out = full_maps.copy()
    mask = (np.random.rand(B,H,W) < prob)
    out[mask] = fog_token
    return out

def mask_fog_realistic(full_map: np.ndarray, player_id: int, fog_token=-2, vision_radius=1):
    """
    Apply realistic fog of war for a specific player:
    - Player can always see their own tiles (including general and cities)
    - Player can see tiles within vision_radius of their tiles
    - Everything else is fogged
    
    full_map: (H, W) array with owner values
    player_id: which player's perspective (1 or 2)
    fog_token: value to use for fogged tiles
    vision_radius: how many tiles away from owned tiles can be seen
    """
    H, W = full_map.shape
    partial = full_map.copy()
    
    # Create mask of tiles owned by this player
    owned_mask = np.zeros((H, W), dtype=bool)
    owned_mask |= (full_map == player_id)  # regular owned tiles
    owned_mask |= (full_map == player_id + 100)  # owned cities
    owned_mask |= (full_map == player_id + 200)  # owned general
    
    # Create vision mask: tiles within vision_radius of owned tiles
    vision_mask = np.zeros((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            if owned_mask[y, x]:
                # Add vision around this owned tile
                for dy in range(-vision_radius, vision_radius + 1):
                    for dx in range(-vision_radius, vision_radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            vision_mask[ny, nx] = True
    
    # Fog everything outside vision
    fog_mask = ~vision_mask
    partial[fog_mask] = fog_token
    
    return partial

def constrain_predictions(pred_map: np.ndarray, partial_map: np.ndarray, fog_token=-2, max_cities=8, territory_balance_tolerance=5):
    """
    Apply game rules to predictions:
    1. Keep all visible tiles from partial_map (don't override known info)
    2. Limit generals: max 1 general per player
    3. Limit total cities to reasonable number (typical map has 2-8 cities)
    4. Balance territory sizes: opponent territory should be within ±tolerance of player 1's territory
    5. If we see a general in visible area, don't predict more of that general type
    """
    result = pred_map.copy()
    H, W = pred_map.shape
    
    # Keep all visible (non-fog) tiles from partial observation
    visible_mask = (partial_map != fog_token)
    result[visible_mask] = partial_map[visible_mask]
    
    # For fogged tiles, constrain predictions
    fog_mask = (partial_map == fog_token)
    
    # For each player, ensure at most 1 general total (visible + fogged)
    for player_id in [1, 2, 3, 4]:
        general_value = 200 + player_id
        
        # Check if this player's general is already visible
        has_visible_general = (general_value in partial_map[visible_mask])
        
        if has_visible_general:
            # General already visible, don't predict it in fogged area
            result[fog_mask & (pred_map == general_value)] = player_id  # convert to regular territory
        else:
            # General not visible, allow at most 1 in fogged area
            general_locs = np.argwhere(fog_mask & (result == general_value))
            if len(general_locs) > 1:
                # Keep only first one, convert rest to regular territory
                for i in range(1, len(general_locs)):
                    y, x = general_locs[i]
                    result[y, x] = player_id
    
    # Constrain number of cities (neutral and captured)
    # Count cities already visible
    visible_cities_count = 0
    for city_val in [100, 101, 102, 103, 104]:  # neutral and captured cities
        visible_cities_count += np.sum(partial_map[visible_mask] == city_val)
    
    # Count predicted cities in fogged area
    predicted_cities = []
    for city_val in [100, 101, 102, 103, 104]:
        city_locs = np.argwhere(fog_mask & (result == city_val))
        for loc in city_locs:
            predicted_cities.append((loc[0], loc[1]))
    
    # If too many cities total, keep only the first max_cities - visible_cities_count
    max_predicted = max(0, max_cities - visible_cities_count)
    if len(predicted_cities) > max_predicted:
        # Remove excess cities (convert to plains)
        for i in range(max_predicted, len(predicted_cities)):
            y, x = predicted_cities[i]
            result[y, x] = 0
    
    # Balance territory sizes between players
    # Count player 1's territory (visible + predicted)
    player1_tiles = np.sum((result == 1) | (result == 101) | (result == 201))
    
    # Count opponent territories in fogged area (player 2 in this case)
    # Separate regular territory from generals when counting
    opponent_tiles_fogged = []  # Regular territory only (not generals)
    for player_id in [2, 3, 4]:  # possible opponents
        # Regular territory
        regular_territory_mask = (result == player_id) | (result == player_id + 100)
        regular_locs = np.argwhere(fog_mask & regular_territory_mask)
        for loc in regular_locs:
            opponent_tiles_fogged.append((loc[0], loc[1], player_id))
        
        # Count generals separately (don't add to removable list)
        general_mask = (result == player_id + 200)
        general_count = np.sum(fog_mask & general_mask)
    
    # Count visible opponent tiles
    visible_opponent_tiles = 0
    for player_id in [2, 3, 4]:
        visible_opponent_tiles += np.sum(
            (partial_map[visible_mask] == player_id) | 
            (partial_map[visible_mask] == player_id + 100) | 
            (partial_map[visible_mask] == player_id + 200)
        )
    
    total_opponent_tiles = visible_opponent_tiles + len(opponent_tiles_fogged)
    
    # If opponent has too many tiles compared to player 1, remove some
    # But never remove generals - only remove regular territory
    max_opponent_tiles = player1_tiles + territory_balance_tolerance
    if total_opponent_tiles > max_opponent_tiles:
        tiles_to_remove = total_opponent_tiles - max_opponent_tiles
        # Remove excess opponent tiles from fogged area (convert to plains)
        # Only removes from opponent_tiles_fogged which excludes generals
        for i in range(min(tiles_to_remove, len(opponent_tiles_fogged))):
            y, x, _ = opponent_tiles_fogged[i]
            result[y, x] = 0
    
    return result


# ---------------------------
# Main training pipeline
# ---------------------------
def main():
    writer = SummaryWriter("runs/dqn_selfplay_demo")
    env = GeneralIOEnv(map_size=MAP_SIZE, max_steps=200, num_players=NUM_PLAYERS)
    agent = DQNAgent(MAP_SIZE, MAP_SIZE)
    snapshot_pool = []

    # train DQN with snapshot self-play
    print("=== Training DQN with snapshot self-play ===")
    ep_rewards = []
    global_step = 0
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        ep_r = 0.0
        # choose opponent policy for this episode
        use_snapshot = (len(snapshot_pool) > 0) and (random.random() < OPP_SAMPLE_FROM_SNAPSHOTS_PCT)
        if use_snapshot:
            opponent = random.choice(snapshot_pool)
        else:
            opponent = None  # will be current agent as opponent? We'll wrap that below
        # if opponent is None, we sample the current agent 20% of time by calling agent.act on opponent obs
        while not done:
            action = agent.act(obs)
            # build opponent callable
            if opponent is None:
                # current agent as opponent: we create a small wrapper
                def opp_policy(o):
                    return agent.act(o, eval_mode=True)
                opp_callable = opp_policy
            else:
                opp_callable = opponent
            next_obs, r, done, _ = env.step(action, player_id=1, opponent_policy=opp_callable)
            agent.store(obs, action, r, next_obs, done)
            loss = agent.learn()
            if loss is not None:
                writer.add_scalar("loss", loss, global_step)
            obs = next_obs
            ep_r += r
            global_step += 1

        # log
        ep_rewards.append(ep_r)
        writer.add_scalar("episode_reward", ep_r, ep)
        writer.add_scalar("epsilon", agent.epsilon(), ep)
        print(f"Episode {ep+1}/{EPISODES}  reward={ep_r:.3f}  eps={agent.epsilon():.3f}")

        # snapshot policy every SNAPSHOT_INTERVAL episodes (store wrapper)
        if (ep+1) % SNAPSHOT_INTERVAL == 0:
            snap = agent.snapshot()
            snapshot_pool.append(snap)
            print(f" Snapshot saved. Pool size: {len(snapshot_pool)}")

    # save trained agent
    os.makedirs("models", exist_ok=True)
    torch.save(agent.qnet.state_dict(), "models/dqn_agent_20x20.pth")
    print("DQN trained and saved.")

    # Collect full maps dataset from mid-game states (more interesting than early game)
    print("=== Collecting full maps for fog predictor training ===")
    collected = []
    for episode_idx in range(500):
        obs = env.reset()
        done = False
        step_count = 0
        # Play for a while to get interesting mid-game states
        while not done:
            a = agent.act(obs, eval_mode=True)
            obs, r, done, _ = env.step(a, player_id=1, opponent_policy=None)
            step_count += 1
            # Collect states from steps 10-100 (mid-game, after some expansion)
            if step_count >= 10 and step_count % 5 == 0:
                collected.append(obs["owner"].copy())
            if step_count >= 100:
                break
        if (episode_idx + 1) % 100 == 0:
            print(f"  Collected {episode_idx + 1}/500 episodes...")
    collected = np.array(collected)  # shape (N,H,W)
    print("Collected maps:", collected.shape)
    
    # Print distribution of values in collected data
    unique, counts = np.unique(collected, return_counts=True)
    print("Training data distribution:")
    for val, cnt in zip(unique, counts):
        print(f"  Value {val}: {cnt} tiles ({100*cnt/collected.size:.2f}%)")

    # Train fog predictor with class weighting
    # Include all possible values: mountains(-1), fog(-2), plains(0), players(1,2), uncaptured city(100), cities(101,102), generals(201,202)
    CLASSES = [-2, -1, 0, 1, 2, 100, 101, 102, 201, 202]  # include all possible map values
    predictor = FogPredictor(MAP_SIZE, MAP_SIZE, CLASSES).to(DEVICE)
    opt = optim.Adam(predictor.parameters(), lr=1e-3)
    
    # Compute class weights to handle imbalance (inverse frequency)
    mapping = {v: i for i, v in enumerate(CLASSES)}
    class_counts = np.zeros(len(CLASSES))
    for val, cnt in zip(unique, counts):
        if val in mapping:
            class_counts[mapping[val]] = cnt
    class_counts = np.maximum(class_counts, 1)  # avoid division by zero
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(CLASSES)  # normalize
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    print(f"Class weights: {dict(zip(CLASSES, class_weights))}")
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    BATCH = 16
    SEQ_LEN = 3
    FOG_TOKEN = -2  # Use correct fog token
    PLAYER_ID = 1  # Train from player 1's perspective
    print("=== Training Fog Predictor (50 epochs) with realistic fog ===")
    print("Note: Computing loss ONLY on fogged tiles (not on visible tiles)")
    for epoch in range(50):
        total_loss = 0.0
        for step in range(50):
            idxs = np.random.randint(0, len(collected), size=BATCH)
            full = collected[idxs]  # (B,H,W)
            # Apply realistic fog from player 1's perspective
            seqs = []
            partial_batch_list = []
            for _ in range(SEQ_LEN):
                partial_batch = []
                for b in range(BATCH):
                    partial = mask_fog_realistic(full[b], player_id=PLAYER_ID, fog_token=FOG_TOKEN, vision_radius=1)
                    partial_batch.append(partial)
                seqs.append(np.stack(partial_batch))
                if _ == 0:  # Save first sequence for masking
                    partial_batch_list = partial_batch
            
            seqs = np.stack(seqs, axis=1)  # (B,T,H,W)
            x = ints_to_onehot(seqs, CLASSES)  # (B,T,C,H,W)
            x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            logits = predictor(x_t)  # (B,C,H,W)
            # target
            mapping = {v:i for i,v in enumerate(CLASSES)}
            tgt = np.vectorize(lambda v: mapping[int(v)])(full)
            tgt_t = torch.tensor(tgt, dtype=torch.long, device=DEVICE)
            
            # Create mask for fogged tiles only (loss computed only on these)
            fog_mask = torch.zeros((BATCH, MAP_SIZE, MAP_SIZE), dtype=torch.bool, device=DEVICE)
            for b in range(BATCH):
                fog_mask[b] = torch.tensor(partial_batch_list[b] == FOG_TOKEN, device=DEVICE)
            
            # Compute loss only on fogged tiles
            loss = 0.0
            for b in range(BATCH):
                if fog_mask[b].any():
                    logits_fogged = logits[b][:, fog_mask[b]]  # (C, num_fogged_tiles)
                    tgt_fogged = tgt_t[b][fog_mask[b]]  # (num_fogged_tiles,)
                    loss += F.cross_entropy(logits_fogged.T, tgt_fogged, weight=class_weights_tensor)
            
            loss = loss / BATCH  # Average over batch
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        avg = total_loss / 50.0
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 - avg_loss {avg:.4f}")
    
    # Save fog predictor model
    torch.save(predictor.state_dict(), "models/fog_predictor_20x20.pth")
    print("Fog predictor trained and saved to models/fog_predictor.pth")

    # Evaluate under fog using predictor
    print("=== Evaluate agent under fog (3 rollouts) ===")
    for rollout in range(3):
        obs = env.reset()
        done = False
        steps = 0
        print(f"--- Rollout {rollout+1} ---")
        while not done and steps < 30:
            steps += 1
            # create partial observation with realistic fog from player 1's perspective
            owner_full = obs["owner"]
            partial = mask_fog_realistic(owner_full, player_id=1, fog_token=FOG_TOKEN, vision_radius=1)
            # create seq_len copies (simple)
            seq = np.stack([partial, partial, partial], axis=0)[None]  # (1,T,H,W)
            x = ints_to_onehot(seq, CLASSES)
            x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                logits = predictor(x_t)
                pred_idx = logits.argmax(1).cpu().numpy()[0]
            idx2cls = {i:c for i,c in enumerate(CLASSES)}
            pred_map = np.vectorize(lambda k: idx2cls[k])(pred_idx)
            
            # Apply constraints to respect game rules
            pred_map = constrain_predictions(pred_map, partial, fog_token=FOG_TOKEN)
            
            # build predicted obs dict (we keep armies as zeros for inference; in practice you might predict armies too)
            pred_obs = {"owner": pred_map.astype(np.int32), "armies": obs["armies"].copy()}
            action = agent.act(pred_obs, eval_mode=True)
            # step environment using that action against a random opponent for test
            next_obs, r, done, _ = env.step(action, player_id=1, opponent_policy=None)
            print(f"Step{steps} Action:{action} Reward:{r:.3f}")
            print("Partial owner:\n", partial)
            print("Pred owner:\n", pred_map)
            obs = next_obs

    writer.close()
    print("Demo complete.")

if __name__ == "__main__":
    main()
