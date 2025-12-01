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
MAP_SIZE = 10
NUM_PLAYERS = 2
EPISODES = 200
SNAPSHOT_INTERVAL = 20
OPP_SAMPLE_FROM_SNAPSHOTS_PCT = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Map encoding:
# -2 : mountain
# -1 : fog (only for predictor inference)
# 0  : neutral
# 1..N : player-owned tile
# 11,12,... : regular base for player1, player2 -> base on tile encoded as owner + 10
# 101,102,... : castle (main base) for player1, player2 -> castle on tile encoded as owner + 100

BASE_OFFSET = 100  # Main castle offset
REGULAR_BASE_OFFSET = 10  # Regular base offset

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
        self.owner = np.zeros((H,W), dtype=np.int32)  # -2 mountains, 0 neutral, 1.. players, 11.. bases
        self.armies = np.zeros((H,W), dtype=np.int32) # army counts
        self.turn = 0

        # place mountains randomly (sparse)
        num_mnts = (H*W)//20
        for _ in range(num_mnts):
            y,x = np.random.randint(0,H), np.random.randint(0,W)
            self.owner[y,x] = -2

        # place castles (main bases) randomly with minimum distance constraint
        min_distance = max(H, W) * 0.4  # at least 40% of map dimension apart
        castle_positions = []
        
        for p in range(1, self.num_players+1):
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                y, x = np.random.randint(0, H), np.random.randint(0, W)
                
                # check if position is valid (not mountain)
                if self.owner[y,x] == -2:
                    continue
                
                # check minimum distance from other castles
                valid_position = True
                for (py, px) in castle_positions:
                    dist = np.sqrt((y - py)**2 + (x - px)**2)
                    if dist < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    self.owner[y,x] = p + BASE_OFFSET  # Castle (main base)
                    self.armies[y,x] = 5  # starting garrison
                    castle_positions.append((y, x))
                    placed = True
                    break
            
            # fallback: if couldn't place randomly, use corners
            if not placed:
                corners = [(0,0),(0,W-1),(H-1,0),(H-1,W-1)]
                y, x = corners[p-1]
                if self.owner[y,x] == -2:
                    self.owner[y,x] = 0
                self.owner[y,x] = p + BASE_OFFSET
                self.armies[y,x] = 5
                castle_positions.append((y, x))

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

    def step(self, player_action:int, player_id=1, opponent_policy=None):
        """
        player_action: integer 0..4 -> global movement direction applied to all owned tiles for that player
        opponent_policy: callable(obs)->action for the opponent (we'll only support 1 opponent in demo)
        Returns: next_obs, reward, done, info
        """
        self.turn += 1
        H,W = self.map_size, self.map_size

        # store prev totals for reward
        prev_total_army = self.armies[self.owner == player_id].sum()
        prev_bases = (self.owner == (player_id + BASE_OFFSET)).sum()

        # Apply player's movement
        self._apply_global_move(player_id, player_action)

        # Opponent(s) moves: for demo we handle single opponent player 2
        if self.num_players >= 2:
            if opponent_policy is None:
                opp_action = random.randint(0, NUM_ACTIONS-1)
            else:
                obs_for_opp = self._get_obs(player_id=2)
                opp_action = opponent_policy(obs_for_opp)
            self._apply_global_move(2, opp_action)

        # produce at bases and castles: each base/castle adds 1 army
        # Check for both castles (>=BASE_OFFSET) and regular bases (>=REGULAR_BASE_OFFSET)
        castle_positions = np.argwhere(self.owner >= BASE_OFFSET)
        regular_base_positions = np.argwhere((self.owner >= REGULAR_BASE_OFFSET) & (self.owner < BASE_OFFSET))
        for (y,x) in castle_positions:
            self.armies[y,x] += 1
        for (y,x) in regular_base_positions:
            self.armies[y,x] += 1

        # clamp armies
        self.armies = np.clip(self.armies, 0, 1000)

        # compute reward for player 1
        my_total_army = self.armies[self.owner == player_id].sum()
        delta_army = int(my_total_army - prev_total_army)

        my_castles = (self.owner == (player_id + BASE_OFFSET)).sum()
        my_bases = ((self.owner >= (player_id + REGULAR_BASE_OFFSET)) & (self.owner < (player_id + BASE_OFFSET))).sum()
        total_bases = my_castles + my_bases
        base_gain = int(total_bases - prev_bases)
        reward = float(delta_army) + 5.0 * float(base_gain)

        # done conditions
        done = False
        if self.turn >= self.max_steps:
            done = True
        # if captured opponent's castle -> win
        opponent_castle = (self.owner == (2 + BASE_OFFSET)).sum()
        if opponent_castle == 0:
            done = True
            reward += 10.0

        info = {}
        return self._get_obs(player_id=player_id), reward, done, info

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

        # iterate all tiles owned by player (including castles, regular bases, and regular owned tiles)
        owned_mask = ( (self.owner == player_id) | 
                      (self.owner == player_id + REGULAR_BASE_OFFSET) | 
                      (self.owner == player_id + BASE_OFFSET) )
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
            if self.owner[ny,nx] == -2:
                self.armies[y,x] += moving
                continue
            # if empty or same owner -> add to incoming
            is_mine = (self.owner[ny,nx] == player_id) or \
                      (self.owner[ny,nx] == player_id + REGULAR_BASE_OFFSET) or \
                      (self.owner[ny,nx] == player_id + BASE_OFFSET)
            if (self.owner[ny,nx] == 0) or is_mine:
                incoming[ny,nx] += moving
            else:
                # combat with enemy/neutral troops: resolve immediately
                defender_owner = self.owner[ny,nx]
                defender_army = int(self.armies[ny,nx])
                if moving > defender_army:
                    # capture tile: remaining armies occupy, owner switches
                    remaining = moving - defender_army
                    # if defender was a castle, convert to your castle
                    if defender_owner >= BASE_OFFSET:
                        # capturing enemy castle -> become your castle
                        self.owner[ny,nx] = player_id + BASE_OFFSET
                    elif defender_owner >= REGULAR_BASE_OFFSET:
                        # capturing regular base -> become your regular base
                        self.owner[ny,nx] = player_id + REGULAR_BASE_OFFSET
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
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        conv_out = 64 * H * W
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        # x shape (B,2,H,W)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

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
        self.eps_start, self.eps_end, self.eps_decay = 1.0, 0.05, 4000

    def act(self, obs_dict, eval_mode=False):
        # build input: 2 channels -> owner (scaled), armies (scaled)
        inp = self._obs_to_tensor(obs_dict)
        if eval_mode or random.random() > self.epsilon():
            with torch.no_grad():
                q = self.qnet(inp)
                return int(q.argmax(1).item())
        else:
            return random.randint(0, NUM_ACTIONS-1)

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
        a_batch = torch.tensor(batch.a, dtype=torch.int64, device=self.device)
        r_batch = torch.tensor(batch.r, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        q_vals = self.qnet(s_batch)
        q_a = q_vals.gather(1, a_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target(s2_batch).max(1)[0]
            target = r_batch + self.gamma * q_next * (1.0 - done_batch)

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
                with torch.no_grad():
                    q = self.net(t)
                    return int(q.argmax(1).item())
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
    torch.save(agent.qnet.state_dict(), "models/dqn_agent.pth")
    print("DQN trained and saved.")

    # Collect full maps dataset
    print("=== Collecting full maps for fog predictor training ===")
    collected = []
    for _ in range(500):
        obs = env.reset()
        done = False
        while not done:
            collected.append(obs["owner"].copy())  # store owner map only for predictor; optionally store armies too
            a = agent.act(obs, eval_mode=True)
            obs, r, done, _ = env.step(a, player_id=1, opponent_policy=None)
    collected = np.array(collected)  # shape (N,H,W)
    print("Collected maps:", collected.shape)

    # Train fog predictor
    # Include all possible values: mountains(-2), fog(-1), neutral(0), players(1,2), regular bases(11,12), castles(101,102)
    CLASSES = [-2, -1, 0, 1, 2, 11, 12, 101, 102]  # include all possible map values
    predictor = FogPredictor(MAP_SIZE, MAP_SIZE, CLASSES).to(DEVICE)
    opt = optim.Adam(predictor.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    BATCH = 16
    SEQ_LEN = 3
    print("=== Training Fog Predictor (5 epochs) ===")
    for epoch in range(5):
        total_loss = 0.0
        for step in range(50):
            idxs = np.random.randint(0, len(collected), size=BATCH)
            full = collected[idxs]  # (B,H,W)
            seqs = [mask_fog_once(full, prob=0.4) for _ in range(SEQ_LEN)]
            seqs = np.stack(seqs, axis=1)  # (B,T,H,W)
            x = ints_to_onehot(seqs, CLASSES)  # (B,T,C,H,W)
            x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            logits = predictor(x_t)  # (B,C,H,W)
            # target
            mapping = {v:i for i,v in enumerate(CLASSES)}
            tgt = np.vectorize(lambda v: mapping[int(v)])(full)
            tgt_t = torch.tensor(tgt, dtype=torch.long, device=DEVICE)
            loss = loss_fn(logits, tgt_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        avg = total_loss / 50.0
        print(f"Epoch {epoch+1}/5 - avg_loss {avg:.4f}")

    # Evaluate under fog using predictor
    print("=== Evaluate agent under fog (3 rollouts) ===")
    for rollout in range(3):
        obs = env.reset()
        done = False
        steps = 0
        print(f"--- Rollout {rollout+1} ---")
        while not done and steps < 30:
            steps += 1
            # create partial observation (mask owner map)
            owner_full = obs["owner"]
            partial = mask_fog_once(owner_full[np.newaxis], prob=0.5)[0]  # (H,W)
            # create seq_len copies (simple)
            seq = np.stack([partial, partial, partial], axis=0)[None]  # (1,T,H,W)
            x = ints_to_onehot(seq, CLASSES)
            x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                logits = predictor(x_t)
                pred_idx = logits.argmax(1).cpu().numpy()[0]
            idx2cls = {i:c for i,c in enumerate(CLASSES)}
            pred_map = np.vectorize(lambda k: idx2cls[k])(pred_idx)
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
