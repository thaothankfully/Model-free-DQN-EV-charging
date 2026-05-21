import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import json

from src.model import QNetwork
from src.memory import Transition, ReplayMemory 
 
class DQNAgent:
    def __init__(self,
                 action_size,
                 prices_history_len,
                 device,
                 hidden_lstm_size=32,
                 hidden_fc_size=32,
                 learning_rate=0.0005,
                 warmup_size=500,
                 gamma=0.99,
                 memory_maxlen = 500000,
                 batch_size=32,
                 target_update_frequency=100,
                 mean_price=None,
                 std_price=None,
                 mean_derivative_price=None,
                 std_derivative_price=None,
                 reward_scale=None):
        
        self.state_size = 3+prices_history_len-1  # 3 for soc, timestep, current_price, and prices_history_len-1 for derivative prices
        self.action_size = action_size
        self.prices_history_len = prices_history_len
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.target_update_frequency = target_update_frequency
        self.mean_price = mean_price
        self.std_price = std_price
        self.mean_derivative_price = mean_derivative_price
        self.std_derivative_price = std_derivative_price
        self.reward_scale = reward_scale

        # Q-Network
        self.network = QNetwork(hidden_lstm_size, hidden_fc_size, action_size).to(device)
        self.target_network = QNetwork(hidden_lstm_size, hidden_fc_size, action_size).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.network.eval()
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayMemory(memory_maxlen)

        self.step = 0

    def _encode_state(self, state):
        """
        Encode the state into a numpy array for DQN input
        state: dict with keys 'soc', 'timestep', 'prices' (array of previous prices)
        returns: numpy array of shape (3,) [soc, timestep, current_price]
        """

        encoded_state = np.empty(self.state_size, dtype=np.float32)

        soc = state['soc']
        timestep = state['timestep']
        current_price = state['prices'][-1]

        time_remaining = 1-((timestep+11) % 24)/23  # Time remaining until 12:00 next day, normlized
        derivative_prices = (np.diff(state['prices']) - self.mean_derivative_price) / self.std_derivative_price
        current_price_norm = (current_price - self.mean_price) / self.std_price

        encoded_state[0] = soc
        encoded_state[1] = time_remaining
        encoded_state[2] = current_price_norm
        encoded_state[3:] = derivative_prices

        return encoded_state

    def select_action(self, state, epsilon=0.):
        """
        Select an action using epsilon-greedy policy
        state: dict representing the current state
        epsilon: float, probability of choosing a random action
        returns: action (int), index of the selected action
        """
        
        self.network.eval()
        encoded_state = self._encode_state(state)

        # Epsilon greedy search
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)    # Choose a random action
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(encoded_state).float().to(self.device)
                q_values = self.network(state_tensor)        # Compute Q-values
                return q_values.argmax().item()    # Returns current best action id

    def train_step(self, state, action, reward, done, next_state):
        """
        Perform a single training step of the DQN agent
        """

        # Encode states
        encoded_state = self._encode_state(state)
        encoded_next_state = self._encode_state(next_state)

        # Store transition in replay memory
        self.memory.push(encoded_state, action, encoded_next_state, reward/self.reward_scale, done) # Normalize reward

        loss_value = 0.

        # Sampling minibatch
        if len(self.memory) >= max(self.warmup_size, self.batch_size):

            self.network.train()

            # Samples a batch of transitions
            transitions = self.memory.sample(self.batch_size)

            # Convert to tensor
            batch = Transition(*zip(*transitions))
            states = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
            actions = torch.tensor(batch.action, dtype=torch.long, device=self.device)
            next_states = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
            rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
            dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
            
            # Compute targets
            with torch.no_grad():
                targets = (rewards + self.gamma * self.target_network(next_states).max(1)[0] * (1.0 - dones))
            # Compute predicted Q-values
            predictions = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute loss
            loss = F.mse_loss(predictions, targets)
            loss_value = loss.item()
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.)
            self.optimizer.step()

        # Hard update: Reset the target network parameters every target_update_frequency steps
        self.step += 1
        if self.step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss_value
    
    def load_model(self, model_path, normalization_consts_path):
        """
        Load the model parameters and normalization statistics from files
        model_path: path to the model file
        normalization_consts_path: path to the JSON file containing normalization constants
        """
        # Load model parameters
        self.network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_network.load_state_dict(self.network.state_dict())

        # Load normalization constants
        with open(normalization_consts_path, 'r') as f:
            stats = json.load(f)
            self.mean_price = stats['mean_price']
            self.std_price = stats['std_price']
            self.mean_derivative_price = stats['mean_derivative_price']
            self.std_derivative_price = stats['std_derivative_price']
            self.reward_scale = stats['reward_scale']


class NaiveAgent():
    def __init__(self, action_size):
        self.action_size = action_size

    def select_action(self, state):
        """
        Select the maximum charging action available
        state: dict representing the current state
        epsilon: float, probability of choosing a random action (not used here)
        returns: action (int), index of the selected action
        """
        return self.action_size - 1  # Always select the maximum charging action
