import torch
import numpy as np
from src.environment import RechargeEnvironment
from src.agent import DQNAgent
from src import utils, config

import sys

def train():

    # Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # Load training data
    df_user_train, df_prices_train =  utils.get_user_prices_data(
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRAIN_END_DATE,
        df_electricity_prices=utils.load_electricity_prices(config.PRICES_DATA_PATH),
        t_a_mean=config.T_A_MEAN,
        t_a_std=config.T_A_STD,
        t_a_min=config.T_A_MIN,
        t_a_max=config.T_A_MAX,
        t_d_mean=config.T_D_MEAN,
        t_d_std=config.T_D_STD,
        t_d_min=config.T_D_MIN,
        t_d_max=config.T_D_MAX,
        soc_mean=config.SOC_MEAN,
        soc_std=config.SOC_STD,
        soc_min=config.SOC_MIN,
        soc_max=config.SOC_MAX,
        prices_history_len=config.PRICES_HISTORY_LEN
    )

    # Compute normlization variables and save it to a JSON file
    mean_price, std_price, mean_derivative_price, std_derivative_price, q95_price = utils.get_price_statistics(df_prices_train)
    reward_scale = q95_price * config.E_MAX
    utils.save_normalization_stats(mean_price, std_price, mean_derivative_price, std_derivative_price, reward_scale)

    # Initialize environment
    env = RechargeEnvironment(actions=config.ACTIONS,
                              e_max=config.E_MAX,
                              tau=config.TAU,
                              df_user=df_user_train,
                              df_prices=df_prices_train,
                              prices_history_len=config.PRICES_HISTORY_LEN)
    
    # Initialize agent
    agent = DQNAgent(action_size=len(config.ACTIONS), 
                     prices_history_len=config.PRICES_HISTORY_LEN,
                     device=device,
                     hidden_lstm_size=config.HIDDEN_LSTM_SIZE,
                     hidden_fc_size=config.HIDDEN_FC_SIZE,
                     learning_rate=config.LR,
                     warmup_size=config.WARMUP_SIZE,
                     gamma=config.GAMMA,
                     memory_maxlen=config.MEMORY_MAXLEN,
                     batch_size=config.BATCH_SIZE,
                     target_update_frequency=config.TARGET_UPDATE_FREQUENCY,
                     mean_price=mean_price,
                     std_price=std_price,
                     mean_derivative_price=mean_derivative_price,
                     std_derivative_price=std_derivative_price,
                     reward_scale=reward_scale)

    losses = []
    scores = []

    # Training loop
    for episode in range(config.NUM_EPISODES):

        # Choose epsilon
        epsilon = utils.epsilon_policy(episode, config.NUM_EPISODES) 

        # Reset environment
        state = env.reset()
        score = 0

        done = False
        while not done:
            # Select action
            action = agent.select_action(state, epsilon=epsilon)

            # Take action
            next_state, reward, done  = env.step(config.ACTIONS[action])

            # Take a training step
            loss = agent.train_step(state, action, reward, done, next_state)
            losses.append(loss)

            state = next_state
            score += reward
        
        scores.append(score)

        # Print episode status
        if (episode+1) % 10 == 0 and episode > 100:
            print(f"\rEpisode {episode+1:<5}/{config.NUM_EPISODES:<5} | "
                f"eps: {epsilon:<5.2f} | "
                f"reward: {np.array(scores[max(0, episode-100):episode]).mean():>8.3f} | "
                f"loss: {np.array(losses[max(0, episode-100):episode]).mean():>8.4f}", end="")
            sys.stdout.flush()
            
    # Save the model
    torch.save(agent.network.state_dict(), config.MODEL_PATH)
    return scores, losses

if __name__ == "__main__":
    scores, losses = train()
