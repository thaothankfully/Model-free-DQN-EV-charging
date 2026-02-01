import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import os
import json
from src import config

# Function to load electricity prices data
def load_electricity_prices(file_path):
    """
    Load electricity prices from a CSV file.
    The CSV file is expected to have two columns: 'datetime' and 'price'.
    'datetime' should be in the format 'YYYY-MM-DD HH:MM:SS'.
    'price' should be expressed in €/kWh
    """
    df_prices = pd.read_csv(file_path)
    df_prices['datetime'] = pd.to_datetime(df_prices['datetime'])
    return df_prices

# Truncated gaussian distribution sampling
def sample_truncated_gaussian(size, mu, sigma, a, b, dtype=np.float32):
    """
    Sample from a truncated Gaussian distribution.
    size: number of samples
    mu: mean
    sigma: standard deviation
    a: minimum value
    b: maximum value
    dtype: data type of the output array
    """
    a_std = (a - mu)/sigma
    b_std = (b - mu)/sigma
    return np.array(truncnorm.rvs(a_std, b_std, loc=mu, scale=sigma, size=size), dtype=dtype)

# Function to get user data and prices data for a given period
def get_user_prices_data(start_date,
                         end_date,
                         df_electricity_prices,
                         t_a_mean,
                         t_a_std,
                         t_a_min,
                         t_a_max,
                         t_d_mean,
                         t_d_std,
                         t_d_min,
                         t_d_max,
                         soc_mean,
                         soc_std,
                         soc_min,
                         soc_max,
                         prices_history_len):
    """
    Generate user data (time of arrival, time of departure, state of charge) and get electricity prices data for the specified period.
    start_date: start date of the period (string 'YYYY-MM-DD')
    end_date: end date of the period (string 'YYYY-MM-DD')
    df_electricity_prices: DataFrame containing electricity prices with 'datetime' and 'price' columns
    t_a_mean, t_a_std, t_a_min, t_a_max: parameters for time of arrival generation
    t_d_mean, t_d_std, t_d_min, t_d_max: parameters for time of departure generation
    soc_mean, soc_std, soc_min, soc_max: parameters for state of charge generation
    prices_history_len: number of hours of price history to include
    return: df_user (DataFrame with user data), df_prices (DataFrame with electricity prices)
    """

    # Compute number of days in the period
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

    # Generate Time of Arrival (t_a), Time of Departure (t_d), and State of Charge (soc)
    t_a = sample_truncated_gaussian(num_days, t_a_mean, t_a_std, t_a_min, t_a_max, dtype=np.int64)
    t_d = sample_truncated_gaussian(num_days, t_d_mean, t_d_std, t_d_min, t_d_max, dtype=np.int64)
    soc = sample_truncated_gaussian(num_days, soc_mean, soc_std, soc_min, soc_max, dtype=np.float32)

    # Create user DataFrame, containing t_a, t_d and soc
    df_user = pd.DataFrame({
        "date": pd.date_range(start=start_date, end=end_date, freq='D'),
        "t_a": t_a,
        "t_d": t_d,
        "soc": soc
    })

    # Get electricity prices data for the period, with extra hours for history
    prices_start_date = pd.to_datetime(start_date+" 13:00:00") - pd.Timedelta(hours=prices_history_len)
    prices_end_date = pd.to_datetime(end_date+" 12:00:00") + pd.Timedelta(days=1)
    df_prices = df_electricity_prices[df_electricity_prices['datetime'].between(prices_start_date, prices_end_date)].reset_index(drop=True)

    return df_user, df_prices

# Function to compute mean and std of prices and derivative prices
def get_price_statistics(df_prices):
    """
    Compute mean and standard deviation of electricity prices and their derivatives, as well as the 95th percentile of prices.
    df_prices: DataFrame containing electricity prices with 'price' column
    return: mean_price, std_price, mean_derivative_price, std_derivative_price, q95_price
    """
    prices = df_prices['price'].values
    derivative_prices = np.diff(prices)

    mean_price = np.mean(prices)
    std_price = np.std(prices)

    mean_derivative_price = np.mean(derivative_prices)
    std_derivative_price = np.std(derivative_prices)

    q95_price = np.quantile(prices, 0.95)

    return mean_price, std_price, mean_derivative_price, std_derivative_price, q95_price

# Save normalization statistics to a JSON file
def save_normalization_stats(mean_price, std_price, mean_derivative_price, std_derivative_price, reward_scale):
    """
    Save normalization statistics to a JSON file.
    mean_price: mean of electricity prices
    std_price: standard deviation of electricity prices
    mean_derivative_price: mean of derivative prices
    std_derivative_price: standard deviation of derivative prices
    reward_scale: scaling factor for rewards
    """
    stats = {"mean_price": mean_price,
             "std_price": std_price,
             "mean_derivative_price": mean_derivative_price,
             "std_derivative_price": std_derivative_price,
             "reward_scale": reward_scale}
    with open(os.path.join(config.PROJECT_ROOT, "models", "normalization_consts.json"), "w") as f:
        json.dump(stats, f)

# Epsilon policy: defines epsilon's value according to the episode
def epsilon_policy(episode, num_episodes):
    """
    Define epsilon value according to the episode number using a exponential decay between 2 checkpoints.
    episode: current episode number
    num_episodes: total number of episodes
    return: epsilon value
    """
    eps_min = 0.05
    checkpoint_1 = 0.1     # Start decreasing epsilon from this checkpoint
    checkpoint_2 = 0.6     # Stop decreasing epsilon from this checkpoint
    progress = episode / num_episodes
    epsilon_decay = np.power(eps_min, 1/((checkpoint_2-checkpoint_1)*num_episodes))
    if progress < checkpoint_1:
        epsilon = 1.
    elif progress < checkpoint_2:
        epsilon = np.power(epsilon_decay, (episode - checkpoint_1*num_episodes))
    else:
        epsilon = eps_min
    return epsilon


# Plot rewards
def plot_rewards(total_rewards, title, n_points=200):
    """
    Plot total rewards per episode, smoothed over a window.
    total_rewards: list of total rewards per episode
    title: title of the plot
    n_points: number of points to use for smoothing
    """
    n = len(total_rewards)
    total_rewards_np = np.array(total_rewards)
    window_size = max(n//n_points, 1)
    total_rewards_smooth = np.zeros((n))
    
    for i in range(n):
        total_rewards_smooth[i] = total_rewards_np[max(i-window_size//2,0):min(i+window_size//2,n-1)].mean()
    plt.plot(total_rewards_smooth)
    # plt.plot(total_rewards)
    plt.xlabel("Episode")
    #plt.yscale("log")
    plt.title(title)
    plt.axis("on")
    plt.show()


# Plot actions, prices and soc
def plot_actions_prices(env, date=None, actions=None, socs=None):
    """
    Plot actions taken, electricity prices and state of charge over an episode.
    env: RechargeEnvironment instance
    date: date of the episode to plot (string 'YYYY-MM-DD'). If None, a random date is chosen.
    actions: array of actions taken during the episode (in kW). If None, no actions are plotted.
    socs: array of state of charge values during the episode (normalized between 0 and 1). If None, no SOC is plotted.
    """

    # Initialize env
    _ = env.reset(date=date)
    t_a = env.t_a
    t_d = env.t_d
    date = env.date
    time_index = env.time_index
    prices = env.df_prices.iloc[time_index:time_index+t_d+24-t_a+1]['price'].values

    # Date range
    start_dt = pd.Timestamp(f"{date} {t_a:02}:00:00")
    end_dt = pd.Timestamp(f"{date} {t_d:02}:00:00") + pd.Timedelta(days=1)
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='h')

    # If there are no actions
    if actions is None:
        actions = np.zeros(len(date_range) - 1)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot electricity prices, red line
    ax1.step(date_range, prices, where="post", label="Prices (€)/kWh", color="red")
    ax1.set_xticks(date_range)
    ax1.set_xticklabels([dt.strftime("%H:%M") for dt in date_range], rotation=45)
    ax1.set_xlim(start_dt, end_dt)
    ax1.set_xlabel("Hour")

    ax1.set_ylabel("Prices (€)/kWh")
    ax1.tick_params(axis='y')

    # Plot actions, blue bars
    bar_centers = date_range[:-1] + pd.Timedelta(minutes=30)  # Centrage des barres
    ax2 = ax1.twinx()
    ax2.bar(bar_centers, actions, width=0.03, align='center', alpha=0.5, color='blue', label="Power (kW)")
    ax2.set_ylabel("Power (kW)")
    ax2.tick_params(axis='y')
    ax2.set_ylim(np.min(config.ACTIONS), np.max(config.ACTIONS))

    # Plot SOCs, in a green dotted line
    if socs is not None:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(("outward", 60))  # Décaler l'axe Y à droite
        ax3.plot(date_range, socs*config.E_MAX, color='green', linestyle='dashed', marker='s', label="SOC (kWh)")
        ax3.set_ylabel("SOC (kWh)", color='green')
        ax3.set_ylim(0, config.E_MAX)
        ax3.tick_params(axis='y')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f"Episode of charge (from {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')})")
    plt.grid()
    plt.show()


# Print a batch of transitions
def print_transition(memory_random_batch,):
    """
    Print a batch of transitions from the replay memory.
    memory_random_batch: list of transitions (state, action, next_state, reward, done)
    """
    for (state, action, next_state, reward , done) in memory_random_batch:
        timestep = (round(((1-state[1])*23))-11)%24
        print(f"---- Transition ----")
        print(f"  SOC: {float(state[0]):.0%}")
        print(f"  Time: {timestep}h") 
        print(f"  Price: {state[2]:.2f}") 
        print(f"  Action: {config.ACTIONS[action]} kW")
        print(f"  Reward: {reward}")
        print(f"  Next SOC: {float(next_state[0]):.0%}")
        print(f"  Done: {done}")
