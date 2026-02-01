import numpy as np
import pandas as pd
import os

# Parameters

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Environment parameters
TAU = 0.05                       # Anxiety rate
E_MAX = 25                      # Maximum battery's energy, in kWh
ACTIONS = np.array([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)   # Available charging power (in kW)
PRICES_HISTORY_LEN = 24         # Number of previous hours' prices to consider

# Prices data parameters
PRICES_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "electricity_prices.csv")
PRICES_TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "electricity_prices_sample.csv")
TRAIN_START_DATE = "2020-01-02"
TRAIN_END_DATE = "2024-12-30"
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-12-30"
assert pd.to_datetime(TRAIN_END_DATE+" 12:00:00") + pd.Timedelta(days=1) < pd.to_datetime(TEST_START_DATE+" 13:00:00")-pd.Timedelta(hours=PRICES_HISTORY_LEN), "Train and test periods are overlapping!"

# Statistical parameters for time of arrival generation
T_A_MEAN = 18                   # Mean time of arrival (in hours)
T_A_STD = 1                     # Standard deviation of time of arrival (in hours)
T_A_MIN = 15                    # Minimum time of arrival (in hours)
T_A_MAX = 21                    # Maximum time of arrival (in hours)

# Statistical parameters for time of departure generation
T_D_MEAN = 8                    # Mean time of departure (in hours)
T_D_STD = 1                     # Standard deviation of time of departure (in hours)
T_D_MIN = 6                     # Minimum time of departure (in hours)
T_D_MAX = 11                    # Maximum time of departure (in hours)

# Statistical parameters for state of charge generation
SOC_MEAN = 0.5                  # Mean state of charge at arrival
SOC_STD = 0.1                   # Standard deviation of state of charge at arrival
SOC_MIN = 0.2                   # Minimum state of charge at arrival
SOC_MAX = 0.8                   # Maximum state of charge at arrival

# Agent parameters
GAMMA = 0.99
HIDDEN_LSTM_SIZE = 32
HIDDEN_FC_SIZE = 32
LR = 0.0005
BATCH_SIZE = 32
NUM_EPISODES = 10000
WARMUP_SIZE = 500
MEMORY_MAXLEN = 500000
TARGET_UPDATE_FREQUENCY = 100
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "dqn_model.pth")