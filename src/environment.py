import numpy as np
import random
import pandas as pd
 
# Define the environment

class RechargeEnvironment():
    def __init__(self, actions, e_max, tau, df_user, df_prices, prices_history_len):  
        """ Environment initialization"""
        super(RechargeEnvironment, self).__init__()  

        self.action_space = actions     # Available actions
        self.n_actions = len(actions)   # Number of available actions
        self.e_max = e_max              # Maximum battery's energy, in kWh
        self.tau = tau                  # Anxiety rate
        self.df_user = df_user          # DataFrame containing user data
        self.df_prices = df_prices      # DataFrame containing electricity prices
        self.prices_history_len = prices_history_len  # Number of previous hours' prices to consider

    def reset(self, date=None):
        """
        Reset to the initial state of the episode starting at date 'date' (string 'YYYY-MM-DD')
        If date is None, a random date is chosen
        return initial_state
        """

        # Select a random date if none is provided
        if date is None:
            # pick a random date and format as "YYYY-MM-DD" string
            date = random.choice(pd.to_datetime(self.df_user['date']).dt.strftime('%Y-%m-%d').unique().tolist())

        # Get user parameters for the selected date
        t_a = self.df_user[self.df_user['date'] == date]['t_a'].values[0]
        t_d = self.df_user[self.df_user['date'] == date]['t_d'].values[0]
        soc = self.df_user[self.df_user['date'] == date]['soc'].values[0]

        # Get the time index corresponding to t_a and date
        time_index = self.df_prices[self.df_prices['datetime'] == date+f" {t_a:02}:00:00"].index[0]

        # Prices history for the episode
        prices = self.df_prices.iloc[time_index-self.prices_history_len+1:time_index+1]['price'].values

        # Initial state
        state = {'soc': soc,
                 'timestep': t_a,
                 'prices': prices}

        # Store variables
        self.date = date
        self.t_a = t_a
        self.t_d = t_d
        self.soc = soc
        self.time_index = time_index
        self.timestep = t_a
        self.prices = prices
        self.state = state

        return self.state
    
    def step(self, action):
        """
        Perform action, move on to the state
        action (float) in kW, charge power
        
        return next_state, reward, done
        """
        soc = self.soc
        timestep = self.timestep
        current_price = self.df_prices.iloc[self.time_index]['price']
        
        # State of charge transition
        next_soc = soc + action / self.e_max
        next_soc = np.clip(next_soc, 0, 1)  # Ensure SOC is within [0, 1]

        # Reward 
        reward = -current_price * self.e_max * (next_soc - soc) # Price paid for charging/discharging
        done = False
        if timestep == self.t_d-1:   # Check if it is a final state
            reward += -self.tau * self.e_max**2 * (1-next_soc)**2 # Penalty due to lack of power
            done = True
        # TODO: Add the degradation battery cost

        # Update time index and timestep
        next_time_index = self.time_index + 1
        next_timestep = self.df_prices.iloc[next_time_index]['datetime'].hour % 24  # Next hour, taking into account summer time change

        # Update prices
        next_prices = self.df_prices.iloc[next_time_index-self.prices_history_len+1:next_time_index+1]['price'].values
        
        # Update new state
        next_state = {'soc': next_soc,
                      'timestep': next_timestep,
                      'prices': next_prices}

        # Update attributes
        self.soc = next_soc
        self.time_index = next_time_index
        self.timestep = next_timestep
        self.prices = next_prices
        self.state = next_state
        
        return next_state, reward, done

    def render(self):
        """Show the current state."""
        print("----- Current state -----")
        print(f"  datetime: {self.state['datetime']} {self.timestep}h")
        print(f"  SOC: {self.state['soc']:.1%} %")
        print(f"  Current electricity price: {self.state['prices'][-1]:.02f} €/kWh")
        print(f"  t_a/t_d = {self.t_a}/{self.t_d}")
