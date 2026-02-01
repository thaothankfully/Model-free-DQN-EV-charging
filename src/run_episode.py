import numpy as np

from src import config

def run_episode(agent, env, date=None):
    
    # Initialize environment
    state = env.reset(date=date)

    # Store actions, SOCs and rewards sequences
    actions = []
    socs = [state['soc']]
    rewards = []

    done = False
    score = 0
    while not done:
        action_id = agent.select_action(state)
        next_state, reward, done = env.step(config.ACTIONS[action_id])
        score += reward
        
        # Store action, SOC and reward
        actions.append(config.ACTIONS[action_id])
        socs.append(next_state['soc'])
        rewards.append(reward)

        state = next_state.copy()

    return np.array(actions), np.array(socs), np.array(rewards), score
