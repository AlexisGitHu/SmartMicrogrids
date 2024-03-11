import numpy as np

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from environment import State, HistoricData,StatesTendency
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from torch import device
import uuid

class Environment(gym.Env):
    def __init__(
        self,
        dataframe_historic_data=HistoricData("datos/clean/merged_data.csv"),
        state_length=7,
        maximum_battery_capacity=2.2,
    ):
        super(Environment, self).__init__()
        self.action_space=gym.spaces.Box(low=-maximum_battery_capacity, high=maximum_battery_capacity, shape=(1,), dtype=np.float32)
        self.observation_space=gym.spaces.Box(low=-100, high=100, shape=(state_length,), dtype=np.float32)

        self.reward_range=(-100, 100)
        self.maximum_battery_capacity = maximum_battery_capacity
        self.dataframe_historic_data = dataframe_historic_data

        self.state =StatesTendency( State(
            historic_data=dataframe_historic_data,
            maximum_battery_power=maximum_battery_capacity,
        ))
        self.current_observation = None

    def flatten(self, xss):
        end=[x for xs in xss for x in xs]
        return end

    def __compute_reward(self, action):
        p = self.state.get_actual_generation()
        f = self.state.get_actual_demand()
        light_pvpc = self.state.get_actual_pvpc()
        battery_capacity = self.state.get_actual_battery_capacity()
        action_value=action[0]
        if p - f > 0:  ## Si la generación es mayor que la demanda
            if action_value < 0:  ## Si decido descargar la batería
                # print("CASO 1")
                reward = 0
            else:
                if (
                    battery_capacity >= self.maximum_battery_capacity
                ):  ## Si la batería está llena
                    
                    if action_value < 1e-4 and action_value > -1e-4: # This avoid probability 0 of event
                        # print("CASO 2")
                        reward = action_value * light_pvpc
                    else:
                        # print("CASO 3")
                        reward = 0
                else:
                    if action_value + battery_capacity > self.maximum_battery_capacity:
                        # print("CASO 5")
                        reward = 0
                        # reward=p-f*light_pvpc-(p-f-action_value)*light_pvpc
                    else:
                        # print("CASO 6")
                        reward = action_value*light_pvpc
        elif p - f < 0:  ## Si la generación es menor que la demanda
            if action_value < 0:
                if battery_capacity!=0:
                    if action_value < -battery_capacity:
                        # print("CASO 7")
                        reward = 0 ## Penalización por descargar demasiado, te premio por descargar algo
                    else:
                        if -action_value+p <= f:
                            reward=-action_value*light_pvpc
                        else:
                            # print("CASO 10")
                            reward=0
                else:
                    reward = 0
            else:
                # print("CASO 12")
                reward = -action_value*light_pvpc 
        else:
            if action_value > 0:
                # print("CASO 13")
                reward = -action_value*light_pvpc ## Penalización por almacenar
            elif action_value < 0:
                # print("CASO 14")
                reward = 0
            else:
                # print("CASO 15")
                reward = 0
        # print(f"Estados: ")
        # print(f"    + PV generated: {p} (kW)")
        # print(f"    + Demand: {f} (kWh)")
        # print(f"    + PVPC light: {light_pvpc} (€/kWh)")
        # print(f"    + Battery power: {battery_capacity} (kW)")
        # print(f"    + Action: {action} (kW)")
        # print(f"    + Reward: {reward} (€/kWh)\n")
        return reward

    def step(self, action):
        terminated=False
        truncated=False
        reward = self.__compute_reward(action)
        new_state = np.array(self.flatten(self.state.next_state(action)),dtype=np.float32)

        return new_state, reward, terminated, truncated, {}

    def reset(self,seed=123,options=None):
        super().reset(seed=seed)
        self.state.reset_state()
        self.current_observation = np.array(self.flatten(self.state.next_state()),dtype=np.float32)
        return self.current_observation, {}
    
    def render(self, action, mode='human'):
        # Render the environment
        # The details of this method will depend on the nature of your environment
        # For a text-based environment, you could simply print some information
        print("Rendering the environment")
    
    def close(self):
        print("Environment is closed")

    def print_actual_state(self,action):
        p = self.state.get_actual_generation()
        f = self.state.get_actual_demand()
        light_pvpc = self.state.get_actual_pvpc()
        battery_capacity = self.state.get_actual_battery_capacity()
        print(f"Estados: ")
        print(f"    + PV generated: {p} (kW)")
        print(f"    + Demand: {f} (kWh)")
        print(f"    + PVPC light: {light_pvpc} (€/kWh)")
        print(f"    + Battery power: {battery_capacity} (kW)")
        print(f"    + Action: {action[0][0]} (kW)")


if __name__ == "__main__":
    tensorboard_logs = "logs/tensorboard/"
    policy_string="MlpPolicy"
    type_of_trial="do_nothing_env"
    trial_name=str(uuid.uuid4())
    device_cpu = device("cpu")
    gym.register("microgrid-v0", entry_point="gym_env:Environment")
    env=gym.make("microgrid-v0")
    env=TimeLimit(env, max_episode_steps=1024)
    monitor_env = Monitor(env, tensorboard_logs)
    model=PPO(policy_string, monitor_env, verbose=1, tensorboard_log=tensorboard_logs,device=device_cpu)
    summary_writer = SummaryWriter(f"{tensorboard_logs}/cumulative_reward/{type_of_trial}/{trial_name}")
    model.learn(total_timesteps=1000000,tb_log_name=f"{type_of_trial}/{policy_string}_{trial_name}")
    model.save(f"models/{type_of_trial}/{policy_string}_{trial_name}")
    vec_env = model.get_env()
    obs=vec_env.reset()
    cumulatve_reward=0
    num_steps=1000
    for i in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        vec_env.env_method('print_actual_state',action)
        obs, reward, done, info = vec_env.step(action)
        print(f"    + Reward: {reward[0]} (€/kWh)\n")

        cumulatve_reward+=reward[0]
        vec_env.render()
    print(f"Cumulative reward: {cumulatve_reward/num_steps} (€/kWh)")
    
    summary_writer.add_scalar('Final reward', cumulatve_reward/num_steps)

    env.close()

    # env.reset()
    # for i in range(100000):
    #     env.step(0)
