import time
import numpy as np
from scipy import stats
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from environment import State, HistoricData, StatesTendency
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO, TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from torch.utils.tensorboard import SummaryWriter
from torch import device
import uuid
import yaml
import pandas as pd
from pingouin import multivariate_normality

with open("config/env_parameters.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


class Environment(gym.Env):
    def __init__(
        self,
        dataframe_historic_data=HistoricData("datos/clean/merged_data.csv"),
        state_length=7,
        maximum_battery_capacity=2.2,
        states_tendency=True,
    ):
        super(Environment, self).__init__()
        if states_tendency:
            self.state = StatesTendency(
                State(
                    historic_data=dataframe_historic_data,
                    maximum_battery_power=maximum_battery_capacity,
                )
            )
            state_length = 7
        else:
            self.state = State(
                historic_data=dataframe_historic_data,
                maximum_battery_power=maximum_battery_capacity,
            )
            state_length = 73

        self.action_space = gym.spaces.Box(
            low=-maximum_battery_capacity,
            high=maximum_battery_capacity,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(state_length,), dtype=np.float32
        )

        self.reward_range = (-100, 100)
        self.maximum_battery_capacity = maximum_battery_capacity
        self.dataframe_historic_data = dataframe_historic_data

        self.current_observation = None

    def flatten(self, xss):
        end = [x for xs in xss for x in xs]
        return end

    def __compute_reward(self, action):
        p = self.state.get_actual_generation()
        f = self.state.get_actual_demand()
        light_pvpc = self.state.get_actual_pvpc()
        battery_capacity = self.state.get_actual_battery_capacity()
        action_value = action[0]
        if p - f > 0:
            if action_value < 0:
                reward = action_value * light_pvpc
            else:
                if battery_capacity >= self.maximum_battery_capacity:
                    reward = -action_value * light_pvpc
                else:
                    if action_value > p - f:
                        reward = (p - f) * light_pvpc + (
                            p - f - action_value
                        ) * light_pvpc
                    else:
                        reward = action_value * light_pvpc
        elif p - f < 0:
            if battery_capacity != 0:
                if action_value < 0:
                    if action_value < -battery_capacity:
                        if action_value < p - f:
                            reward = (
                                -(p - f) * light_pvpc
                                + (battery_capacity + action_value) * light_pvpc
                            )
                        else:
                            reward = (
                                -(action_value) * light_pvpc
                                + (battery_capacity + action_value) * light_pvpc
                            )
                    else:
                        if action_value < p - f:
                            reward = (
                                -(p - f) * light_pvpc
                                - (p - f - action_value) * light_pvpc
                            )
                        else:
                            reward = -action_value * light_pvpc
                else:
                    reward = -action_value * light_pvpc
            else:
                reward = (p - f) * light_pvpc
        else:
            reward = 0
        return reward

    def step(self, action):
        terminated = False
        truncated = False
        reward = self.__compute_reward(action)
        new_state = np.array(
            self.flatten(self.state.next_state(action)), dtype=np.float32
        )

        return new_state, reward, terminated, truncated, {}

    def reset(self, seed=123, options=None):
        super().reset(seed=seed)
        self.state.reset_state()
        self.current_observation = np.array(
            self.flatten(self.state.next_state()), dtype=np.float32
        )
        return self.current_observation, {}

    def render(self, action=None, mode="human"):
        print("Rendering the environment")

    def close(self):
        print("Environment is closed")

    def print_actual_state(self, action):
        p = self.state.get_actual_generation()
        f = self.state.get_actual_demand()
        light_pvpc = self.state.get_actual_pvpc()
        battery_capacity = self.state.get_actual_battery_capacity()
        print(f"Estados: ")
        print(f"    + Generación: {p} (kWh)")
        print(f"    + Demanda: {f} (kWh)")
        print(f"    + PVPC: {light_pvpc} (€/kWh)")
        print(f"    + Capacidad batería: {battery_capacity} (kW)")
        print(f"    + Acción: {action[0][0]} (kW)")


def dummy_agent(
    monitor_env,
    tensorboard_logs,
    type_of_trial,
    trial_name,
    algorithm_name="DummyAgent",
):
    summary_writer = SummaryWriter(
        f"{tensorboard_logs}/cumulative_reward/{algorithm_name}/{type_of_trial}/{trial_name}"
    )
    values_for_mean = []
    for i in range(2000):
        obs = monitor_env.reset()
        cumulatve_reward = 0
        num_steps = 1024
        for j in range(num_steps):
            action, _states = dummy_predict(obs)
            monitor_env.print_actual_state([[action]])
            obs, reward, terminated, truncated, info = monitor_env.step([action])
            obs = (np.array(obs, dtype=np.float32), {})
            print(f"    + Reward: {reward} (€/kWh)\n")

            cumulatve_reward += reward
            monitor_env.render()
        values_for_mean.append(cumulatve_reward / num_steps)
        print(f"Cumulative reward: {cumulatve_reward/num_steps} (€/kWh)")
    print(f"Mean reward dummy agent: {np.mean(values_for_mean)} (€/kWh)")
    summary_writer.add_scalar("Final 1000 iters mean reward", np.mean(values_for_mean))


def dummy_predict(obs):
    p = obs[0][0]
    f = obs[0][2]
    light_pvpc = obs[0][4]
    battery_capacity = obs[0][6]
    if p - f > 0:  ## Si la generación es mayor que la demanda
        action = p - f
    elif p - f < 0:  ## Si la generación es menor que la demanda
        if battery_capacity != 0:
            if -battery_capacity < p - f:
                action = p - f
            else:
                action = -battery_capacity
        else:
            action = 0
    else:
        action = 0
    return action, None


def intelligent_agent(
    model,
    monitor_env,
    policy_string,
    type_of_trial,
    trial_name,
    states_tendency_str,
    algorithm_name="PPO",
):
    eval_callback = EvalCallback(
        monitor_env,
        best_model_save_path=f"./models/best_model_{algorithm_name}/{type_of_trial}/{states_tendency_str}/{trial_name}",
        log_path=f"./logs/eval_logs_{algorithm_name}/{type_of_trial}/{states_tendency_str}/{trial_name}",
        eval_freq=20000,
        deterministic=False,
        render=False,
    )
    model.learn(
        total_timesteps=2000000,
        tb_log_name=f"{algorithm_name}/{type_of_trial}/{states_tendency_str}/{policy_string}_{trial_name}",
        callback=eval_callback,
    )
    vec_env = model.get_env()
    obs = vec_env.reset()
    cumulatve_reward = 0
    num_steps = 1024
    for i in range(num_steps):
        action, _states = model.predict(obs, deterministic=False)
        vec_env.env_method("print_actual_state", action)
        obs, reward, done, info = vec_env.step(action)
        print(f"    + Reward: {reward[0]} (€/kWh)\n")

        cumulatve_reward += reward[0]
        vec_env.render()
    print(f"Cumulative reward: {cumulatve_reward/num_steps} (€/kWh)")

    return f"./models/best_model_{algorithm_name}/{type_of_trial}/{states_tendency_str}/{trial_name}/best_model"


def try_best_intelligent_agent(
    model_trained,
    best_model_path,
    monitor_env,
    device_cpu,
    tensorboard_logs,
    type_of_trial,
    states_tendency_str,
    algorithm_name,
):
    summary_writer = SummaryWriter(
        f"{tensorboard_logs}/cumulative_reward/{algorithm_name}/{type_of_trial}/{states_tendency_str}/{trial_name}"
    )
    model = model_trained.load(best_model_path, env=monitor_env, device=device_cpu)
    values_for_mean = []
    vec_env = model.get_env()
    obs = vec_env.reset()
    num_steps = 1024
    for i in range(1000):
        cumulatve_reward = 0
        obs = vec_env.reset()
        for j in range(num_steps):
            action, _states = model.predict(obs, deterministic=False)
            vec_env.env_method("print_actual_state", action)
            obs, reward, done, info = vec_env.step(action)
            print(f"    + Recompensa: {reward[0]} (€/kWh)\n")

            cumulatve_reward += reward[0]
            vec_env.render()
        values_for_mean.append(cumulatve_reward / num_steps)
        print(f"Cumulative reward: {cumulatve_reward/num_steps} (€/kWh)")
    print(f"Mean reward best agent: {np.mean(values_for_mean)} (€/kWh)")
    summary_writer.add_scalar("Final 1000 iters mean reward", np.mean(values_for_mean))


def recurrent_intelligent_agent(
    monitor_env, policy_string, type_of_trial, trial_name, tensorboard_logs, device_cpu
):
    policy_string = "MlpLstmPolicy"
    model = RecurrentPPO(
        policy_string,
        monitor_env,
        verbose=1,
        tensorboard_log=tensorboard_logs,
        device=device_cpu,
    )
    summary_writer = SummaryWriter(
        f"{tensorboard_logs}/cumulative_reward_recurrent/{type_of_trial}/{trial_name}"
    )
    eval_callback = EvalCallback(
        monitor_env,
        best_model_save_path=f"./models/best_model_recurrent/{type_of_trial}",
        log_path=f"./logs/eval_logs_recurrent/{type_of_trial}",
        eval_freq=20000,
        deterministic=False,
        render=False,
    )
    model.learn(
        total_timesteps=2000000,
        tb_log_name=f"{type_of_trial}/{policy_string}_{trial_name}",
        callback=eval_callback,
    )
    vec_env = model.get_env()
    obs = vec_env.reset()
    cumulatve_reward = 0
    num_steps = 1024
    for i in range(num_steps):
        action, _states = model.predict(obs, deterministic=False)
        vec_env.env_method("print_actual_state", action)
        obs, reward, done, info = vec_env.step(action)
        # print(f"    + Reward: {reward[0]} (€/kWh)\n")

        cumulatve_reward += reward[0]
        vec_env.render()
    print(f"Cumulative reward: {cumulatve_reward/num_steps} (€/kWh)")

    summary_writer.add_scalar("Final reward", cumulatve_reward / num_steps)


def samples_from_predictions(model_trained, best_model_path, monitor_env, device_cpu):
    model = model_trained.load(best_model_path, env=monitor_env, device=device_cpu)
    vector_state_buffer_for = []
    final_states_list = []
    vec_env = model.get_env()
    obs = vec_env.reset()
    range_data = 72
    iterations = 1000
    num_steps = 1024
    for i in range(iterations):
        obs = vec_env.reset()
        vector_state_buffer_for.append(obs[0][0:range_data])
        for j in range(num_steps):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env.step(action)
            if j == num_steps - 1 and i == iterations - 1:
                pass
            else:
                vector_state_buffer_for.append(obs[0][0:range_data])
            vec_env.render()
    for i in vector_state_buffer_for:
        final_states_list.append(i.tolist())
    return final_states_list


def check_normality(
    model,
    monitor_env,
    device_cpu,
    best_model_final_path,
):

    vector_state_buffer = samples_from_predictions(
        model, best_model_final_path, monitor_env, device_cpu
    )
    df = pd.DataFrame(vector_state_buffer)
    print(df)
    result = multivariate_normality(df)
    print(result)


def levenes_test_actions(tensorboard_logs, monitor_env, device_cpu):
    best_model_ppo = "models/best_model_PPO/pos_neg_rewards/StatesTendency/d8dc1a6f-8f36-4499-9927-28e42744040f/best_model.zip"
    best_model_trpo = "models/best_model_TRPO/pos_neg_rewards/StatesTendency/33c8d54c-c839-4e96-9863-cae353ad6a38/best_model.zip"
    best_model_a2c = "models/best_model_A2C/pos_neg_rewards/StatesTendency/a8764d89-ee33-411c-a16b-8c0db8b88ab7/best_model.zip"
    # best_model_recurrent_ppo = "models/best_model_RecurrentPPO/pos_neg_rewards/States/4ea6fa6e-0929-4408-a10d-20edea98a653/best_model.zip"
    best_models_list = [
        best_model_a2c,
        best_model_ppo,
        best_model_trpo,
    ]
    columns = 5
    actions_columns_list = []
    for best_model in best_models_list:
        if "RecurrentPPO" in best_model:
            policy_string = "MlpLstmPolicy"
            algorithm_name = "RecurrentPPO"
        else:
            policy_string = "MlpPolicy"
            if "TRPO" in best_model:
                algorithm_name = "TRPO"
            elif "PPO" in best_model:
                algorithm_name = "PPO"
            elif "A2C" in best_model:
                algorithm_name = "A2C"
        model = eval(algorithm_name)(
            policy_string,
            monitor_env,
            verbose=1,
            tensorboard_log=tensorboard_logs,
            device=device,
        )
        vector_state_buffer = samples_from_predictions(
            model, best_model, monitor_env, device_cpu
        )
        df = pd.DataFrame(vector_state_buffer)
        float_list = pd.to_numeric(df[columns], errors="coerce").dropna().tolist()
        actions_columns_list.append(np.array(float_list))

    stat, p = stats.levene(*actions_columns_list)
    print("Statistic:", stat)
    print("p-value:", p)


if __name__ == "__main__":
    tensorboard_logs = cfg["TENSORBOARD_LOGS"]
    policy_string = cfg["POLICY_STRING"]
    type_of_trial = cfg["TYPE_OF_TRIAL"]
    algorithm_name = cfg["ALGORITHM_NAME"]
    states_tendency_str = "StatesTendency" if cfg["STATES_TENDENCY"] else "States"
    trial_name = str(uuid.uuid4())
    device = device(cfg["DEVICE"])
    best_model_final_path = cfg["BEST_MODEL_FINAL_PATH"]
    gym.register(
        "microgrid-v0",
        entry_point="gym_env:Environment",
        kwargs={
            "states_tendency": cfg["STATES_TENDENCY"],
            "maximum_battery_capacity": cfg["MAXIMUM_BATTERY_CAPACITY"],
        },
    )
    env = gym.make("microgrid-v0")
    env = TimeLimit(env, max_episode_steps=1024)
    monitor_env = Monitor(env, tensorboard_logs)
    model = eval(algorithm_name)(
        policy_string,
        monitor_env,
        verbose=1,
        tensorboard_log=tensorboard_logs,
        device=device,
    )
    best_model_final_path = intelligent_agent(
        model,
        monitor_env,
        policy_string,
        type_of_trial,
        trial_name,
        states_tendency_str,
        algorithm_name=algorithm_name,
    )
    try_best_intelligent_agent(
        model,
        best_model_final_path,
        monitor_env,
        device,
        tensorboard_logs,
        type_of_trial,
        states_tendency_str,
        algorithm_name=algorithm_name,
    )
