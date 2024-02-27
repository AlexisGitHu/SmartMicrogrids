import numpy as np

import pandas as pd
from random import choice as random_choice
from dataframe_handler import get_dataframe_from_file
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs.utils import check_env_specs
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from collections import defaultdict
from torchrl.envs.transforms import TransformedEnv, UnsqueezeTransform, CatTensors
import tqdm
import copy
import time
import sys


class HistoricData:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = get_dataframe_from_file(file_path)
        self.month_choices = sorted(self.data["month"].unique())
        self.day_choices = self.get_day_possible_choices()
        self.hour_choices = sorted(self.data["hour"].unique())
        self.date_value_possible_pairs = self.get_all_values_from_date_tuple()

        self.actual_month = 0
        self.actual_day = 0
        self.actual_hour = 0
        self.actual_final_state_vector = []

    def get_day_possible_choices(self):
        day_choices = {}
        for month in self.month_choices:
            day_choices[month] = sorted(
                self.data[self.data["month"] == month]["day"].unique()
            )
        return day_choices

    def get_random_date(self):
        month = random_choice(self.month_choices)
        day = random_choice(self.day_choices[month])
        hour = random_choice(self.hour_choices)
        return month, day, hour

    def get_all_values_from_date_tuple(self):
        date_value_possible_pairs = {}
        for idx, row in self.data.iterrows():
            if not (row["month"], row["day"], row["hour"]) in date_value_possible_pairs:
                date_value_possible_pairs[(row["month"], row["day"], row["hour"])] = []
            date_value_possible_pairs[(row["month"], row["day"], row["hour"])].append(
                (row["P (kW)"], row["Consumo (kWh)"], row["PVPC (kWh)"])
            )
        return date_value_possible_pairs

    def next_23_hours(self, day, hour):
        next_23_hours = []
        for i in range(1, 24):
            next_hour = (self.actual_hour + i) % 24
            next_day = (self.actual_day + (self.actual_hour + i) // 24) % len(
                self.day_choices[self.actual_month]
            )
            if next_day == 0:
                next_day = 1
            next_month = (
                self.actual_month
                + (self.actual_day + (self.actual_hour + i) // 24)
                // len(self.day_choices[self.actual_month])
            ) % 12
            if next_month == 0:
                next_month = 1
            next_23_hours.append((next_month, next_day, next_hour))
        return next_23_hours

    def create_state_vector(self, month, day, hour):
        next_23_hours = self.next_23_hours(day, hour)
        state_vector = [
            random_choice(self.date_value_possible_pairs[(month, day, hour)])
        ]
        for next_month, next_day, next_hour in next_23_hours:
            state_vector.append(
                random_choice(
                    self.date_value_possible_pairs[(next_month, next_day, next_hour)]
                )
            )
        return state_vector

    def adapt_state_vector_to_state(self, date_tuple):
        final_state_vector = [[], [], []]
        for hour_vector in date_tuple:
            final_state_vector[0].append(hour_vector[0])
            final_state_vector[1].append(hour_vector[1])
            final_state_vector[2].append(hour_vector[2])
        return final_state_vector

    def get_final_state_vector(self, month, day, hour):
        actual_final_state_vector = self.adapt_state_vector_to_state(
            self.create_state_vector(month, day, hour)
        )
        return actual_final_state_vector

    def get_next_value(self, month, day, hour, state_vector):
        new_values = self.adapt_state_vector_to_state(
            [
                random_choice(
                    self.date_value_possible_pairs[
                        (
                            month,
                            day,
                            hour,
                        )
                    ]
                )
            ]
        )
        for i, vector in enumerate(state_vector):
            vector.pop(0)
            vector.append(new_values[i][0])
        return state_vector

    def get_following_state_vector(self, state_vector):
        if self.actual_month == 0:
            self.actual_month = random_choice(self.month_choices)
            self.actual_day = random_choice(self.day_choices[self.actual_month])
            self.actual_hour = random_choice(self.hour_choices)
            actual_final_state_vector = self.get_final_state_vector(
                self.actual_month, self.actual_day, self.actual_hour
            )
        else:
            next_hour = (self.actual_hour + 1) % 24
            next_day = (self.actual_day + (self.actual_hour + 1) // 24) % len(
                self.day_choices[self.actual_month]
            )
            if next_day == 0:
                next_day = 1
            next_month = (
                self.actual_month
                + (self.actual_day + (self.actual_hour + 1) // 24)
                // len(self.day_choices[self.actual_month])
            ) % 12
            if next_month == 0:
                next_month = 1
            self.actual_month = next_month
            self.actual_day = next_day
            self.actual_hour = next_hour
            actual_final_state_vector = self.get_next_value(
                self.actual_month, self.actual_day, self.actual_hour, state_vector
            )
            # print(f"Esto es none?: {actual_final_state_vector}")
        # print(self.actual_month, self.actual_day, self.actual_hour)
        return actual_final_state_vector

    def reset_actual_date(self):
        self.actual_month = 0
        self.actual_day = 0
        self.actual_hour = 0


class State:
    def __init__(
        self,
        historic_data,
        maximum_battery_power=2200,
        trained_weeks=2,
        iteration_interval=24,
    ):
        self.actual_net_power: np.float64 = 0
        self.net_power_window: np.NDArray[np.float64] = np.zeros(23).tolist()
        self.actual_pv_power: np.float64 = 0
        self.pv_power_window: np.NDArray[np.float64] = np.zeros(23).tolist()
        self.actual_light_pvpc_power: np.float64 = 0
        self.light_pvpc_window: np.NDArray[np.float64] = np.zeros(23).tolist()

        self.maximum_battery_power = maximum_battery_power
        self.actual_battery_power = 0
        self.inverse_battery_power = 0
        self.historic_data = historic_data

        self.state_vector = self.create_initial_state_vector()
        self.trained_week = trained_weeks
        self.iteration_interval = iteration_interval
        self.total = self.trained_week * self.iteration_interval
        self.hour_iterator = 0

    def create_initial_state_vector(self):
        state_vector = [
            self.net_power_window,
            self.pv_power_window,
            self.light_pvpc_window,
            [self.actual_battery_power],
        ]
        state_vector[0].insert(0, self.actual_net_power)
        state_vector[1].insert(0, self.actual_pv_power)
        state_vector[2].insert(0, self.actual_light_pvpc_power)
        return state_vector

    def print_batch_ended(self):
        # print("\n", sep=" ")
        # print(f"Terminal state ended with:")
        # print(f"    + PV generated: {self.actual_net_power} (kW)")
        # print(f"    + Demand: {self.actual_pv_power} (kWh)")
        # print(f"    + PVPC light: {self.actual_light_pvpc_power} (€/kWh)")
        # print(f"    + Battery power: {self.actual_battery_power} (kW)\n")
        pass

    def iterate_batch(self):
        if self.hour_iterator == self.total:
            self.historic_data.reset_actual_date()
            self.hour_iterator = 0
            self.print_batch_ended()
        else:
            self.hour_iterator += 1

    def update_class_variables(self, state_vector):
        self.actual_net_power = state_vector[0][0]
        self.net_power_window = np.array(state_vector[0][1:])
        self.actual_pv_power = state_vector[1][0]
        self.pv_power_window = np.array(state_vector[1][1:])
        self.actual_light_pvpc_power = state_vector[2][0]
        self.light_pvpc_window = np.array(state_vector[2][1:])

    def add_actual_battery_value(self, partial_state_vector):
        partial_state_vector.append([self.actual_battery_power])
        return partial_state_vector

    def next_state(self):
        partial_state_vector = self.historic_data.get_following_state_vector(
            self.state_vector[:3]
        )  # Historic data can only return 3 lists of values
        state_vector = self.add_actual_battery_value(partial_state_vector)
        self.update_class_variables(state_vector)
        self.iterate_batch()
        self.state_vector = state_vector
        tensor1 = torch.tensor([self.actual_net_power], dtype=torch.float32)
        tensor2 = torch.tensor(self.net_power_window, dtype=torch.float32).unsqueeze(0)
        tensor3 = torch.tensor([self.actual_pv_power], dtype=torch.float32)
        tensor4 = torch.tensor(self.pv_power_window, dtype=torch.float32).unsqueeze(0)
        tensor5 = torch.tensor([self.actual_light_pvpc_power], dtype=torch.float32)
        tensor6 = torch.tensor(self.light_pvpc_window, dtype=torch.float32).unsqueeze(0)
        tensor7 = torch.tensor([self.actual_battery_power], dtype=torch.float32)

        tensordict = TensorDict(
            {
                "net_power": tensor1,
                "net_power_window": tensor2,
                "pv_power": tensor3,
                "pv_power_window": tensor4,
                "light_pvpc_power": tensor5,
                "light_pvpc_window": tensor6,
                "battery_power": tensor7,
            },
            batch_size=[1],
        )  ## Batch_size value of first dimension of tensor.shape if
        return tensordict


class Environment(EnvBase):
    batch_locked = False

    def __init__(
        self,
        dataframe_historic_data,
        td_params=None,
        maximum_house_power=3.3,
        maximum_pv_power_generation=1.5,
        maximum_battery_power=2.2,
        device=None,
        seed=None,
        alpha=0.6,
    ):
        if td_params is None:
            td_params = self.gen_params()
        super().__init__(device=device, batch_size=[1])
        self.dataframe_historic_data = dataframe_historic_data
        self.state = State(historic_data=dataframe_historic_data)
        self.current_steps = 0
        self.current_observation = None
        self.alpha = alpha
        self.maximum_house_power = maximum_house_power
        self.maximum_pv_power_generation = maximum_pv_power_generation
        self.maximum_battery_power = maximum_battery_power
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)
        # Helpers: _make_step and gen_params
        gen_params = staticmethod(self.gen_params)
        _make_spec = self._make_spec

        # # Mandatory methods: _step, _reset and _set_seed
        _reset = self._reset
        _step = staticmethod(self._step)
        _set_seed = self._set_seed

    def __compute_reward(self, action):
        # Compute the reward
        p = self.state.state_vector[0][0]
        f = self.state.state_vector[1][0]
        reward = (
            -p + f + action["action"]
        ) * self.alpha  ## Esto es lo que hay que cambiar, converge a 0.
        return reward

    def _step(self, action: TensorDict):
        if self.current_steps == 1024:
            done = True
            self.current_steps = 0
        else:
            done = False
            self.current_steps += 1
        reward = self.__compute_reward(action)
        new_state = self.state.next_state()
        new_state["reward"] = torch.tensor([reward])
        new_state["done"] = torch.tensor([done])

        return new_state

    def _reset(self, tensordict: TensorDict = None):
        self.state = State(historic_data=self.dataframe_historic_data)
        self.current_observation = self.state.next_state()
        return self.current_observation

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _make_spec(self, td_params):
        self.observation_spec = CompositeSpec(
            net_power=BoundedTensorSpec(
                shape=(1), dtype=torch.float32, low=0, high=self.maximum_house_power
            ),
            net_power_window=BoundedTensorSpec(
                shape=(1, 23), dtype=torch.float32, low=0, high=self.maximum_house_power
            ),
            pv_power=BoundedTensorSpec(
                shape=(1),
                dtype=torch.float32,
                low=0,
                high=self.maximum_pv_power_generation,
            ),
            pv_power_window=BoundedTensorSpec(
                shape=(1, 23),
                dtype=torch.float32,
                low=0,
                high=self.maximum_pv_power_generation,
            ),
            light_pvpc_power=UnboundedContinuousTensorSpec(
                shape=(1), dtype=torch.float32
            ),
            light_pvpc_window=UnboundedContinuousTensorSpec(
                shape=(1, 23), dtype=torch.float32
            ),
            battery_power=BoundedTensorSpec(
                shape=(1),
                dtype=torch.float32,
                low=0,
                high=self.state.maximum_battery_power,
            ),
            shape=([1]),
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            low=-td_params["battery_power"],
            high=self.maximum_battery_power - td_params["battery_power"],
            shape=(1,),
            dtype=torch.float32,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))

    def gen_params(self, batch_size=None):
        if batch_size is None:
            batch_size = []
        td = TensorDict({"battery_power": torch.tensor([0.0])}, batch_size=[])
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td


def plot():
    import matplotlib
    from matplotlib import pyplot as plt

    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    with plt.ion():
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(logs["return"])
        plt.title("returns")
        plt.xlabel("iteration")
        plt.subplot(1, 2, 2)
        plt.plot(logs["last_reward"])
        plt.title("last reward")
        plt.xlabel("iteration")
        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        plt.show()


if __name__ == "__main__":
    history = HistoricData("datos/clean/merged_data.csv")
    # train_weeks = 2
    # iteration_interval = 24
    # state = State(history, 2.2)
    # for i in range(100000):
    #     state.next_state()
    # history = HistoricData("datos/clean/merged_data.csv")
    # env = Environment(history)
    # env.reset()
    # print(env.reset())

    env = Environment(dataframe_historic_data=history)
    env = TransformedEnv(
        env,
        # ``Unsqueeze`` the observations that we will concatenate
        UnsqueezeTransform(
            unsqueeze_dim=-1,
            in_keys=[
                "net_power",
                "net_power_window",
                "pv_power",
                "pv_power_window",
                "light_pvpc_power",
                "light_pvpc_window",
                "battery_power",
            ],
            in_keys_inv=[
                "net_power",
                "net_power_window",
                "pv_power",
                "pv_power_window",
                "light_pvpc_power",
                "light_pvpc_window",
                "battery_power",
            ],
        ),
    )
    # cat_transform = CatTensors(
    #     in_keys=[
    #         "net_power",
    #         "net_power_window",
    #         "pv_power",
    #         "pv_power_window",
    #         "light_pvpc_power",
    #         "light_pvpc_window",
    #         "battery_power",
    #     ],
    #     dim=-1,
    #     out_key="observation",
    #     del_keys=False,
    # )
    # env.append_transform(cat_transform)
    check_env_specs(env)
    # fake_tensordict = env.fake_tensordict()
    # print(fake_tensordict.batch_size)
    # real_tensordict=env.rollout(3,return_contiguous=True)
    # print(real_tensordict)
    # print(real_tensordict.batch_dims)
    # print("observation_spec:", env.observation_spec)
    # print("state_spec:", env.state_spec)
    # print("reward_spec:", env.reward_spec)

    # tensordict=env.reset()
    # tensordict["action"]=torch.tensor([0.0])
    # tensordict=env.step(tensordict)
    # tensordict["action"]=torch.tensor([0.0])
    # env.step(tensordict)

    # print("observation_spec:", env.observation_spec)
    # print("state_spec:", env.state_spec)
    # print("reward_spec:", env.reward_spec)
    # td = env.reset()
    # print("reset tensordict", td)

    # for i in range(100000):
    #     env.step(0)
