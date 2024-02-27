import numpy as np

import pandas as pd
from random import choice as random_choice
from dataframe_handler import get_dataframe_from_file
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
            # self.progress_bar(
            #     self.hour_iterator,
            # )

    def progress_bar(
        self,
        iteration,
        prefix="",
        suffix="",
        length=50,
        fill="█",
        print_end="\r",
    ):
        percent = f"{iteration}"
        filled_length = int(length * iteration // self.total)
        bar = fill * filled_length + "-" * (length - filled_length)
        sys.stdout.write(
            "\r%s |%s| %s/%s %s" % (prefix, bar, percent, self.total, suffix)
        )
        sys.stdout.flush()

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

    def next_state(self,action=None):
        partial_state_vector = self.historic_data.get_following_state_vector(
            self.state_vector[:3]
        )  # Historic data can only return 3 lists of values
        state_vector = self.add_actual_battery_value(partial_state_vector)
        if action is not None:
            self.actual_battery_power+=action
        self.update_class_variables(state_vector)
        self.iterate_batch()
        self.state_vector = copy.deepcopy(state_vector)
        return state_vector


class Environment:
    def __init__(
        self,
        dataframe_historic_data,
        battery_capacity=2.2,
        alpha=0.6,
    ):

        # self.net_installed_power = net_installed_power
        # self.pv_installed_power = pv_installed_power
        self.battery_capacity = battery_capacity
        # self.inverse_battery_capacity = inverse_battery_capacity
        self.dataframe_historic_data = dataframe_historic_data
        self.state = State(historic_data=dataframe_historic_data)
        self.current_steps = 0
        self.current_observation = None
        self.alpha = alpha

    def flatten(self, xss):
        return [x for xs in xss for x in xs]

    def __compute_reward(self, action):
        # Compute the reward
        p = self.state.state_vector[0][0]
        f = self.state.state_vector[1][0]
        light_pvpc = self.state.state_vector[2][0]
        battery_capacity_difference=self.state.state_vector[3][0]

        reward=(p-f-battery_capacity_difference)*light_pvpc

        return reward

    def step(self, action):
        if self.current_steps == 1024:
            done = True
            self.current_steps = 0
        else:
            done = False
            self.current_steps += 1
        print(action)
        reward = self.__compute_reward(action)
        new_state = self.flatten(self.state.next_state(action))

        return new_state, reward, done

    def reset(self):
        self.state = State(historic_data=self.dataframe_historic_data)
        self.current_observation = self.flatten(self.state.next_state())
        return self.current_observation


if __name__ == "__main__":
    # history = HistoricData("datos/clean/merged_data.csv")
    # train_weeks = 2
    # iteration_interval = 24
    # state = State(history, 2.2)
    # for i in range(100000):
    #     state.next_state()
    history = HistoricData("datos/clean/merged_data.csv")
    env = Environment(history)
    env.reset()
    for i in range(100000):
        env.step(0)
