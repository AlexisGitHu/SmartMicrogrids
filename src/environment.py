import numpy as np

import pandas as pd
from random import choice as random_choice
from dataframe_handler import get_dataframe_from_file
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

    def adapt_state_vector_to_state(self, state_vector):
        final_state_vector = [[], [], []]
        for hour_vector in state_vector:
            final_state_vector[0].append(hour_vector[0])
            final_state_vector[1].append(hour_vector[1])
            final_state_vector[2].append(hour_vector[2])
        return final_state_vector

    def get_final_state_vector(self, month, day, hour):
        actual_final_state_vector = self.adapt_state_vector_to_state(
            self.create_state_vector(month, day, hour)
        )
        return actual_final_state_vector

    def get_next_value(self, month, day, hour):
        state_vector = self.adapt_state_vector_to_state(
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
        for i, vector in enumerate(self.actual_final_state_vector):
            vector.pop(0)
            vector.append(state_vector[i][0])

    def get_following_state_vector(self):
        if self.actual_month == 0:
            self.actual_month = random_choice(self.month_choices)
            self.actual_day = random_choice(self.day_choices[self.actual_month])
            self.actual_hour = random_choice(self.hour_choices)
            self.actual_final_state_vector = self.get_final_state_vector(
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
            self.get_next_value(self.actual_month, self.actual_day, self.actual_hour)

        # print(self.actual_month, self.actual_day, self.actual_hour)
        return self.actual_final_state_vector

    def reset_actual_date(self):
        self.actual_month = 0
        self.actual_day = 0
        self.actual_hour = 0


class State:
    def __init__(self, historic_data, trained_weeks=2, iteration_interval=24):
        self.actual_net_power: np.float64 = 0
        self.net_power_window: np.NDArray[np.float64] = np.zeros(23)
        self.actual_pv_power: np.float64 = 0
        self.pv_power_window: np.NDArray[np.float64] = np.zeros(23)
        self.actual_light_pvpc_power: np.float64 = 0
        self.light_pvpc_window: np.NDArray[np.float64] = np.zeros(23)
        self.battery_power = 0
        self.inverse_battery_power = 0
        self.historic_data = historic_data

        self.trained_week = trained_weeks
        self.iteration_interval = iteration_interval
        self.total = self.trained_week * self.iteration_interval
        self.hour_iterator = 0

    def update(self, action):
        pass

    def print_batch_ended(self):
        print("\n", sep=" ")
        print(f"Terminal state ended with:")
        print(f"    + PV generated: {self.actual_net_power} (kW)")
        print(f"    + Demand: {self.actual_pv_power} (kWh)")
        print(f"    + PVPC light: {self.actual_light_pvpc_power} (€/kWh)\n")

    def iterate_batch(self):
        if self.hour_iterator == self.total:
            self.historic_data.reset_actual_date()
            self.hour_iterator = 0
            self.print_batch_ended()
        else:
            self.hour_iterator += 1
            self.progress_bar(
                self.hour_iterator,
            )

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

    def next_state(self, action=0):
        state_vector = self.historic_data.get_following_state_vector()
        self.update_class_variables(state_vector)
        self.iterate_batch()

    def get_next_power_window(self):
        if self.hour_iterator == 0:
            self.hour_iterator = 1
            return self.net_power_window

    def read_next_state_from_data(self):
        pass


class Environment:
    def __init__(
        self,
        net_installed_power,
        pv_installed_power,
        battery_capacity,
        inverse_battery_capacity,
        dataframe_historic_data,
    ):

        self.net_installed_power = net_installed_power
        self.pv_installed_power = pv_installed_power
        self.battery_capacity = battery_capacity
        self.inverse_battery_capacity = inverse_battery_capacity
        self.state = State(historic_data_file_path=dataframe_historic_data)

    def __compute_reward(self):
        # Compute the reward
        reward = 0
        return reward

    def step(self, action):
        # Update the state
        self.state.update(action)

        # Compute the reward
        reward = self._compute_reward()

        return self.state, reward

    def reset(self):
        self.state = None


if __name__ == "__main__":
    # Load the data
    # net_installed_power = 100
    # pv_installed_power = 100
    # battery_capacity = 100
    # inverse_battery_capacity = 100

    # environment = Environment(
    #     net_installed_power,
    #     pv_installed_power,
    #     battery_capacity,
    #     inverse_battery_capacity,
    #     "datos/clean/merged_data.csv",
    # )

    # # Train the agent
    # for i in range(100):
    #     action = random_choice([0, 1])
    #     state, reward = environment.step(action)
    #     print(f"Step {i}: {state}, {reward}")

    # # Reset the environment
    # environment.reset()
    history = HistoricData("datos/clean/merged_data.csv")
    train_weeks = 2
    iteration_interval = 24
    state = State(history)
    for i in range(100000):
        state.next_state()
