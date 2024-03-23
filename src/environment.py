import numpy as np

from random import choice as random_choice
from dataframe_handler import get_dataframe_from_file
import copy


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
            print(f"Ya han pasado 1024h")
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
            
        return actual_final_state_vector

    def reset_actual_date(self):
        self.actual_month = 0
        self.actual_day = 0
        self.actual_hour = 0


class State:
    def __init__(
        self,
        historic_data,
        maximum_battery_power=2.2,
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


    def update_class_variables(self, state_vector,action):
        self.actual_net_power = state_vector[0][0]
        self.net_power_window = state_vector[0][1:]
        self.actual_pv_power = state_vector[1][0]
        self.pv_power_window = state_vector[1][1:]
        self.actual_light_pvpc_power = state_vector[2][0]
        self.light_pvpc_window = state_vector[2][1:]
        if action is not None:
            if self.actual_battery_power + action[0] < 0:
                self.actual_battery_power = 0
            elif self.actual_battery_power + action[0] > self.maximum_battery_power:
                self.actual_battery_power = self.maximum_battery_power
            else:
                self.actual_battery_power += action[0]

                

    def add_actual_battery_value(self, partial_state_vector):
        partial_state_vector.append([self.actual_battery_power])
        return partial_state_vector


    def next_state(self, action=None):
        partial_state_vector = self.historic_data.get_following_state_vector(
            self.state_vector[:3]
        )  # Historic data can only return 3 lists of values
        self.update_class_variables(partial_state_vector,action)
        state_vector = self.add_actual_battery_value(partial_state_vector)
        self.state_vector = copy.deepcopy(state_vector)
        
        return state_vector

    def reset_state(self):
        self.actual_net_power = 0
        self.net_power_window = np.zeros(23).tolist()
        self.actual_pv_power = 0
        self.pv_power_window = np.zeros(23).tolist()
        self.actual_light_pvpc_power = 0
        self.light_pvpc_window = np.zeros(23).tolist()
        self.actual_battery_power = 0
        self.state_vector = self.create_initial_state_vector()
        self.historic_data.reset_actual_date()

    def get_actual_generation(self):
        return self.state_vector[0][0]
    
    def get_actual_demand(self):
        return self.state_vector[1][0]
    
    def get_actual_pvpc(self):
        return self.state_vector[2][0]
    
    def get_actual_battery_capacity(self):
        return self.state_vector[3][0]

class StatesTendency:
    def __init__(self, state):
        self.state = state
        self.mwa_pvpc=0
        self.state_vector_tendencies: np.NDArray[np.float64] = np.zeros(50).tolist()

    def substitue_window_to_mwa(self,state_vector):
        actual_pvpc=state_vector[2][0]
        state_vector[2]=[actual_pvpc,self.mwa_pvpc]
        return state_vector
    
    def substitute_window_to_tendencies(self,state_vector):
        self.mwa_generation=self.transform_window_to_moving_average(state_vector[0][1:],len(state_vector[0][1:]))
        self.mwa_demand=self.transform_window_to_moving_average(state_vector[1][1:],len(state_vector[1][1:]))
        self.mwa_pvpc=self.transform_window_to_moving_average(state_vector[2][1:],len(state_vector[2][1:]))
        actual_generation=state_vector[0][0]
        state_vector[0]=[actual_generation,self.mwa_generation]
        actual_demand=state_vector[1][0]
        state_vector[1]=[actual_demand,self.mwa_demand]
        actual_pvpc=state_vector[2][0]
        state_vector[2]=[actual_pvpc,self.mwa_pvpc]

        return state_vector

    def transform_window_to_moving_average(self, window, window_size=23):
        assert window_size < 24
        length = len(window[1:window_size+1])
        average=0
        for i in range(length):
            average += window[i+1]*(i+1)/(window_size*(window_size+1)/2)

        return np.float16(average)
    
    
    def next_state(self,action=None):
        state_vector=self.state.next_state(action)
        state_vector=self.substitute_window_to_tendencies(state_vector)
        return state_vector
    
    def reset_state(self):
        self.state.reset_state()
        self.mwa_pvpc=0
    
    def get_actual_generation(self):
        return self.state.state_vector[0][0]
    
    def get_actual_demand(self):
        return self.state.state_vector[1][0]
    
    def get_actual_pvpc(self):
        return self.state.state_vector[2][0]
    
    def get_actual_battery_capacity(self):
        return self.state.state_vector[3][0]
    


if __name__ == "__main__":
    historical=HistoricData("datos/clean/merged_data.csv")
    state=State(historical)
    states_tendency=StatesTendency(state)
    states_tendency.next_state()
    states_tendency.reset_state()
    states_tendency.next_state()

