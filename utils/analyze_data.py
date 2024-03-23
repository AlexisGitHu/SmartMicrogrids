import pandas as pd
from pingouin import multivariate_normality 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import random


class DataAnalyzerConsumption:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, sep=";", header=0)
        self.dates = pd.to_datetime(self.data["Time"], format="%Y-%m-%d %H:%M:%S")
        self.id_mapping = {}

    def describe_data(self):
        print(self.data.describe())

    def plot_time_series_power_consumption(self):
        ids = self.data["ID"].unique()
        for id in ids:
            self.id_mapping[id] = id.split("_")[0]

        self.data["Time"] = pd.to_datetime(
            self.data["Time"], format="%Y-%m-%d %H:%M:%S"
        )
        self.data["Year"] = self.data["Time"].dt.month
        self.data["Month"] = self.data["Time"].dt.year

        self.data["Consumo (kWh)"] = (
            self.data["Consumo (kWh)"].str.replace(",", ".").astype(float)
        )
        grouped_data = self.data.groupby(["ID"])
        # Get unique IDs
        for group_name, group_data in grouped_data:
            y_min = group_data["Consumo (kWh)"].min()
            y_max = group_data["Consumo (kWh)"].max()
            fig, ax = plt.subplots()
            id_data = grouped_data.get_group(group_name[0])
            print(id_data)
            for (month, year), group_data2 in id_data.groupby(["Year", "Month"]):
                month_str = pd.Timestamp(year=year, month=month, day=1).strftime(
                    "%b %Y"
                )
                print(group_data2["Consumo (kWh)"])
                ax.set_ylim(y_min, y_max)
                ax.yaxis.set_major_locator(plt.LinearLocator())
                ax.set_title(f"Consumo residencial: {self.id_mapping[group_name[0]]}")
                ax.legend().set_visible(False)
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Consumo (kWh)")
                ax.plot(
                    group_data2["Time"],
                    group_data2["Consumo (kWh)"],
                    label=month_str,
                    color="#1f77b4",
                )
            plt.xticks(rotation=45)
            plt.savefig(f"graficos/consumo/plot_{group_name[0]}.png", bbox_inches='tight')
            plt.show()


class DataAnalyzerPvpc:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, sep=";", header=0)
        self.dates = pd.to_datetime(self.data["Time"], format="%Y-%m-%d %H:%M:%S")
        self.id_mapping = {}

    def describe_data(self):
        print(self.data.describe())

    def plot_time_series(self):
        self.data["Time"] = pd.to_datetime(
            self.data["Time"], format="%Y-%m-%d %H:%M:%S"
        )
        self.data["Year"] = self.data["Time"].dt.month
        self.data["Month"] = self.data["Time"].dt.year
        self.data["PVPC (kWh)"] = self.data["PVPC (kWh)"].astype(float)
        y_min = self.data["PVPC (kWh)"].min()
        y_max = self.data["PVPC (kWh)"].max()
        fig, ax = plt.subplots()
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.LinearLocator())
        for (month, year), group_data2 in self.data.groupby(["Year", "Month"]):
            month_str = pd.Timestamp(year=year, month=month, day=1).strftime("%b %Y")
            print(group_data2["PVPC (kWh)"])
            ax.plot(
                group_data2["Time"],
                group_data2["PVPC (kWh)"],
                label=month_str,
                color="#1f77b4",
            )
        ax.set_title(f"PVPC - Peninsula Ibérica")
        ax.legend().set_visible(False)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("PVPC (€/kWh)")
        plt.xticks(rotation=45)
        plt.savefig(f"graficos/pvpc/pvpc_time_series.png", bbox_inches='tight')
        plt.show()


class DataAnalyzerRadiation:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, sep=";", header=0)
        self.dates = pd.to_datetime(self.data["Time"], format="%Y-%m-%d %H:%M:%S")
        self.id_mapping = {}

    def describe_data(self):
        print(self.data.describe())

    def plot_time_series(self):
        self.data["Time"] = pd.to_datetime(
            self.data["Time"], format="%Y-%m-%d %H:%M:%S"
        )
        self.data["Year"] = self.data["Time"].dt.month
        self.data["Month"] = self.data["Time"].dt.year
        self.data["P (kW)"] = self.data["P (kW)"].astype(float)
        y_min = self.data["P (kW)"].min()
        y_max = self.data["P (kW)"].max()
        fig, ax = plt.subplots()
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.LinearLocator())
        for (month, year), group_data2 in self.data.groupby(["Year", "Month"]):
            month_str = pd.Timestamp(year=year, month=month, day=1).strftime("%b %Y")
            print(group_data2["P (kW)"])
            ax.plot(
                group_data2["Time"],
                group_data2["P (kW)"],
                label=month_str,
                color="#1f77b4",
            )
        ax.set_title(f"Generación fotovoltaica - Madrid")
        ax.legend().set_visible(False)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("P (kWh)")
        plt.xticks(rotation=45)
        plt.savefig(f"graficos/radiacion/radiacion_time_series.png", bbox_inches='tight')
        plt.show()



if __name__ == "__main__":
    data_consumption = DataAnalyzerConsumption(
        "datos/clean/merged_power_consumption.csv"
    )
    data_pvpc = DataAnalyzerPvpc("datos/clean/merged_pvpc_light.csv")
    data_radiation = DataAnalyzerRadiation("datos/clean/merged_solar_radiation.csv")
    data_consumption.describe_data()
    data_consumption.plot_time_series_power_consumption()
    data_pvpc.plot_time_series()
    data_radiation.plot_time_series()
