import csv
import pandas as pd
import os


def get_csv_files(directory):
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            csv_files.append(os.path.join(directory, file))
    return csv_files


def clean_data(data):
    data = data.dropna()
    data = data.drop_duplicates()
    return data


def change_hour_format(df):
    df["TIME"] = df["TIME"].str.replace("T", " ")
    df["TIME"] = df["TIME"].replace(r"\+.*$", "", regex=True)
    df["Time"] = pd.to_datetime(df["TIME"], format="%Y-%m-%d %H:%M:%S")
    return df


def preprocess_light_pvpc_historic(csv_files):
    for file in csv_files:
        data = pd.read_csv(file, sep=";", encoding="utf-8", header=0)
        data = clean_data(data)
        data = change_hour_format(data)
        data.drop(
            columns=["TIME", "GEOID", "DESCRIPTION", "ID", "LOCATION"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        data.to_csv(
            "datos/clean/merged_pvpc_light.csv",
            index=False,
            sep=";",
        )


def combinar_fecha_hora(row):
    fecha = pd.to_datetime(row["Fecha"], format="%d/%m/%Y")
    hora = row["Hora"] - 1
    return fecha.replace(hour=hora)


def preprocess_power_consumption(csv_files, expected_date_format="%d/%m/%Y"):
    df_empty = pd.DataFrame(columns=["Time", "Consumo (kWh)", "ID"])
    columns_to_remove = ["CUPS", "Metodo_obtencion", "Fecha", "Hora"]
    for file in csv_files:
        data = pd.read_csv(file, sep=";", encoding="utf-8", header=0)
        data = clean_data(data)
        data = data[data["Hora"] != 25]
        data["Time"] = data.apply(combinar_fecha_hora, axis=1)
        data.drop(columns=columns_to_remove, axis=1, inplace=True, errors="ignore")

        # Add extra column with unique id representing the csv_file
        data["ID"] = str(os.path.basename(file))
        # Merge all csv_files into a single one
        df_empty = pd.concat([df_empty, data], ignore_index=True)
    df_empty.to_csv(
        "datos/clean/merged_power_consumption.csv",
        index=False,
        sep=";",
    )


def preprocess_solar_radiation(csv_files, expected_date_format="%Y-%m-%d %H:%M:%S"):
    for file in csv_files:
        data = pd.read_csv(file, sep=";", encoding="utf-8", header=0)
        data = clean_data(data)
        data["Time"] = pd.to_datetime(data["time"], format=expected_date_format)
        data["P"] = data["P"] / 1000
        data.rename(columns={"P": "P (kW)"}, inplace=True)
        data.drop(
            columns=["G(i)", "time", "H_sun", "T2m", "WS10m", "Int"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        data.to_csv(
            "datos/clean/merged_solar_radiation.csv",
            index=False,
            sep=";",
        )


def preprocess(*directories):
    # Preprocess the data can be done in parallel
    for directory in directories:
        csv_files = get_csv_files(directory)
        if "PVPC" in directory:
            preprocess_light_pvpc_historic(csv_files)
        elif "ConsumoFacturado" in directory:
            preprocess_power_consumption(csv_files)
        else:
            preprocess_solar_radiation(csv_files)


def read_csv_files(csv_files):
    dataframes = []
    for file in csv_files:
        data = pd.read_csv(file, sep=";", encoding="utf-8", header=0)
        dataframes.append(data)
    return dataframes


def merging_by_date(*directories):
    dataframes = read_csv_files(directories)
    for i, dataframe in enumerate(dataframes):
        if "ID" in dataframe.columns:
            dataframes[i].drop(columns=["ID"], axis=1, inplace=True, errors="ignore")
        dataframes[i]["month"] = pd.to_datetime(
            dataframe["Time"], format="%Y-%m-%d %H:%M:%S"
        ).dt.month
        dataframes[i]["day"] = pd.to_datetime(
            dataframe["Time"], format="%Y-%m-%d %H:%M:%S"
        ).dt.day
        dataframes[i]["hour"] = pd.to_datetime(
            dataframe["Time"], format="%Y-%m-%d %H:%M:%S"
        ).dt.hour
        dataframes[i].drop(columns=["Time"], axis=1, inplace=True, errors="ignore")

    merged_dataframe = pd.merge(
        dataframes[0], dataframes[1], on=["month", "day", "hour"], how="outer"
    )
    merged_dataframe = pd.merge(
        merged_dataframe, dataframes[2], on=["month", "day", "hour"], how="outer"
    )
    merged_dataframe = merged_dataframe._append(
        {
            "month": 3,
            "day": 27,
            "hour": 2,
            "P (kW)": 0,
            "Consumo (kWh)": "0,115",
            "PVPC (kWh)": 0.3044,
        },
        ignore_index=True,
    )
    merged_dataframe = merged_dataframe.sort_values(by=["month", "day", "hour"])
    merged_dataframe = merged_dataframe.reset_index(drop=True)
    merged_dataframe = clean_data(merged_dataframe)
    merged_dataframe = merged_dataframe[
        ["month", "day", "hour", "P (kW)", "Consumo (kWh)", "PVPC (kWh)"]
    ]
    merged_dataframe["Consumo (kWh)"] = (
        merged_dataframe["Consumo (kWh)"].str.replace(",", ".").astype(float)
    )

    merged_dataframe.to_csv(
        "datos/clean/merged_data.csv",
        index=False,
        sep=";",
    )


# Example usage
if __name__ == "__main__":
    light_pvpc_historic_dir = "datos/raw/PVPC_Luz_Peninsula_2022_2023"
    power_consumption_dir = "datos/raw/ConsumoFacturado"
    solar_radiation_dir = "datos/raw/RadiacionSolar"

    light_pvpc_cleaned_dir = "datos/clean/merged_pvpc_light.csv"
    power_consumption_cleaned_dir = "datos/clean/merged_power_consumption.csv"
    solar_radiation_cleaned_dir = "datos/clean/merged_solar_radiation.csv"

    preprocess(light_pvpc_historic_dir, power_consumption_dir, solar_radiation_dir)
    merging_by_date(
        light_pvpc_cleaned_dir,
        power_consumption_cleaned_dir,
        solar_radiation_cleaned_dir,
    )
