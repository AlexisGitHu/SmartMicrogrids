import pandas as pd

def get_dataframe_from_file(file_path, sep=";", header=0):
    return pd.read_csv(file_path, sep=sep, header=header)