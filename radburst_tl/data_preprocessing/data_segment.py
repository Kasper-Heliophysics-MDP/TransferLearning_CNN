import pandas as pd
import pyreadstat

def read_sps_file(file_path):
    """
    Read an SPSS file and return a DataFrame.

    Parameters:
    file_path (str): Path to the SPSS file.

    Returns:
    pd.DataFrame: DataFrame containing the SPSS data.
    """
    try:
        df = pyreadstat.read_sav(file_path)
        return df
    except Exception as e:
        print(f"Error reading file2 : {e}")
        return None
