import pandas as pd
import chardet

def detect_encoding(file_path):
    """
    Detect the encoding of a file.

    Parameters:
    file_path (str): Path to the file.

    Returns:
    str: Detected encoding.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read(20000)  # 读取文件的前10000字节
        result = chardet.detect(raw_data)
        return result['encoding']

def read_csv_file(file_path):
    """
    Read a CSV file using the detected encoding and return a DataFrame.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

