import pandas as pd
import pickle

def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    指定されたパスからCSVファイルを読み込み、pandas DataFrameとして返します。

    Args:
        file_path (str): 読み込むCSVファイルのパス。
        **kwargs: pandas.read_csvに渡される追加のキーワード引数。

    Returns:
        pd.DataFrame: 読み込まれたデータを含むDataFrame。

    Raises:
        FileNotFoundError: 指定されたファイルが見つからない場合。
        Exception: ファイル読み込み中に他のエラーが発生した場合。
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        raise


def save_csv(df: pd.DataFrame, file_path: str, index: bool = False, **kwargs) -> None:
    """
    pandas DataFrameを指定されたパスにCSVファイルとして保存します。

    Args:
        df (pd.DataFrame): 保存するDataFrame。
        file_path (str): 保存先のCSVファイルのパス。
        index (bool, optional): DataFrameのインデックスをCSVファイルに書き込むかどうか。
                                デフォルトはFalse。
        **kwargs: pandas.DataFrame.to_csvに渡される追加のキーワード引数。

    Raises:
        Exception: ファイル書き込み中にエラーが発生した場合。
    """
    try:
        df.to_csv(file_path, index=index, **kwargs)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving data to {file_path}: {e}")
        raise

def save_pickle(obj: object, file_path: str) -> None:
    """オブジェクトをpickleファイルとして保存します。"""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Successfully saved object to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving object to {file_path}: {e}")
        raise

def load_pickle(file_path: str) -> object:
    """pickleファイルからオブジェクトを読み込みます。"""
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Successfully loaded object from {file_path}")
        return obj
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading object from {file_path}: {e}")
        raise
