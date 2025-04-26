import pandas as pd
import numpy as np
from typing import Optional, Union
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        
    def load_data(self, file_name: str) -> pd.DataFrame:
        """データを読み込む
        
        Args:
            file_name (str): 読み込むファイル名
            
        Returns:
            pd.DataFrame: 読み込んだデータ
        """
        file_path = self.data_dir / file_name
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        return df
    
    def preprocess_data(self, df: pd.DataFrame, 
                       datetime_col: str,
                       target_col: Optional[str] = None) -> pd.DataFrame:
        """データの前処理を行う
        
        Args:
            df (pd.DataFrame): 前処理するデータ
            datetime_col (str): 日時カラム名
            target_col (Optional[str]): 目的変数カラム名
            
        Returns:
            pd.DataFrame: 前処理済みデータ
        """
        # 日時カラムをdatetime型に変換
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # 日時カラムをインデックスに設定
        df = df.set_index(datetime_col)
        
        # 欠損値の処理
        if target_col is not None:
            # 目的変数の欠損値は前方補完
            df[target_col] = df[target_col].fillna(method='ffill')
            
        # その他の欠損値は0で補完
        df = df.fillna(0)
        
        return df 