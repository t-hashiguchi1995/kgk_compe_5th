import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime

class FeatureEngineering:
    def __init__(self, target_col: str):
        self.target_col = target_col
        
    def create_lag_features(self, df: pd.DataFrame, 
                          lags: List[int]) -> pd.DataFrame:
        """ラグ特徴量を作成する
        
        Args:
            df (pd.DataFrame): 元データ
            lags (List[int]): ラグのリスト
            
        Returns:
            pd.DataFrame: ラグ特徴量を追加したデータ
        """
        for lag in lags:
            df[f'{self.target_col}_lag_{lag}'] = df[self.target_col].shift(lag)
            
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間ベースの特徴量を作成する
        
        Args:
            df (pd.DataFrame): 元データ
            
        Returns:
            pd.DataFrame: 時間特徴量を追加したデータ
        """
        # 時間情報の抽出
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear
        
        # 周期的な特徴量の作成
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                              windows: List[int]) -> pd.DataFrame:
        """ローリング統計量特徴量を作成する
        
        Args:
            df (pd.DataFrame): 元データ
            windows (List[int]): ウィンドウサイズのリスト
            
        Returns:
            pd.DataFrame: ローリング特徴量を追加したデータ
        """
        for window in windows:
            df[f'{self.target_col}_rolling_mean_{window}'] = df[self.target_col].rolling(window=window).mean()
            df[f'{self.target_col}_rolling_std_{window}'] = df[self.target_col].rolling(window=window).std()
            df[f'{self.target_col}_rolling_min_{window}'] = df[self.target_col].rolling(window=window).min()
            df[f'{self.target_col}_rolling_max_{window}'] = df[self.target_col].rolling(window=window).max()
            
        return df
    
    def create_all_features(self, df: pd.DataFrame,
                          lags: List[int],
                          windows: List[int]) -> pd.DataFrame:
        """全ての特徴量を作成する
        
        Args:
            df (pd.DataFrame): 元データ
            lags (List[int]): ラグのリスト
            windows (List[int]): ウィンドウサイズのリスト
            
        Returns:
            pd.DataFrame: 全ての特徴量を追加したデータ
        """
        df = self.create_lag_features(df, lags)
        df = self.create_time_features(df)
        df = self.create_rolling_features(df, windows)
        
        # 欠損値の処理
        df = df.fillna(0)
        
        return df 