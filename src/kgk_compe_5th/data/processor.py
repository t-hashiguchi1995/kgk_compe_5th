"""データ処理モジュール

このモジュールは、時系列データの読み込み、前処理、結合などの機能を提供します。
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional
from pathlib import Path


class TimeSeriesDataProcessor:
    """時系列データの処理を行うクラス"""

    def __init__(
        self,
        data_path: Union[str, Path],
        timestamp_col: str,
        target_col: str,
        timezone: Optional[str] = None,
    ):
        """初期化

        Args:
            data_path: データファイルのパス
            timestamp_col: タイムスタンプ列の名前
            target_col: 目的変数列の名前
            timezone: タイムゾーン（オプション）
        """
        self.data_path = Path(data_path)
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.timezone = timezone
        self.data = None

    def load_data(self, file_format: str = "csv") -> pd.DataFrame:
        """データを読み込む

        Args:
            file_format: ファイル形式（"csv"または"parquet"）

        Returns:
            pd.DataFrame: 読み込んだデータ
        """
        if file_format == "csv":
            df = pd.read_csv(self.data_path)
        elif file_format == "parquet":
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # タイムスタンプ列をdatetime型に変換
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        if self.timezone:
            df[self.timestamp_col] = df[self.timestamp_col].dt.tz_localize(self.timezone)

        # タイムスタンプ列をインデックスに設定
        df = df.set_index(self.timestamp_col)
        self.data = df
        return df

    def handle_missing_values(
        self,
        method: str = "ffill",
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """欠損値を処理する

        Args:
            method: 補完方法（"ffill", "bfill", "interpolate"）
            limit: 連続する欠損値の最大補完数
            columns: 処理対象の列（Noneの場合は全列）

        Returns:
            pd.DataFrame: 欠損値を処理したデータ
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.data.copy()
        target_columns = columns if columns is not None else df.columns

        if method == "ffill":
            df[target_columns] = df[target_columns].ffill(limit=limit)
        elif method == "bfill":
            df[target_columns] = df[target_columns].bfill(limit=limit)
        elif method == "interpolate":
            df[target_columns] = df[target_columns].interpolate(limit=limit)
        else:
            raise ValueError(f"Unsupported method: {method}")

        self.data = df
        return df

    def detect_outliers(
        self,
        method: str = "zscore",
        threshold: float = 3.0,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, pd.Series]:
        """外れ値を検出する

        Args:
            method: 検出方法（"zscore"または"iqr"）
            threshold: 閾値
            columns: 処理対象の列（Noneの場合は全列）

        Returns:
            Dict[str, pd.Series]: 各列の外れ値フラグ
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.data.copy()
        target_columns = columns if columns is not None else df.columns
        outliers = {}

        for col in target_columns:
            if method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = z_scores > threshold
            elif method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = (df[col] < (Q1 - threshold * IQR)) | (
                    df[col] > (Q3 + threshold * IQR)
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

        return outliers

    def merge_data(
        self,
        other_df: pd.DataFrame,
        on: Union[str, List[str]],
        how: str = "inner",
    ) -> pd.DataFrame:
        """他のデータフレームと結合する

        Args:
            other_df: 結合するデータフレーム
            on: 結合キー
            how: 結合方法（"inner", "outer", "left", "right"）

        Returns:
            pd.DataFrame: 結合されたデータ
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.data.copy()
        merged_df = df.merge(other_df, on=on, how=how)
        self.data = merged_df
        return merged_df 