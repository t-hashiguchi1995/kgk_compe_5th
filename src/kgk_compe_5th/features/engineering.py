"""特徴量エンジニアリングモジュール

このモジュールは、時系列データの特徴量生成機能を提供します。
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional
from datetime import datetime
import holidays


class TimeSeriesFeatureEngineer:
    """時系列特徴量エンジニアリングを行うクラス"""

    def __init__(self, df: pd.DataFrame):
        """初期化

        Args:
            df: 入力データフレーム（DatetimeIndexを持つ）
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        self.df = df.copy()

    def add_lag_features(
        self,
        target_col: str,
        lags: List[int],
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """ラグ特徴量を追加する

        Args:
            target_col: 目的変数の列名
            lags: ラグのリスト
            fill_method: 欠損値の補完方法

        Returns:
            pd.DataFrame: ラグ特徴量を追加したデータフレーム
        """
        df = self.df.copy()
        for lag in lags:
            col_name = f"{target_col}_lag_{lag}"
            df[col_name] = df[target_col].shift(lag)

        if fill_method == "ffill":
            df = df.ffill()
        elif fill_method == "bfill":
            df = df.bfill()
        elif fill_method == "zero":
            df = df.fillna(0)
        else:
            raise ValueError(f"Unsupported fill method: {fill_method}")

        self.df = df
        return df

    def add_time_features(self) -> pd.DataFrame:
        """時間ベースの特徴量を追加する

        Returns:
            pd.DataFrame: 時間特徴量を追加したデータフレーム
        """
        df = self.df.copy()

        # 基本的な時間特徴量
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["dayofyear"] = df.index.dayofyear
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year
        df["weekofyear"] = df.index.isocalendar().week

        # 周期的な特徴量（sin/cos変換）
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        self.df = df
        return df

    def add_holiday_features(
        self,
        country: str = "JP",
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """祝日特徴量を追加する

        Args:
            country: 国コード
            years: 対象年（Noneの場合はデータから自動判定）

        Returns:
            pd.DataFrame: 祝日特徴量を追加したデータフレーム
        """
        df = self.df.copy()

        if years is None:
            years = list(range(df.index.year.min(), df.index.year.max() + 1))

        # 祝日カレンダーを取得
        holiday_calendar = holidays.CountryHoliday(country, years=years)

        # 祝日フラグを追加
        df["is_holiday"] = df.index.map(lambda x: x in holiday_calendar).astype(int)

        # 祝日前後フラグ（オプション）
        df["is_day_before_holiday"] = df.index.map(
            lambda x: (x + pd.Timedelta(days=1)) in holiday_calendar
        ).astype(int)
        df["is_day_after_holiday"] = df.index.map(
            lambda x: (x - pd.Timedelta(days=1)) in holiday_calendar
        ).astype(int)

        self.df = df
        return df

    def add_rolling_features(
        self,
        target_col: str,
        windows: List[int],
        aggregations: List[str] = ["mean", "std", "min", "max"],
    ) -> pd.DataFrame:
        """ローリング統計量特徴量を追加する

        Args:
            target_col: 目的変数の列名
            windows: ウィンドウサイズのリスト
            aggregations: 集計関数のリスト

        Returns:
            pd.DataFrame: ローリング特徴量を追加したデータフレーム
        """
        df = self.df.copy()

        for window in windows:
            for agg in aggregations:
                col_name = f"{target_col}_rolling_{agg}_{window}"
                if agg == "mean":
                    df[col_name] = df[target_col].rolling(window=window).mean()
                elif agg == "std":
                    df[col_name] = df[target_col].rolling(window=window).std()
                elif agg == "min":
                    df[col_name] = df[target_col].rolling(window=window).min()
                elif agg == "max":
                    df[col_name] = df[target_col].rolling(window=window).max()
                else:
                    raise ValueError(f"Unsupported aggregation: {agg}")

        # 欠損値を前方補完
        df = df.ffill()

        self.df = df
        return df

    def add_expanding_features(
        self,
        target_col: str,
        aggregations: List[str] = ["mean", "std", "min", "max"],
    ) -> pd.DataFrame:
        """拡大統計量特徴量を追加する

        Args:
            target_col: 目的変数の列名
            aggregations: 集計関数のリスト

        Returns:
            pd.DataFrame: 拡大特徴量を追加したデータフレーム
        """
        df = self.df.copy()

        for agg in aggregations:
            col_name = f"{target_col}_expanding_{agg}"
            if agg == "mean":
                df[col_name] = df[target_col].expanding().mean()
            elif agg == "std":
                df[col_name] = df[target_col].expanding().std()
            elif agg == "min":
                df[col_name] = df[target_col].expanding().min()
            elif agg == "max":
                df[col_name] = df[target_col].expanding().max()
            else:
                raise ValueError(f"Unsupported aggregation: {agg}")

        # 欠損値を前方補完
        df = df.ffill()

        self.df = df
        return df 