"""モデリングモジュール

このモジュールは、時系列予測モデルの訓練と評価の機能を提供します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback


class TimeSeriesModelTrainer:
    """時系列予測モデルの訓練を行うクラス"""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        time_col: Optional[str] = None,
    ):
        """初期化

        Args:
            df: 入力データフレーム
            target_col: 目的変数の列名
            feature_cols: 特徴量の列名リスト
            time_col: 時間列の名前（Noneの場合はインデックスを使用）
        """
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.time_col = time_col
        self.model = None
        self.best_params = None

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """データの準備

        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特徴量と目的変数
        """
        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_col].copy()

        if self.time_col is not None:
            X = X.set_index(self.df[self.time_col])

        return X, y

    def create_time_series_split(
        self,
        n_splits: int = 5,
        test_size: int = 1,
        gap: int = 0,
    ) -> TimeSeriesSplit:
        """時系列クロスバリデーションの分割を作成

        Args:
            n_splits: 分割数
            test_size: テストセットのサイズ
            gap: 訓練セットとテストセットの間のギャップ

        Returns:
            TimeSeriesSplit: 時系列分割オブジェクト
        """
        return TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
        )

    def objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        tscv: TimeSeriesSplit,
    ) -> float:
        """Optunaによるハイパーパラメータ最適化の目的関数

        Args:
            trial: Optunaのトライアルオブジェクト
            X: 特徴量
            y: 目的変数
            tscv: 時系列分割オブジェクト

        Returns:
            float: 平均RMSE
        """
        param = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**param)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    LightGBMPruningCallback(trial, "rmse"),
                ],
                early_stopping_rounds=100,
                verbose=False,
            )

            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)

        return np.mean(scores)

    def optimize_hyperparameters(
        self,
        n_trials: int = 100,
        n_splits: int = 5,
        test_size: int = 1,
        gap: int = 0,
    ) -> Dict:
        """ハイパーパラメータの最適化

        Args:
            n_trials: 試行回数
            n_splits: クロスバリデーションの分割数
            test_size: テストセットのサイズ
            gap: 訓練セットとテストセットの間のギャップ

        Returns:
            Dict: 最適なハイパーパラメータ
        """
        X, y = self.prepare_data()
        tscv = self.create_time_series_split(n_splits, test_size, gap)

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(trial, X, y, tscv),
            n_trials=n_trials,
        )

        self.best_params = study.best_params
        return study.best_params

    def train(
        self,
        params: Optional[Dict] = None,
        early_stopping_rounds: int = 100,
    ) -> lgb.LGBMRegressor:
        """モデルの訓練

        Args:
            params: ハイパーパラメータ（Noneの場合は最適化済みのパラメータを使用）
            early_stopping_rounds: 早期停止のラウンド数

        Returns:
            lgb.LGBMRegressor: 訓練済みモデル
        """
        X, y = self.prepare_data()
        tscv = self.create_time_series_split()

        if params is None:
            if self.best_params is None:
                raise ValueError("No parameters provided. Call optimize_hyperparameters first.")
            params = self.best_params

        params["objective"] = "regression"
        params["metric"] = "rmse"
        params["verbosity"] = -1

        models = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
            models.append(model)

        # 最終的なモデルは全データで訓練
        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X, y)
        self.model = final_model

        return final_model

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """モデルの評価

        Args:
            X: 特徴量
            y: 目的変数

        Returns:
            Dict[str, float]: 評価指標
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")

        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測

        Args:
            X: 特徴量

        Returns:
            np.ndarray: 予測値
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")

        return self.model.predict(X) 