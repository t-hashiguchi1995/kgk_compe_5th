"""アンサンブルモジュール

このモジュールは、時系列予測モデルのアンサンブル機能を提供します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


class TimeSeriesStacking:
    """時系列予測のスタッキングアンサンブルを行うクラス"""

    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: Optional[BaseEstimator] = None,
        n_splits: int = 5,
        test_size: int = 1,
        gap: int = 0,
    ):
        """初期化

        Args:
            base_models: ベースモデルのリスト
            meta_model: メタモデル（Noneの場合はRidge回帰を使用）
            n_splits: クロスバリデーションの分割数
            test_size: テストセットのサイズ
            gap: 訓練セットとテストセットの間のギャップ
        """
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.fitted_base_models = []
        self.fitted_meta_model = None

    def _create_time_series_split(self) -> TimeSeriesSplit:
        """時系列クロスバリデーションの分割を作成

        Returns:
            TimeSeriesSplit: 時系列分割オブジェクト
        """
        return TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=self.gap,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "TimeSeriesStacking":
        """モデルの訓練

        Args:
            X: 特徴量
            y: 目的変数
            eval_set: 評価用データセット

        Returns:
            TimeSeriesStacking: 訓練済みのスタッキングモデル
        """
        tscv = self._create_time_series_split()
        oof_predictions = np.zeros((len(X), len(self.base_models)))

        # 各ベースモデルに対してOOF予測を生成
        for i, model in enumerate(self.base_models):
            model_predictions = np.zeros(len(X))
            fitted_models = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # ベースモデルの訓練
                if isinstance(model, lgb.LGBMRegressor):
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=100,
                        verbose=False,
                    )
                else:
                    model.fit(X_train, y_train)

                # 検証セットに対する予測
                model_predictions[val_idx] = model.predict(X_val)
                fitted_models.append(model)

            oof_predictions[:, i] = model_predictions
            self.fitted_base_models.append(fitted_models)

        # メタモデルの訓練
        self.fitted_meta_model = self.meta_model.fit(oof_predictions, y)

        # 評価用データセットがある場合、最終的なベースモデルを訓練
        if eval_set is not None:
            X_eval, y_eval = eval_set
            for i, model in enumerate(self.base_models):
                if isinstance(model, lgb.LGBMRegressor):
                    model.fit(
                        X,
                        y,
                        eval_set=[(X_eval, y_eval)],
                        early_stopping_rounds=100,
                        verbose=False,
                    )
                else:
                    model.fit(X, y)
                self.fitted_base_models[i].append(model)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測

        Args:
            X: 特徴量

        Returns:
            np.ndarray: 予測値
        """
        if not self.fitted_base_models or self.fitted_meta_model is None:
            raise ValueError("Model not trained. Call fit first.")

        # 各ベースモデルの予測を生成
        base_predictions = np.zeros((len(X), len(self.base_models)))
        for i, models in enumerate(self.fitted_base_models):
            # 最後のモデル（全データで訓練されたモデル）を使用
            model = models[-1]
            base_predictions[:, i] = model.predict(X)

        # メタモデルで最終的な予測を生成
        return self.fitted_meta_model.predict(base_predictions)

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
        if not self.fitted_base_models or self.fitted_meta_model is None:
            raise ValueError("Model not trained. Call fit first.")

        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)

        return {
            "mse": mse,
            "rmse": rmse,
        } 