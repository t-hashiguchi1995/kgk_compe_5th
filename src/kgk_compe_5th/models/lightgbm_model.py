import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple
from sklearn.metrics import mean_squared_error

class LightGBMModel:
    def __init__(self, params: Optional[Dict] = None):
        """LightGBMモデルの初期化
        
        Args:
            params (Optional[Dict]): LightGBMのパラメータ
        """
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
           eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
           early_stopping_rounds: Optional[int] = None) -> None:
        """モデルの学習
        
        Args:
            X (pd.DataFrame): 説明変数
            y (pd.Series): 目的変数
            eval_set (Optional[Tuple[pd.DataFrame, pd.Series]]): 評価用データ
            early_stopping_rounds (Optional[int]): 早期停止のラウンド数
        """
        train_data = lgb.Dataset(X, label=y)
        
        if eval_set is not None:
            valid_data = lgb.Dataset(eval_set[0], label=eval_set[1])
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[valid_data],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
        else:
            self.model = lgb.train(
                self.params,
                train_data,
                verbose_eval=False
            )
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測
        
        Args:
            X (pd.DataFrame): 説明変数
            
        Returns:
            np.ndarray: 予測値
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        return self.model.predict(X)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """特徴量の重要度を取得
        
        Returns:
            pd.DataFrame: 特徴量の重要度
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        importance = self.model.feature_importance()
        feature_names = self.model.feature_name()
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False) 