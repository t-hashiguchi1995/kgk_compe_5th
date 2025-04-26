import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from utils.cross_validation import TimeSeriesCV

class StackingModel:
    def __init__(self, base_models: List[BaseEstimator],
                 meta_model: Optional[BaseEstimator] = None):
        """スタッキングモデルの初期化
        
        Args:
            base_models (List[BaseEstimator]): ベースモデルのリスト
            meta_model (Optional[BaseEstimator]): メタモデル
        """
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.n_base_models = len(base_models)
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
           cv: TimeSeriesCV) -> None:
        """スタッキングモデルの学習
        
        Args:
            X (pd.DataFrame): 説明変数
            y (pd.Series): 目的変数
            cv (TimeSeriesCV): 時系列クロスバリデーション
        """
        # OOF予測の生成
        oof_predictions = np.zeros((len(X), self.n_base_models))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            for i, model in enumerate(self.base_models):
                model.fit(X_train, y_train)
                oof_predictions[val_idx, i] = model.predict(X_val)
                
        # メタモデルの学習
        self.meta_model.fit(oof_predictions, y)
        
        # ベースモデルの再学習
        for model in self.base_models:
            model.fit(X, y)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測
        
        Args:
            X (pd.DataFrame): 説明変数
            
        Returns:
            np.ndarray: 予測値
        """
        # ベースモデルの予測
        base_predictions = np.zeros((len(X), self.n_base_models))
        
        for i, model in enumerate(self.base_models):
            base_predictions[:, i] = model.predict(X)
            
        # メタモデルによる予測
        return self.meta_model.predict(base_predictions)
        
    def get_base_model_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """ベースモデルの予測値を取得
        
        Args:
            X (pd.DataFrame): 説明変数
            
        Returns:
            pd.DataFrame: ベースモデルの予測値
        """
        predictions = {}
        
        for i, model in enumerate(self.base_models):
            predictions[f'base_model_{i}'] = model.predict(X)
            
        return pd.DataFrame(predictions) 