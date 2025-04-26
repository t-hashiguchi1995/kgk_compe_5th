"""メインスクリプト

このスクリプトは、時系列予測モデルの構築と評価の一連の流れを実行します。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from data.data_loader import DataLoader
from features.feature_engineering import FeatureEngineering
from utils.cross_validation import TimeSeriesCV
from models.lightgbm_model import LightGBMModel
from models.stacking import StackingModel


def main():
    # データの読み込みと前処理
    data_dir = Path('../../data/raw')
    data_loader = DataLoader(data_dir)
    
    # データの読み込み
    df = data_loader.load_data('train_df.csv')
    
    # 前処理
    df = data_loader.preprocess_data(df, datetime_col='timestamp', target_col='target')
    
    # 特徴量エンジニアリング
    feature_engineering = FeatureEngineering(target_col='target')
    lags = [1, 2, 3, 7, 14, 30]
    windows = [7, 14, 30]
    df = feature_engineering.create_all_features(df, lags=lags, windows=windows)
    
    # 説明変数と目的変数の分離
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 時系列クロスバリデーションの設定
    cv = TimeSeriesCV(n_splits=5, test_size=1, gap=0)
    
    # ベースモデルの設定
    base_models = [
        LightGBMModel(params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05
        }),
        LightGBMModel(params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 63,
            'learning_rate': 0.01
        })
    ]
    
    # スタッキングモデルの学習
    stacking_model = StackingModel(base_models=base_models)
    stacking_model.fit(X, y, cv=cv)
    
    # テストデータの読み込みと前処理
    test_df = data_loader.load_data('test_df.csv')
    test_df = data_loader.preprocess_data(test_df, datetime_col='timestamp')
    test_df = feature_engineering.create_all_features(test_df, lags=lags, windows=windows)
    
    # 予測
    predictions = stacking_model.predict(test_df)
    
    # 予測結果の保存
    submission = pd.DataFrame({
        'timestamp': test_df.index,
        'target': predictions
    })
    submission.to_csv('../data/output/submission.csv', index=False)
    

if __name__ == '__main__':
    main() 