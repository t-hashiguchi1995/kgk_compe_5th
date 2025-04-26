"""メインスクリプト

このスクリプトは、時系列予測モデルの構築と評価の一連の流れを実行します。
実験計画に基づいて、データの読み込み、前処理、特徴量エンジニアリング、
クロスバリデーション、モデリング、評価を行います。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

from data.data_loader import DataLoader
from features.feature_engineering import FeatureEngineering
from utils.cross_validation import TimeSeriesCV
from models.lightgbm_model import LightGBMModel
from models.stacking import StackingModel
from utils.evaluation import evaluate_predictions

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # データの読み込みと前処理
    data_dir = Path('../../data/raw')
    data_loader = DataLoader(data_dir)
    
    # データの読み込み
    logger.info("データの読み込みを開始します")
    df = data_loader.load_data('train_df.csv')
    
    # 前処理
    logger.info("データの前処理を開始します")
    df = data_loader.preprocess_data(df, datetime_col='timestamp', target_col='target')
    
    # 特徴量エンジニアリング
    logger.info("特徴量エンジニアリングを開始します")
    feature_engineering = FeatureEngineering(target_col='target')
    
    # ラグ特徴量の設定
    lags = [1, 2, 3, 7, 14, 30, 60, 90]  # より長期的なラグを追加
    windows = [7, 14, 30, 60, 90]  # より長期的なウィンドウを追加
    
    # 特徴量の生成
    df = feature_engineering.create_all_features(
        df,
        lags=lags,
        windows=windows,
        add_time_features=True,  # 時間ベースの特徴量を追加
        add_holiday_features=True,  # 祝日特徴量を追加
        add_rolling_features=True,  # ローリング統計量を追加
        add_expanding_features=True  # 拡張統計量を追加
    )
    
    # 説明変数と目的変数の分離
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 時系列クロスバリデーションの設定
    logger.info("クロスバリデーションの設定を行います")
    cv = TimeSeriesCV(
        n_splits=5,
        test_size=30,  # テスト期間を30日間に設定
        gap=7,  # ギャップを7日間に設定
        max_train_size=None  # 訓練データサイズの制限なし
    )
    
    # ベースモデルの設定
    logger.info("ベースモデルの設定を行います")
    base_models = [
        LightGBMModel(params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }),
        LightGBMModel(params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 63,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bag_freq': 5,
            'min_data_in_leaf': 30,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2
        })
    ]
    
    # スタッキングモデルの学習
    logger.info("スタッキングモデルの学習を開始します")
    stacking_model = StackingModel(
        base_models=base_models,
        meta_model=LightGBMModel(params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.005,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 20
        })
    )
    
    # モデルの学習と評価
    cv_scores = stacking_model.fit(X, y, cv=cv)
    logger.info(f"クロスバリデーションスコア: {cv_scores}")
    
    # テストデータの読み込みと前処理
    logger.info("テストデータの処理を開始します")
    test_df = data_loader.load_data('test_df.csv')
    test_df = data_loader.preprocess_data(test_df, datetime_col='timestamp')
    test_df = feature_engineering.create_all_features(
        test_df,
        lags=lags,
        windows=windows,
        add_time_features=True,
        add_holiday_features=True,
        add_rolling_features=True,
        add_expanding_features=True
    )
    
    # 予測
    logger.info("テストデータに対する予測を開始します")
    predictions = stacking_model.predict(test_df)
    
    # 予測結果の評価
    logger.info("予測結果の評価を行います")
    evaluation_metrics = evaluate_predictions(
        y_true=test_df['target'],
        y_pred=predictions,
        metrics=['rmse', 'mae', 'mape', 'r2']
    )
    logger.info(f"評価指標: {evaluation_metrics}")
    
    # 予測結果の保存
    logger.info("予測結果を保存します")
    submission = pd.DataFrame({
        'timestamp': test_df.index,
        'target': predictions
    })
    
    # 出力ディレクトリの作成
    output_dir = Path('../../data/output')
    output_dir.mkdir(exist_ok=True)
    
    # タイムスタンプ付きで保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission.to_csv(output_dir / f'submission_{timestamp}.csv', index=False)
    
    # 評価結果の保存
    evaluation_df = pd.DataFrame([evaluation_metrics])
    evaluation_df.to_csv(output_dir / f'evaluation_{timestamp}.csv', index=False)
    
    logger.info("処理が完了しました")

if __name__ == '__main__':
    main() 