import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Optional
from sklearn.model_selection import TimeSeriesSplit

class TimeSeriesCV:
    def __init__(self, n_splits: int = 5, 
                 test_size: int = 1,
                 gap: int = 0,
                 max_train_size: Optional[int] = None):
        """時系列クロスバリデーションの初期化
        
        Args:
            n_splits (int): 分割数
            test_size (int): テストセットのサイズ
            gap (int): 訓練セットとテストセットの間のギャップ
            max_train_size (Optional[int]): 訓練セットの最大サイズ
        """
        self.cv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size
        )
        
    def split(self, X: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """時系列クロスバリデーションの分割を生成
        
        Args:
            X (pd.DataFrame): 分割するデータ
            
        Yields:
            Generator[Tuple[np.ndarray, np.ndarray], None, None]: 訓練インデックスとテストインデックスのタプル
        """
        for train_idx, test_idx in self.cv.split(X):
            yield train_idx, test_idx
            
    def get_n_splits(self) -> int:
        """分割数を取得
        
        Returns:
            int: 分割数
        """
        return self.cv.n_splits 