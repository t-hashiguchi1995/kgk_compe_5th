import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from typing import Optional, List, Tuple

def plot_histogram(df: pd.DataFrame, column: str, bins: int = 10, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    指定された列のヒストグラムをプロットします。

    Args:
        df (pd.DataFrame): データフレーム。
        column (str): ヒストグラムを作成する列名。
        bins (int, optional): ビンの数。デフォルトは10。
        title (str | None, optional): グラフのタイトル。デフォルトはNone。
        xlabel (str | None, optional): X軸のラベル。デフォルトはNone (列名が使用される)。
        ylabel (str | None, optional): Y軸のラベル。デフォルトはNone ('頻度'が使用される)。
        save_path (str | None, optional): グラフを保存するパス。指定しない場合は表示のみ。デフォルトはNone。
    """
    if column not in df.columns:
        print(f"エラー: 列 '{column}' がデータフレームに見つかりません。")
        return
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"エラー: ヒストグラムを作成する列 '{column}' は数値型である必要があります。")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=True)

    plt.title(title if title else f'{column} のヒストグラム')
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel if ylabel else '頻度')
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Y軸のみグリッド表示

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight') # bbox_inches='tight' でラベル切れを防ぐ
            print(f"ヒストグラムを {save_path} に保存しました。")
        except Exception as e:
            print(f"グラフの保存中にエラーが発生しました: {e}")
    else:
        plt.show()
    plt.close() # メモリ解放のため


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str] = None, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    指定された2つの列の散布図をプロットします。オプションで色分けも可能です。

    Args:
        df (pd.DataFrame): データフレーム。
        x_col (str): X軸に使用する列名。
        y_col (str): Y軸に使用する列名。
        hue_col (str | None, optional): 点の色分けに使用する列名。デフォルトはNone。
        title (str | None, optional): グラフのタイトル。デフォルトはNone。
        xlabel (str | None, optional): X軸のラベル。デフォルトはNone (x_colが使用される)。
        ylabel (str | None, optional): Y軸のラベル。デフォルトはNone (y_colが使用される)。
        save_path (str | None, optional): グラフを保存するパス。指定しない場合は表示のみ。デフォルトはNone。
    """
    if x_col not in df.columns:
        print(f"エラー: 列 '{x_col}' がデータフレームに見つかりません。")
        return
    if y_col not in df.columns:
        print(f"エラー: 列 '{y_col}' がデータフレームに見つかりません。")
        return
    if hue_col and hue_col not in df.columns:
        print(f"エラー: 色分け用の列 '{hue_col}' がデータフレームに見つかりません。")
        return


    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.7) # alphaで透明度調整

    plt.title(title if title else f'{x_col} と {y_col} の散布図' + (f' ({hue_col}で色分け)' if hue_col else ''))
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"散布図を {save_path} に保存しました。")
        except Exception as e:
            print(f"グラフの保存中にエラーが発生しました: {e}")
    else:
        plt.show()
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, title: Optional[str] = None, save_path: Optional[str] = None, annot: bool = True, cmap: str = 'coolwarm', figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    データフレームの数値列間の相関ヒートマップをプロットします。

    Args:
        df (pd.DataFrame): データフレーム。
        title (str | None, optional): グラフのタイトル。デフォルトはNone ('相関ヒートマップ')。
        save_path (str | None, optional): グラフを保存するパス。指定しない場合は表示のみ。デフォルトはNone。
        annot (bool, optional): ヒートマップに数値を表示するかどうか。デフォルトはTrue。
        cmap (str, optional): カラーマップ。デフォルトは'coolwarm'。
        figsize (tuple[int, int], optional): グラフのサイズ。デフォルトは(12, 10)。
    """
    # 数値列のみを選択
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("エラー: 相関ヒートマップを作成するための数値列がデータフレームに見つかりません。")
        return
    if numeric_df.shape[1] < 2:
        print("エラー: 相関ヒートマップを作成するには少なくとも2つの数値列が必要です。")
        return

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=".2f", linewidths=.5) # 線を追加して見やすく
    plt.title(title if title else '相関ヒートマップ')
    plt.xticks(rotation=45, ha='right') # ラベルが重ならないように回転
    plt.yticks(rotation=0)
    plt.tight_layout() # レイアウト調整

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"相関ヒートマップを {save_path} に保存しました。")
        except Exception as e:
            print(f"グラフの保存中にエラーが発生しました: {e}")
    else:
        plt.show()
    plt.close()

def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None, save_path: Optional[str] = None, rotation: int = 45) -> None:
    """
    指定された列の箱ひげ図をプロットします。

    Args:
        df (pd.DataFrame): データフレーム。
        x_col (str): X軸に使用するカテゴリカル列名。
        y_col (str): Y軸に使用する数値列名。
        title (str | None, optional): グラフのタイトル。デフォルトはNone。
        xlabel (str | None, optional): X軸のラベル。デフォルトはNone (x_colが使用される)。
        ylabel (str | None, optional): Y軸のラベル。デフォルトはNone (y_colが使用される)。
        save_path (str | None, optional): グラフを保存するパス。指定しない場合は表示のみ。デフォルトはNone。
        rotation (int, optional): X軸ラベルの回転角度。デフォルトは45。
    """
    if x_col not in df.columns:
        print(f"エラー: 列 '{x_col}' がデータフレームに見つかりません。")
        return
    if y_col not in df.columns:
        print(f"エラー: 列 '{y_col}' がデータフレームに見つかりません。")
        return
    if not pd.api.types.is_numeric_dtype(df[y_col]):
         print(f"エラー: Y軸の列 '{y_col}' は数値型である必要があります。")
         return

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x=x_col, y=y_col)

    plt.title(title if title else f'{y_col} by {x_col} の箱ひげ図')
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.xticks(rotation=rotation, ha='right') # X軸ラベルが長い場合に備えて回転
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"箱ひげ図を {save_path} に保存しました。")
        except Exception as e:
            print(f"グラフの保存中にエラーが発生しました: {e}")
    else:
        plt.show()
    plt.close()

def plot_pairplot(df: pd.DataFrame, hue_col: Optional[str] = None, vars: Optional[List[str]] = None, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    データフレームの数値列間のペアプロット（散布図行列）を作成します。

    Args:
        df (pd.DataFrame): データフレーム。
        hue_col (str | None, optional): 点の色分けに使用する列名。デフォルトはNone。
        vars (list[str] | None, optional): プロットする列のリスト。指定しない場合は数値列すべて。デフォルトはNone。
        title (str | None, optional): グラフ全体のタイトル。デフォルトはNone ('ペアプロット')。
        save_path (str | None, optional): グラフを保存するパス。指定しない場合は表示のみ。デフォルトはNone。
    """
    if hue_col and hue_col not in df.columns:
        print(f"エラー: 色分け用の列 '{hue_col}' がデータフレームに見つかりません。")
        return

    plot_df = df
    plot_vars = vars
    if plot_vars:
        missing_vars = [v for v in plot_vars if v not in df.columns]
        if missing_vars:
            print(f"エラー: 列 {missing_vars} がデータフレームに見つかりません。")
            return
        # hue_colもvarsに含まれていない場合は追加する（seabornの仕様）
        if hue_col and hue_col not in plot_vars:
             plot_vars = plot_vars + [hue_col]
        plot_df = df[plot_vars]
    else:
        # varsが指定されていない場合は数値列のみを選択
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
             print("エラー: ペアプロットを作成するための数値列がデータフレームに見つかりません。")
             return
        if hue_col and hue_col not in numeric_cols:
             # hue_colが数値でない場合、それも追加
             plot_vars = numeric_cols + [hue_col]
             plot_df = df[plot_vars]
        elif hue_col: # hue_colが数値の場合
             plot_vars = numeric_cols
             plot_df = df[plot_vars]
        else: # hue_colがない場合
             plot_vars = numeric_cols
             plot_df = df[plot_vars]


    if plot_df.select_dtypes(include=['number']).empty:
        print("エラー: ペアプロットを作成するための数値列がデータフレームに見つかりません。")
        return

    # pairplotに渡すvars引数は数値列のみにする必要がある場合がある
    numeric_plot_vars = plot_df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_plot_vars:
        print("エラー: ペアプロットに表示する数値列がありません。")
        return

    pair_plot = sns.pairplot(plot_df, hue=hue_col, vars=numeric_plot_vars, diag_kind='kde') # 対角はカーネル密度推定

    if title:
        pair_plot.fig.suptitle(title, y=1.02) # タイトルがグラフにかぶらないように調整

    if save_path:
        try:
            pair_plot.savefig(save_path) # pairplotはsavefigメソッドを持つ
            print(f"ペアプロットを {save_path} に保存しました。")
        except Exception as e:
            print(f"グラフの保存中にエラーが発生しました: {e}")
    else:
        plt.show()
    plt.close() # pairplotは内部でfigureを生成するので、plt.close()で閉じる
