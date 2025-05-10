# KgKコンペシーズン5用のリポジトリ
特徴量追加と欠損値に対しての処理を実行。モデルはとりあえずlightgbmでやっている。

大気汚染予測コンペ 入賞戦略_.md: Gemini作成の本コンペ入賞戦略。基本この戦略通りのコードを実装。

時系列予測モデル構築支援.md: Gemini作成の1st stepとして時系列予測モデルを構築する際の方針まとめ


# 環境構築
## uvのインストール
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## python環境のインストール
```bash
uv python install 3.11.7
uv python pin 3.11.7
mkdir project
cd project
uv init --name codebook --package --python 3.11.7
```

## 環境の同期
```bash
uv sync
```

## jupyternotebookカーネルの適用

```bash
uv run ipython kernel install --user --name=kgk
```


# ディレクトリ構造
```
.
├── conf
│   └── config.yaml
├── data
│   ├── model_input
│   │   ├── test_features.pkl
│   │   └── train_features.pkl
│   ├── output
│   │   └── submission_base.csv
│   └── raw
│       ├── sample_submission.csv
│       ├── station_df_latlon.csv
│       ├── station_df.csv
│       ├── test_df.csv
│       ├── train_df.csv
│       └── weather_df.csv
├── notebook
│   ├── 01_eda_feature_enginnering.ipynb
│   └── 02_base_model.ipynb
├── 時系列予測モデル構築支援.md
└── 大気汚染予測コンペ 入賞戦略_.md
```
