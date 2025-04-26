# KgKコンペシーズン5用のリポジトリ


# 環境構築
uvのインストール
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

python環境のインストール
```bash
uv python install 3.11.7
uv python pin 3.11.7
mkdir project
cd project
uv init --name codebook --package --python 3.11.7
```

```bash
uv add jupyterlab numpy pandas ipython scikit-learn lightgbm japanize-matplotlib seaborn wandb
uv add mypy ruff pre-commit taskipy
```

# jupyternotebookカーネルの適用

```bash
uv run ipython kernel install --user --name=kgk
```


# lintとformatの実行
```bash
uv run task lint
uv run task format
```

# コードの型確認
```bash
uv run mypy mypy_example.py
```