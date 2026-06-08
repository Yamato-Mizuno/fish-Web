# 「うぉ！」魚類識別Webアプリ

魚の画像から、淡水魚・海水魚・回遊魚のいずれかを判定するWebアプリ。
授業「データと数理Ⅰ」で学んだ古典的線形代数手法（3カーネル畳み込み・LASSO/Ridge回帰）で動作する。

## ディレクトリ構成

```
uo_app/
├── app.py                       # Flaskバックエンド
├── templates/
│   └── index.html               # メインページ
├── static/
│   ├── style.css                # スタイルシート（うぉ！テーマ）
│   └── script.js                # クライアントサイドJS
├── data/
│   └── pilot_features.csv       # ★ここに既存の特徴量CSVを置く
├── best.pt                      # ★（任意）YOLOv8重み。あれば顔検出が動く
└── README.md
```

## セットアップ

### 1. 必要なライブラリをインストール

```bash
pip install flask opencv-python numpy pandas scikit-learn pillow ultralytics
```

`ultralytics` はYOLO顔検出を使う場合のみ必要。

### 2. データを配置

既存の `pilot_features.csv` を `data/` フォルダに置く：

```bash
mkdir -p data
cp /path/to/pilot_high300/pilot_features.csv data/
```

YOLO顔検出を使いたい場合は `best.pt` をルートに置く：

```bash
cp /path/to/best.pt .
```

### 3. 起動

```bash
python app.py
```

ブラウザで `http://localhost:5000` を開く。

## 使い方

1. トップ画面の「アプリを開始」ボタンをクリック
2. 魚の画像をアップロード（クリック or ドラッグ＆ドロップ）
3. 「判定する」ボタンで分類実行
4. 結果画面で淡水魚・海水魚・回遊魚の確率バーが表示される
5. 「詳細結果を表示」で研究結果の詳細データが見られる

## 推奨される入力画像

- 横向きの魚の顔がよく写った画像
- 背景は何でもOK（学習データもバラバラ）
- best.pt なしで使う場合は、できるだけ魚が中央に来ている画像を推奨

## 技術仕様

### 使用している授業の数理

- **画像 = 行列**: 64×64グレースケール
- **3カーネル畳み込み**: Blur / Kx / Ky の3種類
- **離散微分**: Kx（X方向）・Ky（Y方向）でエッジ検出
- **ベクトル化**: 25次元の特徴ベクトル
- **線形回帰（L2正則化）**: Ridge回帰で分類

### モデル性能

- 訓練サンプル: 約500枚
- balanced accuracy: 約46%（チャンスレベル33%）
- 最重要特徴量グループ: 水平エッジ Ky

## トラブルシューティング

**「pilot_features.csv が見つかりません」**
→ `data/pilot_features.csv` のパスを確認してください。

**「ultralytics が見つかりません」**
→ YOLOを使わない場合は無視してOK。アプリは中央クロップで動作します。

**判定結果が常に同じカテゴリになる**
→ 入力画像によります。学習データの傾向上、海水魚と判定される割合が高めです。

## ライセンス・データソース

- 学習データ: FishNet（公開データセット）
- フォント: Yuji Syuku / Dela Gothic One（Google Fonts）
