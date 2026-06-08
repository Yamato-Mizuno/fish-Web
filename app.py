"""
app.py - 「うぉ！」魚類識別Webアプリ
====================================
画像をアップロードすると、淡水魚・海水魚・回遊魚の確率を返す。
既存のpilot_features.csvからRidgeモデルを構築して使用。

【起動方法】
  pip install flask opencv-python numpy pandas scikit-learn pillow
  python app.py
  → http://localhost:5000 にアクセス

【オプション】
  best.pt（YOLOv8重み）が同フォルダにあれば、自動で顔検出して使用。
  なければ画像中央をクロップして使用。
"""

from pathlib import Path
import base64
import io

import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ═══════════════════════════════════════════════════════════════════
#  授業で学んだ3つの3×3カーネル
# ═══════════════════════════════════════════════════════════════════

KERNEL_BLUR = np.ones((3, 3), dtype=np.float32) / 9.0
KERNEL_KX   = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
KERNEL_KY   = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

FEATURE_NAMES = [
    "orig_mean", "orig_std", "orig_min", "orig_max", "orig_median",
    "blur_mean", "blur_std",
    "kx_abs_mean", "kx_abs_std", "kx_abs_max",
    "ky_abs_mean", "ky_abs_std", "ky_abs_max",
    "edge_mag_mean", "edge_mag_std", "edge_density",
    "edge_dir_horizontal_ratio", "edge_dir_vertical_ratio",
    "edge_dir_diagonal_strength", "edge_dir_entropy",
    "edge_center_ratio", "edge_lr_asymmetry", "edge_tb_asymmetry",
    "texture_contrast", "texture_smoothness",
]

CATEGORIES = ["freshwater", "marine", "migratory"]
CAT_JP     = {"freshwater": "淡水魚", "marine": "海水魚", "migratory": "回遊魚"}


# ═══════════════════════════════════════════════════════════════════
#  起動時：モデル構築
# ═══════════════════════════════════════════════════════════════════

print("=" * 60)
print("モデル初期化中...")
print("=" * 60)

# 設定（環境に合わせて変更可）
FEATURES_CSV = Path("data/pilot_features.csv")
YOLO_WEIGHTS = Path("best.pt")  # 任意

if not FEATURES_CSV.exists():
    raise SystemExit(
        f"❌ {FEATURES_CSV} が見つかりません。\n"
        f"   data/フォルダに pilot_features.csv を配置してください。"
    )

# モデル学習
df = pd.read_csv(FEATURES_CSV)
print(f"  訓練データ: {len(df)}サンプル, {len(FEATURE_NAMES)}特徴量")

X = df[FEATURE_NAMES].to_numpy(dtype=float)
y = df["habitat"].to_numpy()

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("clf",   LogisticRegression(penalty="l2", C=1.0, solver="saga",
                                  max_iter=10000, random_state=42)),
])
pipeline.fit(X, y)
print(f"  ✅ Ridgeモデル構築完了（クラス: {list(pipeline.classes_)}）")

# YOLO（任意）
yolo_model = None
if YOLO_WEIGHTS.exists():
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(str(YOLO_WEIGHTS))
        print(f"  ✅ YOLO顔検出器を読み込み: {YOLO_WEIGHTS}")
    except Exception as e:
        print(f"  ⚠️  YOLO読み込み失敗（手動クロップで動作）: {e}")
else:
    print(f"  ⚠️  best.pt がないため、画像中央を使用します")


# ═══════════════════════════════════════════════════════════════════
#  画像処理関数
# ═══════════════════════════════════════════════════════════════════

def crop_face(img_bgr: np.ndarray) -> np.ndarray:
    """YOLOがあれば顔検出、なければ画像中央をクロップ。"""
    if yolo_model is not None:
        results = yolo_model.predict(img_bgr, conf=0.25, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            best = int(np.argmax(confs))
            x1, y1, x2, y2 = xyxy[best].astype(int)
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cropped = img_bgr[y1:y2, x1:x2]
            if cropped.size > 0:
                return cropped
    # フォールバック：中央クロップ
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    cy, cx = h // 2, w // 2
    half = side // 2
    return img_bgr[cy - half:cy + half, cx - half:cx + half]


def preprocess(face_bgr: np.ndarray, size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def extract_features(img: np.ndarray) -> np.ndarray:
    """25次元特徴ベクトルを抽出（extract_face_features.pyと同じロジック）。"""
    h, w = img.shape
    total = img.size

    blur = cv2.filter2D(img, -1, KERNEL_BLUR)
    kx   = cv2.filter2D(img, -1, KERNEL_KX)
    ky   = cv2.filter2D(img, -1, KERNEL_KY)
    edge = np.sqrt(kx ** 2 + ky ** 2)
    abs_kx, abs_ky = np.abs(kx), np.abs(ky)
    eps = 1e-8

    f_orig = [img.mean(), img.std(), img.min(), img.max(), np.median(img)]
    f_blur = [blur.mean(), blur.std()]
    f_kx   = [abs_kx.mean(), abs_kx.std(), abs_kx.max()]
    f_ky   = [abs_ky.mean(), abs_ky.std(), abs_ky.max()]

    edge_threshold = edge.mean() + edge.std()
    edge_density = (edge > edge_threshold).sum() / total
    f_edge = [edge.mean(), edge.std(), edge_density]

    f_dir_h = (abs_ky > abs_kx).sum() / total
    f_dir_v = (abs_kx > abs_ky).sum() / total
    f_dir_diag = np.minimum(abs_kx, abs_ky).mean()
    angles = np.arctan2(ky, kx)
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + eps)
    f_dir_entropy = -np.sum(hist * np.log(hist + eps))
    f_dir = [f_dir_h, f_dir_v, f_dir_diag, f_dir_entropy]

    cy, cx = h // 2, w // 2
    margin = h // 4
    center = edge[cy - margin:cy + margin, cx - margin:cx + margin]
    f_center = center.mean() / (edge.mean() + eps)
    left_e   = edge[:, :w // 2].mean()
    right_e  = edge[:, w // 2:].mean()
    top_e    = edge[:h // 2, :].mean()
    bottom_e = edge[h // 2:, :].mean()
    f_lr = abs(left_e - right_e) / (left_e + right_e + eps)
    f_tb = abs(top_e - bottom_e) / (top_e + bottom_e + eps)
    f_spatial = [f_center, f_lr, f_tb]

    f_contrast = img.var()
    f_smooth   = 1.0 - 1.0 / (1.0 + img.var() / 255.0 ** 2)
    f_texture = [f_contrast, f_smooth]

    vec = np.array(f_orig + f_blur + f_kx + f_ky + f_edge +
                    f_dir + f_spatial + f_texture, dtype=np.float32)
    return vec


# ═══════════════════════════════════════════════════════════════════
#  Flaskルート
# ═══════════════════════════════════════════════════════════════════

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """画像を受け取り、3カテゴリの確率を返す。"""
    if "image" not in request.files:
        return jsonify({"error": "画像が送信されていません"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # PIL → numpy → BGR
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # クロップ → 前処理 → 特徴抽出
    face = crop_face(img_bgr)
    gray = preprocess(face)
    features = extract_features(gray)

    # 推論
    proba = pipeline.predict_proba([features])[0]
    classes = pipeline.classes_

    result = {
        "probabilities": {
            CAT_JP[c]: float(p) for c, p in zip(classes, proba)
        },
        "predicted": CAT_JP[classes[np.argmax(proba)]],
        "confidence": float(np.max(proba)),
    }

    # クロップ画像をbase64で返す（プレビュー用）
    _, buf = cv2.imencode(".png", face)
    result["cropped_image"] = base64.b64encode(buf.tobytes()).decode("utf-8")

    return jsonify(result)


@app.route("/details")
def details():
    """研究結果の詳細データを返す（プレゼン用の固定値）。"""
    return jsonify({
        "差分画像": {
            "淡水魚 - 海水魚": 40.6,
            "淡水魚 - 回遊魚": 39.9,
            "海水魚 - 回遊魚": 28.2,
        },
        "コサイン類似度行列": {
            "淡水魚": {"淡水魚": 1.0000, "海水魚": 0.9974, "回遊魚": 0.9978},
            "海水魚": {"淡水魚": 0.9974, "海水魚": 1.0000, "回遊魚": 0.9974},
            "回遊魚": {"淡水魚": 0.9978, "海水魚": 0.9974, "回遊魚": 1.0000},
        },
        "ペアごとの類似度": {
            "淡水魚 vs 海水魚": 0.9974,
            "淡水魚 vs 回遊魚": 0.9978,
            "海水魚 vs 回遊魚": 0.9974,
        },
        "最近傍重心の集計": {
            "回遊魚": {"回遊魚": 71, "海水魚": 66, "淡水魚": 28, "total": 165},
            "海水魚": {"回遊魚": 51, "海水魚": 70, "淡水魚": 19, "total": 140},
            "淡水魚": {"回遊魚": 58, "海水魚": 73, "淡水魚": 55, "total": 186},
        },
        "回遊魚個体の内訳": {
            "淡水魚": {"count": 28, "total": 165, "pct": 17.0},
            "海水魚": {"count": 66, "total": 165, "pct": 40.0},
            "回遊魚": {"count": 71, "total": 165, "pct": 43.0},
        },
        "model_info": {
            "model": "Ridge回帰（L2正則化）",
            "features": 25,
            "training_samples": int(len(df)),
            "kernels": ["Blur 3×3", "Kx 3×3（X方向微分）", "Ky 3×3（Y方向微分）"],
        }
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🐟 「うぉ！」魚類識別アプリ 起動中")
    print("=" * 60)
    print("ブラウザで http://localhost:5000 を開いてください\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
