import io
import os
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models
from PIL import Image

app = FastAPI()

# HTML側からのクロスドメインアクセスを許可 (CORS対策)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# ---------------------------------------------------------
# FishNet公式に準じたマルチタスク・モデルの定義
# ---------------------------------------------------------
class FishNetModel(nn.Module):
    def __init__(self, num_orders=4, num_families=3, num_traits=22):
        super(FishNetModel, self).__init__()
        # 特徴抽出バックボーン
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # 最終結合層のバイパス化

        # FishNetのマルチタスクヘッド層
        self.order_layer = nn.Linear(in_features, num_orders)
        self.family_layer = nn.Linear(in_features, num_families)
        self.trait_layer = nn.Linear(in_features, num_traits)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "order": self.order_layer(features),
            "family": self.family_layer(features),
            "traits": self.trait_layer(features)
        }

# ---------------------------------------------------------
# マッピングデータの定義とモデル初期化
# ---------------------------------------------------------
# FishNetデータセットの基準ラベル（必要に応じて拡張してください）
TRAIT_LABELS = ["Freshwater", "Marine", "Brackish", "Pelagic", "Demersal"] + [f"Trait_{i}" for i in range(17)]
ORDER_LABELS = ["Cypriniformes (コイ目)", "Osteoglossiformes (アロワナ目)", "Perciformes (スズキ目)", "Siluriformes (ナマズ目)"]
FAMILY_LABELS = ["Cyprinidae (コイ科)", "Osteoglossidae (アロワナ科)", "Cichlidae (シクリッド科)"]

# 画像前処理パイプライン
transform_pipeline = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# モデルインスタンス化と重みのロード
model = FishNetModel(num_orders=len(ORDER_LABELS), num_families=len(FAMILY_LABELS), num_traits=len(TRAIT_LABELS))

WEIGHTS_FILE = "fishnet_habitat_model.pth"
# 互換性のために、もし以前の2クラス用pthが存在する場合のフォールバック、または新規ウェイト対応
if os.path.exists(WEIGHTS_FILE):
    try:
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
        print(f"成功: '{WEIGHTS_FILE}' をマルチタスクモデルに適用しました。")
    except Exception as e:
        print(f"警告: 重みファイルの読み込みでスキップが発生しました（構造ミスマッチ等）: {e}")

model.to(DEVICE).eval()

# ---------------------------------------------------------
# APIエンドポイント
# ---------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # テンソル変換
    input_tensor = transform_pipeline(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # 確率に変換
        order_probs = torch.softmax(outputs["order"], dim=1).cpu().numpy()[0]
        family_probs = torch.softmax(outputs["family"], dim=1).cpu().numpy()[0]
        trait_probs = torch.sigmoid(outputs["traits"]).cpu().numpy()[0] # 各形質ごとの独立確率
        
    # 最大確率のインデックスを取得
    pred_order_idx = np.argmax(order_probs)
    pred_family_idx = np.argmax(family_probs)
    
    # 形質インデックス0番（Freshwater）の適合スコア
    freshwater_score = float(trait_probs[0])
    is_freshwater = freshwater_score >= 0.5
    
    # フロントエンドへ返す結果オブジェクト
    return {
        "is_freshwater": is_freshwater,
        "freshwater_score": round(freshwater_score * 100, 2),
        "predicted_order": ORDER_LABELS[pred_order_idx],
        "predicted_family": FAMILY_LABELS[pred_family_idx],
        "result_name": "淡水魚 (Freshwater)" if is_freshwater else "淡水魚以外 (Saltwater / Brackish etc.)"
    }