import io
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models
from PIL import Image

app = FastAPI()  # 👈 これが確実に書かれている必要があります！

# HTML側からのアクセスを許可する設定（CORS対策）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AIモデルの準備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 🎯 修正後：0 と 1 の中身を逆にします
CLASS_NAMES = {0: "淡水魚 (Freshwater)", 1: "淡水魚以外 (Saltwater / Brackish etc.)"}

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
if os.path.exists("fishnet_habitat_model.pth"):
    model.load_state_dict(torch.load("fishnet_habitat_model.pth", map_location=DEVICE))
model.to(DEVICE).eval()

# 画像を受け取って判定結果を返すAPIエンドポイント
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    input_tensor = transform_pipeline(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
    return {
        "result_name": CLASS_NAMES[predicted_idx.item()],
        "confidence": round(confidence.item() * 100, 2)
    }