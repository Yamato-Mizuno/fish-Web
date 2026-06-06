import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 1. 基本設定と環境の準備
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# 分類クラスの定義（FishNetのFunctional Traitsに基づく）
CLASS_NAMES = {
    0: "淡水魚以外 (Saltwater / Brackish etc.)", 
    1: "淡水魚 (Freshwater)"
}

# 画像の前処理（PyTorchのモデルが受け付ける形式に変換）
transform_pipeline = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 2. AIモデルのロード（キャッシュ化）
# ==========================================
@st.cache_resource # アプリ起動時に1度だけ読み込み、2回目以降は高速化
def load_fishnet_model():
    """
    ResNet50をベースに、淡水魚(1) or それ以外(0)の2値分類を行うモデルを構築します。
    """
    # 構造を構築 (重みは後からロードするため、ここでは初期状態)
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # FishNetで学習した独自の重みファイル(pth)があれば読み込む
    WEIGHTS_FILE = "fishnet_habitat_model.pth"
    if os.path.exists(WEIGHTS_FILE):
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
        st.sidebar.success(f"学習済み重み '{WEIGHTS_FILE}' を適用しました。")
    else:
        st.sidebar.warning("⚠️ 学習済み重みファイルが見つからないため、ランダム初期値（未学習状態）で動いています。精度の高い判定にはモデルの事前学習が必要です。")
        
    model = model.to(DEVICE)
    model.eval()
    return model

# モデルのインスタンス化
model = load_fishnet_model()

# ==========================================
# 3. WebアプリケーションのUI画面（Streamlit）
# ==========================================
st.set_page_config(page_title="魚類生息地 AI分類器", page_icon="🐟")

st.title("🐟 魚類生息地 AI分類アプリケーション")
st.write("FishNetデータセットの基準に基づき、魚の画像から**「淡水魚」**か**「それ以外（海水魚・汽水魚など）」**かをAIが判定します。")
st.markdown("---")

# ファイルアップローダー
uploaded_file = st.file_uploader(
    "魚の画像をアップロードしてください (PNG, JPG, JPEG)", 
    type=["png", "jpg", "jpeg"]
)

# 画像がアップロードされた場合の処理
if uploaded_file is not None:
    # 1. 画面にアップロード画像を表示
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="アップロードされた画像", width='stretch')
    
    with col2:
        st.subheader("🤖 AIによる解析結果")
        with st.spinner("画像を分析中..."):
            # 2. 画像のテンソル変換とデバイス（CPU/GPU）転送
            input_tensor = transform_pipeline(image).unsqueeze(0).to(DEVICE)
            
            # 3. 推論実行
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
            
            pred_class = predicted_idx.item()
            prob_score = confidence.item() * 100
            
            # 4. 結果を画面に出力
            st.metric(label="判定結果", value=CLASS_NAMES[pred_class])
            st.metric(label="確信度（AIの自信）", value=f"{prob_score:.2f} %")
            
            # 視覚的なフィードバック
            if pred_class == 1:
                st.success("💡 川や湖などに生息する「淡水魚」の可能性が非常に高いです。")
            else:
                st.info("💡 海や河口（汽水域）などに生息する魚の可能性が非常に高いです。")

st.markdown("---")
st.caption("Powered by PyTorch & Streamlit | Dataset: FishNet (ICCV 2023)")