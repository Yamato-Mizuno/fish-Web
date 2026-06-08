// ═══════════════════════════════════════════════════════════════════
//  うぉ！アプリ - クライアントサイドJS
// ═══════════════════════════════════════════════════════════════════

const screens = {
    top:    document.getElementById("screen-top"),
    upload: document.getElementById("screen-upload"),
    result: document.getElementById("screen-result"),
};

const uploadArea = document.getElementById("upload-area");
const imageInput = document.getElementById("image-input");
const preview    = document.getElementById("preview");
const placeholder = document.getElementById("upload-placeholder");
const predictBtn = document.getElementById("predict-btn");

let selectedFile = null;

// ═══════════════════════════════════════════════════════════════════
//  画面切替
// ═══════════════════════════════════════════════════════════════════
function showScreen(name) {
    Object.values(screens).forEach(s => s.classList.remove("active"));
    screens[name].classList.add("active");
}

function goToUpload(event) {
    if (event) event.preventDefault();
    showScreen("upload");
}

function goToTop() {
    showScreen("top");
    resetUpload();
}

function resetUpload() {
    selectedFile = null;
    preview.src = "";
    preview.classList.remove("show");
    placeholder.classList.remove("hidden");
    predictBtn.disabled = true;
    imageInput.value = "";
}

// ═══════════════════════════════════════════════════════════════════
//  画像アップロード
// ═══════════════════════════════════════════════════════════════════
uploadArea.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("画像ファイルを選んでください");
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.add("show");
        placeholder.classList.add("hidden");
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// ═══════════════════════════════════════════════════════════════════
//  判定実行
// ═══════════════════════════════════════════════════════════════════
predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    predictBtn.textContent = "判定中...";

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "判定に失敗しました");
        }
        const data = await res.json();
        showResult(data);
    } catch (e) {
        alert("エラー: " + e.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "判定する";
    }
});

// ═══════════════════════════════════════════════════════════════════
//  結果表示
// ═══════════════════════════════════════════════════════════════════
function showResult(data) {
    document.getElementById("predicted-label").textContent = data.predicted;
    document.getElementById("confidence").textContent =
        (data.confidence * 100).toFixed(1) + "%";

    // クロップ画像表示
    document.getElementById("result-cropped").src =
        "data:image/png;base64," + data.cropped_image;

    // 確率バー
    const bars = document.getElementById("probability-bars");
    bars.innerHTML = "";

    const catKey = { "淡水魚": "freshwater", "海水魚": "marine", "回遊魚": "migratory" };
    const order = ["淡水魚", "海水魚", "回遊魚"];

    order.forEach(cat => {
        const prob = data.probabilities[cat] || 0;
        const pct = (prob * 100).toFixed(1);
        const className = catKey[cat];

        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
            <div class="name">
                <span>${cat}</span>
                <span class="pct">${pct}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill ${className}" style="width: 0%"></div>
            </div>
        `;
        bars.appendChild(row);

        // アニメーション
        setTimeout(() => {
            row.querySelector(".prob-bar-fill").style.width = pct + "%";
        }, 100);
    });

    showScreen("result");
}

// ═══════════════════════════════════════════════════════════════════
//  詳細結果モーダル
// ═══════════════════════════════════════════════════════════════════
async function showDetails() {
    const modal = document.getElementById("details-modal");
    modal.classList.add("show");

    try {
        const res = await fetch("/details");
        const data = await res.json();
        renderDetails(data);
    } catch (e) {
        alert("詳細データの取得に失敗しました");
    }
}

function closeDetails() {
    document.getElementById("details-modal").classList.remove("show");
}

// モーダル外側クリックで閉じる
document.getElementById("details-modal").addEventListener("click", (e) => {
    if (e.target.id === "details-modal") closeDetails();
});

function renderDetails(data) {
    // ── 差分画像テーブル ──
    const diff = data["差分画像"];
    let html = "<tr><th>カテゴリペア</th><th>最大差</th></tr>";
    for (const [pair, val] of Object.entries(diff)) {
        html += `<tr><td>${pair}</td><td>${val.toFixed(1)}</td></tr>`;
    }
    document.getElementById("diff-table").innerHTML = html;

    // ── コサイン類似度行列 ──
    const sim = data["コサイン類似度行列"];
    const cats = Object.keys(sim);
    let simHtml = "<tr><th></th>";
    cats.forEach(c => simHtml += `<th>${c}</th>`);
    simHtml += "</tr>";
    cats.forEach(rowCat => {
        simHtml += `<tr><td>${rowCat}</td>`;
        cats.forEach(colCat => {
            const v = sim[rowCat][colCat];
            simHtml += `<td>${v.toFixed(4)}</td>`;
        });
        simHtml += "</tr>";
    });
    document.getElementById("cosine-table").innerHTML = simHtml;

    // ── ペアごとの類似度 ──
    const pair = data["ペアごとの類似度"];
    const pairList = document.getElementById("pair-list");
    pairList.innerHTML = "";
    for (const [p, v] of Object.entries(pair)) {
        const li = document.createElement("li");
        li.innerHTML = `<strong>${p}:</strong> ${v.toFixed(4)}`;
        pairList.appendChild(li);
    }

    // ── 最近傍重心テーブル ──
    const nearest = data["最近傍重心の集計"];
    let nearestHtml = "<tr><th>真のカテゴリ \\ 最近傍</th><th>回遊魚</th><th>海水魚</th><th>淡水魚</th></tr>";
    ["回遊魚", "海水魚", "淡水魚"].forEach(trueCat => {
        const row = nearest[trueCat];
        nearestHtml += `<tr><td>${trueCat}</td>
            <td>${row["回遊魚"]}</td>
            <td>${row["海水魚"]}</td>
            <td>${row["淡水魚"]}</td></tr>`;
    });
    document.getElementById("nearest-table").innerHTML = nearestHtml;

    // ── 回遊魚個体の内訳 ──
    const breakdown = data["回遊魚個体の内訳"];
    const bdEl = document.getElementById("migratory-breakdown");
    bdEl.innerHTML = "";
    ["淡水魚", "海水魚", "回遊魚"].forEach(cat => {
        const item = breakdown[cat];
        const div = document.createElement("div");
        div.className = "breakdown-item";
        div.innerHTML = `
            <div class="cat-name">→ ${cat}</div>
            <div class="cat-pct">${item.pct}%</div>
            <div class="cat-count">${item.count}/${item.total}</div>
        `;
        bdEl.appendChild(div);
    });

    // ── モデル情報 ──
    const info = data["model_info"];
    const infoEl = document.getElementById("model-info");
    infoEl.innerHTML = `
        <li><strong>モデル:</strong> ${info.model}</li>
        <li><strong>特徴量数:</strong> ${info.features}次元</li>
        <li><strong>訓練サンプル:</strong> ${info.training_samples}枚</li>
        <li><strong>使用カーネル:</strong> ${info.kernels.join(", ")}</li>
    `;
}
