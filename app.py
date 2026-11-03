import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

BASE = Path(__file__).parent
ASSETS = BASE / "assets"

# ----------------------- Model final (dikunci, lihat SERAH_TERIMA) -----------------------
MODEL_PATH = BASE / "efficientnetb0_seed123_best.keras"
META = json.loads((BASE / "class_names.json").read_text())
CLASS_NAMES = META["labels"]  # ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
IMG_SIZE = tuple(META["image_size"])  # (224, 224)

# Ambang "Tidak yakin". WAJIB dipilih dari VALIDATION, bukan test.
# Nilai di bawah hanya placeholder sampai kelompok menetapkan ambang dari validation.
CONF_THRESHOLD = 0.50  # probabilitas tertinggi minimal
MARGIN_THRESHOLD = 0.10  # selisih minimal antara kelas teratas dan kedua

CLASS_INFO = {
    "CBB": "Cassava Bacterial Blight (hawar bakteri)",
    "CBSD": "Cassava Brown Streak Disease (garis cokelat)",
    "CGM": "Cassava Green Mottle (bercak hijau)",
    "CMD": "Cassava Mosaic Disease (mosaik)",
    "Healthy": "Daun sehat",
}

# ----------------------- Palet warna sampeu.ai (tema terang) -----------------------
NAVY, EMERALD, MINT, COBALT = "#00173D", "#059669", "#10B981", "#1D4ED8"
SLATE, BORDER, ICE = "#475569", "#E2E8F0", "#DBEAFE"
STATUS = {  # (teks, latar)
    "healthy": ("#059669", "#D1FAE5"),
    "uncertain": ("#D97706", "#FEF3C7"),
    "disease": ("#DC2626", "#FEE2E2"),
}

CSS = f"""
<style>
h1, h2, h3, h4 {{ color: {NAVY}; }}
.stButton > button {{ background: {EMERALD}; color: #fff; border: 0; border-radius: 10px; }}
.stButton > button:hover {{ background: {MINT}; color: #fff; }}
.badge {{
    display: inline-block; background: {ICE}; color: {COBALT};
    border-radius: 999px; padding: 2px 10px; font-size: 0.8rem;
    font-weight: 600; margin-right: 6px;
}}
.result {{ border-radius: 12px; padding: 12px 14px; font-weight: 600; margin: 8px 0; }}
.bar-row {{ margin: 6px 0; font-size: 0.9rem; color: {SLATE}; }}
.bar-bg {{ background: {BORDER}; border-radius: 999px; height: 8px; overflow: hidden; }}
.bar-fill {{ height: 8px; border-radius: 999px; }}
[data-testid="stFileUploaderDropzone"] {{
    background: #ECFDF5; border: 1px dashed {EMERALD}; border-radius: 12px;
}}
.hint {{ color: {SLATE}; font-size: 0.85rem; }}
</style>
"""


# ----------------------- Model & prediksi -----------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def prepare(pil_image):
    """Sama dengan training: RGB, resize 224x224, float32, efficientnet.preprocess_input."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)
    x = preprocess_input(x)
    return np.expand_dims(x, 0)


def gradcam(model, x, class_idx):
    """Grad-CAM untuk struktur: Input -> efficientnetb0 (bersarang) -> GAP -> Dropout -> Dense."""
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    conv_model = tf.keras.Model(base.input, [base.get_layer("top_conv").output, base.output])
    idx = model.layers.index(base)
    before, after = model.layers[1:idx], model.layers[idx + 1:]

    with tf.GradientTape() as tape:
        h = tf.convert_to_tensor(x)
        for l in before:
            h = l(h)
        conv_out, h = conv_model(h)
        for l in after:
            h = l(h)
        score = h[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]).numpy()
    return cam / (cam.max() + 1e-8)


def overlay(img, cam, alpha=0.4):
    base = img.convert("RGB").resize(IMG_SIZE)
    heat = np.array(Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE)) / 255.0
    color = (colormaps["jet"](heat)[..., :3] * 255).astype("uint8")
    return Image.fromarray(((1 - alpha) * np.array(base) + alpha * color).astype("uint8"))


# ----------------------- Tampilan -----------------------
def result_box(kind, html):
    fg, bg = STATUS[kind]
    st.markdown(
        f'<div class="result" style="color:{fg};background:{bg}">{html}</div>',
        unsafe_allow_html=True,
    )


def bar(label, p, top=False):
    color = EMERALD if top else MINT
    st.markdown(
        f'<div class="bar-row">{label}: <b>{p:.1%}</b><div class="bar-bg">'
        f'<div class="bar-fill" style="width:{p * 100:.1f}%;background:{color}"></div></div></div>',
        unsafe_allow_html=True,
    )


def about():
    with st.expander("Tentang model dan batasan"):
        st.markdown(
            """
**Model:** EfficientNetB0 (seed 123), input 224x224, 5 kelas.

**Hasil test internal (3.209 foto, rata-rata 3 seed):** akurasi 0,7915 dan macro-F1 0,6864.
Recall rata-rata: CBB 0,6667, CBSD 0,6991, CGM 0,6713, CMD 0,8505, Healthy 0,7332.

**Batasan**
- Aplikasi ini alat bantu skrining citra, **bukan diagnosis pasti**.
- Data latih berasal dari survei di Uganda dan belum mewakili kebun di Indonesia.
- Foto lokal tanpa verifikasi penyuluh hanya cocok sebagai evaluasi kualitatif.
- CBB paling sulit dikenali, jadi gejala awal CBB bisa terlewat.
- Label "Tidak yakin" muncul bila probabilitas tertinggi rendah atau dua kelas teratas hampir sama.
"""
        )


def main():
    st.set_page_config(page_title="sampeu.ai", page_icon=Image.open(ASSETS / "symbol.png"))
    st.markdown(CSS, unsafe_allow_html=True)
    st.image(str(ASSETS / "logo.png"), width=300)
    st.markdown('<p class="hint">Deteksi penyakit daun singkong dari foto.</p>', unsafe_allow_html=True)

    show_cam = st.sidebar.checkbox("Tampilkan Grad-CAM", value=True)
    st.sidebar.caption("Model: EfficientNetB0 (seed 123), terkunci.")

    tab_up, tab_cam = st.tabs(["Unggah foto", "Kamera"])
    with tab_up:
        f1 = st.file_uploader("Foto daun", type=["jpg", "jpeg", "png"])
    with tab_cam:
        f2 = st.camera_input("Ambil foto daun")
    file = f1 or f2

    if file is None:
        st.info("Unggah atau ambil foto satu daun singkong, dekat, terang, dan fokus.")
        about()
        return

    img = Image.open(file).convert("RGB")
    st.image(img, caption="Foto masukan", width=300)

    model = load_model()
    x = prepare(img)
    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000

    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    conf, margin = float(probs[top]), float(probs[top] - probs[second])

    size_mb = MODEL_PATH.stat().st_size / 1e6
    st.markdown(
        f'<span class="badge">EfficientNetB0</span>'
        f'<span class="badge">{size_mb:.1f} MB</span>'
        f'<span class="badge">{ms:.0f} ms</span>',
        unsafe_allow_html=True,
    )

    if conf < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
        result_box(
            "uncertain",
            f"Tidak yakin, ambil foto ulang. Dugaan terdekat: {CLASS_NAMES[top]} ({conf:.1%}) "
            f"atau {CLASS_NAMES[second]} ({probs[second]:.1%}).",
        )
    else:
        label = CLASS_NAMES[top]
        result_box("healthy" if label == "Healthy" else "disease",
                   f"{label} - {CLASS_INFO[label]} ({conf:.1%})")

    st.caption("Top-3 prediksi")
    for rank, i in enumerate(order[:3]):
        bar(CLASS_NAMES[i], float(probs[i]), top=(rank == 0))

    if show_cam:
        try:
            cam = gradcam(model, x, top)
            st.image(overlay(img, cam), caption="Grad-CAM (area yang memengaruhi prediksi)")
        except Exception as e:
            st.caption(f"Grad-CAM tidak tersedia ({e}).")

    st.caption("Hasil ini alat bantu skrining, bukan diagnosis pasti. Konsultasikan dengan penyuluh pertanian.")
    about()


main()
