import time

import numpy as np
import streamlit as st
from PIL import Image

from core import (
    BATASAN, CARE_NOTE, CLASS_CARE, CLASS_INFO, CLASS_NAMES, CLASS_SYMPTOMS, CONF_THRESHOLD, EMERALD,
    MARGIN_THRESHOLD, MINT, STATUS, bar, build_report, footer, gradcam, html, load_model, overlay, pct, penyakit_img,
    prepare,
)

html(
    """
    <div class="eyebrow">● Inferensi EfficientNetB0</div>
    <div class="d-hero" style="font-size:clamp(1.8rem,4vw,2.8rem)">Deteksi Penyakit Daun Singkong</div>
    <p class="lead">Unggah foto daun singkong atau ambil lewat kamera untuk diagnosis berbasis EfficientNetB0 dengan visualisasi Grad-CAM dan cara penanganannya.</p>
    """
)

left, right = st.columns([1, 1.35], gap="large")

# ---------------- Kiri: masukan ----------------
with left:
    with st.container(border=True, key="card-input"):
        st.markdown("#### Panel masukan citra")
        tab_up, tab_cam = st.tabs(["Unggah foto", "Kamera"])
        with tab_up:
            f1 = st.file_uploader("Foto daun", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        with tab_cam:
            f2 = st.camera_input("Ambil foto daun", label_visibility="collapsed")
        file = f1 or f2
        img = Image.open(file).convert("RGB") if file is not None else None
        if img is not None:
            st.image(img, caption="Foto masukan", width=360)

    with st.container(border=True, key="card-tips"):
        st.markdown("#### Panduan pengambilan foto")
        html(
            """
            <div class="sub"><b>1. Cahaya merata.</b> Hindari bayangan keras dan cahaya tepat di belakang daun.</div>
            <div class="sub"><b>2. Dekat dan fokus.</b> Daun mengisi sebagian besar bingkai dan tidak buram.</div>
            <div class="sub"><b>3. Satu daun, latar sederhana.</b> Hindari tangan, tanah, atau daun bertumpuk.</div>
            """
        )

# ---------------- Kanan: hasil ----------------
with right:
    if img is None:
        html(
            """
            <div class="card2" style="min-height:260px;display:flex;flex-direction:column;justify-content:center;text-align:center">
            <h4>Hasil akan tampil di sini</h4>
            <p>Unggah atau ambil satu foto daun singkong untuk memulai.</p></div>
            """
        )
    else:
        model = load_model()
        x = prepare(img)
        t0 = time.perf_counter()
        probs = model.predict(x, verbose=0)[0]
        ms = (time.perf_counter() - t0) * 1000

        order = np.argsort(probs)[::-1]
        top, second = int(order[0]), int(order[1])
        conf, margin = float(probs[top]), float(probs[top] - probs[second])
        label = CLASS_NAMES[top]

        if conf < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
            kind = "uncertain"
            title = "Tidak yakin, ambil foto ulang"
            text = (
                f"Dugaan terdekat: {label} ({pct(conf)}) atau {CLASS_NAMES[second]} "
                f"({pct(float(probs[second]))}). Gunakan daun utuh, terang, dan fokus."
            )
        else:
            kind = "healthy" if label == "Healthy" else "disease"
            title = f"{label} · {CLASS_INFO[label]}"
            text = f"Ciri umum: {CLASS_SYMPTOMS[label]}"
        fg, bg = STATUS[kind]
        deg = max(min(conf, 1.0), 0.0) * 100
        html(
            f"""
            <div class="res" style="background:{bg};border-color:{fg}40">
            <div><div class="tag" style="color:{fg}">HASIL PREDIKSI</div>
            <div class="ttl" style="color:{fg}">{title}</div>
            <div class="txt">{text}</div>
            <div style="margin-top:10px"><span class="chip">Waktu inferensi {ms:.0f} ms</span><span class="chip-g">EfficientNetB0</span></div></div>
            <div class="ring" style="background:conic-gradient({fg} {deg:.1f}%, #ffffff 0)">
            <div class="ring-in"><b>{pct(conf)}</b><span>KEYAKINAN</span></div></div></div>
            """
        )

        try:
            cam = gradcam(model, x, top)
            ov = overlay(img, cam)
            cam_err = None
        except Exception as e:  # Grad-CAM gagal tidak boleh menggagalkan hasil utama
            ov, cam_err = None, e

        with st.container(border=True, key="card-cam"):
            st.markdown("#### Visualisasi Grad-CAM")
            mode = st.radio(
                "Tampilan", ["Berdampingan", "Overlay", "Sembunyikan"], horizontal=True,
                label_visibility="collapsed",
            )
            if mode != "Sembunyikan":
                if ov is None:
                    st.caption(f"Grad-CAM tidak tersedia ({cam_err}).")
                else:
                    ov_s = ov.resize((320, 320))
                    if mode == "Berdampingan":
                        a, b = st.columns(2)
                        a.image(img.resize((320, 320)), caption="Citra asli")
                        b.image(ov_s, caption="Peta aktivasi (Grad-CAM)")
                    else:
                        st.image(ov_s, caption="Overlay Grad-CAM")
                    html(
                        """
                        <div class="infobox"><b>Cara membaca Grad-CAM</b>
                        <p>Area hangat (merah/kuning) adalah bagian foto yang paling memengaruhi prediksi.
                        Bila area itu jatuh di latar atau tangan, bukan di daun, anggap hasilnya kurang bisa dipercaya
                        dan ambil foto ulang. Ini bukan penanda lokasi penyakit yang pasti.</p></div>
                        """
                    )

        with st.container(border=True, key="card-prob"):
            st.markdown("#### Distribusi probabilitas")
            for rank, i in enumerate(order):
                bar(f"{CLASS_NAMES[i]} · {CLASS_INFO[CLASS_NAMES[i]]}", float(probs[i]),
                    color=fg if rank == 0 else MINT)

        if kind != "uncertain":
            with st.container(border=True, key="card-care"):
                head = "Rekomendasi Tindakan Agronomi" if label != "Healthy" else "Perawatan Tanaman"
                badge = '<span class="tagc bad">Segera</span>' if kind == "disease" else '<span class="tagc ok">Rutin</span>'
                html(f'<div class="sophead"><h4>{head}</h4>{badge}</div>')
                items = "".join(
                    f'<div class="care-item"><b>{i}.</b> {t}</div>' for i, t in enumerate(CLASS_CARE[label], 1)
                )
                ref = penyakit_img(label)
                if ref:
                    html(
                        f'<div class="care-top"><img class="dphoto" src="{ref}" alt="{label}">'
                        f'<div><div class="k">Contoh foto kelas {label}</div>'
                        f'<p style="margin:4px 0 0 0;color:#475569;font-size:0.9rem">{CLASS_INFO[label]}. '
                        f'{CLASS_SYMPTOMS[label]}</p></div></div>'
                    )
                html(f'<div class="care-grid">{items}</div>')
                st.caption(CARE_NOTE)

        try:
            pdf = build_report(img, ov, probs, order, label, title, kind, ms)
            st.download_button(
                "Unduh Laporan Diagnosis (PDF)", pdf, file_name=f"laporan_sampeu_{label}.pdf",
                mime="application/pdf", type="primary", icon=":material/download:",
            )
        except Exception as e:
            st.caption(f"Laporan PDF tidak tersedia ({e}).")

        st.caption("Hasil ini alat bantu skrining, bukan diagnosis pasti. Konsultasikan dengan penyuluh pertanian.")

with st.expander("Tentang model dan batasan"):
    st.markdown(
        "**Model:** EfficientNetB0 (seed 123), input 224x224, 5 kelas.\n\n"
        "**Batasan**\n" + "\n".join(f"- {b}" for b in BATASAN)
    )

footer()
