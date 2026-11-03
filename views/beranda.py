import streamlit as st

from core import (
    ASSETS, BATASAN, CARE_NOTE, CLASS_CARE, CLASS_INFO, CLASS_SYMPTOMS, CLASS_TAG,
    footer, html, img_uri, penyakit_img,
)


def _photo(code):
    uri = penyakit_img(code)
    if uri:
        return f'<img class="dphoto" src="{uri}" alt="{code}">'
    sym = img_uri(ASSETS / "symbol.png", "image/png")
    return f'<div class="dphoto ph"><img src="{sym}" alt=""><span>{code}</span></div>'


def _care(code):
    return "".join(f"<li>{t}</li>" for t in CLASS_CARE[code])


# ---------------- Hero ----------------
c1, c2 = st.columns([1.2, 1], gap="large", vertical_alignment="center")
with c1:
    html(
        """
        <div class="eyebrow">● Diagnostik pertanian berbasis AI</div>
        <div class="d-hero">Deteksi Cepat Penyakit <span class="g">Daun Singkong</span> dari Foto</div>
        <p class="lead">Klasifikasi daun singkong dengan EfficientNetB0 dan visualisasi Grad-CAM,
        lengkap dengan cara penanganan awal untuk petani dan penyuluh.</p>
        """
    )
    b1, b2, _ = st.columns([1.25, 1.15, 0.8])
    if b1.button("Mulai Deteksi Sekarang", type="primary", icon=":material/photo_camera:"):
        st.switch_page("views/deteksi.py")
    if b2.button("Kenali Tim", icon=":material/groups:"):
        st.switch_page("views/tim.py")
    html(
        """
        <div style="margin-top:14px">
        <span class="chip-g">Grad-CAM</span><span class="chip">5 kondisi daun</span><span class="chip">Cara penanganan</span>
        </div>
        """
    )
with c2:
    hero_photo = penyakit_img("CMD")
    body = (
        f'<img src="{hero_photo}" alt="Contoh daun CMD">' if hero_photo
        else f'<img src="{img_uri(ASSETS / "symbol.png", "image/png")}" alt="" style="object-fit:contain;height:200px">'
    )
    html(
        f"""
        <div class="card2 mock">
        <div class="mtop"><span>Contoh tampilan hasil</span><span class="chip">CMD</span></div>
        {body}
        <div class="mfoot"><span class="chip-g">Grad-CAM overlay</span><span class="chip">EfficientNetB0</span></div>
        </div>
        """
    )

# ---------------- Arsitektur ----------------
html(
    """
    <div class="arch-head"><div><div class="eyebrow2">Cara kerja model</div>
    <div class="d-title">Penyakit Singkong, Dikenali dari Foto Daun</div></div>
    <p>Gejala penyakit pada daun singkong sering mirip satu sama lain dan baru jelas setelah tanaman rusak.
    sampeu.ai membantu skrining awal dari foto, dan Grad-CAM menunjukkan area foto yang memengaruhi prediksi
    sehingga hasilnya bisa diperiksa, bukan sekadar kotak hitam.</p></div>
    """
)
html(
    """
    <div class="grid grid-4">
    <div class="card2 arch"><div class="idx">01</div><h4>Input citra daun</h4><p>Foto diubah ke RGB berukuran 224×224, lalu diproses dengan preprocessing EfficientNet.</p></div>
    <div class="card2 arch"><div class="idx">02</div><h4>Ekstraksi fitur</h4><p>Backbone EfficientNetB0 mengambil pola warna, tekstur, dan bentuk daun.</p></div>
    <div class="card2 arch"><div class="idx">03</div><h4>Klasifikasi softmax</h4><p>Lapisan akhir menghitung probabilitas untuk lima kelas dan memilih yang tertinggi.</p></div>
    <div class="card2 arch"><div class="idx">04</div><h4>Overlay Grad-CAM</h4><p>Peta panas dari lapisan top_conv ditumpuk di atas foto sebagai penjelasan visual.</p></div>
    </div>
    """
)

# ---------------- Katalog penyakit ----------------
html('<div class="eyebrow2" style="margin-top:44px">Taksonomi lapangan</div>'
     '<div class="d-title" style="margin-top:6px">Katalog 5 Kondisi Daun Singkong</div>'
     '<div class="sub">Gejala utama dan cara penanganan awal. Konsultasikan hasil dengan penyuluh pertanian.</div>')


def dis_card(code):
    tag_cls = "ok" if code == "Healthy" else ("bad" if code in ("CBB", "CBSD", "CMD") else "")
    return (
        f'<div class="card2 dcard"><div class="dtop"><span class="tagc {tag_cls}">{CLASS_TAG[code]}</span>'
        f'<span class="code">{code}</span></div>{_photo(code)}<h4>{CLASS_INFO[code]}</h4>'
        f'<div class="lab">Gejala utama</div><p>{CLASS_SYMPTOMS[code]}</p>'
        f'<div class="lab care">Cara penanganan</div><ul>{_care(code)}</ul></div>'
    )


healthy = (
    '<div class="card2 dcard wide"><div class="inner">' + _photo("Healthy") +
    '<div><div class="dtop"><span class="tagc ok">Kondisi daun sehat</span><span class="code">Healthy</span></div>'
    "<h4>Daun sehat (Healthy)</h4>"
    f'<div class="lab">Ciri</div><p>{CLASS_SYMPTOMS["Healthy"]}</p>'
    f'<div class="lab care">Perawatan</div><ul>{_care("Healthy")}</ul></div></div></div>'
)
html('<div class="grid grid-3">' + "".join(dis_card(c) for c in ["CMD", "CBB", "CGM", "CBSD"]) + healthy + "</div>")
st.caption(CARE_NOTE)

# ---------------- Alur kerja ----------------
html('<div style="text-align:center;margin-top:44px"><span class="eyebrow">ALUR KERJA PENGGUNA</span></div>'
     '<div class="d-title" style="text-align:center;margin-top:0">4 Langkah Mudah Deteksi AI di Lapangan</div>'
     '<div class="sub" style="text-align:center">Dirancang intuitif untuk petani dan penyuluh, tanpa perlu keahlian teknis.</div>')
html(
    """
    <div class="grid grid-4">
    <div class="card2"><div class="num">1</div><h4>Unggah citra daun</h4><p>Unggah foto daun singkong atau ambil langsung lewat kamera. Satu daun, dekat, terang, dan fokus.</p><div class="stepf">Format JPG atau PNG</div></div>
    <div class="card2"><div class="num">2</div><h4>Pra-pemrosesan otomatis</h4><p>Foto diubah ke RGB dan ukurannya disesuaikan secara otomatis, tanpa pengaturan apa pun.</p><div class="stepf">Otomatis</div></div>
    <div class="card2"><div class="num">3</div><h4>Inferensi EfficientNetB0</h4><p>Model menghitung probabilitas lima kelas. Bila ragu, aplikasi menyarankan foto ulang.</p><div class="stepf">Keluaran softmax 5 kelas</div></div>
    <div class="card2"><div class="num">4</div><h4>Grad-CAM dan penanganan</h4><p>Peta panas menunjukkan area yang memengaruhi prediksi, disertai cara penanganan awal dan laporan PDF.</p><div class="stepf">Laporan PDF siap unduh</div></div>
    </div>
    """
)

# ---------------- Spesifikasi & batasan ----------------
html('<div class="eyebrow2" style="margin-top:44px">Model terbuka dan transparan</div>'
     '<div class="d-title" style="margin-top:6px">Spesifikasi dan Batasan</div>'
     '<div class="sub">Transparan tentang apa yang dikerjakan model dan apa yang belum.</div>')
s1, s2 = st.columns(2, gap="medium")
with s1:
    html(
        """
        <div class="card2"><h4>Spesifikasi model</h4>
        <table class="spec">
        <tr><td>Model</td><td>EfficientNetB0, checkpoint seed 123</td></tr>
        <tr><td>Dataset</td><td>Cassava Leaf Disease Classification (Kaggle), 21.397 foto</td></tr>
        <tr><td>Pembagian data</td><td>70/15/15, seed 42 (validasi 3.210, uji 3.209)</td></tr>
        <tr><td>Masukan</td><td>RGB 224×224, efficientnet.preprocess_input</td></tr>
        <tr><td>Keluaran</td><td>5 kelas (softmax)</td></tr>
        <tr><td>Pelatihan</td><td>Dua fase (LR 1e-3 lalu 1e-4), early stopping</td></tr>
        <tr><td>Berkas model</td><td>.keras, 48,9 MB</td></tr>
        </table></div>
        """
    )
with s2:
    items = "".join(f"<li>{b}</li>" for b in BATASAN).replace("**", "")
    html(f'<div class="card2"><h4>Batasan operasional</h4><ul style="color:#475569;font-size:0.92rem;line-height:1.6;padding-left:18px;margin:8px 0 0 0">{items}</ul></div>')

# ---------------- CTA ----------------
html(
    """
    <div class="cta2"><span class="eb">MODUL DETEKSI SIAP DIPAKAI</span>
    <div class="t">Periksa Daun Singkong Anda Hari Ini</div>
    <div class="p">Unggah satu foto daun atau gunakan kamera, lalu lihat kelas, probabilitas, Grad-CAM, dan cara penanganannya.</div></div>
    """
)
if st.button("Buka Modul Deteksi", type="primary", icon=":material/arrow_forward:", key="cta"):
    st.switch_page("views/deteksi.py")

footer()
