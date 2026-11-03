import streamlit as st

from core import ASSETS, footer, html, img_uri

# ----- Ubah di sini bila ada perubahan nama, peran, atau foto (berkas di assets/team/) -----
TEAM = [
    {"nama": "Muhammad Adam Ilyasa", "peran": "Ketua Kelompok", "model": "Model MobileNetV2", "foto": "adam.jpg"},
    {"nama": "Daffa' Raihandi", "peran": "Integrasi Aplikasi", "model": "Model EfficientNetB0", "foto": "daffa.jpg"},
    {"nama": "Muhamad Arif Darmawan", "peran": "Pengelola Protokol", "model": "Model ResNet50", "foto": "arif.jpg"},
]
DOSEN = {"nama": "Dodik Ariyanto, S.T.P., M.Sc.", "peran": "Dosen Pengampu", "foto": "dodik.jpg"}
GITHUB = "https://github.com/Wandarix/Sampeu.ai"
# -------------------------------------------------------------------------------------------

html(
    """
    <div style="text-align:center">
    <div class="eyebrow">● Penelitian &amp; pengembangan · Kelompok 11 Kelas A</div>
    <div class="d-hero">Tim di Balik <span class="g">sampeu.ai</span></div>
    <p class="lead" style="margin:0 auto 12px auto">Proyek mata kuliah Machine Learning (Semester 5):
    Klasifikasi Penyakit Daun Singkong pada Data Tidak Seimbang - Perbandingan MobileNetV2, ResNet50, dan EfficientNetB0.</p>
    <div><span class="chip-g">Data Cassava (Kaggle)</span><span class="chip">Grad-CAM</span><span class="chip">Streamlit</span></div>
    </div>
    """
)

html(
    """
    <div class="grid grid-3">
    <div class="card2"><div class="k">Latar belakang</div><h4 style="margin-top:6px">Gejala yang mirip</h4>
    <p>Gejala penyakit pada daun singkong sering mirip satu sama lain. sampeu.ai mencoba membantu skrining awal lewat foto,
    dengan Grad-CAM agar hasilnya bisa diperiksa.</p></div>
    <div class="card2"><div class="k">Dataset</div><h4 style="margin-top:6px">Cassava Leaf Disease</h4>
    <p>Dataset Kaggle dengan 21.397 foto dan 5 kelas (CBB, CBSD, CGM, CMD, Healthy). Distribusi kelasnya tidak seimbang.</p></div>
    <div class="card2"><div class="k">Model aplikasi</div><h4 style="margin-top:6px">EfficientNetB0</h4>
    <p>Tiga arsitektur dibandingkan pada tahap eksperimen. Aplikasi memakai EfficientNetB0 checkpoint seed 123.</p></div>
    </div>
    """
)

html('<div class="d-title" style="text-align:center">Anggota Kelompok</div>')
cards = ""
for m in TEAM:
    cards += (
        f'<div class="photo"><img src="{img_uri(ASSETS / "team" / m["foto"])}" alt="{m["nama"]}">'
        f'<div class="ov"><span class="role">{m["peran"]}</span>'
        f'<div class="nm">{m["nama"]}</div><div class="mdl">{m["model"]}</div></div></div>'
    )
html(f'<div class="grid grid-3">{cards}</div>')

html('<div class="d-title" style="text-align:center">Dosen Pengampu</div>')
html(
    f"""
    <div class="card2 lect" style="max-width:560px;margin:0 auto">
    <img src="{img_uri(ASSETS / "team" / DOSEN["foto"])}" alt="{DOSEN["nama"]}">
    <div><h4 style="margin:0">{DOSEN["nama"]}</h4><p>{DOSEN["peran"]}</p>
    <div class="tags"><span class="tag-o">Machine Learning</span><span class="tag-o">Supervisi akademik</span></div></div>
    </div>
    """
)

html('<div class="d-title" style="text-align:center">Teknologi yang Dipakai</div>')
html(
    """
    <div class="grid grid-4">
    <div class="card2"><div class="k">Model</div><h4 style="margin-top:6px">EfficientNetB0</h4><p>Backbone klasifikasi dengan 5 kelas keluaran.</p></div>
    <div class="card2"><div class="k">Framework</div><h4 style="margin-top:6px">TensorFlow / Keras</h4><p>Pelatihan dan inferensi model (.keras).</p></div>
    <div class="card2"><div class="k">Penjelasan</div><h4 style="margin-top:6px">Grad-CAM</h4><p>Peta panas area foto yang memengaruhi prediksi.</p></div>
    <div class="card2"><div class="k">Antarmuka</div><h4 style="margin-top:6px">Streamlit</h4><p>Aplikasi web yang berjalan di Streamlit Community Cloud.</p></div>
    </div>
    """
)

html(
    """
    <div class="cta2"><span class="eb">KODE TERBUKA</span>
    <div class="t">Ingin Melihat atau Mencoba Kodenya?</div>
    <div class="p">Seluruh kode aplikasi sampeu.ai tersedia di GitHub untuk dipelajari dan dikembangkan lebih lanjut.</div></div>
    """
)
st.link_button("Lihat kode di GitHub", GITHUB, icon=":material/code:", type="primary")

footer()
