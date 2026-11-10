"""Bagian bersama sampeu.ai: konfigurasi, model, prediksi, Grad-CAM, gaya."""
import base64
import json
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib import colormaps
from PIL import Image

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
# Ciri umum untuk pengenalan awal (bukan panduan diagnosis)
CLASS_SYMPTOMS = {
    "CBB": "Bercak bersudut berair pada daun, lalu layu dan mati pucuk.",
    "CBSD": "Warna kuning mengikuti tulang daun; garis cokelat dapat muncul pada batang.",
    "CGM": "Bercak mosaik hijau muda dan tua; daun bisa mengecil.",
    "CMD": "Mosaik kuning-hijau, daun keriting atau terpelintir.",
    "Healthy": "Warna hijau merata tanpa bercak atau perubahan bentuk.",
}

# Cara penanganan awal (ringkasan umum; disusun dari fact sheet Pacific Pests, Pathogens & Weeds (CABI),
# UF/IFAS EDIS, dan laporan Crop Protection Programme UK). Bukan pengganti saran penyuluh.
CLASS_CARE = {
    "CMD": [
        "Tanam stek dari tanaman sehat tanpa gejala, dan pilih varietas tahan CMD bila tersedia.",
        "Cabut dan musnahkan tanaman bergejala bila jumlahnya masih sedikit.",
        "Pantau kutu kebul (Bemisia tabaci), serangga penular virus ini.",
    ],
    "CBB": [
        "Pakai stek dari tanaman sehat; bagian batang yang tua lebih baik.",
        "Bersihkan alat kebun dengan disinfektan dan hindari masuk kebun saat tanaman basah.",
        "Setelah panen, bakar atau kubur sisa tanaman, lalu beri jeda tanam singkong di lahan yang sama.",
    ],
    "CGM": [
        "Pilih stek tanpa gejala; cabut dan musnahkan tanaman bergejala sejak dini.",
        "Periksa bagian bawah daun: gejala serupa bisa disebabkan tungau hijau singkong.",
        "Hindari menyemprot pestisida tanpa saran penyuluh, karena dapat membunuh musuh alami tungau.",
    ],
    "CBSD": [
        "Gunakan stek bebas virus dan varietas tahan atau toleran CBSD bila tersedia.",
        "Cabut tanaman bergejala dan kendalikan kutu kebul penular virus.",
        "Jangan memakai stek dari kebun yang terinfeksi; kerusakan umbi sering baru terlihat saat panen.",
    ],
    "Healthy": [
        "Pantau daun secara rutin dan foto ulang bila muncul bercak atau perubahan bentuk.",
        "Gunakan stek sehat dan jaga kebersihan alat kebun.",
        "Hindari membawa stek atau tanaman dari kebun yang bergejala.",
    ],
}
CLASS_TAG = {"CMD": "Virus · kutu kebul", "CBB": "Bakteri", "CGM": "Bercak hijau", "CBSD": "Virus · kutu kebul", "Healthy": "Daun sehat"}
CARE_NOTE = (
    "Ringkasan umum dari fact sheet Pacific Pests, Pathogens & Weeds (CABI), UF/IFAS EDIS, dan laporan "
    "Crop Protection Programme (UK). Bukan pengganti saran penyuluh; konsultasikan dengan penyuluh atau dinas pertanian setempat."
)


BATASAN = [
    "Aplikasi ini alat bantu skrining citra, **bukan diagnosis pasti**.",
    "Data latih berasal dari survei di Uganda dan belum mewakili kebun di Indonesia.",
    "Foto lokal tanpa verifikasi penyuluh hanya cocok sebagai evaluasi kualitatif.",
    "Penyakit dengan gejala awal samar (misalnya CBB) bisa terlewat, jadi tetap periksa kebun secara langsung.",
    'Label "Tidak yakin" muncul bila probabilitas tertinggi rendah atau dua kelas teratas hampir sama.',
]

# ----------------------- Palet warna sampeu.ai -----------------------
NAVY, EMERALD, MINT, COBALT = "#00173D", "#059669", "#10B981", "#1D4ED8"
SLATE, BORDER, ICE, MINT_TINT = "#475569", "#E2E8F0", "#DBEAFE", "#ECFDF5"
STATUS = {  # (teks, latar)
    "healthy": ("#059669", "#D1FAE5"),
    "uncertain": ("#B45309", "#FEF3C7"),
    "disease": ("#DC2626", "#FEE2E2"),
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; background-color: #F8FAFC;
    background-image: linear-gradient(rgba(248,250,252,0.62), rgba(248,250,252,0.62)), url("%POLA%");
    background-size: auto, 440px 440px; }
.block-container { max-width: 1200px; padding-top: 3.2rem; }
[data-testid="stHeader"] {
    background: rgba(255,255,255,0.78); backdrop-filter: blur(20px) saturate(180%);
    border-bottom: 1px solid %BORDER%;
}
h1, h2, h3, h4, .pjs { font-family: 'Plus Jakarta Sans', sans-serif !important; color: %NAVY%; letter-spacing: -0.02em; }

[data-testid="stBaseButton-primary"] {
    background: %EMERALD%; border: 0; border-radius: 999px; font-weight: 600; padding: 0.55rem 1.4rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.25);
}
[data-testid="stBaseButton-primary"]:hover { background: %EMERALD%; box-shadow: 0 4px 14px rgba(5,150,105,0.3); }
[data-testid="stBaseButton-secondary"] {
    border-radius: 999px; border: 1px solid %BORDER%; background: rgba(255,255,255,0.8); font-weight: 600;
}
[data-testid="stBaseButton-secondary"]:hover { background: %ICE%; border-color: %ICE%; color: %NAVY%; }
[data-testid="stFileUploaderDropzone"] {
    background: %MINT_TINT%; border: 1px dashed %EMERALD%; border-radius: 16px;
}
div[class*="st-key-card-"] {
    background: #fff; border: 1px solid %BORDER% !important; border-radius: 20px;
    padding: 0.9rem 1rem 1.2rem 1rem !important; overflow: hidden; box-sizing: border-box; min-width: 0; box-shadow: 0 4px 20px -2px rgba(0,23,61,0.04), 0 2px 6px -1px rgba(0,23,61,0.02);
}

/* Halaman deteksi: di bawah 992px kolom ditumpuk agar foto/hasil tidak saling menimpa */
.res { flex-wrap: wrap; }
.res > div:first-child { min-width: 0; flex: 1 1 220px; }
.res .ttl { overflow-wrap: anywhere; }
div[class*="st-key-det-layout"] [data-testid="stImage"] img { max-width: 100%; height: auto; }
div[class*="st-key-det-layout"] [data-testid="stColumn"] { min-width: 0; }
div[class*="st-key-card-"] > [data-testid="stVerticalBlock"], div[class*="st-key-card-"] [data-testid="stMarkdownContainer"] { min-width: 0; max-width: 100%; }
.infobox, .bar-row { box-sizing: border-box; max-width: 100%; }
.bar-row .top span { min-width: 0; overflow-wrap: anywhere; padding-right: 8px; }
@media (max-width: 992px) {
    div[class*="st-key-det-layout"] [data-testid="stHorizontalBlock"]:not([data-testid="stColumn"] [data-testid="stHorizontalBlock"]) {
        flex-direction: column !important; gap: 1rem !important;
    }
    div[class*="st-key-det-layout"] [data-testid="stHorizontalBlock"]:not([data-testid="stColumn"] [data-testid="stHorizontalBlock"]) > [data-testid="stColumn"] {
        width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important;
    }
}
/* Sembunyikan ikon tautan (anchor) otomatis di samping judul */
[data-testid="stHeaderActionElements"], .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, .stMarkdown h4 a { display: none !important; }
.linkcard { display: block; text-decoration: none !important; color: #00173D !important; transition: transform .2s ease, box-shadow .2s ease; }
.linkcard:hover { transform: translateY(-2px); box-shadow: 0 10px 28px -8px rgba(5,150,105,0.25); }
.linkcard { cursor: pointer; }
 .linkcard:hover, .linkcard:visited, .linkcard:active { color: #00173D !important; text-decoration: none !important; }
   .linkcard h4, .linkcard p { color: #00173D !important; }
.linkcard h4 { display: flex; align-items: center; gap: 8px; }
.linkcard .lnk { width: 18px; height: 18px; flex: none; background: %EMERALD%; opacity: 0; transform: translateX(-4px);
    transition: opacity .2s ease, transform .2s ease;
    -webkit-mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'><path d='M10 13a5 5 0 0 0 7.07 0l3-3a5 5 0 0 0-7.07-7.07l-1.5 1.5'/><path d='M14 11a5 5 0 0 0-7.07 0l-3 3a5 5 0 0 0 7.07 7.07l1.5-1.5'/></svg>") center / contain no-repeat;
    mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'><path d='M10 13a5 5 0 0 0 7.07 0l3-3a5 5 0 0 0-7.07-7.07l-1.5 1.5'/><path d='M14 11a5 5 0 0 0-7.07 0l-3 3a5 5 0 0 0 7.07 7.07l1.5-1.5'/></svg>") center / contain no-repeat; }
.linkcard:hover .lnk, .linkcard:focus-visible .lnk { opacity: 1; transform: none; }

.eyebrow {
    display: inline-block; background: %MINT_TINT%; color: %EMERALD%; border: 1px solid #A7F3D0;
    border-radius: 999px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600; margin-bottom: 14px;
}
.d-hero { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; color: %NAVY%;
    font-size: clamp(2rem, 4.6vw, 3.4rem); line-height: 1.1; letter-spacing: -0.03em; margin-bottom: 14px; }
.d-hero .g { color: %EMERALD%; }
.d-title { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: %NAVY%;
    font-size: clamp(1.4rem, 3vw, 2.1rem); line-height: 1.2; letter-spacing: -0.025em; margin: 36px 0 6px 0; }
.lead { color: %SLATE%; font-size: 1.05rem; line-height: 1.65; max-width: 620px; margin-bottom: 16px; }
.sub { color: %SLATE%; font-size: 0.95rem; margin-bottom: 14px; }
.chip { display: inline-block; background: %ICE%; color: %COBALT%; border-radius: 999px;
    padding: 3px 12px; font-size: 0.78rem; font-weight: 600; margin: 0 6px 6px 0; }
.chip-g { display: inline-block; background: %MINT_TINT%; color: %EMERALD%; border-radius: 999px;
    padding: 3px 12px; font-size: 0.78rem; font-weight: 600; margin: 0 6px 6px 0; }

.grid { display: grid; gap: 16px; margin: 10px 0 8px 0; }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }
@media (max-width: 900px) {
    .grid-3, .grid-4 { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 600px) {
    .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
}
.card2 {
    background: #fff; border: 1px solid %BORDER%; border-radius: 20px; padding: 20px 22px;
    box-shadow: 0 4px 20px -2px rgba(0,23,61,0.04), 0 2px 6px -1px rgba(0,23,61,0.02);
}
.card2 h4 { margin: 0 0 6px 0; font-size: 1.05rem; }
.card2 p { margin: 0; color: %SLATE%; font-size: 0.92rem; line-height: 1.55; }
.k { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; color: %SLATE%; text-transform: uppercase; }
.v { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2rem; font-weight: 800; color: %NAVY%; margin: 4px 0; font-variant-numeric: tabular-nums; }
.s { font-size: 0.82rem; color: %SLATE%; }
.num { display: inline-flex; width: 32px; height: 32px; align-items: center; justify-content: center;
    background: %EMERALD%; color: #fff; border-radius: 50%; font-weight: 700; margin-bottom: 10px; font-size: 0.9rem; }
.dis { border-left: 4px solid %EMERALD%; }
.dis .code { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; color: %COBALT%; }
.dis.bad { border-left-color: #DC2626; }
.vis { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; text-align: center; min-height: 260px;
    background: linear-gradient(135deg, %MINT_TINT% 0%, %ICE% 100%); }
.vis img { width: 150px; height: 150px; object-fit: contain; }
.spec { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.spec td { padding: 9px 0; border-bottom: 1px solid %BORDER%; vertical-align: top; }
.spec td:first-child { color: %SLATE%; width: 36%; padding-right: 12px; }
.spec td:last-child { color: %NAVY%; font-weight: 500; }
.cta { background: linear-gradient(135deg, %NAVY% 0%, #0B2A5B 100%); border-radius: 24px; padding: 36px 32px; margin: 36px 0 12px 0; }
.cta .t { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; color: #fff; font-size: 1.7rem; margin-bottom: 8px; }
.cta .p { color: #CBD5E1; max-width: 560px; }

.bar-row { margin: 8px 0; font-size: 0.9rem; color: %SLATE%; }
.bar-row .top { display: flex; justify-content: space-between; margin-bottom: 4px; }
.bar-bg { background: #E6EEFF; border-radius: 999px; height: 8px; overflow: hidden; }
.bar-fill { height: 8px; border-radius: 999px; }

.res { display: flex; justify-content: space-between; align-items: center; gap: 18px;
    border: 1px solid; border-radius: 22px; padding: 22px 24px; margin-bottom: 14px; }
.res .tag { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; }
.res .ttl { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.5rem; line-height: 1.2; margin: 6px 0; }
.res .txt { color: %SLATE%; font-size: 0.92rem; }
.ring { flex: none; width: 118px; height: 118px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
.ring-in { width: 88px; height: 88px; border-radius: 50%; background: #fff; display: flex; flex-direction: column;
    align-items: center; justify-content: center; }
.ring-in b { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.05rem; color: %NAVY%; }
.ring-in span { font-size: 0.62rem; letter-spacing: 0.08em; color: %SLATE%; }
@media (max-width: 600px) { .res { flex-direction: column-reverse; align-items: flex-start; } }

.photo { position: relative; border-radius: 24px; overflow: hidden; aspect-ratio: 3/4; background: #E2E8F0;
    cursor: pointer; outline: none; isolation: isolate; -webkit-tap-highlight-color: transparent; }
.photo:focus-visible { box-shadow: 0 0 0 3px %MINT%; }
.photo img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; display: block;
    transition: opacity .45s ease, transform .7s ease; }
.photo img.p2 { opacity: 0; }
.photo:hover img.p2, .photo:focus img.p2 { opacity: 1; }
.photo:hover img.p1, .photo:focus img.p1 { transform: scale(1.04); }
.photo .shade { position: absolute; inset: 0; z-index: 1; pointer-events: none; }
.photo .shade::before, .photo .shade::after { content: ""; position: absolute; inset: 0; transition: opacity .35s ease; }
.photo .shade::before { background: linear-gradient(to top, rgba(0,23,61,0.88) 0%, rgba(0,23,61,0) 45%); }
.photo .shade::after { opacity: 0;
    background: linear-gradient(to top, rgba(0,23,61,0.96) 0%, rgba(0,23,61,0.88) 36%, rgba(0,23,61,0.4) 62%, rgba(0,23,61,0) 90%); }
.photo:hover .shade::before, .photo:focus .shade::before { opacity: 0; }
.photo:hover .shade::after, .photo:focus .shade::after { opacity: 1; }
.photo .ov { position: absolute; inset: auto 0 0 0; z-index: 2; padding: 60px 18px 16px 18px; }
.photo .role { display: inline-block; background: %MINT%; color: %NAVY%; border-radius: 999px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 700; margin-bottom: 6px; }
.photo .nm { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #fff; font-size: 1.05rem; line-height: 1.25; }
.photo .nim { display: none; color: #A7F3D0; font-size: 0.76rem; font-weight: 600; letter-spacing: 0.04em; margin-top: 2px; }
.photo .mdl { color: #CBD5E1; font-size: 0.8rem; margin-top: 2px; }
.photo .jobs { display: grid; grid-template-rows: 0fr; opacity: 0; transition: grid-template-rows .4s ease, opacity .3s ease; }
.photo .jobs > div { overflow: hidden; min-height: 0; }
.photo .jt { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #6CF8BB; margin: 12px 0 7px 0; }
.photo .jobs ul { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 6px; }
.photo .jobs li { background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.24); color: #fff;
    border-radius: 999px; padding: 3px 10px; font-size: 0.76rem; font-weight: 500; line-height: 1.35; }
.photo:hover .jobs, .photo:focus .jobs { grid-template-rows: 1fr; opacity: 1; }
.photo:hover .nim, .photo:focus .nim { display: block; }
.photo:hover .mdl, .photo:focus .mdl { display: none; }
.lect { display: flex; gap: 20px; align-items: center; }
.lect img { width: 110px; height: 128px; border-radius: 18px; object-fit: cover; border: 2px solid %MINT_TINT%; }
.footer { color: %SLATE%; font-size: 0.8rem; text-align: center; margin: 32px 0 8px 0; }

/* Footer menempel ke dasar layar: tanpa padding bawah, dan footer didorong ke bawah pada halaman pendek. */
.block-container { padding-top: 5.6rem !important; padding-bottom: 0 !important; }
.block-container > [data-testid="stVerticalBlock"] {
    min-height: calc(100vh - 5.6rem); min-height: calc(100dvh - 5.6rem);
}
.block-container > [data-testid="stVerticalBlock"] > div[class*="st-key-sitefoot"],
.block-container > [data-testid="stVerticalBlock"] > *:has(div[class*="st-key-sitefoot"]) { margin-top: auto !important; }

[data-testid="stHeader"] { background: transparent !important; backdrop-filter: none !important; border-bottom: 0 !important; }

div[class*="st-key-topnav"] {
    position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
    width: min(1000px, calc(100vw - 28px)); z-index: 1000001;
    background: rgba(255,255,255,0.78); backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border: 1px solid rgba(226,232,240,0.9); border-top-color: rgba(255,255,255,0.95);
    border-radius: 999px; padding: 7px 10px 7px 18px;
    box-shadow: 0 16px 40px -8px rgba(0,23,61,0.14), 0 4px 12px -2px rgba(0,23,61,0.05);
}
div[class*="st-key-topnav"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.25rem !important; align-items: center; }
div[class*="st-key-topnav"] [data-testid="stColumn"] { min-width: 0 !important; }
div[class*="st-key-topnav"] a {
    border-radius: 999px !important; padding: 8px 16px !important; justify-content: center; gap: 6px;
    text-decoration: none !important; background: transparent !important;
    transition: background .18s ease, box-shadow .18s ease, transform .18s ease;
}
div[class*="st-key-topnav"] a p, div[class*="st-key-topnav"] a span { color: %SLATE% !important; font-weight: 600; font-size: 0.9rem; white-space: nowrap; }
div[class*="st-key-topnav"] a:hover { background: %MINT_TINT% !important; transform: translateY(-1px); }
div[class*="st-key-topnav"] a:hover p, div[class*="st-key-topnav"] a:hover span { color: %NAVY% !important; }
div[class*="st-key-navlink-on"] a { background: #6CF8BB !important; box-shadow: 0 2px 10px rgba(16,185,129,0.28); }
div[class*="st-key-navlink-on"] a p, div[class*="st-key-navlink-on"] a span { color: %NAVY% !important; font-weight: 700; }
.navlogo-wrap { position: fixed; left: 18px; top: 50%; transform: translateY(-50%); z-index: 2; display: flex; align-items: center;
    height: 32px; margin: 0; padding: 0; }
.navlogo { height: 32px; width: auto; display: block; margin: 0; }
/* Navbar mobile memakai deretan tombol sendiri (bukan kolom desktop yang menyempit). */
@media (min-width: 761px) {
    div[class*="st-key-mobilenav"], div[class*="st-key-topnav"] > *:has(> div[class*="st-key-mobilenav"]) { display: none !important; }
}
@media (max-width: 760px) {
    .block-container { padding-top: 5.0rem !important; padding-bottom: 0 !important; }
    .block-container > [data-testid="stVerticalBlock"] {
        min-height: calc(100vh - 5rem); min-height: calc(100dvh - 5rem);
    }
    div[class*="st-key-topnav"] {
        top: 8px; width: calc(100vw - 20px); max-width: 520px;
        padding: 5px 6px; border-radius: 26px;
    }
    /* Sembunyikan baris kolom desktop (logo, tiga tautan, tombol CTA). */
    div[class*="st-key-topnav"] [data-testid="stHorizontalBlock"],
    div[class*="st-key-topnav"] > *:has(> [data-testid="stHorizontalBlock"]) { display: none !important; }

    div[class*="st-key-mobilenav"] {
        display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 4px !important; width: 100%;
    }
    div[class*="st-key-mobilenav"] > * { min-width: 0 !important; width: 100% !important; }
    div[class*="st-key-mobilenav"] [data-testid="stElementContainer"],
    div[class*="st-key-mobilenav"] [data-testid="stButton"],
    div[class*="st-key-mobilenav"] div[class*="st-key-mbtn"] { width: 100% !important; min-width: 0 !important; }
    div[class*="st-key-mobilenav"] button {
        width: 100% !important; min-height: 44px; padding: 6px 4px !important; border-radius: 999px !important;
        border: 0 !important; background: transparent !important; box-shadow: none !important;
        display: flex; align-items: center; justify-content: center;
    }
    div[class*="st-key-mobilenav"] button * { max-width: 100%; overflow: visible !important; }
    div[class*="st-key-mobilenav"] button p {
        margin: 0 !important; text-align: center; white-space: normal; line-height: 1.15 !important;
        font-size: clamp(0.7rem, 3vw, 0.86rem) !important; font-weight: 600; color: %SLATE% !important;
    }
    div[class*="st-key-mobilenav"] button:hover, div[class*="st-key-mobilenav"] button:active { background: %MINT_TINT% !important; }
    div[class*="st-key-mbtn-on"] button { background: #6CF8BB !important; box-shadow: 0 2px 10px rgba(16,185,129,0.28) !important; }
    div[class*="st-key-mbtn-on"] button p { color: %NAVY% !important; font-weight: 700; }

    /* Footer menempel di dasar layar dengan sudut atas membulat. */
    div[class*="st-key-sitefoot"] { padding: 24px 22px calc(14px + env(safe-area-inset-bottom, 0px)) !important; }
    div[class*="st-key-sitefoot"] [data-testid="stHorizontalBlock"] { gap: 1.1rem !important; }
    div[class*="st-key-sitefoot"] [data-testid="stColumn"] { min-width: 0 !important; }
    div[class*="st-key-sitefoot"] .fdesc { max-width: none; }
    div[class*="st-key-sitefoot"] .fbar { line-height: 1.5; }
}

.grid-cat { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.grid-cat > .dcard { grid-column: span 2; }
.grid-cat > .dcard:nth-child(4) { grid-column: 2 / span 2; }
.grid-cat > .dcard:nth-child(5) { grid-column: 4 / span 2; }
@media (max-width: 900px) {
    .grid-cat { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid-cat > .dcard:nth-child(n) { grid-column: auto; }
    .grid-cat > .dcard:nth-child(5) { grid-column: 1 / -1; }
}
@media (max-width: 600px) {
    .grid-cat { grid-template-columns: 1fr; }
    .grid-cat > .dcard:nth-child(5) { grid-column: auto; }
}

/* Ikon daun/kuncup dan ornamen sisi hero */
.sthead { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
.ico { width: 44px; height: 44px; object-fit: contain; flex: 0 0 auto; }
.cd { display: inline-flex; align-items: center; gap: 6px; }
.ico-s { width: 26px; height: 26px; object-fit: contain; }
.stApp { --ow: clamp(64px, calc(80px + (100vw - 1200px) / 2 - 22px), 136px); }
@media (min-width: 1280px) {
    .stApp::before, .stApp::after { content: ""; position: fixed; top: 0; height: 100vh; z-index: 1; pointer-events: none;
        width: var(--ow); background-repeat: repeat-y; background-size: var(--ow) auto; opacity: 0.95; }
    .stApp::before { left: 0; background-image: url("%ORN_L%"); background-position: left top; }
    .stApp::after { right: 0; background-image: url("%ORN_R%"); background-position: right top; }
}

/* Aset ilustrasi: hero, judul bagian, ornamen, dan Sibi di hasil */
.hero-art { position: relative; border-radius: 28px; overflow: hidden; aspect-ratio: 5 / 4; background: #E6F4EC;
    border: 1px solid %BORDER%; box-shadow: 0 24px 48px -24px rgba(0,23,61,0.25); }
.hero-art .bg { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; display: block; }
.hero-art .sibi-h { position: absolute; left: 50%; bottom: 5%; width: 46%; height: auto; transform: translateX(-50%);
    filter: drop-shadow(0 12px 12px rgba(0,23,61,0.18)); animation: sibi-bob 3.2s ease-in-out infinite; }
.hero-art .hi { position: absolute; top: 9%; left: 7%; max-width: 62%; background: #fff; border: 1px solid #BBF7D0;
    border-radius: 18px 18px 18px 4px; padding: 10px 16px; font-weight: 600; font-size: 0.95rem; line-height: 1.4; color: %NAVY%;
    box-shadow: 0 8px 20px -8px rgba(0,23,61,0.18); }
@keyframes sibi-bob { 0%, 100% { transform: translateX(-50%) translateY(0); } 50% { transform: translateX(-50%) translateY(-9px); } }
.sech { display: flex; align-items: center; justify-content: space-between; gap: 28px; margin: 44px 0 14px 0; }
.sech .tx { flex: 1 1 50%; min-width: 0; }
.sech .art { flex: 0 1 40%; min-width: 0; }
.sech .art img, .banner, .empty-art { display: block; width: 100%; height: auto; border-radius: 22px; border: 1px solid %BORDER%;
    box-shadow: 0 14px 30px -18px rgba(0,23,61,0.28); }
.banner { width: min(100%, 640px); margin: 18px auto 22px auto; }
.empty-art { max-width: 420px; margin: 0 auto 16px auto; }
.orn { display: block; width: min(100%, 760px); height: auto; margin: 34px auto 6px auto; }
@media (max-width: 760px) {
    .sech { flex-direction: column; align-items: stretch; gap: 16px; }
    .sech .art { order: 2; }
    .hero-art { aspect-ratio: 1 / 1; }
    .hero-art .hi { font-size: 0.86rem; padding: 8px 13px; top: 6%; max-width: 70%; }
    .hero-art .sibi-h { width: 50%; }
}
.say { display: flex; align-items: center; gap: 14px; margin: 0 0 12px 0; }
.say img { width: 84px; height: 84px; object-fit: contain; flex: 0 0 auto; }
.say .bub { position: relative; background: #fff; border: 1px solid #BBF7D0; border-radius: 18px; padding: 11px 16px;
    font-weight: 600; font-size: 0.95rem; line-height: 1.4; color: %NAVY%; box-shadow: 0 6px 18px -8px rgba(0,23,61,0.15); }
.say .bub::before { content: ""; position: absolute; left: -7px; top: 50%; width: 12px; height: 12px; background: #fff;
    border-left: 1px solid #BBF7D0; border-bottom: 1px solid #BBF7D0; transform: translateY(-50%) rotate(45deg); }
@media (prefers-reduced-motion: reduce) { .hero-art .sibi-h { animation: none; } }

/* Maskot Sibi saat menunggu hasil deteksi */
.sibi { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 12px; padding: 26px 20px 22px;
    background: linear-gradient(135deg, %MINT_TINT% 0%, %ICE% 100%); min-height: 260px; justify-content: center; }
.sibi img { width: 150px; height: auto; display: block; transform-origin: 50% 90%; animation: sibi-float 1.9s ease-in-out infinite; }
.sibi .bubble { position: relative; background: #fff; border: 1px solid #BBF7D0; border-radius: 18px; padding: 10px 18px;
    font-weight: 600; color: %NAVY%; font-size: 0.98rem; line-height: 1.4; max-width: 340px;
    box-shadow: 0 6px 18px -6px rgba(0,23,61,0.12); }
.sibi .bubble::before { content: ""; position: absolute; top: -7px; left: 50%; width: 12px; height: 12px; background: #fff;
    border-left: 1px solid #BBF7D0; border-top: 1px solid #BBF7D0; transform: translateX(-50%) rotate(45deg); }
.sibi p { margin: 0; color: %SLATE%; font-size: 0.86rem; }
.sibi .dots { display: inline-flex; gap: 3px; margin-left: 5px; vertical-align: baseline; }
.sibi .dots i { width: 5px; height: 5px; border-radius: 50%; background: %EMERALD%; display: inline-block; animation: sibi-dot 1.2s ease-in-out infinite; }
.sibi .dots i:nth-child(2) { animation-delay: .18s; }
.sibi .dots i:nth-child(3) { animation-delay: .36s; }
@keyframes sibi-float { 0%, 100% { transform: translateY(0) rotate(-2deg); } 50% { transform: translateY(-12px) rotate(2deg); } }
@keyframes sibi-dot { 0%, 80%, 100% { opacity: .25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }
@media (prefers-reduced-motion: reduce) { .sibi img, .sibi .dots i { animation: none; } }

.dcard { display: flex; flex-direction: column; gap: 0; }
.dcard .dtop { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.tagc { font-size: 0.72rem; font-weight: 700; border-radius: 999px; padding: 3px 10px; background: %ICE%; color: %COBALT%; }
.tagc.bad { background: #FEE2E2; color: #B91C1C; }
.tagc.ok { background: #D1FAE5; color: %EMERALD%; }
.dcard .code { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; color: %SLATE%; }
.dphoto { width: 100%; height: 170px; object-fit: cover; border-radius: 14px; display: block; margin-bottom: 12px; background: #E6EEFF; }
.dphoto.ph { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px;
    background: linear-gradient(135deg, %MINT_TINT% 0%, %ICE% 100%); color: %SLATE%; font-size: 0.78rem; font-weight: 600; }
.dphoto.ph img { width: 54px; height: 54px; object-fit: contain; border-radius: 0; margin: 0; }
.dcard h4 { margin: 0 0 8px 0; }
.dcard .lab { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: %NAVY%; margin: 10px 0 4px 0; }
.dcard .lab.care { color: %COBALT%; }
.dcard ul { margin: 0; padding-left: 18px; color: %SLATE%; font-size: 0.88rem; line-height: 1.55; }
.dcard ul li { margin-bottom: 3px; }
.wide { grid-column: span 2; }
.wide .inner { display: grid; grid-template-columns: 1fr 1.2fr; gap: 18px; align-items: start; }
.wide .dphoto { height: 100%; min-height: 220px; margin-bottom: 0; }
@media (max-width: 600px) { .wide { grid-column: span 1; } .wide .inner { grid-template-columns: 1fr; } }
.stepf { border-top: 1px solid %BORDER%; margin-top: 12px; padding-top: 8px; font-size: 0.78rem; font-weight: 600; color: %SLATE%; }
.care-top { display: grid; grid-template-columns: 200px 1fr; gap: 16px; align-items: start; margin-bottom: 4px; }
.care-top .dphoto { height: 140px; margin-bottom: 0; }
@media (max-width: 800px) { .care-top { grid-template-columns: 1fr; } }
.care-grid { display: grid; gap: 12px; grid-template-columns: repeat(3, 1fr); margin: 6px 0 10px 0; }
.care-item { background: %MINT_TINT%; border: 1px solid #D1FAE5; border-radius: 14px; padding: 12px 14px; color: %NAVY%; font-size: 0.88rem; line-height: 1.5; }
.care-item b { color: %EMERALD%; }
@media (max-width: 800px) { .care-grid { grid-template-columns: 1fr; } }
.mock { padding: 14px; }
.mock .mtop { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-size: 0.78rem; font-weight: 700; color: %SLATE%; letter-spacing: 0.04em; text-transform: uppercase; }
.mock img { width: 100%; height: 250px; object-fit: cover; border-radius: 14px; display: block; }
.mock .mfoot { margin-top: 10px; }
.arch-head { display: grid; grid-template-columns: 1fr 1.2fr; gap: 28px; align-items: end; margin: 44px 0 14px 0; }
.arch-head .d-title { margin: 6px 0 0 0; }
.arch-head p { color: %SLATE%; font-size: 0.98rem; line-height: 1.65; margin: 0; }
@media (max-width: 800px) { .arch-head { grid-template-columns: 1fr; gap: 10px; } }
.arch .idx { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; color: %EMERALD%; font-size: 0.95rem; margin-bottom: 8px; }
.eyebrow2 { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; color: %EMERALD%; text-transform: uppercase; }
.infobox { background: %ICE%; border: 1px solid #BFDBFE; border-radius: 16px; padding: 14px 16px; margin-top: 10px; }
.infobox b { color: %COBALT%; font-family: 'Plus Jakarta Sans', sans-serif; }
.infobox p { margin: 4px 0 0 0; color: %NAVY%; font-size: 0.88rem; line-height: 1.55; }
.sophead { display: flex; justify-content: space-between; align-items: center; margin: 2px 0 8px 0; }
.sophead h4 { margin: 0; font-size: 1.05rem; }
.cta2 { background: linear-gradient(135deg, %NAVY% 0%, #0B2A5B 100%); border-radius: 28px; padding: 40px 36px; margin: 40px 0 14px 0; }
.cta2 .eb { display: inline-block; background: rgba(108,248,187,0.14); color: #6CF8BB; border-radius: 999px; padding: 4px 14px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 14px; }
.cta2 .t { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; color: #fff; font-size: clamp(1.5rem, 3vw, 2.1rem); line-height: 1.2; margin-bottom: 10px; max-width: 640px; }
.cta2 .p { color: #CBD5E1; max-width: 560px; line-height: 1.6; }
.tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.tag-o { border: 1px solid %BORDER%; background: #F8FAFC; color: %SLATE%; border-radius: 999px; padding: 3px 12px; font-size: 0.76rem; font-weight: 600; }
div[class*="st-key-sitefoot"] { background: #EFF4FF; border-radius: 28px 28px 0 0; padding: 28px 32px calc(16px + env(safe-area-inset-bottom, 0px)) 32px; margin-top: 40px; margin-bottom: 0 !important; }
div[class*="st-key-sitefoot"] a { text-decoration: none !important; padding: 0 !important; min-height: 0 !important; background: transparent !important; justify-content: flex-start; }
div[class*="st-key-sitefoot"] a p { color: %SLATE% !important; font-size: 0.9rem; }
div[class*="st-key-sitefoot"] a:hover p { color: %EMERALD% !important; }
.fcol-t { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: %NAVY%; margin-bottom: 8px; }
.fdesc { color: %SLATE%; font-size: 0.9rem; line-height: 1.6; max-width: 420px; margin-top: 8px; }
.fbar { border-top: 1px solid #D5E3FC; margin-top: 14px; padding-top: 12px; color: %SLATE%; font-size: 0.78rem; text-align: center; }

/* Panel bagian bergradien + efek pop-up saat di-scroll */
div[class*="st-key-sec"] { position: relative; border-radius: 32px; padding: 14px 34px 30px 34px; margin: 26px 0;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 24px 48px -34px rgba(0,23,61,0.28), inset 0 1px 0 rgba(255,255,255,0.9); }
div[class*="st-key-secmodel"], div[class*="st-key-secinfo"] { background: linear-gradient(135deg, #DDEAFF 0%, #F1F6FF 48%, #E1FAEF 100%); }
div[class*="st-key-seckatalog"], div[class*="st-key-secdosen"] { background: linear-gradient(135deg, #FFF0D6 0%, #FFF9EE 42%, #E3F9EE 100%); }
div[class*="st-key-secalur"], div[class*="st-key-secanggota"] { background: linear-gradient(160deg, #D8F8E8 0%, #F0FDF7 50%, #DEE9FF 100%); }
div[class*="st-key-secspek"], div[class*="st-key-sectek"] { background: linear-gradient(135deg, #E8E7FF 0%, #F4F5FF 45%, #D9F2FF 100%); }
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-card-input"]) {
    background: linear-gradient(160deg, #D8F8E8 0%, #F0FDF7 46%, #DEE9FF 100%); border-radius: 32px; padding: 26px 26px 30px 26px;
    margin: 12px 0 26px 0; border: 1px solid rgba(255,255,255,0.8); box-shadow: 0 24px 48px -34px rgba(0,23,61,0.28); }
div[class*="st-key-sec"] .sech, div[class*="st-key-sec"] .arch-head { margin-top: 14px; }
div[class*="st-key-sec"] .d-title { margin-top: 14px; }
@media (max-width: 760px) {
    div[class*="st-key-sec"] { padding: 8px 14px 20px 14px; border-radius: 24px; margin: 18px 0; }
    div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-card-input"]) { padding: 16px 12px 20px 12px; border-radius: 24px; }
}
@supports (animation-timeline: view()) {
    @keyframes pop-up {
        0% { opacity: 0; translate: 0 46px; scale: 0.94; }
        60% { opacity: 1; }
        82% { scale: 1.012; }
        100% { opacity: 1; translate: 0 0; scale: 1; }
    }
    @keyframes pop-panel { from { translate: 0 44px; scale: 0.97; } to { translate: 0 0; scale: 1; } }
    div[class*="st-key-sec"] { animation: pop-panel ease-out both; animation-timeline: view(); animation-range: entry 0% entry 25%; }
    :is(.card2, .sech, .arch-head, .banner, .orn, .d-title, .hero-art, .cta2, .photo, .say) {
        animation: pop-up ease-out both; animation-timeline: view(); animation-range: entry 0% entry 60%; }
    .grid > :nth-child(2) { animation-range: entry 6% entry 66%; }
    .grid > :nth-child(3) { animation-range: entry 12% entry 72%; }
    .grid > :nth-child(4) { animation-range: entry 18% entry 78%; }
    .grid > :nth-child(5) { animation-range: entry 24% entry 84%; }
    @media (prefers-reduced-motion: reduce) {
        div[class*="st-key-sec"], :is(.card2, .sech, .arch-head, .banner, .orn, .d-title, .hero-art, .cta2, .photo, .say) { animation: none; }
    }
}
</style>
"""

CSS = (
    _CSS.replace("%NAVY%", NAVY).replace("%EMERALD%", EMERALD).replace("%MINT_TINT%", MINT_TINT)
    .replace("%MINT%", MINT).replace("%COBALT%", COBALT).replace("%SLATE%", SLATE)
    .replace("%BORDER%", BORDER).replace("%ICE%", ICE)
)


def inject_css():
    st.markdown(
        CSS.replace("%POLA%", art("pola.jpg")).replace("%ORN_L%", art("ornamen-kiri.webp", "image/webp"))
        .replace("%ORN_R%", art("ornamen-kanan.webp", "image/webp")),
        unsafe_allow_html=True,
    )


def html_str(s):
    """Rapikan HTML (tanpa indentasi/baris kosong) agar tidak dibaca sebagai blok kode markdown."""
    return "\n".join(l.strip() for l in s.strip().splitlines() if l.strip())


def html(s):
    """Render HTML tanpa indentasi/baris kosong."""
    st.markdown(html_str(s), unsafe_allow_html=True)


def sibi_loading():
    """Kartu maskot Sibi untuk tampilan menunggu hasil deteksi (ditaruh di st.empty())."""
    return html_str(
        f'''
        <div class="card2 sibi" role="status" aria-live="polite">
        <img src="{art("sibi_scan.webp", "image/webp")}" alt="Sibi, maskot sampeu.ai">
        <div class="bubble">Sebentar ya, lagi dicek dulu sama Sibi<span class="dots"><i></i><i></i><i></i></span></div>
        <p>Sibi lagi memeriksa daun dan menyiapkan peta Grad-CAM.</p>
        </div>
        '''
    )


@st.cache_data
def img_uri(path, mime="image/jpeg"):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def art(name, mime="image/jpeg"):
    """Data-URI untuk berkas di assets/ilustrasi/."""
    return img_uri(ASSETS / "ilustrasi" / name, mime)


def sibi_say(kind):
    """Sibi dengan ekspresi sesuai hasil: healthy, disease, atau uncertain."""
    pose, text = {
        "healthy": ("sibi_senang.webp", "Daunnya sehat! Tetap rawat rutin ya."),
        "disease": ("sibi_laporan.webp", "Sibi menemukan tanda penyakit. Cek cara penanganannya di bawah ya."),
        "uncertain": ("sibi_bingung.webp", "Hmm, Sibi kurang yakin. Coba foto ulang yang lebih jelas ya."),
    }[kind]
    return html_str(
        f'''
        <div class="say"><img src="{art(pose, "image/webp")}" alt="Sibi">
        <div class="bub">{text}</div></div>
        '''
    )


def penyakit_img(code):
    """Foto contoh per kelas (assets/penyakit/<KELAS>.jpg|jpeg|png), atau None bila belum ada."""
    for ext, mime in (("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("png", "image/png")):
        p = ASSETS / "penyakit" / f"{code}.{ext}"
        if p.exists():
            return img_uri(p, mime)
    return None


def render_nav(active):
    """Navbar pil melayang. Halaman aktif ditandai lewat key kontainer (CSS: st-key-navlink-on-*)."""
    items = [("views/beranda.py", "Beranda", "home", ":material/home:"),
             ("views/deteksi.py", "Deteksi Penyakit", "detect", ":material/biotech:"),
             ("views/tim.py", "Tim Kami", "team", ":material/groups:")]
    with st.container(key="topnav"):
        cols = st.columns([1.5, 1.0, 1.55, 1.1], vertical_alignment="center")
        with cols[0]:
            st.markdown(
                f'<div class="navlogo-wrap"><img class="navlogo" src="{img_uri(ASSETS / "logo.png", "image/png")}" alt="sampeu.ai"></div>',
                unsafe_allow_html=True,
            )
        for col, (path, label, slug, icon) in zip(cols[1:4], items):
            with col:
                with st.container(key=f"navlink-{'on' if label == active else 'off'}-{slug}"):
                    st.page_link(path, label=label, icon=icon)
        # Navigasi mobile: tombol biasa + st.switch_page (CSS menampilkannya hanya di layar sempit).
        with st.container(key="mobilenav"):
            for path, label, slug, _ in items:
                with st.container(key=f"mbtn-{'on' if label == active else 'off'}-{slug}"):
                    if st.button(label, key=f"msw-{slug}") and label != active:
                        st.switch_page(path)


# ----------------------- Model & prediksi (TensorFlow dimuat malas) -----------------------
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def prepare(pil_image):
    """Sama dengan training: RGB, resize 224x224, float32, efficientnet.preprocess_input."""
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = pil_image.convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)
    x = preprocess_input(x)
    return np.expand_dims(x, 0)


def gradcam(model, x, class_idx):
    """Grad-CAM untuk struktur: Input -> efficientnetb0 (bersarang) -> GAP -> Dropout -> Dense."""
    import tensorflow as tf

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


# ----------------------- Komponen tampilan -----------------------
def pct(p):
    """Persentase untuk ditampilkan; hindari kesan 'pasti' pada nilai ekstrem."""
    if p > 0.999:
        return ">99,9%"
    if p < 0.001:
        return "<0,1%"
    return f"{p * 100:.1f}%".replace(".", ",")


def bar(label, p, color=MINT, text=None):
    html(
        f'<div class="bar-row"><div class="top"><span>{label}</span><b>{text or pct(p)}</b></div>'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{p * 100:.1f}%;background:{color}"></div></div></div>'
    )


def footer():
    with st.container(key="sitefoot"):
        c1, c2, c3 = st.columns([2, 1, 1.2], gap="large")
        with c1:
            html(
                f'''
                <img class="navlogo" style="margin-left:0" src="{img_uri(ASSETS / "logo.png", "image/png")}" alt="sampeu.ai">
                <div class="fdesc">Alat bantu skrining penyakit daun singkong berbasis citra: kelas, probabilitas,
                peta Grad-CAM, dan cara penanganan awal dalam satu halaman.</div>
                '''
            )
        with c2:
            html('<div class="fcol-t">Navigasi cepat</div>')
            st.page_link("views/beranda.py", label="Beranda")
            st.page_link("views/deteksi.py", label="Deteksi Penyakit")
            st.page_link("views/tim.py", label="Tim Kami")
        with c3:
            html(
                '''
                <div class="fcol-t">Catatan penting</div>
                <div class="fdesc" style="margin-top:0">Hasil hanya skrining awal, bukan diagnosis pasti.
                Konsultasikan dengan penyuluh pertanian setempat.</div>
                '''
            )
        html('<div class="fbar">© 2026 sampeu.ai · Kelompok 11 Kelas A, Machine Learning</div>')


def build_report(img, ov, probs, order, label, title, kind, ms):
    """Laporan diagnosis PDF satu halaman (matplotlib, tanpa dependensi baru)."""
    import io
    import textwrap
    from datetime import datetime

    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8.27, 11.69))
    FigureCanvasPdf(fig)
    fg = STATUS[kind][0]
    fig.text(0.07, 0.955, "sampeu.ai", fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.07, 0.932, "Laporan Deteksi Penyakit Daun Singkong", fontsize=11, color=SLATE)
    fig.text(0.93, 0.955, datetime.now().strftime("%d-%m-%Y %H:%M"), fontsize=9, color=SLATE, ha="right")
    fig.text(0.07, 0.885, "HASIL PREDIKSI", fontsize=8, fontweight="bold", color=fg)
    fig.text(0.07, 0.86, "\n".join(textwrap.wrap(title, 60)), fontsize=15, fontweight="bold", color=fg, va="top")
    fig.text(0.07, 0.815, f"Waktu inferensi {ms:.0f} ms · EfficientNetB0 · Alat bantu skrining", fontsize=8.5, color=SLATE)

    for x0, im, cap in ((0.07, img, "Citra asli"), (0.52, ov, "Peta aktivasi (Grad-CAM)")):
        if im is None:
            continue
        ax = fig.add_axes([x0, 0.52, 0.41, 0.25])
        ax.imshow(im.convert("RGB").resize((320, 320)))
        ax.axis("off")
        ax.set_title(cap, fontsize=9, color=SLATE)

    ax = fig.add_axes([0.28, 0.33, 0.62, 0.14])
    names = [CLASS_NAMES[i] for i in order][::-1]
    vals = [float(probs[i]) * 100 for i in order][::-1]
    ax.barh(names, vals, color=[EMERALD if n == names[-1] else "#A7F3D0" for n in names])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probabilitas (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.07, 0.49, "Distribusi probabilitas", fontsize=10, fontweight="bold", color=NAVY)

    y = 0.27
    if kind != "uncertain":
        fig.text(0.07, y, "Cara penanganan awal" if label != "Healthy" else "Perawatan", fontsize=10, fontweight="bold", color=NAVY)
        y -= 0.03
        for i, t in enumerate(CLASS_CARE[label], 1):
            lines = textwrap.wrap(t, 95)
            fig.text(0.07, y, f"{i}. " + lines[0], fontsize=8.5, color=NAVY, va="top")
            for extra in lines[1:]:
                y -= 0.017
                fig.text(0.095, y, extra, fontsize=8.5, color=NAVY, va="top")
            y -= 0.032
    note = textwrap.fill(
        "Hasil ini alat bantu skrining, bukan diagnosis pasti. " + CARE_NOTE, 120
    )
    fig.text(0.07, 0.07, note, fontsize=7, color=SLATE, va="top")
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    return buf.getvalue()
