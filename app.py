import streamlit as st
from PIL import Image

from core import ASSETS, inject_css, render_nav

# True  = navbar pil melayang buatan sendiri (mirip desain Stitch)
# False = navbar atas bawaan Streamlit (cadangan bila navbar kustom bermasalah)
USE_CUSTOM_NAV = True

st.set_page_config(
    page_title="sampeu.ai",
    page_icon=Image.open(ASSETS / "symbol.png"),
    layout="wide",
)
inject_css()

pages = [
    st.Page("views/beranda.py", title="Beranda", icon=":material/home:", default=True),
    st.Page("views/deteksi.py", title="Deteksi Penyakit", icon=":material/photo_camera:"),
    st.Page("views/tim.py", title="Tim Kami", icon=":material/groups:"),
]

if USE_CUSTOM_NAV:
    pg = st.navigation(pages, position="hidden")
    render_nav(pg.title)
else:
    st.logo(str(ASSETS / "logo.png"), icon_image=str(ASSETS / "symbol.png"), size="large")
    pg = st.navigation(pages, position="top")
pg.run()
