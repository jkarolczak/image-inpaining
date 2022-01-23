import streamlit as st

from inference import *
from pages import *


def main():
    st.set_page_config(
        page_title="Face inpainting",
        page_icon="🧑",
        layout="wide",
    )
    pages = {
        'Problem and data': problem,
        'Environment': environment,
        'Inferece': inference
    }
    name = st.sidebar.radio('Menu', pages.keys(), index=0)
    pages[name]()

if __name__ == '__main__':
    main()  