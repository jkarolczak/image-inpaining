import io
import sys
from PIL import Image

import numpy as np
import torch
import streamlit as st

sys.path.append('.')
import src.models
from src.models import Generator


def _get_generator() -> Generator:
    return src.models.utils.deserialize(Generator, 'state_dict.pt', 'cpu', 'report')
    
    
def _inpaint(input: bytes) -> np.ndarray:
    img = np.asarray(Image.open(io.BytesIO(input)))
    img = torch.tensor(img).unsqueeze(0).float()
    model = _get_generator()
    img = model(img)    
    img = img.squeeze(0).detach().int().numpy()
    img = np.clip(img, 0, 255)
    return img
    
    

def inference():
    top = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        placeholder_in = st.empty()
        
    with col2:
        placeholder_out = st.empty()
    
    col1, col2 = top.columns(2)
    with col1:
        st.header("Input")
        st.markdown("Upload an image below to perform inference")
        form = st.form("inference_form")
        file = form.file_uploader("Input data", type=['jpg', 'png'])
        if form.form_submit_button():
            try:
                content = file.getvalue()
                placeholder_in.image(content)
                inpainted = _inpaint(content)
                placeholder_out.image(inpainted)
                    
            except:
                st.error("Uploading file is mandatory!")
    with col2:
        st.header("Output")
        st.markdown("Inference results will be displayed below after uploading file on the left-hand-side of the screen.")
        
    