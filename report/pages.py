import streamlit as st

def environment():
    st.header("Environment used for conducting experiments")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hardware")
        st.markdown(r"""
            - CPU
                - AMD Ryzen 5 3600X
                - Base clock speed - 3.8 GHz
                - 12 threads
                - Cache 384KB/3MB/32MB
            - RAM
                - 16 GB
                - Memory speed - 3200MHz
            - GPU
                - NVIDIA GeForce GTX960
                - 4 GB RAM
                - 1024 CUDA cores
                - Base clock speed - 1178 MHz
        """)
        
    with col2:
        st.subheader("Software")
        st.markdown(r"""
            - Ubuntu 20.04
            - CUDA Toolkit 11.5
            - DVC
            - Python 3.9 with the following packages:
                - `PyTorch` (with `cudatoolkit`) - implementation of neural networks
                - `numpy` - generating dataset, manipulating the data
                - `pandas` - managing datasets
                - `opencv` - manipulating pictures, reading and writing images to files
                - `scikit-learn` - splitting dataset
                - `streamlit` - deploying the final model in an accesbile way for end-users and writing this report
                - `neptune-client` - tracking experiments using [neptune.ai](https://neptune.ai/)
                - `matplotlib.pyplot` - visualizing the results 
        """)
        
def problem():
    st.header("Problem and dataset")
    st.markdown("""
        We try to address problem of image inpainting, precisely face inpainting. We use [CelebFaces dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) - it consists of 202,599 images of size 178x218 px. From this dataset we generate our own dataset by removing arbitrary area.
    """)