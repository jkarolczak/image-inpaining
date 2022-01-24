import streamlit as st


def bibliography() -> None:
    st.header("Sources")
    st.markdown("""
    - Generative Face Completion, Yijun Li, Sifei Li, Jimei Yang, Ming-Hsuan Yang, 2017, ([arXiv](https://arxiv.org/pdf/1704.05838.pdf)) - inspiration for a general GAN framework
    - Very Deep Convolutional Networks for Large-Scale Image Recognition, Karen Simonyan, Andrew Zisserman, 2015 ([arXiv](https://arxiv.org/pdf/1409.1556.pdf)) - inspiration for the generator architecture (VGG-19)
    - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Alec Radford, Luke Metz, Soumith Chintala, 2016, ([arXiv](https://arxiv.org/pdf/1511.06434.pdf), [PyTorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)) - inspiration for the training loop
    """)


def environment() -> None:
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
                - `matplotlib.pyplot` - visualizing results 
        """)
        
        
def problem() -> None:
    st.header("Problem and dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            We try to address problem of image inpainting, precisely face inpainting. We use [CelebFaces dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) - it consists of 202,599 images of size 218x178 px. From this dataset we generate our own dataset by removing (or actually substituting pixels with `(255, 255, 255)`) an arbitrary area from each image. The area is an rectange of width and hight from range 25-55px (height and width may differ). To increase probability of covering a part of a face the area the mask is put only in a region apart 35px from vertical borders and 45px from horizontal borders.
        """)
    with col2:
        st.image('report/roi.png')