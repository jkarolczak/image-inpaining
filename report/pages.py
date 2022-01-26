import pandas as pd
import plotly.express as px
import streamlit as st


def approach() -> None:
    st.header("Our approach")
    pd.options.plotting.backend = "plotly"
    st.markdown("""
        Our solution belongs to a generative adversarial network class. The solution consists of three networks:
        - `netG` - neural network acting as a generator. This network is inspired with VGG-19 architecture. It consists of 5831043 parameters (about 23.32 MB).
    """)
    with st.expander("Generator graph"):
        with open('report/images/Generator.png', 'rb') as file:
            st.download_button("Download netG.png", data=file, file_name="netG.png", mime='image/png')
            st.image('report/images/Generator.png')
    st.markdown("""
        - `netGD` - neural network acting as a global discriminator evaluating whether the whole image seems to be real. This network consists of 3 identical (in terms of the architecture, not parameters values) branches. Each branch extracts 256 features from the input image - it is done with convolutional layers (mixed with ReLU and local maximum pooling) and channel-wise global pooling with maxiumim function at the end of each branch. Later on attention mechanism is applied to these features. The network is parametrized by 3715905 values (about 14.86 MB).
    """)
    with st.expander("Global discriminator graph"):
        with open('report/images/GlobalDiscriminator.png', 'rb') as file:
            st.download_button("Download netGD.png", data=file, file_name="netGD.png", mime='image/png')
            st.image('report/images/GlobalDiscriminator.png')
    st.markdown("""
        - `netLD` - neural network acting as a local discriminator evaluating whether the erased area seems to be real. It is done by expansion of channels to 64. It's performed only on the snippet, not the whole image. After on each channel three global poolings are applied using `min`, `mean` and `max` functions. This way 192 values are obtained. They are passed through linear layers and that's how predictions are obtained. This network consists of 118785 weights (about 0.47 MB).
    """)
    with st.expander("Local discriminator graph"):
        with open('report/images/LocalDiscriminator.png', 'rb') as file:
            st.download_button("Download netLD.png", data=file, file_name="netLD.png", mime='image/png')
            st.image('report/images/LocalDiscriminator.png')

    st.header("Training - stage 2")
    df_lr = pd.read_csv('report/dataframes/stage2.csv')
    fig = df_lr.plot()
    fig.update_xaxes(title="epoch")
    fig.update_yaxes(title="binary cross entropy")
    st.plotly_chart(fig)


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
        st.markdown("""
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
        

def experiments() -> None:
    st.header("Conducted experiments")
    pd.options.plotting.backend = "plotly"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stage 1 - learning rate")
        df_lr = pd.read_csv('report/dataframes/lr.csv')
        fig = df_lr.plot()
        fig.update_xaxes(title="epoch")
        fig.update_yaxes(title="mean absolute error")
        st.plotly_chart(fig)
        
    with col2:
        st.subheader("Stage 1 - optimizers")
        df_lr = pd.read_csv('report/dataframes/optimizer.csv')
        fig = df_lr.plot()
        fig.update_xaxes(title="epoch")
        fig.update_yaxes(title="mean absolute error")
        st.plotly_chart(fig)
        
        
def problem() -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Problem and dataset")
        st.markdown("""
            We try to address problem of image inpainting, precisely face inpainting. We use [CelebFaces dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) - it consists of 202,599 images of size 218x178 px. From this dataset we generate our own dataset by removing (or actually substituting pixels with `(255, 255, 255)`) an arbitrary area from each image. The area is an rectange of width and hight from range 25-55px (height and width may differ). To increase probability of covering a part of a face the area the mask is put only in a region apart 35px from vertical borders and 45px from horizontal borders.
        """)
    with col2:
        st.image('report/images/roi.png')
    
    st.subheader("Exemplary images")
    cols = st.columns(5)
    for idx in range(5):
        cols[idx].image(f'report/images/img{idx + 1}.jpg')
    