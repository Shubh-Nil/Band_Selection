import os
import glob

import streamlit as st
import scipy.io
import numpy as np

import train
from train import genetic_algorithm
from utils import *


def get_rgb_composite(X: np.ndarray, selected_bands = None) -> np.ndarray:
    """Make a quick RGB composite by picking three bands and normalizing."""
    H, W, C = X.shape
    # pick three representative bands
    if selected_bands is not None:
        b0, b1, b2 = selected_bands[:3]
    else:    
        # b0, b1, b2 = (0, C // 2, C - 1)
        b0, b1, b2 = (15, 30, 95)

    rgb = np.stack([X[:, :, b0], X[:, :, b1], X[:, :, b2]], axis=-1)
    # scale to [0,1]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)
    return rgb


def main():
    st.title("Hyperspectral Band Selection Demo")

    # ─── Discover all datasets in data/ ───
    subfolders = [f for f in glob.glob("data/*") if os.path.isdir(f)]
    datasets = {}
    for subfolder in subfolders:
        corrected_files = glob.glob(os.path.join(subfolder, "*_corrected.mat"))
        if len(corrected_files) != 1:
            continue
        name = os.path.basename(subfolder)
        datasets[name] = corrected_files[0]                                         # path of the image dataset (not the ground truth label)


    if not datasets:
        st.error("No `*_corrected.mat` files found in the data/ folder.")
        return

    # user selects which dataset
    choice = st.selectbox("Choose a dataset", list(datasets.keys()))
    img_path = datasets[choice]

    # load data
    X, y = read_HSI(img_path)
    H, W, C = X.shape
    st.write(f"Image shape: **{H} x {W} x {C}** (H x W x C)")

    # show a quick RGB composite
    st.image(get_rgb_composite(X), caption="RGB Composite of the HSI", use_container_width=True)

    # ─── GA parameters ───
    st.sidebar.header("Genetic Algorithm Params")
    num_bands   = st.sidebar.slider("Bands to select", 4, C, min(10, C), 1)
    pop_size    = st.sidebar.number_input("Population size", 4, 256, 128, 4)
    generations = st.sidebar.number_input("Generations",      1, 200, 150, 1)

    # ─── Run everything ───
    if st.button("Run Selection"):
        with st.spinner("Running GA..."):
            selected = genetic_algorithm(
                Image = img_path,
                NUM_BANDS = num_bands,
                POP_SIZE = pop_size,
                GENERATIONS = generations,
            )
        st.success("Band selection done!")
        st.write("Selected bands:", sorted(selected.tolist()))

        col1, col2 = st.columns(2)
        with col1:
            st.image(get_rgb_composite(X), caption="Original RGB Composite", use_container_width=True)
        with col2:
            st.image(get_rgb_composite(X, selected), caption="RGB Composite from Selected Bands Map", use_container_width=True)

if __name__ == "__main__":
    main()
