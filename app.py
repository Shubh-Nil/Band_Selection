import os
import glob

import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from train import genetic_algorithm
from utils import *


def classify_hsi(X: np.ndarray,
                 y: np.ndarray,
                 selected_bands: list[int] | None = None,
                 n_classes: int | None = None) -> np.ndarray:
    """
    Perform pixel-wise classification using either all bands
    or a subset of bands. Returns an H×W×3 RGB map.
    """
    H, W, C = X.shape

    # 1) Flatten
    X_flat = X.reshape(-1, C)      # shape (H*W, C)
    y_flat = y.flatten()           # shape (H*W,)

    # 2) Mask for training
    valid = y_flat > 0
    X_train = X_flat[valid]
    y_train = y_flat[valid]

    # 3) Choose features for training
    if selected_bands is None:
        X_feat_train = X_train          # use all C bands
        X_feat_full  = X_flat           # full image features
    else:
        X_feat_train = X_train[:, selected_bands]
        X_feat_full  = X_flat[:, selected_bands]

    # 4) Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_feat_train, y_train)

    # 5) Predict on every pixel
    y_pred_flat = clf.predict(X_feat_full)    # shape (H*W,)

    # 6) Reshape back to (H, W)
    y_pred = y_pred_flat.reshape(H, W)

    # 7) (Optional) reset background to zero
    y_pred[y == 0] = 0

    # 8) Color map
    n_cl = n_classes or int(y_pred.max())
    cmap = plt.get_cmap('tab20', n_cl)
    norm = mcolors.BoundaryNorm(np.arange(n_cl+1)-0.5, cmap.N)
    color_map = cmap(norm(y_pred))[..., :3]  # drop alpha

    return color_map


def get_rgb_composite(X: np.ndarray, selected_bands = None, use_pca: bool = True) -> np.ndarray:
    """Make a quick RGB composite by picking three bands and normalizing."""
    H, W, C = X.shape

    if use_pca:
        # Determine which bands to use for PCA
        if selected_bands is not None:
            # Use only selected bands for PCA
            X_sel = np.stack([X[:, :, b] for b in selected_bands], axis=-1)
        else:
            # Use all bands for PCA
            X_sel = X

        # Reshape to (num_pixels, num_bands)
        X_reshaped = X_sel.reshape(-1, X_sel.shape[2])

        # Apply PCA to reduce to 3 components
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_reshaped)  # shape: (H*W, 3)

        # Reshape back to image form
        rgb = X_pca.reshape(H, W, 3)

        # Normalize each channel to [0,1]
        rgb_min = rgb.min(axis=(0, 1), keepdims=True)
        rgb_max = rgb.max(axis=(0, 1), keepdims=True)
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-12)


    # pick three representative bands
    if selected_bands is not None:
        # b0, b1, b2 = selected_bands[:3]
        b0, b1, b2 = np.random.choice(selected_bands, size=3, replace=False)
    else:    
        # b0, b1, b2 = (0, C // 2, C - 1)
        # b0, b1, b2 = (15, 30, 95)
        b0, b1, b2 = np.random.choice(C, size=3, replace=False)

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
    generations = st.sidebar.number_input("Generations",      1, 200, 60, 1)

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

        # ─── RGB Visuals ───
        col1, col2 = st.columns(2)
        with col1:
            st.image(get_rgb_composite(X), caption="Original RGB Composite", use_container_width=True)
        with col2:
            st.image(get_rgb_composite(X, selected), caption="RGB Composite from Selected Bands Map", use_container_width=True)

        # ─── Classification Map ───
        st.header("Classification Maps")
        col3, col4 = st.columns(2)
        with col3:
            st.image(classify_hsi(X, y, selected_bands=None), caption="Classification on Selected Bands", use_container_width=True)
        with col4:
            st.image(classify_hsi(X, y, selected_bands=selected), caption="Classification on All Bands", use_container_width=True)

if __name__ == "__main__":
    main()






# import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     # 1) Path to your .mat file
#     mat_file = "data/Indian_pines/Indian_pines_corrected.mat"

#     # 2) Which bands to use as R, G, B?
#     #    If you leave this None, it'll default to [15, 30, 95].
#     selected_bands = [41,27,74]

#     # 3) Where to save your composite
#     out_path = "indian_pines_rgb.png"
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

#     # --- load, composite, save ---
#     X,_ = read_HSI(mat_file)                     # → (145, 145, 200)
#     rgb = get_rgb_composite(X, selected_bands)  # → (145, 145, 3), values in [0,1]

#     # save with matplotlib (auto‐scales [0,1]→[0,255])
#     plt.imsave(out_path, rgb)
#     print(f"Saved RGB composite of bands {selected_bands} to {out_path}")