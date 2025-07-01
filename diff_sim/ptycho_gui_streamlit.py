import streamlit as st
import torch
import numpy as np
import scipy.io as sio
from scipy.ndimage import rotate, zoom
import matplotlib.pyplot as plt
import random

st.title("Ptychography Simulation GUI")

# --- All user controls at the top ---
scan_number = st.number_input("Scan number", min_value=0, max_value=9999, value=438)
object_file = st.text_input("Object .mat file", value=f"/net/micdata/data2/12IDC/2025_Feb/results/ZC4_/fly{scan_number:03d}/roi1_Ndp512/MLc_L1_p10_g1000_Ndp256_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm//Niter2000.mat")
probe_file = st.text_input("Probe .mat file", value=f"/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly888/roi0_Ndp1280/MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/Niter200.mat")
probe_ideal_file = st.text_input("Probe ideal .npy file", value="/home/beams0/PTYCHOSAXS/NN/probe_pinhole_bw0.2_wl1.24e-10_ps0.15_gs1280x1280.npy")
zoom_factor = st.slider("Zoom factor", 0.5, 1.5, 1.0, 0.01)
angle = st.slider("Rotation angle (deg)", 0, 360, 0, 1)
batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=8)
scan_step_div = st.number_input("Scan step divisor", min_value=2, max_value=64, value=16)
segment_size = st.number_input("Segment size", min_value=32, max_value=1024, value=256)
pattern_index = st.number_input("Select pattern index", min_value=0, value=0, step=1)

run_button = st.button("Run Simulation")

if run_button:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- Load object ---
    object_img = torch.tensor(sio.loadmat(object_file)['object'][:,:], dtype=torch.cfloat, device=device)
    # --- Apply zoom and rotation ---
    object_np = object_img.cpu().numpy()
    zoomed = zoom(object_np, (zoom_factor, zoom_factor), order=1)
    rotated = rotate(zoomed, angle, reshape=False, order=1)
    object_img = torch.tensor(rotated, dtype=torch.cfloat, device=device)
    # --- Load probe ---
    probe = torch.tensor(sio.loadmat(probe_file)['probe'][:,:,0], dtype=torch.cfloat, device=device)
    probe_ideal = torch.tensor(np.load(probe_ideal_file), dtype=torch.cfloat, device=device)
    probe_size = probe.shape[0]
    # --- Pad object if needed ---
    if object_img.shape[0] < probe.shape[0]:
        pad_height = (2*probe.shape[0] - object_img.shape[0]) // 2
        pad_width = (2*probe.shape[1] - object_img.shape[1]) // 2
        y = torch.linspace(-1, 1, object_img.shape[0])
        x = torch.linspace(-1, 1, object_img.shape[1])
        X, Y = torch.meshgrid(x, y, indexing='xy')
        R = torch.sqrt(X**2 + Y**2).to(device)
        edge_vignette = (1 - R**2).clamp(0, 1)
        object_img = object_img * edge_vignette
        object_img = torch.nn.functional.pad(object_img, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)
        y = torch.linspace(-1, 1, object_img.shape[0])
        x = torch.linspace(-1, 1, object_img.shape[1])
        X, Y = torch.meshgrid(x, y, indexing='xy')
        R = torch.sqrt(X**2 + Y**2).to(device)
        full_vignette = (1 - R**2).clamp(0, 1)
        object_img = object_img * full_vignette

    # --- Scan positions ---
    scan_step = probe.shape[0] // scan_step_div
    scan_positions = []
    for y in range(0, object_img.shape[0]-probe_size+1, scan_step):
        for x in range(0, object_img.shape[1]-probe_size+1, scan_step):
            scan_positions.append([y, x])
    scan_positions = torch.tensor(scan_positions, device=device)
    # --- Batch processing ---
    patterns, dps, dps_ideal = [], [], []
    for i in range(0, len(scan_positions), batch_size):
        batch = scan_positions[i:i+batch_size]
        batch_size_actual = len(batch)
        object_patches = torch.zeros((batch_size_actual, probe_size, probe_size), dtype=torch.cfloat, device=device)
        for j, pos in enumerate(batch):
            y, x = pos
            object_patches[j] = object_img[y:y+probe_size, x:x+probe_size]
        y = torch.linspace(-1, 1, probe_size)
        x = torch.linspace(-1, 1, probe_size)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        R = torch.sqrt(X**2 + Y**2).to(device)
        probe_vignette = (1 - R**2).clamp(0, 1)
        exit_waves = object_patches * probe * probe_vignette
        exit_waves_ideal = object_patches * probe_ideal * probe_vignette
        dp = torch.fft.fftshift(torch.fft.fft2(exit_waves, norm='ortho'), dim=(-2, -1))
        intensity = torch.abs(dp) ** 2
        dp_ideal = torch.fft.fft2(exit_waves_ideal, norm='ortho')
        intensity_ideal = torch.abs(dp_ideal) ** 2
        patterns.extend(exit_waves.detach().cpu())
        dps.extend(intensity.detach().cpu().numpy())
        dps_ideal.extend(intensity_ideal.detach().cpu().numpy())

    # --- Plotting ---
    st.subheader("Summed Diffraction Patterns")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(np.sum(dps, axis=0), norm='log', cmap='jet')
    ax[0].set_title("Summed DP")
    ax[1].imshow(np.sum(dps_ideal, axis=0), norm='log', cmap='jet')
    ax[1].set_title("Summed Ideal DP")
    st.pyplot(fig)

    # --- Preprocessing and single pattern selection ---
    st.subheader("Preprocessed Single Diffraction Pattern")
    def preprocess_dps(dp):
        dp_log = np.log10(dp)  # avoid log(0)
        dp_norm = (dp_log - np.min(dp_log)) / (np.max(dp_log) - np.min(dp_log))
        return dp_norm
    max_idx = len(dps) - 1
    idx = pattern_index
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(preprocess_dps(dps[idx]), cmap='jet')
    ax[0].set_title(f"Preprocessed DP (index {idx})")
    ax[1].imshow(preprocess_dps(dps_ideal[idx]), cmap='jet')
    ax[1].set_title(f"Preprocessed Ideal DP (index {idx})")
    st.pyplot(fig)

    # --- Segmented summed diffraction patterns ---
    st.subheader("Segmented Summed Diffraction Patterns")
    summed_dp = np.sum(dps, axis=0)
    summed_dp_ideal = np.sum(dps_ideal, axis=0)
    n_segments_y = summed_dp.shape[0] // segment_size
    n_segments_x = summed_dp.shape[1] // segment_size
    # Simulated
    st.write("Segmented Summed DP")
    fig, axes = plt.subplots(n_segments_y, n_segments_x, figsize=(3*n_segments_x, 3*n_segments_y))
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            y_start = i * segment_size
            y_end = (i + 1) * segment_size
            x_start = j * segment_size
            x_end = (j + 1) * segment_size
            seg = summed_dp[y_start:y_end, x_start:x_end]
            axes[i, j].imshow(preprocess_dps(seg), cmap='jet')
            axes[i, j].axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    # Ideal
    st.write("Segmented Summed Ideal DP")
    fig, axes = plt.subplots(n_segments_y, n_segments_x, figsize=(3*n_segments_x, 3*n_segments_y))
    for i in range(n_segments_y):
        for j in range(n_segments_x):
            y_start = i * segment_size
            y_end = (i + 1) * segment_size
            x_start = j * segment_size
            x_end = (j + 1) * segment_size
            seg = summed_dp_ideal[y_start:y_end, x_start:x_end]
            axes[i, j].imshow(preprocess_dps(seg), cmap='jet')
            axes[i, j].axis('off')
    plt.tight_layout()
    st.pyplot(fig)