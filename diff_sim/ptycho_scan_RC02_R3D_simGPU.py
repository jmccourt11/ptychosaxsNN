# coding: utf-8
#%%
import scipy.io as sio
import numpy as np
import cupy as cp
from cupyx.scipy.fft import fft2, fftshift, ifft2
from cupyx.scipy.signal import convolve2d as conv2
from cupyx.scipy.ndimage import rotate as rotate_gpu
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from matplotlib import colors
import random
import h5py
import pdb
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
from cupyx.scipy.ndimage import zoom as zoom_gpu


import sys
import importlib
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)

from cupyx.scipy.ndimage import gaussian_filter


#%%
def hanning_gpu(image):
    # GPU version of hanning window
    xs = cp.hanning(image.shape[0])
    ys = cp.hanning(image.shape[1])
    temp = cp.outer(xs, ys)
    return temp

def vignette_gpu(image):
    # GPU version of vignette
    rows, cols = image.shape
    X, Y = cp.meshgrid(cp.linspace(-1, 1, cols), cp.linspace(-1, 1, rows))
    distance = cp.maximum(cp.abs(X), cp.abs(Y))
    vignette_mask = cp.clip(1 - distance, 0, 1)
    vignette_image = image * vignette_mask
    return vignette_image

def load_and_prepare_data():
    sample_dir = 'RC02_R3D_'
    #base_directory = '/scratch/2025_Feb/'
    base_directory = '/net/micdata/data2/12IDC/2024_Dec/results/'
    recon_path = 'MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g50_Ndp1280_mom0.5_bg0.1_vp4_vi_mm/'
    scan_number = 888

    # # Load data and move to GPU
    # with h5py.File(f"{base_directory}S{scan_number:04d}/{recon_path}/recon_Niter1000.h5", 'r') as f:
    #     ob = f['object'][()]
    #     pb = f['probe'][()]
    ob = sio.loadmat(f"{base_directory}/{sample_dir}/fly{scan_number:03d}/roi0_Ndp1280/{recon_path}/Niter200.mat")
    ob_w = cp.array(ob['object'])
    pb = cp.array(ob['probe'])
    
    ob_w = ob
    #pb1 = pb[:,:,0,0]
    pb1 = np.sum(pb, axis=(2))[:,:,0]
    
    return ob_w, pb1

def rotate_lattice_gpu(amplitude_3d, angles):
    """Rotate 3D lattice on GPU using CuPy's rotation function"""
    # Use angles directly in degrees like the CPU version
    rotated = rotate_gpu(amplitude_3d, angle=angles[0], axes=(1, 2), reshape=False)
    rotated = rotate_gpu(rotated, angle=angles[1], axes=(0, 2), reshape=False)
    rotated = rotate_gpu(rotated, angle=angles[2], axes=(0, 1), reshape=False)
    return rotated

def bin_ndarray(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray to a new shape by averaging or summing.
    """
    shape = ndarray.shape
    assert len(shape) == len(new_shape)
    compression_pairs = [(d, c//d) for d, c in zip(new_shape, shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray
def reference_DP():
    sample_dir = 'RC02_R3D_'
    #base_directory = '/scratch/2025_Feb/'
    base_directory = '/net/micdata/data2/12IDC/2024_Dec/results/'
    scan_number = 888
    filename=f'{base_directory}/{sample_dir}/fly{scan_number:03d}/data_roi0_Ndp1280_dp.hdf5'
    with h5py.File(filename, 'r') as f:
        data=f['dp'][()]
    return data

def soft_disk(xx, yy, x, y, pr, edge_width=2.0):
    r = cp.sqrt((xx-x)**2 + (yy-y)**2)
    return 0.5 * (1 + cp.tanh((pr - r) / edge_width))
# --- 2D Complex Lattice Generation Function ---
def generate_2d_complex_lattice(
    shape=(1280, 1280),
    particle_spacing=8,
    amplitude_noise=0.1,
    phase_noise=0.2,
    particle_density=2.0,
    spacing_noise=0.0,
    radius_noise=0.0,
    apply_window=True,
    rotation_angle=None
):
    """
    Generate a 2D complex lattice with amplitude/phase noise, Hanning, vignette windowing,
    and optional particle spacing/radius noise.
    """
    amplitude_2d = cp.zeros(shape, dtype=cp.complex64)
    phase_2d = cp.zeros(shape, dtype=cp.complex64)
    x = 0
    # Calculate total number of iterations for progress bar
    total_x_steps = int(shape[0] / particle_spacing)
    total_y_steps = int(shape[1] / particle_spacing)
    total_steps = total_x_steps * total_y_steps
    
    with tqdm(total=total_steps, desc="Generating lattice") as pbar:
        while x < shape[0]:
            # Add noise to spacing for this column
            actual_spacing_x = particle_spacing + cp.random.uniform(-spacing_noise, spacing_noise)
            y = 0
            while y < shape[1]:
                # Add noise to spacing for this row
                actual_spacing_y = particle_spacing + cp.random.uniform(-spacing_noise, spacing_noise)
                # Add noise to radius for this particle
                particle_radius = (particle_spacing / 2) + cp.random.uniform(-radius_noise, radius_noise)
                pr = float(particle_radius)
                xx, yy = cp.meshgrid(
                    cp.arange(float(x-pr), float(x+pr)),
                    cp.arange(float(y-pr), float(y+pr))
                )
                circle = soft_disk(xx, yy, x, y, pr)
                x_min = max(0, int(float(x-pr)))
                x_max = min(shape[0], int(float(x+pr)))
                y_min = max(0, int(float(y-pr)))
                y_max = min(shape[1], int(float(y+pr)))
                amp_variation = 1.0 + cp.random.uniform(-amplitude_noise, amplitude_noise)
                phase_variation = 1.0 + cp.random.uniform(-phase_noise, phase_noise)
                region_amp = amplitude_2d[y_min:y_max, x_min:x_max]
                region_phase = phase_2d[y_min:y_max, x_min:x_max]
                region_amp += particle_density * amp_variation * circle[:(y_max-y_min), :(x_max-x_min)]
                region_phase += particle_density * phase_variation * circle[:(y_max-y_min), :(x_max-x_min)]
                amplitude_2d[y_min:y_max, x_min:x_max] = region_amp
                phase_2d[y_min:y_max, x_min:x_max] = region_phase
                y += actual_spacing_y
                pbar.update(1)
            x += actual_spacing_x
    # Only apply window if requested
    if apply_window:
        amplitude_2d = hanning_gpu(amplitude_2d) * amplitude_2d
        amplitude_2d = vignette_gpu(amplitude_2d)
        amplitude_2d /= cp.max(amplitude_2d)
        phase_2d = hanning_gpu(phase_2d) * phase_2d
        phase_2d = vignette_gpu(phase_2d)
        phase_2d /= cp.max(phase_2d)
    else:
        amplitude_2d /= cp.max(amplitude_2d)
        phase_2d /= cp.max(phase_2d)
    # Create complex lattice
    complex_lattice = amplitude_2d * cp.exp(1j * phase_2d)
    return complex_lattice

def tukey2d(shape, alpha=0.5):
    """Create a 2D Tukey window with tunable alpha (blend strength)."""
    from scipy.signal.windows import tukey
    win1d_x = tukey(shape[0], alpha)
    win1d_y = tukey(shape[1], alpha)
    win2d = np.outer(win1d_x, win1d_y)
    return win2d

def generate_multi_domain_lattice(
    shape=(1280, 1280),
    domain_grid=(2, 2),
    domain_params=None,
    blend_alpha=0.5
):
    """
    Generate a 2D complex lattice with multiple domains, each with its own lattice parameters.
    - shape: overall lattice shape
    - domain_grid: (rows, cols) of domains
    - domain_params: list of dicts with keys matching generate_2d_complex_lattice params
    """
    domain_rows, domain_cols = domain_grid
    domain_height = shape[0] // domain_rows
    domain_width = shape[1] // domain_cols

    # If no params provided, use random ones for each domain
    if domain_params is None:
        domain_params = []
        for _ in range(domain_rows * domain_cols):
            domain_params.append({
                'shape': (domain_height, domain_width),
                'particle_spacing': 4, # np.random.uniform(5, 12),
                'amplitude_noise': 0.5,#np.random.uniform(0.3, 0.5),
                'phase_noise': 0.5,#np.random.uniform(0.3, 0.5),
                'particle_density': 1.0, # np.random.uniform(1.0, 4.0),
                'spacing_noise': 0.0,
                'radius_noise': 0.0,
                'apply_window': False,  # Do not apply window per domain
                'rotation_angle': np.random.uniform(0, 360)  # random rotation for each domain
            })

    # Create the full lattice
    full_lattice = cp.zeros(shape, dtype=cp.complex64)

    # Fill each domain
    idx = 0
    for i in range(domain_rows):
        for j in range(domain_cols):
            y_start = i * domain_height
            y_end = (i + 1) * domain_height
            x_start = j * domain_width
            x_end = (j + 1) * domain_width
            params = domain_params[idx]
            params['shape'] = (y_end - y_start, x_end - x_start)
            params['apply_window'] = False  # Ensure window is not applied per domain
            domain_lattice = generate_2d_complex_lattice(**params)
            # Rotate if specified
            rotation_angle = params.get('rotation_angle', 0)
            if rotation_angle != 0:
                domain_lattice = rotate_gpu(domain_lattice, angle=rotation_angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0)
            # Apply tunable Tukey window to domain
            win = cp.array(tukey2d(domain_lattice.shape, alpha=blend_alpha))
            domain_lattice *= win
            full_lattice[y_start:y_end, x_start:x_end] = domain_lattice
            idx += 1

    # Apply Hanning and vignette to the full lattice (amplitude and phase)
    amplitude = cp.abs(full_lattice)
    phase = cp.angle(full_lattice)
    amplitude = hanning_gpu(amplitude) * amplitude
    amplitude = vignette_gpu(amplitude)
    amplitude /= cp.max(amplitude)
    phase = hanning_gpu(phase) * phase
    phase = vignette_gpu(phase)
    phase /= cp.max(phase)
    full_lattice = amplitude * cp.exp(1j * phase)

    return full_lattice,domain_params


#%%
def main():
    # GPU device selection
    gpu_id = 1  # Change this to select different GPU (0, 1, 2, etc.)
    with cp.cuda.Device(gpu_id):
        # Initialize parameters
        dpsize = 1280
        sim_size = 1280
        grid_size = sim_size * 3
        nsteps = 5
        nscans = 1
        plot = True
        total_plot = False
        total = False
        save = False   
        save_total = False
        noise_on = False
        resize_pbp = False
        dr = 'DELETE'
        # Load and prepare data
        ob_w, pb1 = load_and_prepare_data()
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(np.abs(cp.asnumpy(pb1)))
            ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(cp.asnumpy(pb1)))), norm=colors.LogNorm(), cmap='jet')
            plt.show()
        # Load pinhole and PSF
        pbp = cp.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_bw0.2_wl1.24e-10_ps0.15_gs1280x1280.npy')
        psf_pinhole = cp.abs(cp.fft.ifft2(cp.load('/home/beams/PTYCHOSAXS/NN/probe_pinhole_bw0.2_wl1.24e-10_ps0.15_gs256x256.npy')))
        #pb1 = cp.array(np.abs(np.fft.fftshift(np.fft.fft2(cp.asnumpy(pb1))))[1280//2-64:1280//2+64,1280//2-64:1280//2+64])
        pb1 = cp.array(pb1)
        
        # Load mask
        detector_mask = ~cp.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_sum_RC02_R3D_1280.npy')
        # Remove all 3D lattice code and references
        count = 1
        # Define scan pattern - center in the middle of the grid
        center_x = grid_size // 2
        center_y = grid_size // 2
        center_concentration = 0.8
        scan_range = int(1280 * center_concentration)
        start_x = center_x - scan_range // 2
        start_y = center_y - scan_range // 2
        step_size_x = scan_range // (nsteps-1) if nsteps > 1 else 0
        step_size_y = scan_range // (nsteps-1) if nsteps > 1 else 0
        # Main simulation loop
        #for l in range(nscans):
        for l in tqdm(range(nscans)):
            total_intensity = cp.zeros((dpsize, dpsize))
            total_intensity_conv = cp.zeros((dpsize, dpsize))
            # --- Generate 2D complex lattice ---
            # particles_2d = generate_2d_complex_lattice(shape=(1280, 1280),
            #                         particle_spacing=6,
            #                         amplitude_noise=0.5, #random.uniform(0.0,0.5),
            #                         phase_noise=0.5, #random.uniform(0.0,0.5),
            #                         particle_density=1.0,
            #                         spacing_noise=0.0,
            #                         radius_noise=0.0)
            particles_2d,domain_params = generate_multi_domain_lattice(shape=(1280, 1280),
                                                domain_grid=(2, 2),
                                                domain_params=None,
                                                blend_alpha=0.5)
            for k in range(nsteps):
                for i in range(nsteps):

                    # Pad on GPU
                    padding = (grid_size - 1280) // 2
                    pad_value = 0#cp.min(particles_2d)
                    bkg = pad_value
                    particles_padded = cp.pad(particles_2d + bkg, pad_width=padding, mode='constant', constant_values=pad_value)
                    # Add noise if enabled
                    if noise_on:
                        noise = cp.random.uniform(-1, 1, (grid_size, grid_size))/100
                        particles_padded += noise
                    # Calculate probe position
                    probe_center_x = start_x + i * step_size_x
                    probe_center_y = start_y + k * step_size_y
                    p_hw = int(pb1.shape[0]/2)
                    p_hw = int(pbp.shape[0]/2)
                    # Extract illuminated region
                    ob_w_2 = particles_padded
                    ob_e_2 = ob_w_2[probe_center_x-p_hw:probe_center_x+p_hw, 
                                  probe_center_y-p_hw:probe_center_y+p_hw] * \
                             hanning_gpu(ob_w_2[probe_center_x-p_hw:probe_center_x+p_hw,
                                             probe_center_y-p_hw:probe_center_y+p_hw])
                    # Calculate diffraction patterns on GPU
                    psi_k_2_ideal = fft2(ob_e_2 * pbp)
                    pinhole_DP = fftshift(fft2(ob_e_2))
                    psi_k_2 = fftshift(fft2(pb1 * ob_e_2))
                    #psi_k_2 = conv2(psi_k_2_ideal, pb1, 'same', boundary='symm')
                    # Calculate intensities
                    conv_DP = cp.abs(psi_k_2)**2*detector_mask
                    pinhole_DP = cp.abs(psi_k_2_ideal)**2
                    
                    # Calculate convolution of pinhole diffraction pattern with probe
                    conv_DP_pinhole = conv2(pinhole_DP, cp.abs(cp.fft.fftshift(cp.fft.fft2(pb1))), 'same', boundary='symm')
                    conv_DP_pinhole *= detector_mask
                    pinhole_DP_extra_conv =conv2(pinhole_DP, cp.abs(cp.fft.fft2(pbp)), 'same', boundary='symm')
                    
                    # Apply Gaussian blur to pinhole diffraction pattern
                    n_pixels=16
                    pinhole_DP_extra_conv = gaussian_filter(pinhole_DP_extra_conv, sigma=n_pixels/2.355)
                    
                    # Bin diffraction patterns back to dpsize if simulation size is different
                    if sim_size != dpsize:
                        bin_factor = sim_size // dpsize
                        conv_DP = bin_ndarray(conv_DP, (dpsize, dpsize), operation='sum')
                        pinhole_DP = bin_ndarray(pinhole_DP, (dpsize, dpsize), operation='sum')
                        pinhole_DP_extra_conv = bin_ndarray(pinhole_DP_extra_conv, (dpsize, dpsize), operation='sum')
                    # Resize and accumulate total intensities
                    if total:
                        total_intensity += cp.array(resize(cp.asnumpy(pinhole_DP), (256,256), preserve_range=True, anti_aliasing=True))
                        total_intensity_conv += cp.array(resize(cp.asnumpy(conv_DP), (256,256), preserve_range=True, anti_aliasing=True))
                        
                    
                    if plot:
                        fig,ax=plt.subplots(2,2,figsize=(10,10))
                        ax[0][0].imshow(np.abs(cp.asnumpy(pinhole_DP)),norm=colors.LogNorm(),cmap='jet')
                        ax[0][1].imshow(np.abs(cp.asnumpy(pinhole_DP_extra_conv)),norm=colors.LogNorm(),cmap='jet')
                        #ax[1][0].imshow(np.abs(cp.asnumpy(conv_DP)),norm=colors.LogNorm(),cmap='jet')
                        ax[1][0].imshow(np.abs(cp.asnumpy(conv_DP_pinhole)),norm=colors.LogNorm(),cmap='jet')
                        ax[1][1].imshow(np.abs(cp.asnumpy(ob_e_2)),cmap='gray')
                        ax[1][1].imshow(np.abs(cp.asnumpy(pb1)),cmap='gray',alpha=0.2)
                        plt.show()
                    # Split diffraction patterns into 5x5 grid of 256x256 segments
                    segments_pinhole = np.zeros((5, 5, 256, 256))
                    segments_pinhole_extra = cp.zeros((5, 5, 256, 256))
                    segments_conv = cp.zeros((5, 5, 256, 256))
                    segments_conv_pinhole = np.zeros((5, 5, 256, 256))
                    pinhole_cpu = cp.asnumpy(pinhole_DP)
                    conv_cpu = cp.asnumpy(conv_DP)
                    conv_pinhole_cpu = cp.asnumpy(conv_DP_pinhole)
                    for i_seg in range(5):
                        for j_seg in range(5):
                            y_start = i_seg * 256
                            y_end = (i_seg + 1) * 256
                            x_start = j_seg * 256
                            x_end = (j_seg + 1) * 256
                            segments_pinhole[i_seg,j_seg] = pinhole_cpu[y_start:y_end, x_start:x_end]
                            segments_pinhole_extra[i_seg,j_seg] = pinhole_DP_extra_conv[y_start:y_end, x_start:x_end]
                            segments_conv[i_seg,j_seg] = conv_DP[y_start:y_end, x_start:x_end]
                            segments_conv_pinhole[i_seg,j_seg] = conv_pinhole_cpu[y_start:y_end, x_start:x_end]
                    if plot:
                        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
                        fig.suptitle('Segmented Pinhole Diffraction Pattern')
                        for i_seg in range(5):
                            for j_seg in range(5):
                                im = axes[i_seg,j_seg].imshow(segments_pinhole[i_seg,j_seg], norm=colors.LogNorm(), cmap='jet')
                                plt.colorbar(im, ax=axes[i_seg,j_seg])
                                axes[i_seg,j_seg].axis('off')
                        plt.tight_layout()
                        plt.show()
                        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
                        fig.suptitle('Segmented Extra Convolution Pattern')
                        for i_seg in range(5):
                            for j_seg in range(5):
                                # segments_pinhole_extra[i_seg,j_seg] = conv2(segments_pinhole_extra[i_seg,j_seg], cp.abs(psf_pinhole), 'same', boundary='symm')
                                im = axes[i_seg,j_seg].imshow(cp.asnumpy(segments_pinhole_extra[i_seg,j_seg]), norm=colors.LogNorm(), cmap='jet')
                                plt.colorbar(im, ax=axes[i_seg,j_seg])
                                axes[i_seg,j_seg].axis('off')
                        plt.tight_layout()
                        plt.show()
                        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
                        fig.suptitle('Segmented Convolution Pattern')
                        for i_seg in range(5):
                            for j_seg in range(5):
                                segments_conv[i_seg,j_seg] = conv2(segments_conv[i_seg,j_seg], cp.abs(psf_pinhole), 'same', boundary='symm')
                                im = axes[i_seg,j_seg].imshow(cp.asnumpy(segments_conv[i_seg,j_seg]), norm=colors.LogNorm(), cmap='jet')
                                plt.colorbar(im, ax=axes[i_seg,j_seg])
                                axes[i_seg,j_seg].axis('off')
                        plt.tight_layout()
                        plt.show()
                        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
                        fig.suptitle('Segmented Convolution Pinhole Diffraction Pattern')
                        for i_seg in range(5):
                            for j_seg in range(5):
                                im = axes[i_seg,j_seg].imshow(segments_conv_pinhole[i_seg,j_seg], norm=colors.LogNorm(), cmap='jet')
                                plt.colorbar(im, ax=axes[i_seg,j_seg])
                                axes[i_seg,j_seg].axis('off')
                        plt.tight_layout()
                        plt.show()
                        
                    if save:
                        for i_seg in range(5):
                            for j_seg in range(5):
                                if i_seg == 0 or i_seg == 4 or j_seg == 0 or j_seg == 4 or (i_seg == 2 and j_seg == 2):
                                    segment_filename = f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/{dr}/output_hanning_conv_{count:05d}.npz'
                                    np.savez(segment_filename,
                                            pinholeDP=segments_pinhole[i_seg,j_seg],
                                            pinholeDP_extra_conv=cp.asnumpy(segments_pinhole_extra[i_seg,j_seg]),
                                            convDP=cp.asnumpy(segments_conv[i_seg,j_seg]),
                                            convDP_pinhole=cp.asnumpy(segments_conv_pinhole[i_seg,j_seg]))
                                    #np.save(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/{dr}/lattice_domain_params_{count:05d}.npy',domain_params)
                                    print(f'saved: {segment_filename}')
                                    count += 1
            if total_plot:
                plt.figure(figsize=(12,5))
                plt.subplot(121)
                plt.imshow(cp.asnumpy(total_intensity), norm=colors.LogNorm(), cmap='jet')
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(cp.asnumpy(total_intensity_conv), norm=colors.LogNorm(), cmap='jet')
                plt.colorbar()
                plt.show()

#%%
def clear_gpu_memory():
    """Clear GPU memory by emptying the cache and garbage collecting"""
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Always clear GPU memory, even if there's an error
        clear_gpu_memory()
#%%
clear_gpu_memory()










# %%
# Initialize arrays to store full diffraction patterns
full_pinholeDP = np.zeros((1280, 1280))  # 17 patterns
full_pinholeDP_extra_conv = np.zeros((1280, 1280))
full_convDP = np.zeros((1280, 1280))
full_convDP_pinhole = np.zeros((1280, 1280))

# Load and reconstruct each pattern
# Create temporary 5x5 grid of segments
segments_pinhole = np.zeros((5, 5, 256, 256))
segments_pinhole_extra = np.zeros((5, 5, 256, 256))
segments_conv = np.zeros((5, 5, 256, 256))
segments_conv_pinhole = np.zeros((5, 5, 256, 256))
scan_index=random.randint(0,30)
num_patterns=range(0,25)
for pattern in num_patterns:
    index=pattern*17+scan_index*425
 # Load saved segments
    for i_seg in range(5):
        for j_seg in range(5):
            if i_seg == 0 or i_seg == 4 or j_seg == 0 or j_seg == 4 or (i_seg == 2 and j_seg == 2):
                index += 1
                filename = f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/DELETE2/output_hanning_conv_{index:05d}.npz'
                print(filename)
                data = np.load(filename)
                #segments_pinhole[i_seg, j_seg] += np.maximum(np.mean(data['pinholeDP']), data['pinholeDP']-np.mean(data['pinholeDP']))
                mean = np.mean(data['pinholeDP'])
                std = np.std(data['pinholeDP'])
                z = (data['pinholeDP'] - mean) / std
                #z_clipped = np.clip(z, 0, None)  # Only keep positive deviations
                segments_pinhole[i_seg, j_seg] += z#_clipped
                #segments_pinhole_extra[i_seg, j_seg] += np.maximum(np.mean(data['pinholeDP_extra_conv']), data['pinholeDP_extra_conv']-np.mean(data['pinholeDP_extra_conv']))
                #segments_pinhole[i_seg,j_seg] += data['pinholeDP']
                
                #segments_conv[i_seg, j_seg] += data['convDP']
                mean = np.mean(data['convDP'])
                std = np.std(data['convDP'])
                z = (data['convDP'] - mean) / std
                #z_clipped = np.clip(z, 0, None)  # Only keep positive deviations
                segments_conv[i_seg, j_seg] += z#_clipped
                
                # segments_conv[i_seg,j_seg] += data['convDP']
                segments_conv_pinhole[i_seg, j_seg] += data['convDP_pinhole']
            else:
                continue


# Reconstruct full 1280x1280 pattern
for i_seg in range(5):
    for j_seg in range(5):
        y_start = i_seg * 256
        y_end = (i_seg + 1) * 256
        x_start = j_seg * 256
        x_end = (j_seg + 1) * 256
        
        full_pinholeDP[y_start:y_end, x_start:x_end] = segments_pinhole[i_seg, j_seg]
        full_pinholeDP_extra_conv[y_start:y_end, x_start:x_end] = segments_pinhole_extra[i_seg, j_seg]
        full_convDP[y_start:y_end, x_start:x_end] = segments_conv[i_seg, j_seg]
        full_convDP_pinhole[y_start:y_end, x_start:x_end] = segments_conv_pinhole[i_seg, j_seg]
    
    
# Plot full diffraction patterns
fig,ax=plt.subplots(2,2,figsize=(10,10))
ax[0][0].imshow(full_convDP_pinhole,norm=colors.LogNorm(),cmap='jet')
ax[0][1].imshow(full_convDP,norm=colors.LogNorm(),cmap='jet')
ax[1][0].imshow(full_pinholeDP,norm=colors.LogNorm(),cmap='jet')
ax[1][1].imshow(np.ones_like(full_pinholeDP),norm=colors.LogNorm(),cmap='jet')
#ax[1][1].imshow(full_pinholeDP_extra_conv,norm=colors.LogNorm(),cmap='jet')
plt.show()
# Segment and plot the full diffraction patterns as 5x5 grids
segment_size = 256
# for dp_name, full_dp in zip([
#     'Full Convolution Pinhole DP',
#     'Full Convolution DP',
#     'Full Pinhole DP',
#     'Full Pinhole Extra Conv DP'],
#     [full_convDP_pinhole, full_convDP, full_pinholeDP, full_pinholeDP_extra_conv]):
    
for dp_name, full_dp in zip([
    'Full Convolution Pinhole DP',
    'Full Convolution DP',
    'Full Pinhole DP'],
    [full_convDP_pinhole, full_convDP, full_pinholeDP]):
    segments = np.zeros((5, 5, segment_size, segment_size))
    for i in range(5):
        for j in range(5):
            y_start = i * segment_size
            y_end = (i + 1) * segment_size
            x_start = j * segment_size
            x_end = (j + 1) * segment_size
            segments[i, j] = full_dp[y_start:y_end, x_start:x_end]
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle(f'Segmented {dp_name}')
    for i in range(5):
        for j in range(5):
            im = axes[i, j].imshow(np.log10(segments[i, j]+1), cmap='jet')
            plt.colorbar(im, ax=axes[i, j])
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

#%%


#%%
ref_DP=reference_DP()

#%%
# Create segments array
segments_ref = np.zeros((5, 5, 256, 256))
index=552

# Segment the reference diffraction pattern
for i in range(5):
    for j in range(5):
        y_start = i * 256
        y_end = (i + 1) * 256
        x_start = j * 256
        x_end = (j + 1) * 256
        segments_ref[i,j] = ref_DP[index, y_start:y_end, x_start:x_end]

# Plot segmented reference diffraction pattern
fig, axes = plt.subplots(5, 5, figsize=(15, 15))
fig.suptitle('Segmented Reference Diffraction Pattern')
for i in range(5):
    for j in range(5):
        im = axes[i,j].imshow(segments_ref[i,j], norm=colors.LogNorm(), cmap='jet')
        plt.colorbar(im, ax=axes[i,j])
        axes[i,j].axis('off')
plt.tight_layout()
plt.show()

# Also show the original plots
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(np.sum(ref_DP,axis=0),norm=colors.LogNorm(),cmap='jet')
ax[1].imshow(ref_DP[index],norm=colors.LogNorm(),cmap='jet')
plt.show()

# %%

sample_dir = 'RC02_R3D_'
#base_directory = '/scratch/2025_Feb/'
base_directory = '/net/micdata/data2/12IDC/2024_Dec/results/'
recon_path = 'MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g50_Ndp1280_mom0.5_bg0.1_vp4_vi_mm/'
scan_number = 888

# # Load data and move to GPU
# with h5py.File(f"{base_directory}S{scan_number:04d}/{recon_path}/recon_Niter1000.h5", 'r') as f:
#     ob = f['object'][()]
#     pb = f['probe'][()]
ob = sio.loadmat(f"{base_directory}/{sample_dir}/fly{scan_number:03d}/roi0_Ndp1280/{recon_path}/Niter200.mat")

#%%
ob_w = np.array(ob['object'])
pb = np.array(ob['probe'])

ob_w = ob
pb1 = pb[:,:,0,0];#np.sum(pb, axis=(2))[:,:,0]  # shape: (1280, 1280)
fig,ax=plt.subplots(1,3,figsize=(10,5))
crop_size =256
ax[0].imshow(np.abs(pb1))
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(pb1))),norm=colors.LogNorm(),cmap='jet')
ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(pb1))[1280//2-crop_size//2:1280//2+crop_size//2,1280//2-crop_size//2:1280//2+crop_size//2]),norm=colors.LogNorm(),cmap='jet')
plt.show()


























# %%


import torch
import scipy.io as sio
import random
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from tqdm import tqdm
from scipy.ndimage import rotate
from scipy.ndimage import zoom
#%%
# Setup GPU
device = torch.device('cuda')

# Create random object
#object_img = torch.randn(1, 1, 1280, 1280, dtype=torch.cfloat, device=device)

# Define scan number
scan_number = 438#888

# Load object from file
#object_img = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly{scan_number:03d}/roi0_Ndp1280/MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/Niter200.mat")['object'][:,:], dtype=torch.cfloat, device=device)
object_img = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2025_Feb/results/ZC4_/fly{scan_number:03d}/roi1_Ndp512/MLc_L1_p10_g1000_Ndp256_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm//Niter2000.mat")['object'][:,:], dtype=torch.cfloat, device=device)

# Random rotation angle between 0 and 360 degrees
angle = random.uniform(0, 360)

# Random zoom factor between 0.8 and 1.2
zoom_factor = random.uniform(0.8, 1.2)

# Convert to numpy for rotation and zoom
object_np = object_img.cpu().numpy()

# Apply zoom using scipy.ndimage.zoom
zoomed = zoom(object_np, (zoom_factor, zoom_factor), order=1)

# Apply rotation using scipy.ndimage.rotate
rotated = rotate(zoomed, angle, reshape=False, order=1)

# Convert back to torch tensor
object_img = torch.tensor(rotated, dtype=torch.cfloat, device=device)
print("Original object shape:", object_img.shape)


scan_number=888
# Load probe from file and move to GPU
probe = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly{scan_number:03d}/roi0_Ndp1280/MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/Niter200.mat")['probe'][:,:,0], dtype=torch.cfloat, device=device)
print("Probe shape:", probe.shape)

#load the probe from the file
probe_ideal = np.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_bw0.2_wl1.24e-10_ps0.15_gs1280x1280.npy')
probe_ideal = torch.tensor(probe_ideal, dtype=torch.cfloat, device=device)
        


# Pad object to match probe size
if object_img.shape[0] < probe.shape[0]:
    # Calculate padding dimensions
    pad_height = (2*probe.shape[0] - object_img.shape[0]) // 2
    pad_width = (2*probe.shape[1] - object_img.shape[1]) // 2
    
    # Create vignette for edges
    y = torch.linspace(-1, 1, object_img.shape[0])
    x = torch.linspace(-1, 1, object_img.shape[1])
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X**2 + Y**2).to(device)
    edge_vignette = (1 - R**2).clamp(0, 1)
    
    # Apply edge vignette
    object_img = object_img * edge_vignette
    
    # Pad the object
    object_img = torch.nn.functional.pad(object_img, 
                                    (pad_width, pad_width, pad_height, pad_height),
                                    mode='constant', value=0)
    
    # Create vignette for full padded object
    y = torch.linspace(-1, 1, object_img.shape[0])
    x = torch.linspace(-1, 1, object_img.shape[1])
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X**2 + Y**2).to(device)
    full_vignette = (1 - R**2).clamp(0, 1)
    
    # Apply full vignette
    object_img = object_img * full_vignette
    
    print("Padded object shape:", object_img.shape)



# Define scan parameters
scan_step = probe.shape[0]//random.randint(12,36) # Step size between scan positions
patch_size = 128  # Size of patches to process at once
probe_size = probe.shape[0]


# Calculate scan positions
scan_positions = []
for y in range(0, object_img.shape[0]-probe_size+1, scan_step):
    for x in range(0, object_img.shape[1]-probe_size+1, scan_step):
        scan_positions.append([y, x])
scan_positions = torch.tensor(scan_positions, device=device)

# Initialize list to store result for each scan position
patterns = []
dps = []
dps_ideal = []
batch_size = 16  # Adjust as needed for your GPU

# Batch processing of scan positions
for i in tqdm(range(0, len(scan_positions), batch_size), desc="Batch scanning"):
    batch = scan_positions[i:i+batch_size]
    batch_size_actual = len(batch)
    # Prepare tensor for object patches
    object_patches = torch.zeros((batch_size_actual, probe_size, probe_size), dtype=torch.cfloat, device=device)
    for j, pos in enumerate(batch):
        y, x = pos
        object_patches[j] = object_img[y:y+probe_size, x:x+probe_size]
    # Multiply by probe (broadcasted)
    # Create vignette for probe size
    y = torch.linspace(-1, 1, probe_size)
    x = torch.linspace(-1, 1, probe_size)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X**2 + Y**2).to(device)
    probe_vignette = (1 - R**2).clamp(0, 1)
    
    # Apply vignette to both exit waves
    exit_waves = object_patches * probe * probe_vignette  # shape: (batch_size, probe_size, probe_size)
    exit_waves_ideal = object_patches * probe_ideal * probe_vignette
    # FFT and intensity
    dp = torch.fft.fftshift(torch.fft.fft2(exit_waves, norm='ortho'), dim=(-2, -1))
    intensity = torch.abs(dp) ** 2  # shape: (batch_size, probe_size, probe_size)
    #dp_ideal = torch.fft.fftshift(torch.fft.fft2(exit_waves_ideal, norm='ortho'), dim=(-2, -1))
    dp_ideal = torch.fft.fft2(exit_waves_ideal, norm='ortho')
    intensity_ideal = torch.abs(dp_ideal) ** 2  # shape: (batch_size, probe_size, probe_size)
    # Store results
    patterns.extend(exit_waves.detach().cpu())
    dps.extend(intensity.detach().cpu().numpy())
    dps_ideal.extend(intensity_ideal.detach().cpu().numpy())

# --- Plotting example ---
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
im1=ax[0][0].imshow(np.sum(dps, axis=0), norm=colors.LogNorm(), cmap='jet')
im2=ax[0][1].imshow(np.sum(dps_ideal, axis=0), norm=colors.LogNorm(), cmap='jet')
ri = random.randint(0, len(dps)-1)
im3=ax[1][0].imshow(dps[ri], norm=colors.LogNorm(), cmap='jet')
im4=ax[1][1].imshow(dps_ideal[ri], norm=colors.LogNorm(), cmap='jet')
plt.colorbar(im1)
plt.colorbar(im2)
plt.colorbar(im3)
plt.colorbar(im4)
plt.show()




# %%
def preprocess_dps(dp):
    # Take log10 and normalize to 0-1 range
    dp_log = np.log10(dp)
    dp_norm = (dp_log - np.min(dp_log)) / (np.max(dp_log) - np.min(dp_log))
    return dp_norm
fig,ax=plt.subplots(1,2,figsize=(10,5))
ri = random.randint(0, len(dps)-1)
im1=ax[0].imshow(preprocess_dps(dps[ri]), cmap='jet')
im2=ax[1].imshow(preprocess_dps(dps_ideal[ri]), cmap='jet')
plt.colorbar(im1)
plt.colorbar(im2)
plt.show()
# %%




# Segment and plot summed diffraction patterns
segment_size = 256

# Sum up all diffraction patterns
summed_dp = np.sum(dps, axis=0)
summed_dp_ideal = np.sum(dps_ideal, axis=0)

# Create segments arrays
segments_dp = np.zeros((5, 5, segment_size, segment_size))
segments_dp_ideal = np.zeros((5, 5, segment_size, segment_size))

# Segment the summed patterns
for i in range(5):
    for j in range(5):
        y_start = i * segment_size
        y_end = (i + 1) * segment_size
        x_start = j * segment_size 
        x_end = (j + 1) * segment_size
        segments_dp[i,j] = summed_dp[y_start:y_end, x_start:x_end]
        segments_dp_ideal[i,j] = summed_dp_ideal[y_start:y_end, x_start:x_end]

# Plot segmented patterns
for dp_name, segments in zip(['Segmented Summed DP', 'Segmented Summed Ideal DP'], 
                           [segments_dp, segments_dp_ideal]):
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle(dp_name)
    for i in range(5):
        for j in range(5):
            im = axes[i,j].imshow(preprocess_dps(segments[i,j]), cmap='jet')
            plt.colorbar(im, ax=axes[i,j])
            axes[i,j].axis('off')
    plt.tight_layout()
    plt.show()