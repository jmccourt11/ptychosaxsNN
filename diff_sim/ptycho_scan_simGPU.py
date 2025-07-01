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

import sys
import importlib
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)

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
    sample_dir = 'ZC4_'
    base_directory = '/scratch/2025_Feb/'
    recon_path = 'MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_model_rotation_scale_shear_asymmetry_noModelCon_vi_mm/MLc_L1_p10_g200_Ndp256_mom0.5_pc800_model_scale_asymmetry_rotation_shear_vp4_vi_mm'
    scan_number = 439

    # Load data and move to GPU
    ob = sio.loadmat(f"{base_directory}/results/{sample_dir}/fly{scan_number}/roi0_Ndp256/{recon_path}/Niter1000.mat")
    ob_w = cp.array(ob['object'])
    pb = cp.array(ob['probe'])
    pb1 = pb[:,:,0,0]
    
    return ob_w, pb1

def rotate_lattice_gpu(amplitude_3d, angles):
    """Rotate 3D lattice on GPU using CuPy's rotation function"""
    # Use angles directly in degrees like the CPU version
    rotated = rotate_gpu(amplitude_3d, angle=angles[0], axes=(1, 2), reshape=False)
    rotated = rotate_gpu(rotated, angle=angles[1], axes=(0, 2), reshape=False)
    rotated = rotate_gpu(rotated, angle=angles[2], axes=(0, 1), reshape=False)
    return rotated

#%%
def main():
    # GPU device selection
    gpu_id = 2  # Change this to select different GPU (0, 1, 2, etc.)
    with cp.cuda.Device(gpu_id):
        # Initialize parameters
        # lattice_size = 400
        # grid_size = 1024
        # lattice_spacing = 6
        # radius = lattice_spacing/2
        # lattice_type = 'SC'
        # center_concentration = 0.5
        lattice_size = 480
        grid_size = 1024
        lattice_spacing = 12
        radius = lattice_spacing/2
        lattice_type = 'FCC'
        center_concentration = 0.6
        
        dpsize = 256
        nsteps = 3
        nscans = 1
        plot = True
        total_plot = True
        total = False
        save = False
        save_total = False
        noise_on = False
        resize_pbp = False
        dr='DELETE'
        
        # Load and prepare data
        ob_w, pb1 = load_and_prepare_data()
        if plot:
            plt.imshow(np.abs(cp.asnumpy(pb1)))
            plt.show()
        
        # Load pinhole and PSF
        pbp = cp.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_complex_256x256_bw0.75.npy')
        psf_pinhole = cp.abs(cp.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole.npy'))
        #psf_pinhole=cp.fft.fft2(pbp)
        
        # Load lattice and move to GPU
        amplitude_3d = cp.array(np.load(f'lattices/lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_type{lattice_type}.npy'))

        # Initialize count
        count = 1

        # Define scan pattern
        center_x = 512
        center_y = 512
        center_concentration = center_concentration
        scan_range = int(lattice_size * center_concentration)
        start_x = center_x - scan_range // 2
        start_y = center_y - scan_range // 2
        step_size_x = scan_range // (nsteps-1) if nsteps > 1 else 0
        step_size_y = scan_range // (nsteps-1) if nsteps > 1 else 0
        
        # Main simulation loop
        for l in tqdm(range(nscans)):
            # Initialize arrays
            total_intensity = np.zeros((dpsize, dpsize))
            total_intensity_conv = np.zeros((dpsize, dpsize))
            
            # Random rotation angles
            rotation_angles = [random.randint(0,90), random.randint(0,90), random.randint(0,90)]
            
            # Rotate lattice on GPU using the new function
            amplitude_3d_rotated = rotate_lattice_gpu(amplitude_3d, rotation_angles)

            for k in range(nsteps):
                for i in range(nsteps):
                    # Project and process on GPU
                    amplitude_2d = cp.sum(amplitude_3d_rotated, axis=2)
                    amplitude_2d = hanning_gpu(amplitude_2d) * amplitude_2d
                    amplitude_2d /= cp.max(amplitude_2d)
                    
                    # Create phase object
                    particles_2d = cp.exp(1j * amplitude_2d)
                    particles_2d = hanning_gpu(particles_2d) * particles_2d
                    particles_2d = vignette_gpu(particles_2d)
                    particles_2d = hanning_gpu(particles_2d) * particles_2d

                    # Pad on GPU
                    padding = (grid_size - lattice_size) // 2
                    pad_value = 1
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

                    # Extract illuminated region
                    ob_w_2 = particles_padded
                    ob_e_2 = ob_w_2[probe_center_x-p_hw:probe_center_x+p_hw, 
                                  probe_center_y-p_hw:probe_center_y+p_hw] * \
                             hanning_gpu(ob_w_2[probe_center_x-p_hw:probe_center_x+p_hw,
                                             probe_center_y-p_hw:probe_center_y+p_hw])
                    # if plot:
                    #     plt.imshow(np.abs(cp.asnumpy(ob_e_2)))
                    #     plt.show()  
                    #     plt.imshow(np.abs(cp.asnumpy(ob_w_2)))
                    #     plt.show()
                    
                    # Calculate diffraction patterns on GPU
                    psi_k_2_ideal = fft2(ob_e_2 * pbp)
                    pinhole_DP = conv2(cp.abs(psi_k_2_ideal), cp.abs(psf_pinhole), 'same', boundary='symm')
                    psi_k_2 = fftshift(fft2(pb1 * ob_e_2))

                    # Calculate intensities
                    conv_DP = cp.abs(psi_k_2)**2
                    pinhole_DP_extra_conv = cp.abs(pinhole_DP)**2
                    pinhole_DP = cp.abs(psi_k_2_ideal)**2

                    # Resize and accumulate total intensities
                    # if total:
                    #     total_intensity += cp.array(resize(cp.asnumpy(pinhole_DP), (256,256), preserve_range=True, anti_aliasing=True))
                    #     total_intensity_conv += cp.array(resize(cp.asnumpy(conv_DP), (256,256), preserve_range=True, anti_aliasing=True))
                    if total_plot:
                        total_intensity += cp.asnumpy(pinhole_DP_extra_conv)
                        total_intensity_conv += cp.asnumpy(conv_DP)
                    # # Save results if enabled
                    # if save:
                    #     if not os.path.exists(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_typeSC'):
                    #         os.makedirs(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_typeSC')
                        
                    #     filename = f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_typeSC/output_hanning_conv_{count:05d}.npz'
                    #     np.savez(filename,
                    #             pinholeDP=cp.asnumpy(pinhole_DP),
                    #             pinholeDP_extra_conv=cp.asnumpy(pinhole_DP_extra_conv),
                    #             convDP=cp.asnumpy(conv_DP),
                    #             obj=cp.asnumpy(ob_e_2),
                    #             probe=cp.asnumpy(pb1))
                    if plot:
                        fig,ax=plt.subplots(2,2)
                        ax[0][0].imshow(np.abs(cp.asnumpy(pinhole_DP)),norm=colors.LogNorm(),cmap='jet')
                        ax[0][1].imshow(np.abs(cp.asnumpy(pinhole_DP_extra_conv)),norm=colors.LogNorm(),cmap='jet')
                        ax[1][0].imshow(np.abs(cp.asnumpy(conv_DP)),norm=colors.LogNorm(),cmap='jet')
                        #ax[1][1].imshow(np.abs(cp.asnumpy(ob_e_2)),cmap='gray')
                        ax[1][1].imshow(np.abs(cp.asnumpy(pb1)),cmap='Reds',alpha=1.0)
                        plt.show()
                        
                    if save:
                        filename = f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/{dr}/output_hanning_conv_{count:05d}.npz'
                        np.savez(filename,
                                pinholeDP=cp.asnumpy(pinhole_DP),
                                pinholeDP_extra_conv=cp.asnumpy(pinhole_DP_extra_conv),
                                convDP=cp.asnumpy(conv_DP),
                                obj=cp.asnumpy(ob_w_2),
                                probe=cp.asnumpy(pb1))
                        #print(f'saved: {filename}')
                    count += 1

            # Plot results if enabled
            if total_plot:
                plt.figure(figsize=(12,5))
                plt.subplot(121)
                plt.imshow(np.abs(cp.asnumpy(total_intensity)), cmap='jet')
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(np.abs(cp.asnumpy(total_intensity_conv)), cmap='jet')
                plt.colorbar()
                plt.show()

if __name__ == "__main__":
    main() 
# %%
tindex=209
test=np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/ZC4/output_hanning_conv_{tindex:05d}.npz')['pinholeDP_extra_conv']
plt.imshow(test,norm=colors.LogNorm(),cmap='jet')
plt.show()
# %%
