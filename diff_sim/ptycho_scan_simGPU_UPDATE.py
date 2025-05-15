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

def load_and_prepare_data(base_directory,sample_dir,recon_path,scan_number,Niter):
    # sample_dir = 'RC02_R3D_'
    # #base_directory = '/scratch/2025_Feb/'
    # base_directory = '/net/micdata/data2/12IDC/2024_Dec/results/'
    # recon_path = 'MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g50_Ndp1280_mom0.5_bg0.1_vp4_vi_mm/'
    # scan_number = 888

    # # Load data and move to GPU
    # with h5py.File(f"{base_directory}S{scan_number:04d}/{recon_path}/recon_Niter1000.h5", 'r') as f:
    #     ob = f['object'][()]
    #     pb = f['probe'][()]
    ob = sio.loadmat(f"{base_directory}/{sample_dir}/fly{scan_number:03d}/{recon_path}/Niter{Niter}.mat")
    ob_w = cp.array(ob['object'])
    pb = cp.array(ob['probe'])
    
    ob_w = ob
    pb1 = pb[:,:,0,0]
    
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


#%%
def main():
    # GPU device selection
    gpu_id = 3  # Change this to select different GPU (0, 1, 2, etc.)
    with cp.cuda.Device(gpu_id):
        # Initialize parameters
        lattice_size = 400
        dpsize = 256 # Final output size
        sim_size =256#1024  # Size for simulation (can be larger than dpsize)
        grid_size = sim_size * 4  # Double the simulation size to allow for padding
        lattice_spacing = 6
        radius = lattice_spacing/2
        nsteps = 3
        nscans = 1#1200
        zoom_lattice = True
        zoom_factor =2
        plot = True
        total_plot = False
        total = False
        save = False
        save_total = False
        noise_on = False
        resize_pbp = False
        dr='TEMP'
        
        # Sample info for probe and object
        sample_dir = 'ZCB_9_3D_'
        base_directory = '/net/micdata/data2/12IDC/2025_Feb/results/'
        recon_path = 'roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/'
        Niter = 1000
        scan_number = 5065
        
        # Load and prepare data
        ob_w, pb1 = load_and_prepare_data(base_directory,sample_dir,recon_path,scan_number,Niter)

        if plot:
            plt.imshow(np.abs(cp.asnumpy(pb1)))
            plt.show()
        
        # Load pinhole and PSF
        #pbp = cp.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_bw0.75_wl1.24e-10_ps0.15_gs1280x1280.npy')
        pbp = cp.load('/home/beams/PTYCHOSAXS/NN/probe_pinhole_complex_256x256_bw0.75.npy')
        psf_pinhole = cp.abs(cp.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole.npy'))
        psf_pinhole = cp.array(resize(cp.asnumpy(psf_pinhole), (sim_size, sim_size), preserve_range=True, anti_aliasing=True))        
        pb1=cp.array(pb1)
        
        # Load lattice and move to GPU
        #amplitude_3d = cp.array(np.load(f'lattices/lattice_ls{lattice_size}_gs1024_lsp{lattice_spacing}_r{radius}_typeSC.npy'))
        amplitude_3d = cp.array(np.load(f'/home/beams/PTYCHOSAXS/NN/lattice_ls{lattice_size}_gs1024_lsp{lattice_spacing}_r{radius}_typeSC.npy'))
        
        original_lattice_size = lattice_size
        if zoom_lattice:
            amplitude_3d = zoom_gpu(amplitude_3d, (zoom_factor, zoom_factor, zoom_factor), order=1)
            lattice_size = int(original_lattice_size * zoom_factor)
        else:
            amplitude_3d = amplitude_3d
        # Initialize count
        count = 1

        # Define scan pattern - center in the middle of the grid
        center_x = grid_size // 2
        center_y = grid_size // 2
        center_concentration = 0.5
        scan_range = int(lattice_size * center_concentration)
        start_x = center_x - scan_range // 2
        start_y = center_y - scan_range // 2
        step_size_x = scan_range // (nsteps-1) if nsteps > 1 else 0
        step_size_y = scan_range // (nsteps-1) if nsteps > 1 else 0
        
        # Main simulation loop
        for l in tqdm(range(nscans)):
            # Initialize arrays
            total_intensity = cp.zeros((dpsize, dpsize))
            total_intensity_conv = cp.zeros((dpsize, dpsize))
            
            # Random rotation angles
            rotation_angles = [random.randint(0,90), random.randint(0,90), random.randint(0,90)]
            
            # Rotate lattice on GPU using the new function
            amplitude_3d_rotated = rotate_lattice_gpu(amplitude_3d, rotation_angles)
            if plot:
                plt.imshow(np.abs(cp.asnumpy(fftshift(fft2(cp.sum(amplitude_3d_rotated[:,:,:],axis=2))))),norm=colors.LogNorm(),cmap='jet')
                plt.colorbar()
                plt.show()
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
                    if plot:
                        plt.imshow(np.abs(cp.asnumpy(ob_e_2)))
                        plt.show()  
                        plt.imshow(np.abs(cp.asnumpy(ob_w_2)))
                        plt.show()
                    
                    # Calculate diffraction patterns on GPU
                    psi_k_2_ideal = fft2(ob_e_2 * pbp)
                    pinhole_DP = conv2(cp.abs(psi_k_2_ideal), cp.abs(psf_pinhole), 'same', boundary='symm')
                    #pinhole_DP = fftshift(fft2(ob_e_2))
                    psi_k_2 = fftshift(fft2(pb1 * ob_e_2))

                    # Calculate intensities
                    conv_DP = cp.abs(psi_k_2)**2
                    pinhole_DP_extra_conv = cp.abs(pinhole_DP)**2
                    pinhole_DP = cp.abs(psi_k_2_ideal)**2

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
                        ax[1][1].imshow(np.abs(cp.asnumpy(ob_e_2)),cmap='gray')
                        ax[1][1].imshow(np.abs(cp.asnumpy(pb1)),cmap='gray',alpha=0.2)
                        plt.show()
                        
                    if save:
                        filename = f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/{dr}/output_hanning_conv_{count:05d}.npz'
                        np.savez(filename,
                                pinholeDP=cp.asnumpy(pinhole_DP),
                                #pinholeDP_extra_conv=cp.asnumpy(pinhole_DP_extra_conv),
                                pinholeDP_raw_FFT=cp.asnumpy(pinhole_DP_extra_conv),
                                convDP=cp.asnumpy(conv_DP),
                                obj=cp.asnumpy(ob_w_2),
                                probe=cp.asnumpy(pb1))
                        #print(f'saved: {filename}')
                    count += 1

            # Plot results if enabled
            if total_plot:
                plt.figure(figsize=(12,5))
                plt.subplot(121)
                plt.imshow(cp.asnumpy(total_intensity), norm=colors.LogNorm(), cmap='jet')
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(cp.asnumpy(total_intensity_conv), norm=colors.LogNorm(), cmap='jet')
                plt.colorbar()
                plt.show()

if __name__ == "__main__":
    main() 
# %%
sim_index=random.randint(0,1200) # simulation index, rotation of lattice index
scan_frames=9 # number of frames per scan
scan_index=1 # scan index
tindex=scan_index+scan_frames*sim_index
test=np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/DELETE/output_hanning_conv_{tindex:05d}.npz')
fig,ax=plt.subplots(2,2,figsize=(10,10))
ax[0][0].imshow(test['pinholeDP'],norm=colors.LogNorm(),cmap='jet')
ax[0][1].imshow(test['pinholeDP_extra_conv'],norm=colors.LogNorm(),cmap='jet')
ax[1][0].imshow(test['convDP'],norm=colors.LogNorm(),cmap='jet')
ax[1][1].imshow(np.abs(test['probe']),cmap='gray',alpha=0.2)
plt.show()
# %%
plt.imsho
# %%



# # %%
