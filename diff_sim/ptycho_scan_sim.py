# coding: utf-8
#%%
import scipy.io as sio
import numpy as np
from pylab import *
import scipy.fft as spf
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from tqdm import tqdm
from cupyx.scipy.signal import convolve2d as conv2
import cupy as cp
from skimage.transform import resize
from matplotlib import colors
import random
import h5py
import pdb
import os
def hanning(image):
    #circular mask of radius=radius over image 
    xs=np.hanning(image.shape[0])
    ys=np.hanning(image.shape[1])
    temp=np.outer(xs,ys)
    return temp


def vignette(image):
    # Vignette lattice
    # Get the dimensions of the image
    rows, cols = image.shape

    # Create a meshgrid of X and Y coordinates
    X, Y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))

    # Create the square vignette mask
    distance = np.maximum(np.abs(X), np.abs(Y))
    vignette_mask = np.clip(1 - distance, 0, 1)

    # Apply the vignette mask to the image
    vignette_image = image * vignette_mask
    
    return vignette_image

#########################################################################
#ob=sio.loadmat('/mnt/micdata2/12IDC/2021_Nov/results/ML_recon/tomo_scan3/scan1538/roi0_Ndp512/MLc_L1_p10_g10_Ndp512_mom0.5_pc0_noModelCon_bg0.01_vp5_vi_mm_ff/Niter500.mat')

# # For Cindy data
# ob=sio.loadmat('/net/micdata/data2/12IDC/2021_Nov/results/ML_recon/tomo_scan3/scan1060/roi0_Ndp512/MLc_L1_p10_g10_Ndp512_mom0.5_pc0_noModelCon_bg0.01_vp5_vi_mm_ff/Niter500.mat')

# ob_roi=ob['object_roi']
# matshow(abs(ob_roi))
# plt.show()

# ob_w=ob['object']
# matshow(abs(ob_w))
# plt.show()

# pb=ob['probe']
# pb1=pb[:,:,0,0]
# matshow(abs(pb1))
# plt.show()

# center=[550,550]
# p_hw=int(pb1.shape[0]/2)
# ob_e=ob_w[center[0]-p_hw:center[0]+p_hw,center[1]-p_hw:center[1]+p_hw]
# matshow(angle(ob_e))
# plt.show()


# fig, ax=plt.subplots()
# ax.imshow(angle(ob_e),cmap='gray')
# ax.imshow(abs(pb1),cmap='Reds', alpha=0.5)


# psi_k=spf.fftshift(spf.fft2(pb1*ob_e))
# matshow(log(abs(psi_k)**2+1))
# plt.show()

# psi_k2=spf.fftshift(spf.fft2(pb1*ob_e/abs(ob_e)))
# matshow(log(abs(psi_k2)**2+1))
# plt.show()
#%%

def load_zhihua_ptychi(path,key):
    with h5py.File(path, 'r') as f:
        obj = f[key][()]
    return obj
pb=load_zhihua_ptychi("/net/micdata/data2/12IDC/2025_Feb/ptychi_recons/S5045/Ndp256_LSQML_c1000_m0.5_p15_cp_mm_opr3_ic_pc_ul2/recon_Niter1000.h5","probe")
ob=load_zhihua_ptychi("/net/micdata/data2/12IDC/2025_Feb/ptychi_recons/S5045/Ndp256_LSQML_c1000_m0.5_p15_cp_mm_opr3_ic_pc_ul2/recon_Niter1000.h5","object")

ob_w=ob[0]
plt.figure()
plt.imshow(abs(ob_w))
plt.show()

pb1=pb[0,0,:,:]
plt.figure()
plt.imshow(abs(spf.fftshift(spf.fft2(pb1))),norm=colors.LogNorm(),cmap='jet')
plt.show()

#%%
ob=sio.loadmat("/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly5102/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/Niter1000.mat")
ob_w=ob['object']
matshow(np.angle(ob_w)[588-256:588+256,528-256:528+256],cmap='gray')
plt.axis('off')
plt.savefig('ob_w.pdf',dpi=300)
plt.show()

#%%
pb=ob['probe']
pb1=pb[:,:,0,0]
im1=plt.imshow(abs(spf.fftshift(spf.fft2(pb1))),norm=colors.LogNorm(),cmap='jet')
#plt.colorbar(im1)
plt.axis('off')
#plt.savefig('probe.pdf',dpi=300)
plt.show()

#%%


# ob_w=ob[0]
# # plt.figure()
# # plt.imshow(abs(ob_w))
# # plt.show()

# pb1=pb[0,0,:,:]
# plt.figure()
# plt.imshow(abs(spf.fftshift(spf.fft2(pb1))),norm=colors.LogNorm(),cmap='jet')
# plt.show()

center=[ob_w.shape[0]//2,ob_w.shape[1]//2]
center=[550,500]
p_hw=int(pb1.shape[0]/2)
ob_e=ob_w[center[0]-p_hw:center[0]+p_hw,center[1]-p_hw:center[1]+p_hw]
# plt.figure()
# plt.imshow(angle(ob_e))
# plt.show()


fig, ax=plt.subplots()
ax.imshow(angle(ob_e),cmap='gray')
ax.imshow(abs(pb1),cmap='Reds', alpha=0.5)
plt.show()

psi_k=spf.fftshift(spf.fft2(pb1*ob_e))
# plt.figure()
# plt.imshow(log(abs(psi_k)**2+1))
# plt.show()

psi_k2=spf.fftshift(spf.fft2(pb1*ob_e/abs(ob_e)))
# plt.figure()
# plt.imshow(log(abs(psi_k2)**2+1))
# plt.show()
#%%

###################################################################################

#save_lattice=True
## Parameters
lattice_size = 400  # Lattice size (128x128)
grid_size = 1024  # Final image size after padding
lattice_spacing =6 # Distance between the centers of nanoparticles
radius = lattice_spacing/2  # Radius of the spherical nanoparticles

## Create a 3D grid for the lattice
#x = np.linspace(-lattice_size/2, lattice_size/2, lattice_size)
#y = np.linspace(-lattice_size/2, lattice_size/2, lattice_size)
#z = np.linspace(-lattice_size/2, lattice_size/2, lattice_size)
#X, Y, Z = np.meshgrid(x, y, z)

## Initialize 3D amplitude array for the lattice
#amplitude_3d = np.zeros((lattice_size, lattice_size, lattice_size))


## Populate the 3D grid with spherical nanoparticles in a simple cubic lattice
#for i in tqdm(range(-lattice_size//2, lattice_size//2, lattice_spacing)):
#    for j in tqdm(range(-lattice_size//2, lattice_size//2, lattice_spacing)):
#        for k in tqdm(range(-lattice_size//2, lattice_size//2, lattice_spacing)):
#            distance_from_center = np.sqrt((X - i)**2 + (Y - j)**2 + (Z - k)**2)
#            mask = distance_from_center <= radius
#            amplitude_3d[mask] = np.exp(-((distance_from_center[mask]) / radius)**2)  # Gaussian amplitude
# 
#if save_lattice:           
#    np.save(f'lattice_SC_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}.npy',amplitude_3d)

#total_intensity=np.zeros((256,256))
#total_intensity_conv=np.zeros((256,256))
count=1
plot_all=True
plot=True  
total_plot=True
total=True
nsteps=3
nscans=1
num_simdps=nsteps**2*nscans
random_placed=False
save=False
save_total=False
noise_on=False
dr=0#32
dpsize=256
resize_pbp=False#True

#load pinhole
#pbp=np.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_complex_256x256.npy')
pbp=np.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_complex_256x256_bw0.75.npy')
#pbp=np.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_complex.npy')

if resize_pbp:
    # THIS NEEDS TO BE THROUGHLY CHECKED
    pbp=resize(np.abs(np.real(pbp)+np.imag(pbp)),(256,256),preserve_range=True,anti_aliasing=True)
    # pbp_real = resize(np.real(pbp), (dpsize,dpsize), preserve_range=True, anti_aliasing=True)
    # pbp_imag = resize(np.imag(pbp), (dpsize,dpsize), preserve_range=True, anti_aliasing=True)
    # pbp = pbp_real + 1j * pbp_imag
#matshow(abs(fft2(pbp))**2)
#plt.show()


psf_pinhole=cp.abs(cp.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole.npy'))

#load lattice
amplitude_3d=np.load(f'lattices/lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_typeSC.npy')

#%%
def calculate_rotation_angles(h, k, l):
    """
    Calculate rotation angles to align a specific (hkl) plane with the beam direction [0,0,1]
    for a cubic lattice.
    
    Parameters:
    -----------
    h, k, l : int
        Miller indices of the desired reflection
    
    Returns:
    --------
    tuple
        Rotation angles (alpha, beta, gamma) in degrees
    """
    # Normalize the vector
    norm = np.sqrt(h**2 + k**2 + l**2)
    h, k, l = h/norm, k/norm, l/norm
    
    # Calculate rotation angles
    # First rotation: around z-axis to align projection with x-z plane
    alpha = np.arctan2(k, h) * 180/np.pi
    
    # Second rotation: around y-axis to align with z-axis
    beta = np.arccos(l) * 180/np.pi
    
    # Third rotation: around z-axis to set final orientation
    gamma = 0  # Can be adjusted if specific in-plane orientation is needed
    
    return (alpha, beta, gamma)

# Define scan pattern - probe moves across the lattice
# Start position in the padded grid - centered with offset
center_x = 512  # Center of the grid
center_y = 512
# Control parameter for scan concentration (smaller = more concentrated to center)
# Range: 0.1 (very close to center) to 1.0 (full lattice scan)
center_concentration = 0.5  # Adjust this value to control offset
scan_range = int(lattice_size * center_concentration)  # Adjustable scan range

# Calculate starting position with offset to center the scan pattern
start_x = center_x - scan_range // 2
start_y = center_y - scan_range // 2

# Step size for scanning (smaller steps to stay closer to center)
step_size_x = scan_range // (nsteps-1) if nsteps > 1 else 0
step_size_y = scan_range // (nsteps-1) if nsteps > 1 else 0

# Define the Miller indices for the desired reflection
hr, kr, lr = 1, 0, 0  # Example: (111) reflection
rotation_angles = calculate_rotation_angles(hr, kr, lr)

for l in tqdm(range(0,nscans)):
    total_intensity=np.zeros((dpsize,dpsize))
    total_intensity_conv=np.zeros((dpsize,dpsize))
    
    # Use the calculated rotation angles instead of random ones
    amplitude_3d_rotated = rotate(amplitude_3d, angle=rotation_angles[0], axes=(1, 2), reshape=False)
    amplitude_3d_rotated = rotate(amplitude_3d_rotated, angle=rotation_angles[1], axes=(0, 2), reshape=False)
    amplitude_3d_rotated = rotate(amplitude_3d_rotated, angle=rotation_angles[2], axes=(0, 1), reshape=False)
    cstart=count
    for k in range(0,nsteps):
        for i in range(0,nsteps):
            
            # Project the 3D rotated grid to 2D by summing along one axis (e.g., Z-axis)
            amplitude_2d = np.sum(amplitude_3d_rotated, axis=2)

            # Hanning of the projection to smoothen edge features
            amplitude_2d = hanning(amplitude_2d)*amplitude_2d

            # Normalize amplitude
            amplitude_2d /= np.max(amplitude_2d)
            

    #        # Define phase based on 2D coordinates
    #        x_2d = np.linspace(-lattice_size/2, lattice_size/2, lattice_size)
    #        y_2d = np.linspace(-lattice_size/2, lattice_size/2, lattice_size)
    #        X_2d, Y_2d = np.meshgrid(x_2d, y_2d)
    #        phase_2d = np.sin(2 * np.pi * (X_2d + Y_2d) / lattice_spacing)


            # Create complex array with amplitude and phase
        #    sigma=1#0.2
        #    particles_2d = sigma*amplitude_2d * np.exp(1j * phase_2d)
            
            # Just phase object
            particles_2d =  np.exp(1j * amplitude_2d)
            
            # Hanning so that amplitude also has Hanning
            particles_2d = hanning(particles_2d)*particles_2d

            # Vignette lattice
            # Get the dimensions of the image
            image=particles_2d
            rows, cols = image.shape

            # Create a meshgrid of X and Y coordinates
            X, Y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))

            # Create the square vignette mask
            distance = np.maximum(np.abs(X), np.abs(Y))
            vignette_mask = np.clip(1 - distance, 0, 1)

            # Apply the vignette mask to the image
            vignette_image = image * vignette_mask

            # Set 2D lattice to vignette image
            particles_2d=vignette_image
            
            # Hanning again
            particles_2d=hanning(particles_2d)*particles_2d


            # Pad the 128x128 array to 1024x1024
            padding = (grid_size - lattice_size) // 2


            #constant_values for padding needs to be =1 to acquire zeroth order probe peak features
            pad_value=1
            bkg=pad_value
            #amplitude_padded = np.pad(amplitude_2d, pad_width=padding, mode='constant', constant_values=pad_value)
            #phase_padded = np.pad(phase_2d, pad_width=padding, mode='constant', constant_values=pad_value)
            particles_padded = np.pad(particles_2d+bkg, pad_width=padding, mode='constant', constant_values=pad_value)
            
            

            # Generate a 2D array of random numbers between -1 and 1 for noise
            # Noise makes simulated object background similar to reconstructed object background
            if noise_on:
                noise = np.random.uniform(-1, 1, (grid_size, grid_size))/100
                particles_padded+=noise
            
            

            if random_placed:
                center=[random.randint(256+128,768-128),random.randint(256+128,768-128)]
                ob_w_2=particles_padded
                ob_e_2=ob_w_2[center[0]-p_hw:center[0]+p_hw,center[1]-p_hw:center[1]+p_hw]
            else:
                # Calculate probe position for this scan point
                probe_center_x = start_x + i * step_size_x
                probe_center_y = start_y + k * step_size_y
                
                # Extract the region where the probe illuminates the object
                ob_w_2 = particles_padded
                ob_e_2 = ob_w_2[probe_center_x-p_hw:probe_center_x+p_hw, 
                                probe_center_y-p_hw:probe_center_y+p_hw] * \
                        hanning(ob_w_2[probe_center_x-p_hw:probe_center_x+p_hw,
                                    probe_center_y-p_hw:probe_center_y+p_hw])
        
            if plot_all:
                matshow(angle(ob_e_2),cmap='gray')
                plt.show()
                    
                if k==1 and i==1:
                    matshow(abs(pb1),cmap='Reds')
                    plt.axis('off')
                    plt.savefig(f'pb1_{hr}_{kr}_{lr}.pdf',dpi=300)
                    plt.show()
                #%%
            if plot_all:
                fig, ax=plt.subplots()
                ax.imshow(angle(ob_e_2),cmap='gray')
                ax.imshow(abs(pb1),cmap='Reds', alpha=0.7)
                
                fig, ax=plt.subplots()
                ax.imshow(angle(ob_e_2),cmap='gray')
                ax.imshow(abs(pbp),cmap='Reds',alpha=0.7)

                plt.show()


            # THIS NEEDS TO BE THROUGHLY CHECKED
            psi_k_2_ideal=spf.fft2(ob_e_2*pbp)
            #psi_k_2_ideal=spf.fftshift(spf.fft2(ob_e_2*pbp))
            
            
            if plot_all:
                matshow(log(abs(psi_k_2_ideal)**2))
                plt.show()
            #psi_k_2_ideal=spf.fftshift(spf.fft2(ob_e_2))
            #matshow(log(abs(psi_k_2_ideal)**2+1))
            
            #psi_k_2_ideal=spf.fftshift(spf.fft2(ob_w_2))
            #matshow(log(abs(psi_k_2_ideal)**2+1))
            
            psi_k_2_ideal=cp.array(psi_k_2_ideal)
            psf_pinhole=cp.array(psf_pinhole)
            #psf_pinhole=cp.array(fft2(pbp))
            pinhole_DP=conv2(np.abs(psi_k_2_ideal),np.abs(psf_pinhole),'same', boundary='symm')
            pinhole_DP=pinhole_DP.get()
            psi_k_2_ideal=psi_k_2_ideal.get()
            psf_pinhole=psf_pinhole.get()
        #    matshow(log(abs(pinhole_DP)**2+1))
            

            psi_k_2=spf.fftshift(spf.fft2(pb1*ob_e_2))
            # matshow(log(abs(psi_k_2)**2))
            # plt.show()
            
            if plot:
                fig,ax=plt.subplots(2,3,figsize=(15,5))
                im1=ax[0][0].imshow(abs(psi_k_2)**2,norm=colors.LogNorm(),cmap='jet')
                im2=ax[0][1].imshow(abs(pinhole_DP)**2,norm=colors.LogNorm(),cmap='jet')
                im3=ax[0][2].imshow(abs(psi_k_2_ideal)**2,norm=colors.LogNorm(),cmap='jet')
                im4=ax[1][0].imshow(resize(abs(psi_k_2)**2,(256,256),preserve_range=True,anti_aliasing=True),norm=colors.LogNorm(),cmap='jet')
                im5=ax[1][1].imshow(resize(abs(pinhole_DP)**2,(256,256),preserve_range=True,anti_aliasing=True),norm=colors.LogNorm(),cmap='jet')
                im6=ax[1][2].imshow(resize(abs(psi_k_2_ideal)**2,(256,256),preserve_range=True,anti_aliasing=True),norm=colors.LogNorm(),cmap='jet')
                plt.colorbar(im1)
                plt.colorbar(im2)
                plt.colorbar(im3)
                plt.colorbar(im4)
                plt.colorbar(im5)
                plt.colorbar(im6)
                plt.show()
            if total:
                total_intensity+=resize(abs(psi_k_2_ideal)**2,(256,256),preserve_range=True,anti_aliasing=True)
                total_intensity_conv+=resize(abs(psi_k_2)**2,(256,256),preserve_range=True,anti_aliasing=True)
            # conv_DP=resize(abs(psi_k_2)**2,(256,256),preserve_range=True,anti_aliasing=True)
            # pinhole_DP=resize(abs(psi_k_2_ideal)**2,(256,256),preserve_range=True,anti_aliasing=True)
            # pinhole_DP_extra_conv=resize(abs(pinhole_DP)**2,(256,256),preserve_range=True,anti_aliasing=True)
            
            conv_DP=abs(psi_k_2)**2
            pinhole_DP_extra_conv=abs(pinhole_DP)**2
            pinhole_DP=abs(psi_k_2_ideal)**2

            
            # fig,ax=plt.subplots(1,3)
            # ax[0].imshow(conv_DP,norm=colors.LogNorm(),cmap='jet')
            # ax[1].imshow(pinhole_DP,norm=colors.LogNorm(),cmap='jet')
            # ax[2].imshow(pinhole_DP_extra_conv,norm=colors.LogNorm(),cmap='jet')
            # plt.show()
            
            #filename='/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/{}/output_hanning_conv_{:05d}.npz'.format(dr,count)
            if not os.path.exists(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_typeSC'):
                os.makedirs(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_typeSC')    
            
            filename=f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_typeSC/output_hanning_conv_{hr}_{kr}_{lr}_{count:05d}.npz'
            print(filename)
            
            if save:
                np.savez(filename,pinholeDP=pinhole_DP,pinholeDP_extra_conv=pinhole_DP_extra_conv,convDP=conv_DP,obj=ob_e_2,probe=pb1)
                print(f"saved: {filename}")

            count+=1
    cend=count-1
    if save_total:
        filename_total='/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/{}/output_hanning_conv_TOTAL_{:05d}_{:05d}.npz'.format(dr,cstart,cend)
        
        np.savez(filename_total,pinholeDP=total_intensity,convDP=total_intensity_conv,probe=pb1)
        print(f"saved: {filename_total}")
        
    # Plot sum of scanned diffraction patterns
    if total_plot:
        fig,ax=plt.subplots(1,2)
        im1=ax[0].imshow(total_intensity,norm=colors.LogNorm(),cmap='jet')
        im2=ax[1].imshow(total_intensity_conv,norm=colors.LogNorm(),cmap='jet')
        plt.colorbar(im1)
        plt.colorbar(im2)
        plt.show()

# %%
