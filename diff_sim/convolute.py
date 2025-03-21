# coding: utf-8
#%%
from cupyx.scipy.signal import convolve2d as conv2
from scipy.signal import convolve2d as conv2np
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import window
from skimage.transform import resize
from scipy import signal
from numpy.fft import fftn, fftshift
import os
from matplotlib import colors
from scipy.io import loadmat
import h5py
plt.rcParams['image.cmap'] = 'jet'

def flip180(arr):
    #inverts 2D array, used to invert probe array for Richardson Lucy deconvoltuion algorithm
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def hanning(image):
    #circular mask of radius=radius over image 
    xs=np.hanning(image.shape[0])
    ys=np.hanning(image.shape[1])
    temp=np.outer(xs,ys)
    return temp


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    
    
save=True #save files
plot=False
dr='17' #save location
data_location='/mnt/micdata2/12IDC/ptychosaxs/data/diff_sim/5/'

data_location='/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/5/'
dpsize=256#512#1408
endsize=256#512
resize_probe=False

with cp.cuda.Device(1): #select last GPU (i.e. 4 gpus, index of last one is 3)
    #probe=loadmat('/net/micdata/data2/12IDC/2024_Dec/results/RC_01_/fly315/roi0_Ndp1024/MLc_L1_p10_g200_Ndp512_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp1024_mom0.5_pc200_model_scale_asymmetry_rotation_shear_maxPosError200nm_noModelCon_bg0.1_vi_mm/Niter400.mat')['probe'].T[0].T
    #probe=loadmat('/net/micdata/data2/12IDC/2024_Dec/results/RC_01_/fly318/roi0_Ndp256/MLc_L1_p10_g1000_Ndp256_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/Niter1000.mat')['probe'].T[0].T
    
    # #Cindy probe
    # probe=loadmat('/net/micdata/data2/12IDC/2024_Dec/results/RC_01_/fly308/roi0_Ndp512/MLc_L1_p10_gInf_Ndp256_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/MLs_L1_p10_g100_Ndp512_pc100_model_scale_asymmetry_rotation_shear_maxPosError200nm_noModelCon_bg0.1_vi_mm/Niter1000.mat')['probe'].T[0].T

    # #Zhihua probe
    # probe=loadmat("/net/micdata/data2/12IDC/2024_Dec/results/JM02_3D_/fly482/roi2_Ndp1024/MLc_L1_p10_gInf_Ndp256_mom0.5_pc100_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g400_Ndp512_mom0.5_pc400_noModelCon_bg0.1_vp4_vi_mm/Niter1000.mat")['probe'].T[0][0].T
    # print(probe.shape)
    
    #Zhihua probe ZCB_9_3D_
    def load_zhihua_probe_ptychi(probe_path):
        with h5py.File(probe_path, 'r') as f:
            probe = f['probe'][()][0,0]
        return probe
    
    probe_file='/net/micdata/data2/12IDC/2025_Feb/ptychi_recons/S5045/Ndp256_LSQML_c1000_m0.5_p15_cp_mm_opr3_ic_pc_ul2/recon_Niter1000.h5'
    probe=load_zhihua_probe_ptychi(probe_file)
    
    if save:
        np.save('/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/{}/probe_{}.npy'.format(dr,probe_file.split('/')[-5]+probe_file.split('/')[-4]+'_'+probe_file.split('/')[-3]+'_'+probe_file.split('/')[-2]+'_'+probe_file.split('/')[-1]),probe)
    fig,ax=plt.subplots(1,2)
    ax[0].imshow(np.abs(probe))
    ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(probe))),norm=colors.LogNorm())
    plt.show()
    
    if resize_probe:
        probe=resize(probe,(dpsize,dpsize),preserve_range=True,anti_aliasing=True)
    
    #mask=loadmat('/mnt/micdata2/12IDC/2024_Dec/results/JM02_3D_/mask_1024.mat')['mask']
    #mask=np.load('/mnt/micdata2/12IDC/2024_Dec/mask1408.npy')
    #mask=np.ones(mask.shape)-mask
    #mask=resize(mask,(endsize,endsize),preserve_range=True,anti_aliasing=True)
    
    probe = np.asarray(np.fft.fftshift(np.fft.fft2(probe)))
    if resize_probe:
        probe=resize(probe,(endsize,endsize),preserve_range=True,anti_aliasing=True)#*mask
    
    #offset=1e-8
    probe=cp.asarray(probe)#+offset)
    
    psf_pinhole=np.load('/home/beams/PTYCHOSAXS/NN/probe_pinhole.npy')
    psf_pinhole=resize(psf_pinhole,(endsize,endsize),preserve_range=True,anti_aliasing=True)

    
    
    for i,x in enumerate(tqdm(os.listdir(data_location)[:10000])):
        # Load simulated diffraction patterns
        dp=np.load(data_location+x)
        ideal_DP=dp['pinholeDP']
        #ideal_DP=cp.load('/mnt/micdata2/12IDC/ptychosaxs/data/diff_sim/2/output_ideal_{:05d}.npz'.format(i))['idealDP']
        #ideal_DP=cp.load('/mnt/micdata2/12IDC/ptychosaxs/data/diff_sim/2/output_hanning_conv_{:05d}.npz'.format(i+1))['pinholeDP']s
        ideal_DP=resize(ideal_DP,(endsize,endsize),preserve_range=True,anti_aliasing=True)
        
        #psf_pinhole=fftn(psf_pinhole)
        psf_pinhole=cp.asarray(psf_pinhole)
        ideal_DP=cp.asarray(ideal_DP)
        
        # Convolute ideal diffraction patterns with simulated
        pinhole_DP=conv2(ideal_DP,psf_pinhole,'same', boundary='symm')

        # Convolute ideal with reconstructed probe
        #probe = loadmat('/mnt/micdata2/12IDC/2024_Dec/results/RC_01_/fly312/roi0_Ndp512/MLc_L1_p10_gInf_Ndp256_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/MLs_L1_p10_g100_Ndp512_pc100_model_scale_asymmetry_rotation_shear_maxPosError200nm_noModelCon_bg0.1_vi_mm/Niter1000.mat')['probe'].T[0].T
        #probe = loadmat('/mnt/micdata2/12IDC/2024_Dec/results/RC_01_/fly315/roi0_Ndp1024/MLc_L1_p10_g200_Ndp512_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/Niter2000.mat')['probe'].T[0].T
        #probe=loadmat('/net/micdata/data2/12IDC/2024_Dec/results/RC_01_/fly315/roi0_Ndp1024/MLc_L1_p10_g200_Ndp512_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp1024_mom0.5_pc200_model_scale_asymmetry_rotation_shear_maxPosError200nm_noModelCon_bg0.1_vi_mm/Niter400.mat')['probe'].T[0].T

        
        #probe=cp.asarray(resize(np.fft.fftshift(np.fft.fft2(probe)),(1024,1024),preserve_range=True,anti_aliasing=True))

        conv_DP=conv2(ideal_DP,np.abs(probe)**2,'same',boundary='symm')
        
        # Create a distance map to focus away from the center (
        height, width = ideal_DP.shape
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.meshgrid(np.arange(height, dtype=np.float32), 
                                            np.arange(width, dtype=np.float32))

        # Calculate the distance from the center for each pixel (float type)
        qpixel=(2/height) #2 length^(-1) per pixel
        distance_map = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)*qpixel  #q distance map


        # Getting from GPU
        ideal_DP=ideal_DP.get()
        pinhole_DP=pinhole_DP.get()
        #conv_DP=dp['convDP'].get()
        conv_DP=conv_DP.get()
        #conv_DP=conv_DP*mask
#        conv_DP=resize(conv_DP,(endsize,endsize),preserve_range=False,anti_aliasing=True)
#        fig,ax=plt.subplots(1,3)
#        ax[0].imshow(conv_DP,norm=colors.LogNorm())
#        conv_DP=resize(conv_DP,(dpsize,dpsize),preserve_range=True,anti_aliasing=True)*mask
#        ax[1].imshow(conv_DP,norm=colors.LogNorm())
#        conv_DP=resize(conv_DP,(endsize,endsize),preserve_range=False,anti_aliasing=True)
#        ax[2].imshow(conv_DP,norm=colors.LogNorm());
#        plt.show();
        psf_pinhole=psf_pinhole.get()
        #probe = probe.get()
        
#        plt.figure()
#        plt.imshow(np.abs(probe)**2,norm=colors.LogNorm())
#        plt.colorbar()
#        plt.title('Probe Reconstructed FT')
#        plt.show()
        # Porod scaling of intensities (q^4)
        ideal_DP=np.abs(ideal_DP)**2#*distance_map**4
        conv_DP = np.abs(conv_DP)**2#*distance_map**4

        # Plot
        if plot:
            fig,ax=plt.subplots(1,3)
            im1=ax[0].imshow(np.abs(pinhole_DP)**2,norm=colors.LogNorm())
            im2=ax[1].imshow(ideal_DP,norm=colors.LogNorm())
            #im3=ax[2].imshow(np.abs(psf_pinhole.get())**2,norm=colors.LogNorm())
            im3=ax[2].imshow(conv_DP,norm=colors.LogNorm())
            #im4=ax[3].imshow(distance_map**4)
            plt.colorbar(im1)
            plt.colorbar(im2)
            plt.colorbar(im3)
            #plt.colorbar(im4)
            plt.show()

        if save:
            #np.savez('/mnt/micdata2/12IDC/ptychosaxs/data/diff_sim/{}/output_hanning_conv_{:05d}.npz'.format(dr,i+1),pinholeDP=ideal_DP,convDP=conv_DP)
            np.savez('/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/{}/output_hanning_conv_{:05d}.npz'.format(dr,i+1),pinholeDP=ideal_DP,convDP=conv_DP)
        
#        if plot:
#            print('plotting')
#            fig,ax=plt.subplots(2,2,layout='constrained');ax[0][0].imshow(np.abs(pinhole_DP)**2,norm=colors.LogNorm());ax[1][0].imshow(np.abs(conv_DP)**2,norm=colors.LogNorm());ax[0][1].imshow(np.abs(psf_pinhole)**2,norm=colors.LogNorm());ax[1][1].imshow(np.abs(ideal_DP)**2,norm=colors.LogNorm());plt.show()
# %%
