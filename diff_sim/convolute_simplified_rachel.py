# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.io import loadmat
from skimage.transform import resize
from cupyx.scipy.signal import convolve2d as conv2
import cupy as cp

probe=loadmat('/net/micdata/data2/12IDC/2024_Dec/results/RC_01_/fly308/roi0_Ndp512/MLc_L1_p10_gInf_Ndp256_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/MLs_L1_p10_g100_Ndp512_pc100_model_scale_asymmetry_rotation_shear_maxPosError200nm_noModelCon_bg0.1_vi_mm/Niter1000.mat')['probe'].T[0].T


dpsize=1408
probe=resize(probe,(dpsize,dpsize),preserve_range=True,anti_aliasing=True)
probe = np.asarray(np.fft.fftshift(np.fft.fft2(probe)))

endsize=512
probe=resize(probe,(endsize,endsize),preserve_range=True,anti_aliasing=True)

ideal_DP=np.load('/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/5/output_hanning_conv_00001.npz')['pinholeDP']
ideal_DP=resize(ideal_DP,(endsize,endsize),preserve_range=True,anti_aliasing=True)

#mask=np.load('/net/micdata/data2/12IDC/2024_Dec/mask1408.npy')
#mask=np.ones(mask.shape)-mask
#mask=resize(mask,(endsize,endsize),preserve_range=True,anti_aliasing=True)

#conv_DP=conv2(cp.array(ideal_DP),cp.array(np.abs(probe)**2*mask),'same',boundary='symm')
conv_DP=conv2(cp.array(ideal_DP),cp.array(np.abs(probe)**2),'same',boundary='symm')

fig,ax=plt.subplots(1,2);ax[0].imshow(conv_DP.get(),cmap='jet',norm=colors.LogNorm());ax[1].imshow(np.load('/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/14/output_hanning_conv_00001.npz')['convDP'],cmap='jet',norm=colors.LogNorm());plt.show()
