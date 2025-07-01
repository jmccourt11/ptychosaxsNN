#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import os
import h5py
import cupy as cp
import scipy.io as sio
# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import utils.ptychosaxsNN_utils as ptNN_U




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

scan_number=888
base_path='/net/micdata/data2/12IDC/2024_Dec/ptycho/'
data = ptNN_U.load_h5_scan_to_npy(base_path, scan_number, plot=False, point_data=True)
plt.imshow(data[0],norm=colors.LogNorm())
plt.colorbar()
plt.show()
dps=data.copy()
# Create a mask for the data
# Center and ncols and nrows will be different for different scans
# ZC4
#%%

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
ob_w = np.array(ob['object'])
pb = np.array(ob['probe'])

ob_w = ob
pb1 = pb[:,:,0,0]

ncols=37#41#36
nrows=26#31#29
center=(734,745)
dps_size = dps[0].shape
center_offset_y=dps_size[0]//2-center[0]
center_offset_x=dps_size[1]//2-center[1]
dpsize = 1280

dps_cropped = dps[:, 
    dps_size[0]//2-center_offset_y - dpsize//2:dps_size[0]//2-center_offset_y + dpsize//2,
    dps_size[1]//2-center_offset_x - dpsize//2:dps_size[1]//2-center_offset_x + dpsize//2
]

# Remove hot pixels
for i, dp in enumerate(dps_cropped):
    dp[dp >= 2**16-1] = np.min(dp)

fig,axs=plt.subplots(1,4,figsize=(20,10))
axs[0].imshow(np.sum(dps_cropped,axis=0),norm=colors.LogNorm())
mask=np.sum(dps_cropped,axis=0)<=0
im1=axs[1].imshow(mask)
im2=axs[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(pb1)))*~mask,norm=colors.LogNorm())
im3=axs[3].imshow((np.abs(np.fft.fftshift(np.fft.fft2(pb1)))*~mask)[center[0]-256:center[0]+256,center[1]-256:center[1]+256],norm=colors.LogNorm())
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.colorbar(im3,ax=axs[3])
plt.show()

save_mask=False
if save_mask:
np.save('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_sum_RC02_R3D_1280.npy',mask)





















#%%
# BINNING 1280 to 256 MASK

bin_factor = 256
dps_binned = bin_ndarray(dps_cropped[0], (bin_factor, bin_factor), operation='sum')

# Create initial mask (True everywhere)
mask = np.ones_like(dps_binned, dtype=bool)

# Remove negative values
#mask[dps_binned < 0] = False

# Find horizontal and vertical stripes of zeros
# For horizontal stripes
for i in range(bin_factor):
    if np.all(dps_binned[i, :] == 0):
        mask[i, :] = False

# For vertical stripes
for j in range(bin_factor):
    if np.all(dps_binned[:, j] == 0):
        mask[:, j] = False

# Visualize the original data, mask, and masked data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Original data
im1 = ax1.imshow(dps_binned, norm=colors.LogNorm())
ax1.set_title('Original Data')
plt.colorbar(im1, ax=ax1)

# Mask
im2 = ax2.imshow(mask, cmap='gray')
ax2.set_title('Mask')
plt.colorbar(im2, ax=ax2)

# Masked data - convert to float first
masked_data = dps_binned.astype(float)
masked_data[~mask] = np.nan
im3 = ax3.imshow(masked_data, norm=colors.LogNorm())
ax3.set_title('Masked Data')
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.show()

save_mask=True
if save_mask:
    mask_filename = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_RC02_R3D_1280.npy'
    np.save(mask_filename, mask)
# %%
