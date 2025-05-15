
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import os
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

scan_number=62
base_path='/scratch/2025_Feb/ptycho/'
data = ptNN_U.load_h5_scan_to_npy(base_path, scan_number, plot=False, point_data=True)
plt.imshow(data[0],norm=colors.LogNorm())
plt.colorbar()
plt.show()

# Create a mask for the data
# Center and ncols and nrows will be different for different scans
#%%
# ZC4
dps=data.copy()

#%%

ncols=41#36
nrows=31#29
center=(718,742)
dps_size = dps[0].shape
center_offset_y=dps_size[0]//2-center[0]
center_offset_x=dps_size[1]//2-center[1]
dpsize = 1024

dps_cropped = dps[:, 
    dps_size[0]//2-center_offset_y - dpsize//2:dps_size[0]//2-center_offset_y + dpsize//2,
    dps_size[1]//2-center_offset_x - dpsize//2:dps_size[1]//2-center_offset_x + dpsize//2
]

# Remove hot pixels
for i, dp in enumerate(dps_cropped):
    dp[dp >= 2**16-1] = np.min(dp)


#%%
bin_factor = 256
dps_binned = bin_ndarray(dps_cropped[0], (bin_factor, bin_factor), operation='sum')

plt.imshow(dps_binned,norm=colors.LogNorm())
plt.colorbar()
plt.show()
#%%

plt.imshow(dps_cropped[0],norm=colors.LogNorm())
plt.plot(dpsize//2, dpsize//2, 'rx', markersize=10, markeredgewidth=2, label='Center')
plt.legend()
plt.colorbar()
plt.show()

#%%
mask=dps_binned<=0
plt.imshow(mask)
plt.colorbar()
plt.show()

save_mask=False
if save_mask:
    mask_filename = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZC4.npy'
    np.save(mask_filename, mask)
# %%
