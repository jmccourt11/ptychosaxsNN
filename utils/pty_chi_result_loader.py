#%%
import tifffile as tif
import numpy as np
import os
from pathlib import Path    
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import importlib
import h5py
sys.path.append('/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/')
import utils.ptychosaxsNN_utils as ptNN_U
importlib.reload(ptNN_U)

def load_ptychi_result_tiff(file_path):
    """
    Load the chi result from the file path
    """
    chi_result = tif.imread(file_path)
    return chi_result


# GUI Box 1
# Information about the scan
results_dir = '/net/micdata/data2/12IDC/2025_Feb/ZCB_9_3D_/'
base_path = '/net/micdata/data2/12IDC/2025_Feb/ptychi_recons'
base_path_ptycho='/net/micdata/data2/12IDC/2025_Feb/ptycho/'
scan_number = 5102
frame_idx = 666  # Choose which frame to display
#reconstruction_name = 'Ndp128_LSQML_c1000_m0.5_p15_cp_mm_opr3_ic_pc_ul2'
reconstruction_name = 'Ndp128_LSQML_c1000_m0.5_p10_cp_mm_opr3_ic_pc_ul2'
niter_name='Niter1000'
center_x = 517
center_y = 575
dpsize=128


# GUI Box 2
# Load sum image
file_path = f'{base_path}/S{scan_number}/{reconstruction_name}/dp_sum.tiff'
chi_result = load_ptychi_result_tiff(file_path).T
fig, axs = plt.subplots(1, chi_result.shape[0], figsize=(10, 10))
for i in range(chi_result.shape[0]):
    axs[i].imshow(chi_result[i],norm=colors.LogNorm())
    axs[i].axis('off')
axs[0].set_title('Sum of diffraction patterns used for reconstruction')
plt.tight_layout()
plt.show()


# GUI Box 4
# Load object phase
object_path = f'{base_path}/S{scan_number}/{reconstruction_name}/object_ph/object_ph_{niter_name}.tiff'
object_phase = load_ptychi_result_tiff(object_path)
plt.title('Object phase')
plt.imshow(object_phase,cmap='gray')
plt.colorbar()
plt.tight_layout()
plt.show()


# GUI Box 5
# Load probe magnitude
probe_path = f'{base_path}/S{scan_number}/{reconstruction_name}/probe_mag/probe_mag_{niter_name}.tiff'
probe_mag = load_ptychi_result_tiff(probe_path)
plt.title('Probe magnitude')
plt.imshow(probe_mag)
plt.tight_layout()
plt.show()


# GUI Box 6
# Load sum diffraction pattern from raw data
dps = ptNN_U.load_h5_scan_to_npy(base_path_ptycho, scan_number, plot=False,point_data=True)
dps=dps[:,center_x-dpsize//2:center_x+dpsize//2,center_y-dpsize//2:center_y+dpsize//2]

plt.title(f'Sum of diffraction patterns from raw data ({dpsize}x{dpsize})')
plt.imshow(np.sum(dps,axis=0),norm=colors.LogNorm(),cmap='jet')
plt.colorbar()
plt.tight_layout()
plt.show()



# GUI Box 7
# Grid image of all diffraction patterns in scan
grid_size_row = 29
grid_size_col = 36

# Create subplot with 2 images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot individual frame
ax1.imshow(dps[frame_idx], norm=colors.LogNorm(), cmap='jet')
ax1.set_title(f'Frame {frame_idx}')
fig.colorbar(ax1.images[0], ax=ax1)

# Create and plot grid image
image_size = dps[0].shape
grid_image = np.zeros((grid_size_row * image_size[0], grid_size_col * image_size[1]))

for j in range(grid_size_row):
    for i in range(grid_size_col):
        image_idx = j * grid_size_col + i
        if image_idx < len(dps):
            grid_image[
                j * image_size[0]:(j + 1) * image_size[0],
                i * image_size[1]:(i + 1) * image_size[1]
            ] = dps[image_idx]

# Plot grid image
ax2.imshow(grid_image, norm=colors.LogNorm(), cmap='jet')
ax2.set_title('Grid image of all diffraction patterns in scan')
fig.colorbar(ax2.images[0], ax=ax2)

# Calculate position of selected frame in grid
row_idx = frame_idx // grid_size_col
col_idx = frame_idx % grid_size_col

# Draw rectangle around selected frame
rect = plt.Rectangle(
    (col_idx * image_size[1], row_idx * image_size[0]),
    image_size[1], image_size[0],
    fill=False, color='red', linewidth=2
)
ax2.add_patch(rect)

plt.tight_layout()
plt.show()


file_path = f'{base_path}/S{scan_number}/{reconstruction_name}/recon_{niter_name}.h5'

with h5py.File(file_path, 'r') as f:
    print(list(f.keys()))
    pix_size = f['obj_pixel_size_m'][()]
print(f'Pixel size: {pix_size*10**9} nm')

# %%
