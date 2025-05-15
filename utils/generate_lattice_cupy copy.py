#%%
import numpy as np
import cupy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import matplotlib.colors as colors
import os
import sys
import importlib
#%%
# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)
#%%
def fast_generate_lattice_gpu(lattice_size=72, lattice_spacing=3, radius=1.5, lattice_type='sc'):
    """
    Efficiently generate a 3D lattice of spheres using CuPy (GPU-accelerated NumPy-like library).
    
    Parameters:
    lattice_size (int): The size of the lattice (along one dimension).
    lattice_spacing (int): The spacing between particle centers.
    radius (float): The radius of each spherical particle.
    lattice_type (str): The type of lattice ('sc', 'bcc', 'fcc').

    Returns:
    amplitude_3d (cupy.ndarray): 3D array representing the lattice.
    """
    # Create a 3D grid of coordinates using CuPy
    x = cp.arange(-lattice_size//2, lattice_size//2)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')

    # Initialize an empty amplitude array
    amplitude_3d = cp.zeros((lattice_size, lattice_size, lattice_size), dtype=cp.float32)

    # Define lattice point offsets based on lattice type
    if lattice_type == 'SC':
        offsets = [(0, 0, 0)]
    elif lattice_type == 'BCC':
        offsets = [(0, 0, 0), (lattice_spacing//2, lattice_spacing//2, lattice_spacing//2)]
    elif lattice_type == 'FCC':
        offsets = [(0, 0, 0),
                   (lattice_spacing//2, 0, 0),
                   (0, lattice_spacing//2, 0),
                   (0, 0, lattice_spacing//2)]
    else:
        raise ValueError(f"Unsupported lattice type: {lattice_type}")

    # Generate positions for spheres
    positions = cp.arange(-lattice_size//2, lattice_size//2, lattice_spacing)

    # Create spheres by iterating over all positions and offsets
    for offset in offsets:
        for i in tqdm(positions):
            for j in positions:
                for k in positions:
                    # Calculate the distance of the grid points from the current sphere center
                    dx = X - (i + offset[0])
                    dy = Y - (j + offset[1])
                    dz = Z - (k + offset[2])
                    distance = cp.sqrt(dx**2 + dy**2 + dz**2)

                    # Use broadcasting to set values inside the sphere to 1
                    amplitude_3d[distance <= radius] = 1.0

    return amplitude_3d


if __name__=='__main__':
    with cp.cuda.Device(2):
        # Example usage:
        lattice_size = 500  # Lattice size (NxN)
        grid_size = 512  # Final image size after padding
        lattice_spacing = 25  # Distance between the centers of nanoparticles (25nm)
        radius = lattice_spacing / 2  # Radius of the spherical nanoparticles (12.5nm)

        # Generate the lattice on the GPU
        lattice_types = ['SC']
        for lattice_type in lattice_types:
            # Generate the lattice
            lattice_gpu = fast_generate_lattice_gpu(lattice_size=lattice_size, lattice_spacing=lattice_spacing, radius=radius, lattice_type=lattice_type)

            # Compute FFT
            fft_gpu = cp.fft.fftn(lattice_gpu)
            fft_shifted = cp.fft.fftshift(fft_gpu)
            fft_abs = cp.abs(fft_shifted)
            
            # Move to CPU for plotting
            fft_cpu = cp.asnumpy(fft_abs)
            
            # Plot the FFT
            plt.figure(figsize=(10, 8))
            plt.imshow(np.log10(fft_cpu[lattice_size//2, :, :] + 1), cmap='viridis')
            plt.colorbar(label='log10(Intensity)')
            plt.title(f'FFT of {lattice_type} Lattice (Central Slice)')
            plt.xlabel('q_x')
            plt.ylabel('q_y')
            plt.show()

            # Save the generated lattice
            save_lattice = False  # Enable saving
            if save_lattice:
                lattice_cpu = cp.asnumpy(lattice_gpu)           
                np.save(f'lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_type{lattice_type}.npy', lattice_cpu)
                print(f'Lattice saved to lattice_ls{lattice_size}_gs{grid_size}_lsp{lattice_spacing}_r{radius}_type{lattice_type}.npy')

        # If needed, move the result back to the CPU (e.g., for saving or visualization)


# %%
from scipy.ndimage import zoom
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

def hanning(image):
    # GPU version of hanning window
    xs = np.hanning(image.shape[0])
    ys = np.hanning(image.shape[1])
    temp = np.outer(xs, ys)
    return temp

desired_scan_num = 62  # <-- set this to the scan number you want

# Center of dp
center=(718,742)
dpsize=1024
bin_size=4


dps=ptNN_U.load_h5_scan_to_npy( '/scratch/2025_Feb/ptycho/', 62, plot=False, point_data=True)

#%%
# Find the index in the angles/probes/projections lists
idx = np.where(scan_nums == desired_scan_num)[0][0]

# Extract the probe, projection, and angle for the desired scan number
probe = probes[idx]
projection = projections[idx]
angle = angles[idx]

fig,ax=plt.subplots(1,4,figsize=(15,6))
with h5py.File(f'/scratch/2025_Feb/ptycho/{scan_num}/{sample_name}{scan_num:03d}_00021_00021.h5', 'r') as f:
    dp=f['/entry/data/data'][()][center[0]-dpsize//2:center[0]+dpsize//2,center[1]-dpsize//2:center[1]+dpsize//2]
    dp_binned=bin_ndarray(dp, (dpsize//bin_size,dpsize//bin_size), 'sum')
    ax[0].imshow(dp_binned,norm=colors.LogNorm())
probe_resized = zoom(probe[:, :, 0, 0], (dpsize / probe.shape[0], dpsize / probe.shape[1]), order=1)
ax[1].imshow(np.abs(probe_resized))
probe_resized_fft=np.fft.fftshift(np.fft.fft2(probe_resized))
ax[2].imshow(np.abs(probe_resized_fft))


 
amplitude_3d = np.load(f'/home/beams/PTYCHOSAXS/NN/lattice_ls400_gs1024_lsp6_r3.0_typeSC.npy')
# Project and process on GPU
amplitude_2d = np.sum(amplitude_3d, axis=2)
amplitude_2d = hanning(amplitude_2d) * amplitude_2d
amplitude_2d /= np.max(amplitude_2d)

# Create phase object
particles_2d = np.exp(1j * amplitude_2d)
particles_2d = hanning(particles_2d) * particles_2d

# Pad on GPU
padding = (1024 - 400) // 2
pad_value = 1
bkg = pad_value
particles_padded = np.pad(particles_2d + bkg, pad_width=padding, mode='constant', constant_values=pad_value)

       
# Pad projection to match probe size
projection_resized = np.zeros_like(probe_resized)
orig_size = projection.shape
start_x = (probe_resized.shape[0] - orig_size[0]) // 2
start_y = (probe_resized.shape[1] - orig_size[1]) // 2
projection_resized[start_x:start_x+orig_size[0], start_y:start_y+orig_size[1]] = np.angle(projection)*hanning(projection)
projection_resized = projection_resized*particles_padded









# Multiply probe and projection then take FFT
multiplied = probe_resized * projection_resized
multiplied_fft = np.fft.fftshift(np.fft.fft2(multiplied))

# Add another subplot to show result
ax[3].imshow(np.abs(multiplied_fft),norm=colors.LogNorm())
ax[3].set_title('FFT of probe × projection')

plt.show()