#%%
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
import tifffile
# Setup GPU
device = torch.device('cuda')

#%%
probe = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly888/roi0_Ndp512/MLc_L1_p10_gInf_Ndp128_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/Niter1000.mat")['probe'][:,:,0], dtype=torch.cfloat, device=device)

#%%
# Resize probe in fourier space
probe_FFT = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(probe, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

# Create padded array of target size (256x256)
target_size = 256
pad_size = (target_size - probe_FFT.shape[0]) // 2
padded_FFT = torch.zeros((target_size, target_size), dtype=torch.cfloat, device=device)
padded_FFT[pad_size:pad_size+probe_FFT.shape[0], pad_size:pad_size+probe_FFT.shape[1]] = probe_FFT

# Transform back to real space
probe_resized = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(padded_FFT, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

#Vignette the resized probe
y = torch.linspace(-1, 1, probe_resized.shape[0])
x = torch.linspace(-1, 1, probe_resized.shape[1])
X, Y = torch.meshgrid(x, y, indexing='xy')
R = torch.sqrt(X**2 + Y**2).to(device)
probe_resized = probe_resized * (1 - R**2).clamp(0, 1)

probe_resized_FFT = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(probe_resized, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

# Load probe from file and move to GPU
# probe_256x256 = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly888/roi0_Ndp512/MLc_L1_p10_g50_Ndp256_mom0.5_pc0_noModelCon_bg0.1_vi_mm/Niter1000.mat")['probe'][:,:,0], dtype=torch.cfloat, device=device)
probe_256x256 = probe_resized

# Load and pad the lattice
lattice = np.load('/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/diff_sim/lattices/lattice_ls400_gs1024_lsp6_r3.0_typeSC.npy')

# Create 3D vignette mask
x, y, z = np.meshgrid(np.linspace(-1, 1, lattice.shape[0]),
                     np.linspace(-1, 1, lattice.shape[1]), 
                     np.linspace(-1, 1, lattice.shape[2]))
distance = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)])
vignette_mask = np.clip(1 - distance, 0, 1)

# Apply vignette to lattice
lattice = lattice * vignette_mask

# Project lattice along z-axis and convert to torch tensor
lattice_2d = torch.tensor(np.sum(lattice, axis=1), dtype=torch.float32, device=device)

lattice_size = lattice_2d.shape[0]

# Resize 2D lattice to target size
lattice_2d = torch.nn.functional.interpolate(lattice_2d.unsqueeze(0).unsqueeze(0), 
                                           size=(target_size, target_size), 
                                           mode='bilinear', align_corners=False)
lattice_2d = lattice_2d.squeeze(0).squeeze(0)

# Vignette the lattice
y = torch.linspace(-1, 1, lattice_2d.shape[0])
x = torch.linspace(-1, 1, lattice_2d.shape[1])
X, Y = torch.meshgrid(x, y, indexing='xy')
R = torch.sqrt(X**2 + Y**2).to(device)
lattice_2d = lattice_2d * (1 - R**2).clamp(0, 1)

# The lattice is now already at target size, so no padding needed
lattice_padded = lattice_2d

# Multiply probe with projected lattice in real space
probe_lattice = probe_resized * lattice_padded

# Calculate diffraction pattern (FFT of product)
diffraction = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(probe_lattice, dim=(-2, -1)), norm='ortho'), dim=(-2, -1)))**2

#%%
count = 0
num_sims = 1
while count < num_sims:
    count += 1
    
    object_img = lattice_padded
    scan_number = 888
    probe = probe_resized
    
    # Load the probe from the file
    probe_ideal = np.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_bw0.2_wl1.24e-10_ps0.15_gs256x256.npy')
    probe_ideal = torch.tensor(probe_ideal, dtype=torch.cfloat, device=device)

    # Random rotation angle between 0 and 360 degrees
    angle = random.uniform(0, 360)
    zoom_factor = 1.5

    # Convert to numpy for rotation and zoom
    object_np = object_img.cpu().numpy()

    # Apply zoom using scipy.ndimage.zoom
    zoomed = zoom(object_np, (zoom_factor, zoom_factor), order=1)

    # Apply rotation using scipy.ndimage.rotate
    rotated = rotate(zoomed, angle, reshape=False, order=1)

    # Convert back to torch tensor
    object_img = torch.tensor(rotated, dtype=torch.cfloat, device=device)
            
    # Pad object to match probe size
    if object_img.shape[0] < probe.shape[0]:
        # Calculate padding dimensions
        pad_height = int((1.5*probe.shape[0] - object_img.shape[0]) // 2)
        pad_width = int((1.5*probe.shape[1] - object_img.shape[1]) // 2)
        
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

    # Define scan parameters - maintain similar density as 1280 case
    scan_step = probe.shape[0]//random.randint(12, 18)  # Adjusted for 256 size
    probe_size = probe.shape[0]

    # Create a dictionary to store simulation parameters
    sim_params = {
        'scan_step': scan_step,
        'probe_size': probe_size,
        'angle': angle,
        'zoom_factor': zoom_factor
    }

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
    batch_size = 32  # Adjust as needed for your GPU

    # Batch processing of scan positions
    for i in tqdm(range(0, len(scan_positions), batch_size), desc="Batch scanning"):
        batch = scan_positions[i:i+batch_size]
        batch_size_actual = len(batch)
        
        # Prepare tensor for object patches
        object_patches = torch.zeros((batch_size_actual, probe_size, probe_size), dtype=torch.cfloat, device=device)
        for j, pos in enumerate(batch):
            y, x = pos
            object_patches[j] = object_img[y:y+probe_size, x:x+probe_size]
            
        # Create vignette for probe size
        y = torch.linspace(-1, 1, probe_size)
        x = torch.linspace(-1, 1, probe_size)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        R = torch.sqrt(X**2 + Y**2).to(device)
        probe_vignette = (1 - R**2).clamp(0, 1)
        
        # Apply vignette to both exit waves
        exit_waves = object_patches * probe * probe_vignette
        exit_waves_ideal = object_patches * probe_vignette
        
        # FFT and intensity
        dp = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        intensity = torch.abs(dp) ** 2
        dp_ideal = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves_ideal, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        intensity_ideal = torch.abs(dp_ideal) ** 2
        
        # Store results
        patterns.extend(exit_waves.detach().cpu())
        dps.extend(intensity.detach().cpu().numpy())
        dps_ideal.extend(intensity_ideal.detach().cpu().numpy())

    # Save the diffraction patterns
    dp_count = (count-1) * len(scan_positions)
    for scan_idx in tqdm(range(len(dps)), desc="Saving patterns"):
        dp_count += 1
        
        # Get scan position for this pattern
        pos = scan_positions[scan_idx].cpu().numpy()
        y, x = pos
        
        # Save pattern
        pattern_filename = f'/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/output_256_conv_{dp_count:05d}.npz'
        
        np.savez(pattern_filename,
                pinholeDP=dps_ideal[scan_idx],
                convDP=dps[scan_idx],
                scan_position=[y, x],
                scan_index=scan_idx)
        
        if dp_count % 100 == 0:  # Print progress every 100 files
            print(f'saved: {pattern_filename} (scan {scan_idx}: {x}, {y})')

# Visualization code for 256x256 case
# Plot probe and FFT
fig,ax=plt.subplots(1,3,figsize=(10,5))
ax[0].imshow(np.abs(probe.cpu().numpy()))
ax[1].imshow(np.angle(probe.cpu().numpy()))
ax[2].imshow(np.abs(probe_FFT.cpu().numpy()),norm=colors.LogNorm())
plt.show()

fig,ax=plt.subplots(1,3,figsize=(10,5))
ax[0].imshow(np.abs(probe_resized.cpu().numpy()))
ax[1].imshow(np.angle(probe_resized.cpu().numpy()))
ax[2].imshow(np.abs(probe_resized_FFT.cpu().numpy()),norm=colors.LogNorm())
plt.show()

# Plot results of lattice and diffraction
fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(np.abs(probe_lattice.cpu().numpy()))
ax[0].set_title('Probe * Lattice (Real Space)')
ax[1].imshow(diffraction.cpu().numpy(), norm=colors.LogNorm())
ax[1].set_title('Diffraction Pattern')
ax[2].imshow(lattice_padded.cpu().numpy())
ax[2].set_title('Projected Lattice')
plt.show()

# Plot example diffraction patterns
plot_example=True
if plot_example:
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    im1=ax[0][0].imshow(np.sum(dps, axis=0), norm=colors.LogNorm(), cmap='jet')
    im2=ax[0][1].imshow(np.sum(dps_ideal, axis=0), norm=colors.LogNorm(), cmap='jet')
    ri = random.randint(0, len(dps)-1)
    im3=ax[1][0].imshow(dps[ri], norm=colors.LogNorm(), cmap='jet')
    im4=ax[1][1].imshow(dps_ideal[ri], norm=colors.LogNorm(), cmap='jet')
    plt.colorbar(im1, ax=ax[0][0])
    plt.colorbar(im2, ax=ax[0][1])
    plt.colorbar(im3, ax=ax[1][0])
    plt.colorbar(im4, ax=ax[1][1])
    ax[0][0].set_title('Sum of Convolved DPs')
    ax[0][1].set_title('Sum of Ideal DPs')
    ax[1][0].set_title(f'Single Convolved DP (idx: {ri})')
    ax[1][1].set_title(f'Single Ideal DP (idx: {ri})')
    plt.tight_layout()
    plt.show()

# Visualization of scan positions
print(f"Total number of scan positions: {len(scan_positions)}")
print(f"Scan step size: {scan_step} pixels")
print(f"Probe size: {probe_size} pixels")

# Plot overview of all scan positions on the object
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# Left plot: Object with scan position grid overlay
ax[0].imshow(np.abs(object_img.cpu().numpy()), cmap='gray')
ax[0].set_title('Object with All Scan Positions')

# Draw rectangles for each scan position
for i, pos in enumerate(scan_positions.cpu().numpy()):
    y, x = pos
    rect = patches.Rectangle((x, y), probe_size, probe_size, 
                           linewidth=1, edgecolor='red', facecolor='none', alpha=0.3)
    ax[0].add_patch(rect)
    
    # Add position number for first few positions
    if i < 10:
        ax[0].text(x + probe_size/2, y + probe_size/2, str(i), 
                  color='yellow', fontsize=8, ha='center', va='center')

ax[0].set_xlabel('X position (pixels)')
ax[0].set_ylabel('Y position (pixels)')

# Right plot: Probe shape for reference
ax[1].imshow(np.abs(probe.cpu().numpy()), cmap='viridis')
ax[1].set_title('Probe Shape')
ax[1].set_xlabel('X (pixels)')
ax[1].set_ylabel('Y (pixels)')

plt.tight_layout()
plt.show()

# Show detailed view of randomly selected scan positions
n_positions_to_show = min(3, len(scan_positions))
# Randomly select positions to show
random_indices = random.sample(range(len(scan_positions)), n_positions_to_show)

fig, axes = plt.subplots(1, n_positions_to_show, figsize=(15, 10))
if n_positions_to_show == 1:
    axes = [axes]

for plot_idx, scan_idx in enumerate(random_indices):
    pos = scan_positions[scan_idx].cpu().numpy()
    y, x = pos
    
    # Extract the object patch at this position
    object_patch = object_img[y:y+probe_size, x:x+probe_size].cpu().numpy()
    
    # Show the object patch with probe overlay
    axes[plot_idx].imshow(np.abs(object_patch), cmap='gray', alpha=0.7)
    
    # Overlay the probe (with transparency)
    probe_abs = np.abs(probe.cpu().numpy())
    probe_normalized = probe_abs / np.max(probe_abs)
    axes[plot_idx].imshow(probe_normalized, cmap='Reds', alpha=0.4)
    
    axes[plot_idx].set_title(f'Position {scan_idx}: ({x}, {y})')
    axes[plot_idx].set_xlabel('X (pixels)')
    axes[plot_idx].set_ylabel('Y (pixels)')
    
    # Add crosshairs at center
    center_x, center_y = probe_size//2, probe_size//2
    axes[plot_idx].axhline(y=center_y, color='cyan', linestyle='--', alpha=0.7)
    axes[plot_idx].axvline(x=center_x, color='cyan', linestyle='--', alpha=0.7)

plt.suptitle('Probe-Object Overlap for Random Scan Positions\n(Gray: Object, Red: Probe, Cyan: Center)', fontsize=14)
plt.tight_layout()
plt.show()

# Show corresponding diffraction patterns for these positions
fig, axes = plt.subplots(3, n_positions_to_show, figsize=(18, 12))
if n_positions_to_show == 1:
    axes = axes.reshape(-1, 1)

for plot_idx, scan_idx in enumerate(random_indices):
    pos = scan_positions[scan_idx].cpu().numpy()
    y, x = pos
    
    # Top row: Convolved diffraction patterns
    im1 = axes[0, plot_idx].imshow(dps[scan_idx], norm=colors.LogNorm(), cmap='jet')
    axes[0, plot_idx].set_title(f'Convolved DP {scan_idx}\nScan pos: ({x}, {y})')
    axes[0, plot_idx].set_xlabel('Qx')
    axes[0, plot_idx].set_ylabel('Qy')
    plt.colorbar(im1, ax=axes[0, plot_idx], shrink=0.8)
    
    # Middle row: Ideal diffraction patterns
    im2 = axes[1, plot_idx].imshow(dps_ideal[scan_idx], norm=colors.LogNorm(), cmap='jet')
    axes[1, plot_idx].set_title(f'Ideal DP {scan_idx}\nScan pos: ({x}, {y})')
    axes[1, plot_idx].set_xlabel('Qx')
    axes[1, plot_idx].set_ylabel('Qy')
    plt.colorbar(im2, ax=axes[1, plot_idx], shrink=0.8)
    
    # Bottom row: Difference (ratio) between convolved and ideal
    ratio = dps[scan_idx] / (dps_ideal[scan_idx] + 1e-10)  # Add small value to avoid division by zero
    im3 = axes[2, plot_idx].imshow(ratio, norm=colors.LogNorm(), cmap='RdBu_r')
    axes[2, plot_idx].set_title(f'Ratio (Conv/Ideal) {scan_idx}')
    axes[2, plot_idx].set_xlabel('Qx')
    axes[2, plot_idx].set_ylabel('Qy')
    plt.colorbar(im3, ax=axes[2, plot_idx], shrink=0.8)

# Add row labels
axes[0, 0].text(-0.3, 0.5, 'Convolved\nDiffraction\nPatterns', 
                transform=axes[0, 0].transAxes, rotation=90, 
                va='center', ha='center', fontsize=12, fontweight='bold')
axes[1, 0].text(-0.3, 0.5, 'Ideal\nDiffraction\nPatterns', 
                transform=axes[1, 0].transAxes, rotation=90, 
                va='center', ha='center', fontsize=12, fontweight='bold')
axes[2, 0].text(-0.3, 0.5, 'Ratio\n(Conv/Ideal)', 
                transform=axes[2, 0].transAxes, rotation=90, 
                va='center', ha='center', fontsize=12, fontweight='bold')

plt.suptitle('Diffraction Patterns Comparison for Random Scan Positions', fontsize=16)
plt.tight_layout()
plt.show()

# %%
