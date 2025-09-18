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

def upsample_probe(base_probe, target_size, base_size=128):
    """
    Upsample probe in Fourier space to maintain probe properties
    """
    # Convert to Fourier space
    probe_FFT = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(base_probe, dim=(-2, -1)), 
                                                 norm='ortho'), dim=(-2, -1))
    
    # Create padded array of target size
    pad_size = (target_size - probe_FFT.shape[0]) // 2
    padded_FFT = torch.zeros((target_size, target_size), dtype=torch.cfloat, device=device)
    padded_FFT[pad_size:pad_size+probe_FFT.shape[0], 
              pad_size:pad_size+probe_FFT.shape[1]] = probe_FFT
    
    # Transform back to real space
    probe_resized = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(padded_FFT, dim=(-2, -1)), 
                                                       norm='ortho'), dim=(-2, -1))
    
    # Apply vignette
    y = torch.linspace(-1, 1, probe_resized.shape[0])
    x = torch.linspace(-1, 1, probe_resized.shape[1])
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X**2 + Y**2).to(device)
    probe_resized = probe_resized * (1 - R**2).clamp(0, 1)
    
    return probe_resized

def scale_lattice(lattice_3d, target_size, base_size=128):
    """
    Scale lattice to maintain consistent feature sizes
    """
    # Project 3D lattice to 2D
    lattice_2d = np.sum(lattice_3d, axis=2)
    
    # Convert to tensor
    lattice_tensor = torch.tensor(lattice_2d, device=device, dtype=torch.float32)
    
    # Scale the lattice
    lattice_scaled = torch.nn.functional.interpolate(lattice_tensor.unsqueeze(0).unsqueeze(0),
                                                   size=(target_size, target_size),
                                                   mode='bilinear',
                                                   align_corners=True).squeeze()
    
    return lattice_scaled

def get_scan_step(target_size, base_size=128):
    """
    Calculate scan step to maintain consistent sampling density
    """
    scale_factor = target_size / base_size
    base_steps = random.randint(12, 15)  # Base number of steps for 128x128
    return max(1, int(target_size // (base_steps * scale_factor)))

def plot_feature_comparison(lattices, probes, sizes):
    """
    Plot lattices and probes at different resolutions
    to verify consistent physical feature sizes
    """
    fig, axes = plt.subplots(len(sizes), 3, figsize=(15, 5*len(sizes)))
    if len(sizes) == 1:
        axes = axes.reshape(1, -1)
    
    for i, size in enumerate(sizes):
        # Plot lattice
        axes[i,0].imshow(lattices[i].cpu().numpy())
        axes[i,0].set_title(f'{size}x{size} Lattice')
        
        # Plot probe
        axes[i,1].imshow(np.abs(probes[i].cpu().numpy()))
        axes[i,1].set_title(f'{size}x{size} Probe')
        
        # Plot overlay
        axes[i,2].imshow(lattices[i].cpu().numpy())
        axes[i,2].imshow(np.abs(probes[i].cpu().numpy()), alpha=0.5)
        axes[i,2].set_title(f'{size}x{size} Overlay')
    
    plt.tight_layout()
    plt.show()

def simulate_pattern(target_size, base_size=128):
    """
    Simulate diffraction pattern for a given target size
    while maintaining consistent physical feature sizes
    """
    # Load base probe (128x128)
    base_probe = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly888/roi0_Ndp512/MLc_L1_p10_gInf_Ndp128_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/Niter1000.mat")['probe'][:,:,0], 
                            dtype=torch.cfloat, device=device)
    
    # Load lattice
    #lattice = tifffile.imread('/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/diff_sim/lattices/clathrate_II_simulated_800x800x800_24x24x24unitcells.tif')
    lattice = tifffile.imread('/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/diff_sim/lattices/clathrate_II_simulated_400x400x400_24x24x24unitcells.tif')
    
    
    # Create 3D vignette mask for lattice
    x, y, z = np.meshgrid(np.linspace(-1, 1, lattice.shape[0]),
                         np.linspace(-1, 1, lattice.shape[1]), 
                         np.linspace(-1, 1, lattice.shape[2]))
    distance = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)])
    vignette_mask = np.clip(1 - distance, 0, 1)
    
    # Apply vignette to lattice
    lattice = lattice * vignette_mask
    
    # Scale probe and lattice to target size
    probe = upsample_probe(base_probe, target_size, base_size)
    lattice_2d = scale_lattice(lattice, target_size, base_size)
    
    # Apply vignette to both probe and lattice
    y = torch.linspace(-1, 1, target_size)
    x = torch.linspace(-1, 1, target_size)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X**2 + Y**2).to(device)
    vignette = (1 - R**2).clamp(0, 1)
    
    probe = probe * vignette
    lattice_2d = lattice_2d * vignette
    
    # Calculate scan parameters
    scan_step = get_scan_step(target_size, base_size)
    probe_size = target_size
    
    # Prepare object (rotated and zoomed lattice)
    angle = 0  # Fixed angle for now
    zoom_factor = 1.5
    
    # Convert to numpy for rotation and zoom
    object_np = lattice_2d.cpu().numpy()
    zoomed = zoom(object_np, (zoom_factor, zoom_factor), order=1)
    rotated = rotate(zoomed, angle, reshape=False, order=1)
    object_img = torch.tensor(rotated, dtype=torch.cfloat, device=device)
    
    # Pad object if needed
    if object_img.shape[0] < probe_size:
        pad_height = int((1.5*probe_size - object_img.shape[0]) // 2)
        pad_width = int((1.5*probe_size - object_img.shape[1]) // 2)
        object_img = torch.nn.functional.pad(object_img, 
                                           (pad_width, pad_width, pad_height, pad_height),
                                           mode='constant', value=0)
        object_img = object_img * vignette
    
    # Calculate scan positions
    scan_positions = []
    for y in range(0, object_img.shape[0]-probe_size+1, scan_step):
        for x in range(0, object_img.shape[1]-probe_size+1, scan_step):
            scan_positions.append([y, x])
    scan_positions = torch.tensor(scan_positions, device=device)
    
    # Initialize storage
    patterns = []
    dps = []
    dps_ideal = []
    batch_size = 32
    
    # Batch processing
    for i in tqdm(range(0, len(scan_positions), batch_size), desc=f"Processing {target_size}x{target_size}"):
        batch = scan_positions[i:i+batch_size]
        batch_size_actual = len(batch)
        
        # Prepare patches
        object_patches = torch.zeros((batch_size_actual, probe_size, probe_size), 
                                   dtype=torch.cfloat, device=device)
        for j, pos in enumerate(batch):
            y, x = pos
            object_patches[j] = object_img[y:y+probe_size, x:x+probe_size]
        
        # Apply probe and calculate diffraction patterns
        exit_waves = object_patches * probe * vignette
        exit_waves_ideal = object_patches * vignette
        
        # FFT and intensity
        dp = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves, dim=(-2, -1)), 
                                              norm='ortho'), dim=(-2, -1))
        intensity = torch.abs(dp) ** 2
        
        dp_ideal = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves_ideal, dim=(-2, -1)), 
                                                    norm='ortho'), dim=(-2, -1))
        intensity_ideal = torch.abs(dp_ideal) ** 2
        
        # Store results
        patterns.extend(exit_waves.detach().cpu())
        dps.extend(intensity.detach().cpu().numpy())
        dps_ideal.extend(intensity_ideal.detach().cpu().numpy())
    
    return {
        'probe': probe,
        'lattice': lattice_2d,
        'object': object_img,
        'scan_positions': scan_positions,
        'patterns': patterns,
        'dps': dps,
        'dps_ideal': dps_ideal,
        'scan_step': scan_step
    }

#%%
# Test the simulation at different scales
target_sizes = [128, 256,512,1280]  # Can extend to [128, 256, 512, 1280]
results = {}

for size in target_sizes:
    print(f"\nSimulating {size}x{size} pattern...")
    results[size] = simulate_pattern(size)

# Plot comparison of features across scales
lattices = [results[size]['lattice'] for size in target_sizes]
probes = [results[size]['probe'] for size in target_sizes]
plot_feature_comparison(lattices, probes, target_sizes)

# Plot example diffraction patterns for each size
for size in target_sizes:
    result = results[size]
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    im1 = ax[0,0].imshow(np.sum(result['dps'], axis=0), norm=colors.LogNorm(), cmap='jet')
    im2 = ax[0,1].imshow(np.sum(result['dps_ideal'], axis=0), norm=colors.LogNorm(), cmap='jet')
    
    ri = random.randint(0, len(result['dps'])-1)
    im3 = ax[1,0].imshow(result['dps'][ri], norm=colors.LogNorm(), cmap='jet')
    im4 = ax[1,1].imshow(result['dps_ideal'][ri], norm=colors.LogNorm(), cmap='jet')
    
    plt.colorbar(im1, ax=ax[0,0])
    plt.colorbar(im2, ax=ax[0,1])
    plt.colorbar(im3, ax=ax[1,0])
    plt.colorbar(im4, ax=ax[1,1])
    
    ax[0,0].set_title(f'{size}x{size} Sum of Convolved DPs')
    ax[0,1].set_title(f'{size}x{size} Sum of Ideal DPs')
    ax[1,0].set_title(f'{size}x{size} Single Convolved DP')
    ax[1,1].set_title(f'{size}x{size} Single Ideal DP')
    
    plt.suptitle(f'Diffraction Patterns for {size}x{size}')
    plt.tight_layout()
    plt.show()

    # Plot scan positions
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    ax[0].imshow(np.abs(result['object'].cpu().numpy()), cmap='gray')
    ax[0].set_title(f'{size}x{size} Object with Scan Positions')
    
    for i, pos in enumerate(result['scan_positions'].cpu().numpy()):
        y, x = pos
        rect = patches.Rectangle((x, y), size, size, 
                               linewidth=1, edgecolor='red', facecolor='none', alpha=0.3)
        ax[0].add_patch(rect)
        
        if i < 10:
            ax[0].text(x + size/2, y + size/2, str(i), 
                      color='yellow', fontsize=8, ha='center', va='center')
    
    ax[1].imshow(np.abs(result['probe'].cpu().numpy()), cmap='viridis')
    ax[1].set_title(f'{size}x{size} Probe')
    
    plt.tight_layout()
    plt.show()

# Save patterns
for size in target_sizes:
    result = results[size]
    dp_count = 0
    
    for scan_idx in tqdm(range(len(result['dps'])), desc=f"Saving {size}x{size} patterns"):
        dp_count += 1
        pos = result['scan_positions'][scan_idx].cpu().numpy()
        y, x = pos
        
        pattern_filename = f'/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/output_{size}_conv_{dp_count:05d}.npz'
        
        # np.savez(pattern_filename,
        #         pinholeDP=result['dps_ideal'][scan_idx],
        #         convDP=result['dps'][scan_idx],
        #         scan_position=[y, x],
        #         scan_index=scan_idx)
        
        # if dp_count % 100 == 0:
        #     print(f'saved: {pattern_filename} (scan {scan_idx}: {x}, {y})')

#%%
