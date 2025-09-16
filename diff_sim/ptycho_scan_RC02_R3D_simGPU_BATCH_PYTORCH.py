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

# Resize probe in fourier space
probe_FFT = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(probe, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

# Create padded array of target size 1280x1280
target_size = 1280
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

# Load probe from file and move to GPU
probe_1280x1280 = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly888/roi0_Ndp1280/MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/Niter200.mat")['probe'][:,:,0], dtype=torch.cfloat, device=device)


# Load and pad the lattice
# Load lattice
lattice = np.load('/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/diff_sim/lattices/lattice_ls400_gs1024_lsp6_r3.0_typeSC.npy')

# Create 3D vignette mask
x, y, z = np.meshgrid(np.linspace(-1, 1, lattice.shape[0]),
                     np.linspace(-1, 1, lattice.shape[1]), 
                     np.linspace(-1, 1, lattice.shape[2]))
distance = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)])
vignette_mask = np.clip(1 - distance, 0, 1)

# Apply vignette to lattice
lattice = lattice * vignette_mask
print(lattice.shape)

# Project lattice along z-axis and convert to torch tensor
lattice_2d = torch.tensor(np.sum(lattice, axis=1), dtype=torch.float32, device=device)


lattice_size=lattice_2d.shape[0]

# Resize 2D lattice to n times the size
n=2
lattice_2d = torch.nn.functional.interpolate(lattice_2d.unsqueeze(0).unsqueeze(0), size=(lattice_size*n, lattice_size*n), mode='bilinear', align_corners=False)
lattice_2d = lattice_2d.squeeze(0).squeeze(0)

# Vignette the lattice
y = torch.linspace(-1, 1, lattice_2d.shape[0])
x = torch.linspace(-1, 1, lattice_2d.shape[1])
X, Y = torch.meshgrid(x, y, indexing='xy')
R = torch.sqrt(X**2 + Y**2).to(device)
lattice_2d = lattice_2d * (1 - R**2).clamp(0, 1)

# New lattice size
lattice_size=lattice_2d.shape[0]


# Pad lattice to target size
lattice_padded = torch.zeros((target_size, target_size), device=device)
lattice_padded[target_size//2-lattice_size//2:target_size//2+lattice_size//2,target_size//2-lattice_size//2:target_size//2+lattice_size//2] = lattice_2d


# Multiply probe with projected lattice in real space
probe_lattice = probe_resized * lattice_padded
#probe_lattice = probe_1280x1280 * lattice_padded

# Calculate diffraction pattern (FFT of product)
diffraction = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(probe_lattice, dim=(-2, -1)), norm='ortho'), dim=(-2, -1)))**2

# Plot results
fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(np.abs(probe_lattice.cpu().numpy()))
ax[0].set_title('Probe * Lattice (Real Space)')
ax[1].imshow(diffraction.cpu().numpy(), norm=colors.LogNorm())
ax[1].set_title('Diffraction Pattern')
ax[2].imshow(lattice_padded.cpu().numpy())
ax[2].set_title('Projected Lattice')
plt.show()




#%%
lattice=tifffile.imread('/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/diff_sim/lattices/clathrate_II_simulated_800x800x800_24x24x24unitcells.tif')
lattice=tifffile.imread('/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/diff_sim/lattices/clathrateRBP_800x800x800_12x12x12unitcells_RBP.tif')


# Create 3D vignette mask
x, y, z = np.meshgrid(np.linspace(-1, 1, lattice.shape[0]),
                     np.linspace(-1, 1, lattice.shape[1]), 
                     np.linspace(-1, 1, lattice.shape[2]))
distance = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)])
vignette_mask = np.clip(1 - distance, 0, 1)

# Apply vignette to lattice
lattice = lattice * vignette_mask

# Apply random 3D rotation to the lattice before projection
# Generate random rotation angles for each axis (in degrees)
angle_x = random.uniform(0, 360)
angle_y = random.uniform(0, 360) 
angle_z = random.uniform(0, 360)
angle_x=0
angle_y=0
angle_z=0

print(f"3D Rotation angles: X={angle_x:.1f}°, Y={angle_y:.1f}°, Z={angle_z:.1f}°")

# Rotate the 3D lattice around each axis
print("Rotating lattice around X-axis")
lattice_rotated = rotate(lattice, angle_x, axes=(1, 2), reshape=False, order=1)  # Rotate around X-axis
print("Rotating lattice around Y-axis")
lattice_rotated = rotate(lattice_rotated, angle_y, axes=(0, 2), reshape=False, order=1)  # Rotate around Y-axis
print("Rotating lattice around Z-axis")
lattice_rotated = rotate(lattice_rotated, angle_z, axes=(0, 1), reshape=False, order=1)  # Rotate around Z-axis

# Project lattice along z-axis and convert to torch tensor
lattice_2d = torch.tensor(np.sum(lattice_rotated, axis=2), dtype=torch.float32, device=device)
lattice_size=lattice_2d.shape[0]

# Resize 2D lattice to n times the size
n=1
lattice_2d = torch.nn.functional.interpolate(lattice_2d.unsqueeze(0).unsqueeze(0), size=(lattice_size*n, lattice_size*n), mode='bilinear', align_corners=False)
lattice_2d = lattice_2d.squeeze(0).squeeze(0)
# # Rotate the 2D lattice by random angles
# angle = random.uniform(0, 360)
# # Convert torch tensor to numpy array for rotation
# lattice_2d_np = lattice_2d.cpu().numpy()
# # Rotate the lattice
# lattice_2d_rotated = rotate(lattice_2d_np, angle, reshape=False, order=1)
# # Convert back to torch tensor on GPU
# lattice_2d = torch.tensor(lattice_2d_rotated, dtype=torch.float32, device=device)
#%%
fig,ax=plt.subplots(1,2,figsize=(30,15))
ax[0].imshow(lattice_2d.cpu().numpy())
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(lattice_2d.cpu().numpy())))**2,norm=colors.LogNorm())
plt.show()

#%%
# Vignette the lattice
y = torch.linspace(-1, 1, lattice_2d.shape[0])
x = torch.linspace(-1, 1, lattice_2d.shape[1])
X, Y = torch.meshgrid(x, y, indexing='xy')
R = torch.sqrt(X**2 + Y**2).to(device)
lattice_2d = lattice_2d * (1 - R**2).clamp(0, 1)

# New lattice size
lattice_size=lattice_2d.shape[0]

# Pad lattice to target size
lattice_padded = torch.zeros((target_size, target_size), device=device)
lattice_padded[target_size//2-lattice_size//2:target_size//2+lattice_size//2,target_size//2-lattice_size//2:target_size//2+lattice_size//2] = lattice_2d

# Multiply probe with projected lattice in real space
#probe_lattice = probe_resized * lattice_padded
probe_lattice = probe_1280x1280 * lattice_padded

# Calculate diffraction pattern (FFT of product)
diffraction = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(probe_lattice, dim=(-2, -1)), norm='ortho'), dim=(-2, -1)))

# # Scale diffraction pattern by radius from the center of pattern
# radius = target_size//2
# y = torch.linspace(-1, 1, diffraction.shape[0])
# x = torch.linspace(-1, 1, diffraction.shape[1])
# X, Y = torch.meshgrid(x, y, indexing='xy')
# R = torch.sqrt((X*1e-3)**2 + (Y*1e-3)**2).to(device)
# diffraction = diffraction * ((1 - R**2).clamp(0, 1))

# Plot results
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(np.abs(probe_resized.cpu().numpy()))
ax[1].imshow(np.abs(lattice_padded.cpu().numpy()))
plt.show()

fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(np.abs(probe_lattice.cpu().numpy()))
ax[0].set_title('Probe * Lattice (Real Space)')
ax[1].imshow(diffraction.cpu().numpy(), norm=colors.LogNorm())
ax[1].set_title('Diffraction Pattern')
ax[2].imshow(lattice_padded.cpu().numpy())
ax[2].set_title('Projected Lattice')
plt.show()


#%%
count=0
num_sims=1
while count<num_sims:
    count+=1
    
        
    # Create random object
    #object_img = torch.randn(1, 1, 1280, 1280, dtype=torch.cfloat, device=device)

    # Define scan number
    scan_number = 438#888

    # Load object from file
    #object_img = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly{scan_number:03d}/roi0_Ndp1280/MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/Niter200.mat")['object'][:,:], dtype=torch.cfloat, device=device)
    #object_img = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2025_Feb/results/ZC4_/fly{scan_number:03d}/roi1_Ndp512/MLc_L1_p10_g1000_Ndp256_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm//Niter2000.mat")['object'][:,:], dtype=torch.cfloat, device=device)
    object_img=lattice_padded

    scan_number=888
    # Load probe from file and move to GPU
    #probe = torch.tensor(sio.loadmat(f"/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly{scan_number:03d}/roi0_Ndp1280/MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/Niter200.mat")['probe'][:,:,0], dtype=torch.cfloat, device=device)
    probe = probe_resized
    print("Probe shape:", probe.shape)

    #load the probe from the file
    probe_ideal = np.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_bw0.2_wl1.24e-10_ps0.15_gs1280x1280.npy')
    probe_ideal = torch.tensor(probe_ideal, dtype=torch.cfloat, device=device)

    # Random rotation angle between 0 and 360 degrees
    angle = random.uniform(0, 360)

    # Random zoom factor between 0.8 and 1.2
    #zoom_factor = random.uniform(0.5, 1.2)
    zoom_factor=1.5#0.8

    # Convert to numpy for rotation and zoom
    object_np = object_img.cpu().numpy()
    print("Object shape:", object_np.shape)

    # Apply zoom using scipy.ndimage.zoom
    zoomed = zoom(object_np, (zoom_factor, zoom_factor), order=1)
    print("Zoomed object shape:", zoomed.shape)

    # Apply rotation using scipy.ndimage.rotate
    rotated = rotate(zoomed, angle, reshape=False, order=1)

    # Convert back to torch tensor
    object_img = torch.tensor(rotated, dtype=torch.cfloat, device=device)
    print("Original object shape:", object_img.shape)
            
    # Pad object to match probe size
    if object_img.shape[0] < probe.shape[0]:
        # Calculate padding dimensions
        pad_height = int((1.5*probe.shape[0] - object_img.shape[0]) // 2)
        pad_width = int((1.5*probe.shape[1] - object_img.shape[1]) // 2)
        
        print(f'pad_height: {pad_height}, pad_width: {pad_width}')
        
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
        
        print("Padded object shape:", object_img.shape)



    # Define scan parameters
    scan_step = probe.shape[0]//random.randint(18,36) # Step size between scan positions
    #scan_step = probe.shape[0]//36
    patch_size = 128  # Size of patches to process at once
    probe_size = probe.shape[0]

    # Create a dictionary to store simulation parameters
    sim_params = {
        'scan_step': scan_step,
        'patch_size': patch_size,
        'probe_size': probe_size,
        'angle': angle,
        'zoom_factor': zoom_factor
    }
    print("sim_params", sim_params)

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
    phase_diffs = []
    phase_ideals = []
    amps_diffs = []
    amps_ideals = []
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
        # Multiply by probe (broadcasted)
        # Create vignette for probe size
        y = torch.linspace(-1, 1, probe_size)
        x = torch.linspace(-1, 1, probe_size)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        R = torch.sqrt(X**2 + Y**2).to(device)
        probe_vignette = (1 - R**2).clamp(0, 1)
        
        # Apply vignette to both exit waves
        exit_waves = object_patches * probe * probe_vignette  # shape: (batch_size, probe_size, probe_size)
        exit_waves_ideal = object_patches*probe_vignette #* probe_ideal * probe_vignette
        # FFT and intensity
        dp = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        intensity = torch.abs(dp) ** 2  # shape: (batch_size, probe_size, probe_size)
        dp_ideal = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves_ideal, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        #dp_ideal = torch.fft.fft2(torch.fft.fftshift(exit_waves_ideal, dim=(-2, -1)), norm='ortho')
        intensity_ideal = torch.abs(dp_ideal) ** 2  # shape: (batch_size, probe_size, probe_size)
        
        # Calculate amps and phase of diffraction patterns
        amps_ideal = torch.abs(dp_ideal)
        amps = torch.abs(dp)
        phase_ideal = torch.angle(dp_ideal)
        phase_diff = torch.angle(dp)
        # Store results
        patterns.extend(exit_waves.detach().cpu())
        dps.extend(intensity.detach().cpu().numpy())
        dps_ideal.extend(intensity_ideal.detach().cpu().numpy())
        phase_diffs.extend(phase_diff.detach().cpu().numpy())
        phase_ideals.extend(phase_ideal.detach().cpu().numpy())
        amps_diffs.extend(amps.detach().cpu().numpy())
        amps_ideals.extend(amps_ideal.detach().cpu().numpy())
    
    # --- Plotting example ---
    plot_example=True
    if plot_example:
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        im1=ax[0][0].imshow(np.sum(dps, axis=0), norm=colors.LogNorm(), cmap='jet')
        im2=ax[0][1].imshow(np.sum(dps_ideal, axis=0), norm=colors.LogNorm(), cmap='jet')
        ri = random.randint(0, len(dps)-1)
        im3=ax[1][0].imshow(dps[ri], norm=colors.LogNorm(), cmap='jet')
        im4=ax[1][1].imshow(dps_ideal[ri], norm=colors.LogNorm(), cmap='jet')
        plt.colorbar(im1)
        plt.colorbar(im2)
        plt.colorbar(im3)
        plt.colorbar(im4)
        plt.show()
    
    # --- Visualization of probe/object scan positions ---
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
    axes = axes.flatten()
    
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
    
    plt.suptitle('Probe-Object Overlap for 6 Random Scan Positions\n(Gray: Object, Red: Probe, Cyan: Center)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Show corresponding diffraction patterns for these positions
    fig, axes = plt.subplots(3, n_positions_to_show, figsize=(18, 12))
    
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
    
    plt.suptitle('Diffraction Patterns Comparison for 6 Random Scan Positions', fontsize=16)
    plt.tight_layout()
    plt.show()




    # # %%
    def preprocess_dps(dp):
        # Take log10 and normalize to 0-1 range
        dp_log = np.log10(dp)
        dp_norm = (dp_log - np.min(dp_log)) / (np.max(dp_log) - np.min(dp_log))
        return dp_norm
    # fig,ax=plt.subplots(3,2,figsize=(15,10))
    # ri = random.randint(0, len(dps)-1)
    # im1=ax[0][0].imshow(preprocess_dps(dps[ri]), cmap='jet')
    # im2=ax[0][1].imshow(preprocess_dps(dps_ideal[ri]), cmap='jet')
    # im3=ax[1][0].imshow(phase_diffs[ri][1280//2-512//2:1280//2+512//2,1280//2-512//2:1280//2+512//2], cmap='jet')
    # im4=ax[1][1].imshow(phase_ideals[ri][1280// 2-512//2:1280//2+512//2,1280//2-512//2:1280//2+512//2], cmap='jet')
    # im5=ax[2][0].imshow(amps_diffs[ri][1280//2-512//2:1280//2+512//2,1280//2-512//2:1280//2+512//2], cmap='jet',norm=colors.LogNorm())
    # im6=ax[2][1].imshow(amps_ideals[ri][1280//2-512//2:1280//2+512//2,1280//2-512//2:1280//2+512//2], cmap='jet',norm=colors.LogNorm())
    # plt.colorbar(im1)
    # plt.colorbar(im2)
    # plt.colorbar(im3)
    # plt.colorbar(im4)
    # plt.colorbar(im5)
    # plt.colorbar(im6)
    # plt.show()
    # # %%




    # Choose what to save: 'summed' for segmented summed patterns, 'individual' for all scan patterns
    save_mode = 'individual'  # Change this to 'individual' to save all scan patterns
    
    if save_mode == 'summed':
        # Segment and plot summed diffraction patterns
        segment_size = 256

        # Sum up all diffraction patterns
        summed_dp = np.sum(dps, axis=0)
        summed_dp_ideal = np.sum(dps_ideal, axis=0)
        
        #load mask
        mask = ~np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_sum_RC02_R3D_1280.npy')
        summed_dp = summed_dp*mask
        
        # Create segments arrays
        segments_dp = np.zeros((5, 5, segment_size, segment_size))
        segments_dp_ideal = np.zeros((5, 5, segment_size, segment_size))
        segments_dp_ideal_conv=np.zeros((5, 5, segment_size, segment_size))

        # Segment the summed patterns
        for i in tqdm(range(5)):
            for j in range(5):
                y_start = i * segment_size
                y_end = (i + 1) * segment_size
                x_start = j * segment_size 
                x_end = (j + 1) * segment_size
                segments_dp[i,j] = summed_dp[y_start:y_end, x_start:x_end]
                segments_dp_ideal[i,j] = summed_dp_ideal[y_start:y_end, x_start:x_end]
        
        # Plot segmented patterns
        dp_count = (count-1)*25
        for dp_name, segments in zip(['Segmented Summed DP', 'Segmented Summed Ideal DP'], 
                                [segments_dp, segments_dp_ideal]):
            fig, axes = plt.subplots(5, 5, figsize=(15, 15))
            fig.suptitle(dp_name)
            for i in range(5):
                for j in range(5):
                    im = axes[i,j].imshow(segments[i,j], cmap='jet',norm=colors.LogNorm())
                    plt.colorbar(im, ax=axes[i,j])
                    axes[i,j].axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save segmented patterns
        print(f"Saving {len(segments_dp.flatten())} segmented patterns...")
        dp_count = (count-1)*25
        for i in range(5):
            for j in range(5):
                dp_count += 1
                segment_filename = f'/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/output_hanning_conv_{dp_count:05d}.npz'
                np.savez(segment_filename,
                        pinholeDP=segments_dp_ideal[i,j],
                        convDP=segments_dp[i,j])
                print(f'saved: {segment_filename}')
    
    elif save_mode == 'individual':
        # Save all individual scan diffraction patterns as segmented pieces
        segment_size = 256
        segments_per_pattern = 25  # 5x5 grid
        total_segments = len(dps) * segments_per_pattern
        print(f"Saving {len(dps)} individual scan patterns as {total_segments} segmented pieces...")
        
        dp_count = (count-1) * total_segments
                #load mask
        mask = ~np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_sum_RC02_R3D_1280.npy')
        dps = dps*mask
        # Optional: Plot a sample of individual patterns
        plot_individual_sample = True
        if plot_individual_sample:
            # Show a 5x5 grid of random individual patterns
            sample_indices = random.sample(range(len(dps)), min(25, len(dps)))
            fig, axes = plt.subplots(5, 5, figsize=(15, 15))
            fig.suptitle('Sample Individual Diffraction Patterns')
            axes = axes.flatten()
            
            for plot_idx, dp_idx in enumerate(sample_indices):
                if plot_idx < 25:  # Only plot up to 25
                    im = axes[plot_idx].imshow(dps[dp_idx], cmap='jet', norm=colors.LogNorm())
                    pos = scan_positions[dp_idx].cpu().numpy()
                    y, x = pos
                    axes[plot_idx].set_title(f'Scan {dp_idx}: ({x},{y})', fontsize=8)
                    axes[plot_idx].axis('off')
            
            # Hide unused subplots
            for plot_idx in range(len(sample_indices), 25):
                axes[plot_idx].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Save all individual patterns as segmented pieces
        for scan_idx in tqdm(range(len(dps)), desc="Saving individual segmented patterns"):
            # Get the individual diffraction patterns
            dp_conv = dps[scan_idx]
            dp_ideal = dps_ideal[scan_idx]
            
            # Get scan position for this pattern
            pos = scan_positions[scan_idx].cpu().numpy()
            y, x = pos
            
            # Segment each individual pattern into 5x5 grid
            for i in range(5):
                for j in range(5):
                    dp_count += 1
                    
                    # Calculate segment boundaries
                    y_start = i * segment_size
                    y_end = (i + 1) * segment_size
                    x_start = j * segment_size 
                    x_end = (j + 1) * segment_size
                    
                    # Extract segments
                    segment_conv = dp_conv[y_start:y_end, x_start:x_end]
                    segment_ideal = dp_ideal[y_start:y_end, x_start:x_end]
                    
                    # Save segmented pattern
                    pattern_filename = f'/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/output_hanning_conv_{dp_count:05d}.npz'
                    
                    np.savez(pattern_filename,
                            pinholeDP=segment_ideal,
                            convDP=segment_conv,
                            scan_position=[y, x],  # Save scan position info
                            scan_index=scan_idx,   # Save scan index
                            segment_position=[i, j])  # Save segment position within pattern
                    
                    if dp_count % 100 == 0:  # Print progress every 100 files
                        print(f'saved: {pattern_filename} (scan {scan_idx}: {x}, {y}, segment {i},{j})')
    
    else:
        print(f"Invalid save_mode: {save_mode}. Use 'summed' or 'individual'.")

      























#%%
# TESTING LOADING SAVED DATA

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import random
import sys
import importlib
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)


# Load the saved .npz file
ri=random.randint(0,7225)#12500)
print(f'ri: {ri}')
#file_path = f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/TEMP/output_hanning_conv_{ri:05d}.npz'
file_path = f'/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/output_hanning_conv_{ri:05d}.npz'
data = np.load(file_path)

# Extract the pinholeDP and convDP arrays
pinhole_dp = data['pinholeDP']
conv_dp = data['convDP']


def preprocess_dps(dp):
    # Take log10 and normalize to 0-1 range
    dp=dp-np.mean(dp)
    dp_log = ptNN_U.log10_custom(dp)
    dp_norm = (dp_log - np.min(dp_log)) / (np.max(dp_log) - np.min(dp_log))
    return dp_norm

# Plot the pinholeDP and convDP arrays
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
im1=ax[0][0].imshow(preprocess_dps(pinhole_dp), cmap='jet')
im2=ax[0][1].imshow(preprocess_dps(conv_dp), cmap='jet')
im3=ax[1][0].imshow(pinhole_dp, cmap='jet',norm=colors.LogNorm())
im4=ax[1][1].imshow(conv_dp, cmap='jet',norm=colors.LogNorm())
plt.colorbar(im1)
plt.colorbar(im2)
plt.colorbar(im3)
plt.colorbar(im4)
plt.show()


# %%
