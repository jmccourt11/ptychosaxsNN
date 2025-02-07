#%%
import plotly.graph_objects as go
import numpy as np
import tifffile
from ipywidgets import interact, FloatSlider, IntSlider
from plotly.subplots import make_subplots
import scipy.ndimage as ndi
from plotly.subplots import make_subplots
import pdb
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from scipy.ndimage import maximum_filter

def find_peaks_3d(magnitude, threshold=0.1, sigma=1):
    # Apply Gaussian filter to smooth the data
    smoothed = gaussian_filter(magnitude, sigma=sigma)
    
    # Apply a threshold
    max_intensity = smoothed.max()
    threshold_value = max_intensity * threshold
    mask = smoothed > threshold_value
    
    # Use maximum filter to find local maxima
    local_max = maximum_filter(smoothed, size=3) == smoothed
    
    # Combine mask and local maxima
    peaks = mask & local_max
    
    # Label the peaks
    labeled, num_features = label(peaks)
    
    # Extract peak positions
    peak_positions = np.argwhere(peaks)
    
    return peak_positions, smoothed[peaks]


def calculate_orientation(projection, kx, ky):
    """Calculate primary orientation in a 2D projection using center of mass"""
    # Find center indices
    center_x = len(kx) // 2
    center_y = len(ky) // 2
    
    # Create coordinate grids relative to center
    y_coords, x_coords = np.indices(projection.shape)
    x_coords = x_coords - center_x
    y_coords = y_coords - center_y
    
    # Find points above threshold
    max_val = np.max(projection)
    threshold_mask = projection > (max_val * 0.1)  # Example threshold
    
    # Create circular mask to exclude the central region
    radius = 5  # Example radius
    r = np.sqrt(x_coords**2 + y_coords**2)
    central_mask = r > radius
    mask = threshold_mask & central_mask
    
    if np.sum(mask) < 2:  # Need at least 2 points
        return None, None, None
    
    # Calculate center of mass
    x_com, y_com = calculate_center_of_mass(projection * mask)
    
    # Offset the COM by the center of the image
    x_com -= center_x
    y_com -= center_y
    
    # Calculate angle from center of mass
    angle = np.arctan2(y_com, x_com)
    
    # Calculate magnitude (distance from center)
    magnitude = np.sqrt(x_com**2 + y_com**2)
    
    # Convert to real frequencies
    x_freq = x_com * (kx[1] - kx[0])
    y_freq = y_com * (ky[1] - ky[0])

    return angle, magnitude, (x_freq, y_freq)

def calculate_center_of_mass(image):
    """
    Calculate the center of mass of a 2D image.

    Args:
        image (np.ndarray): 2D array representing the image.

    Returns:
        tuple: (x_com, y_com) coordinates of the center of mass.
    """
    # Create coordinate grids
    y_indices, x_indices = np.indices(image.shape)
    
    # Calculate total mass
    image = np.nan_to_num(image, nan=0.0)
    total_mass = np.sum(image)
    
    if total_mass == 0:
        raise ValueError("The total mass of the image is zero, cannot compute center of mass.")
    
    # Calculate center of mass
    x_com = np.sum(x_indices * image) / total_mass
    y_com = np.sum(y_indices * image) / total_mass
    
    return x_com, y_com

def extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx):
    vz, vy, vx = voxel_results['voxel_size']
    nz, ny, nx = tomo_data.shape
    
    # Calculate the start and end indices for each dimension
    z_start = z_idx * vz
    z_end = min((z_idx + 1) * vz, nz)
    y_start = y_idx * vy
    y_end = min((y_idx + 1) * vy, ny)
    x_start = x_idx * vx
    x_end = min((x_idx + 1) * vx, nx)
    
    # Extract the region
    region = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Pad the region if it's smaller than the voxel size
    pad_width = ((0, vz - region.shape[0]), 
                 (0, vy - region.shape[1]), 
                 (0, vx - region.shape[2]))
    region_padded = np.pad(region, pad_width, mode='constant', constant_values=0)
    
    return region_padded

def compute_fft(region, use_vignette=False):
    if use_vignette:
        vignette = create_3d_vignette(region.shape)
        region_to_fft = region * vignette
    else:
        region_to_fft = region
    
    fft_3d = np.fft.fftn(region_to_fft)
    fft_3d_shifted = np.fft.fftshift(fft_3d)
    magnitude = np.abs(fft_3d_shifted)
    
    kz = np.fft.fftshift(np.fft.fftfreq(region.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(region.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(region.shape[2]))
    
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    
    return magnitude, KX, KY, KZ

def create_3d_fft_plot(magnitude, KX, KY, KZ, fft_threshold):
    max_magnitude = np.max(magnitude)
    threshold = max_magnitude * fft_threshold
    mask = magnitude > threshold
    
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=KX[mask],
        y=KY[mask],
        z=KZ[mask],
        mode='markers',
        marker=dict(
            size=5,
            color=np.log10(magnitude[mask] + 1),
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title='Log Magnitude')
        )
    )])
    
    fig_3d.update_layout(
        title="3D FFT of Voxel",
        scene=dict(
            xaxis_title="kx",
            yaxis_title="ky",
            zaxis_title="kz",
            aspectmode='cube',
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        width=800, height=800
    )
    
    return fig_3d

def create_2d_projections(magnitude):
    proj_xy = np.max(magnitude, axis=0)
    proj_xz = np.max(magnitude, axis=1)
    proj_yz = np.max(magnitude, axis=2)
    return proj_xy, proj_xz, proj_yz

def calculate_and_plot_orientations(proj_xy, proj_xz, proj_yz, kx, ky, kz):
    angle_xy, mag_xy, freq_xy = calculate_orientation(proj_xy, kx, ky)
    angle_xz, mag_xz, freq_xz = calculate_orientation(proj_xz, kx, kz)
    angle_yz, mag_yz, freq_yz = calculate_orientation(proj_yz, ky, kz)
    
    fig_xy = plot_projection(proj_xy, kx, ky, angle_xy, mag_xy, freq_xy, "XY Projection")
    fig_xz = plot_projection(proj_xz, kx, kz, angle_xz, mag_xz, freq_xz, "XZ Projection")
    fig_yz = plot_projection(proj_yz, ky, kz, angle_yz, mag_yz, freq_yz, "YZ Projection")
    
    return fig_xy, fig_xz, fig_yz

def analyze_voxel_fourier(tomo_data, voxel_results, z_idx, y_idx, x_idx, fft_threshold=1e-3, use_vignette=False, overlay_octants=False, plot_projections=False):
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft(region, use_vignette)
    
    fig_3d = create_3d_fft_plot(magnitude, KX, KY, KZ, fft_threshold)
    
    if overlay_octants:
        octant_intensities = calculate_octant_intensities(magnitude, KX, KY, KZ)
        kx_vector, ky_vector, kz_vector = calculate_orientation_vector_from_octants(octant_intensities, KX, KY, KZ)
        plot_orientation_vector_on_fft(fig_3d, kx_vector, ky_vector, kz_vector)
    
    proj_xy, proj_xz, proj_yz = create_2d_projections(magnitude)
    fig_xy, fig_xz, fig_yz = calculate_and_plot_orientations(proj_xy, proj_xz, proj_yz, KX[0,0,:], KY[0,:,0], KZ[:,0,0])

    return fig_3d, fig_xy, fig_yz, fig_xz

def plot_3D_tomogram(tomo_data, intensity_threshold=0.1):
    """
    Create efficient 3D visualization of tomogram
    
    Args:
        tomo_data (np.ndarray): 3D tomographic data
        intensity_threshold (float): Threshold relative to max intensity (0-1)
    """
    # Ensure the shape is interpreted correctly
    nz, ny, nx = tomo_data.shape  # Assuming (z, y, x) order
    z = np.arange(nz)
    y = np.arange(ny)
    x = np.arange(nx)
    
    # Create meshgrid
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')  # Ensure correct order
    
    # Apply threshold
    max_intensity = tomo_data.max()
    threshold = max_intensity * intensity_threshold
    mask = tomo_data > threshold
    
    # Get masked coordinates and intensities
    x_plot = X[mask]
    y_plot = Y[mask]
    z_plot = Z[mask]
    intensities_plot = tomo_data[mask]
    
    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        mode='markers',
        marker=dict(
            size=3,
            color=intensities_plot,
            colorscale='Greys',  # 'Viridis',
            opacity=0.2,
            colorbar=dict(title='Intensity')
        ),
        hovertemplate=(
            "x: %{x}<br>" +
            "y: %{y}<br>" +
            "z: %{z}<br>" +
            "Intensity: %{marker.color:.1f}<br>" +
            "<extra></extra>"
        )
    )])
    
    # Update layout
    fig.update_layout(
        title="3D Tomogram Visualization",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode='cube',
            camera=dict(
                eye=dict(x=2, y=2, z=2)
            ),
            bgcolor='white'
        ),
        width=800,
        height=800
    )
    
    return fig

def analyze_tomogram_voxels(tomo_data, voxel_size=(10, 10, 10)):
    """
    Break tomogram into voxels and analyze, including partial voxels at edges
    
    Args:
        tomo_data (np.ndarray): 3D tomographic data
        voxel_size (tuple): Size of voxels in (z, y, x)
    
    Returns:
        dict: Voxel analysis results
    """
    nz, ny, nx = tomo_data.shape
    vz, vy, vx = voxel_size
    
    # Calculate number of voxels needed to cover entire volume
    n_voxels_z = int(np.ceil(nz / vz))
    n_voxels_y = int(np.ceil(ny / vy))
    n_voxels_x = int(np.ceil(nx / vx))
    
    # Initialize arrays for voxel statistics
    voxel_means = np.zeros((n_voxels_z, n_voxels_y, n_voxels_x))
    voxel_maxes = np.zeros_like(voxel_means)
    voxel_stds = np.zeros_like(voxel_means)
    
    # Calculate statistics for each voxel
    for iz in range(n_voxels_z):
        z_start = iz * vz
        z_end = min((iz + 1) * vz, nz)
        
        for iy in range(n_voxels_y):
            y_start = iy * vy
            y_end = min((iy + 1) * vy, ny)
            
            for ix in range(n_voxels_x):
                x_start = ix * vx
                x_end = min((ix + 1) * vx, nx)
                
                voxel = tomo_data[z_start:z_end, 
                                y_start:y_end, 
                                x_start:x_end]
                
                voxel_means[iz, iy, ix] = np.mean(voxel)
                voxel_maxes[iz, iy, ix] = np.max(voxel)
                voxel_stds[iz, iy, ix] = np.std(voxel)
    
    return {
        'means': voxel_means,
        'maxes': voxel_maxes,
        'stds': voxel_stds,
        'voxel_size': voxel_size,
        'n_voxels': (n_voxels_z, n_voxels_y, n_voxels_x)
    }

def plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold=0.1):
    """
    Plot a specific voxel region and its location in the full tomogram
    """
    vz, vy, vx = voxel_results['voxel_size']
    
    # Extract voxel region
    region = tomo_data[z_idx*vz:(z_idx+1)*vz, 
                      y_idx*vy:(y_idx+1)*vy, 
                      x_idx*vx:(x_idx+1)*vx]
    
    # Create coordinate arrays for this region
    z, y, x = np.meshgrid(np.arange(vz), np.arange(vy), np.arange(vx), indexing='ij')
    
    # Apply threshold to region
    max_intensity = region.max()
    threshold = max_intensity * intensity_threshold
    mask = region > threshold
    
    # Get masked coordinates and intensities for region
    x_plot = x[mask] + x_idx*vx
    y_plot = y[mask] + y_idx*vy
    z_plot = z[mask] + z_idx*vz
    intensities_plot = region[mask]
    
    # Create figure with two subplots
    fig = go.Figure()
    
    # Plot 1: Voxel region
    fig.add_trace(go.Scatter3d(
        x=x_plot, y=y_plot, z=z_plot,
        mode='markers',
        marker=dict(
            size=2,
            color=intensities_plot,
            colorscale='Viridis',
            opacity=0.6#,
            #colorbar=dict(title='Intensity', x=0.45)
        ),
        name='Voxel Region',
        showlegend=False
    ))
    
    # Plot 2: Full tomogram with highlighted region
    # Apply threshold to full tomogram
    full_max = tomo_data.max()
    full_threshold = full_max * intensity_threshold
    full_mask = tomo_data > full_threshold
    
    # Create meshgrid for full tomogram - corrected indexing
    Z, Y, X = np.meshgrid(np.arange(tomo_data.shape[0]), 
                         np.arange(tomo_data.shape[1]), 
                         np.arange(tomo_data.shape[2]), 
                         indexing='ij')
    
    # Add full tomogram points with low opacity
    fig.add_trace(go.Scatter3d(
        x=X[full_mask],
        y=Y[full_mask],
        z=Z[full_mask],
        mode='markers',
        marker=dict(
            size=2,
            color='gray',
            opacity=0.1
        ),
        name='Full Tomogram',
        showlegend=False
    ))
    
    # Add box to highlight voxel region
    box_x = [x_idx*vx, (x_idx+1)*vx, (x_idx+1)*vx, x_idx*vx, x_idx*vx, 
            x_idx*vx, (x_idx+1)*vx, (x_idx+1)*vx, x_idx*vx, x_idx*vx]
    box_y = [y_idx*vy, y_idx*vy, (y_idx+1)*vy, (y_idx+1)*vy, y_idx*vy,
            y_idx*vy, y_idx*vy, (y_idx+1)*vy, (y_idx+1)*vy, y_idx*vy]
    box_z = [z_idx*vz, z_idx*vz, z_idx*vz, z_idx*vz, z_idx*vz,
            (z_idx+1)*vz, (z_idx+1)*vz, (z_idx+1)*vz, (z_idx+1)*vz, (z_idx+1)*vz]
    
    fig.add_trace(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode='lines',
        line=dict(color='red', width=4),
        name='Region Box',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Voxel Region (z={z_idx}, y={y_idx}, x={x_idx})",
        scene=dict(
            aspectmode='cube',
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        width=1000, height=800,
        showlegend=False
    )
    
    # Create local view figure
    fig_local = go.Figure(data=[go.Scatter3d(
        x=x[mask], y=y[mask], z=z[mask],
        mode='markers',
        marker=dict(
            size=10,
            color=region[mask],
            colorscale='Viridis',
            opacity=0.6#,
            #colorbar=dict(title='Intensity')
        )
    )])
    
    fig_local.update_layout(
        title=f"Local Voxel View (z={z_idx}, y={y_idx}, x={x_idx})",
        scene=dict(
            aspectmode='cube',
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        width=800, height=800
    )
    
    return fig, fig_local

def create_3d_vignette(shape):
    """
    Create a 3D cosine window vignette
    
    Args:
        shape (tuple): Shape of the 3D volume (z, y, x)
    
    Returns:
        np.ndarray: 3D vignette array
    """
    z, y, x = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    # Create normalized coordinates (-1 to 1)
    z_norm = 2 * z / (shape[0] - 1) - 1
    y_norm = 2 * y / (shape[1] - 1) - 1
    x_norm = 2 * x / (shape[2] - 1) - 1
    
    # Calculate radial distance from center (squared)
    r_squared = x_norm**2 + y_norm**2 + z_norm**2
    
    # Create cosine window
    vignette = np.cos(np.pi/2 * np.sqrt(r_squared))
    vignette = np.clip(vignette, 0, 1)
    
    return vignette



def calculate_octant_intensities(magnitude, KX, KY, KZ):
    """
    Calculate the total intensity for each of the 8 octants in the 3D FFT magnitude.
    
    Args:
        magnitude (np.ndarray): 3D FFT magnitude.
        KX, KY, KZ (np.ndarray): Frequency coordinates.
    
    Returns:
        dict: Total intensity for each octant.
    """
    # Initialize dictionary to store total intensity for each octant
    octant_intensities = {
        '+++' : 0,
        '++-' : 0,
        '+-+' : 0,
        '+--' : 0,
        '-++' : 0,
        '-+-' : 0,
        '--+' : 0,
        '---' : 0
    }
    
    # Determine the octant for each point and sum the magnitudes
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            for k in range(magnitude.shape[2]):
                key = ('+' if KX[i, j, k] >= 0 else '-') + \
                      ('+' if KY[i, j, k] >= 0 else '-') + \
                      ('+' if KZ[i, j, k] >= 0 else '-')
                octant_intensities[key] += magnitude[i, j, k]
    
    return octant_intensities

def calculate_orientation_vector_from_octants(octant_intensities, KX, KY, KZ):
    """
    Calculate the orientation vector pointing to the octant with the highest intensity.
    
    Args:
        octant_intensities (dict): Total intensity for each octant.
        KX, KY, KZ (np.ndarray): Frequency coordinates.
    
    Returns:
        tuple: (kx_vector, ky_vector, kz_vector) coordinates of the orientation vector.
    """
    # Find the octant with the highest intensity
    max_octant = max(octant_intensities, key=octant_intensities.get)
    
    # Define the center of each octant in frequency space
    kx_center, ky_center, kz_center = 0, 0, 0
    kx_max, ky_max, kz_max = KX.max(), KY.max(), KZ.max()
    kx_min, ky_min, kz_min = KX.min(), KY.min(), KZ.min()

    octant_centers = {
        '+++': ((kx_max + kx_center) / 2, (ky_max + ky_center) / 2, (kz_max + kz_center) / 2),
        '++-': ((kx_max + kx_center) / 2, (ky_max + ky_center) / 2, (kz_min + kz_center) / 2),
        '+-+': ((kx_max + kx_center) / 2, (ky_min + ky_center) / 2, (kz_max + kz_center) / 2),
        '+--': ((kx_max + kx_center) / 2, (ky_min + ky_center) / 2, (kz_min + kz_center) / 2),
        '-++': ((kx_min + kx_center) / 2, (ky_max + ky_center) / 2, (kz_max + kz_center) / 2),
        '-+-': ((kx_min + kx_center) / 2, (ky_max + ky_center) / 2, (kz_min + kz_center) / 2),
        '--+': ((kx_min + kx_center) / 2, (ky_min + ky_center) / 2, (kz_max + kz_center) / 2),
        '---': ((kx_min + kx_center) / 2, (ky_min + ky_center) / 2, (kz_min + kz_center) / 2)
    }
    
    # Get the center of the octant with the highest intensity
    kx_vector, ky_vector, kz_vector = octant_centers[max_octant]
    
    return kx_vector, ky_vector, kz_vector

def plot_orientation_vector_on_fft(fig, kx_vector, ky_vector, kz_vector):
    """
    Add an orientation vector to the 3D FFT plot.
    
    Args:
        fig (go.Figure): Plotly figure.
        kx_vector, ky_vector, kz_vector (float): Coordinates of the orientation vector.
    """
    fig.add_trace(go.Cone(
        x=[0], y=[0], z=[0],
        u=[kx_vector], v=[ky_vector], w=[kz_vector],
        sizemode="absolute",
        sizeref=0.1,
        anchor="tail",
        colorscale='Reds',
        showscale=False
    ))

def plot_projection(projection, kx, ky, angle, magnitude, freq_coords, title):
    """Plot a 2D projection with an orientation vector."""
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=projection,
                   x=kx, y=ky,
                   colorscale='Viridis',
                   showscale=False)
    )
    
    if angle is not None:
        x_com, y_com = freq_coords
        arrow_length = 0.25 * magnitude
        
        x_end = x_com + arrow_length * np.cos(angle)
        y_end = y_com + arrow_length * np.sin(angle)
        
        fig.add_trace(
            go.Scatter(
                x=[x_com, x_end],
                y=[y_com, y_end],
                mode='lines',
                line=dict(color='red', width=4),
                showlegend=False
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="kx",
        yaxis_title="ky",
        width=600, height=600
    )
    
    return fig

def calculate_wedge_intensities_with_radius(array, num_wedges=32, min_radius=2.5, max_radius=7.5):
    """
    Calculate the intensity of discrete wedges on a 2D array within a radius range.
    
    Args:
        array (np.ndarray): 2D array representing the image.
        num_wedges (int): Number of wedges to divide the array into.
        min_radius (float): Minimum radius for the wedge calculation.
        max_radius (float): Maximum radius for the wedge calculation.
    
    Returns:
        list: Intensities of each wedge.
    """
    # Get the center of the array
    center_y, center_x = np.array(array.shape) // 2
    
    # Create coordinate grids
    y_indices, x_indices = np.indices(array.shape)
    x_indices = x_indices - center_x
    y_indices = y_indices - center_y
    
    # Calculate distances and angles for each point
    distances = np.sqrt(x_indices**2 + y_indices**2)
    angles = np.arctan2(y_indices, x_indices)
    
    # Normalize angles to [0, 2*pi]
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    # Calculate wedge boundaries
    wedge_boundaries = np.linspace(0, 2 * np.pi, num_wedges + 1)
    
    # Calculate intensities for each wedge within the radius range
    wedge_intensities = []
    for i in range(num_wedges):
        # Create a mask for the current wedge and radius range
        mask = ((angles >= wedge_boundaries[i]) & (angles < wedge_boundaries[i + 1]) &
                (distances >= min_radius) & (distances <= max_radius))
        
        # Sum the intensities within the wedge
        wedge_intensity = np.sum(array[mask])
        wedge_intensities.append(wedge_intensity)
    
    return wedge_intensities

def visualize_wedges(array, num_wedges=32, min_radius=2.5, max_radius=7.5, show_first_wedge=False):
    """
    Visualize the wedges within a specified radius range on a 2D array.
    
    Args:
        array (np.ndarray): 2D array representing the image.
        num_wedges (int): Number of wedges to divide the array into.
        min_radius (float): Minimum radius for the wedge calculation.
        max_radius (float): Maximum radius for the wedge calculation.
        show_first_wedge (bool): Whether to overlay only the first wedge.
    """
    # Get the center of the array
    center_y, center_x = np.array(array.shape) // 2
    
    # Create coordinate grids
    y_indices, x_indices = np.indices(array.shape)
    x_indices = x_indices - center_x
    y_indices = y_indices - center_y
    
    # Calculate distances and angles for each point
    distances = np.sqrt(x_indices**2 + y_indices**2)
    angles = np.arctan2(y_indices, x_indices)
    
    # Normalize angles to [0, 2*pi]
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    # Calculate wedge boundaries
    wedge_boundaries = np.linspace(0, 2 * np.pi, num_wedges + 1)
    
    # Create a mask for the wedges within the radius range
    mask = np.zeros_like(array, dtype=bool)
    for i in range(num_wedges):
        wedge_mask = ((angles >= wedge_boundaries[i]) & (angles < wedge_boundaries[i + 1]) &
                      (distances >= min_radius) & (distances <= max_radius))
        mask |= wedge_mask
    
    # Create a mask for the first wedge
    first_wedge_mask = ((angles >= wedge_boundaries[0]) & (angles < wedge_boundaries[1]) &
                        (distances >= min_radius) & (distances <= max_radius))
    
    # Plot the original array
    plt.imshow(array, cmap='gray', origin='lower')
    plt.colorbar(label='Intensity')
    
    # Overlay the wedge mask
    if show_first_wedge:
        plt.imshow(first_wedge_mask, cmap='cool', alpha=0.8, origin='lower')
    else:
        plt.imshow(mask, cmap='cool', alpha=0.5, origin='lower')
    
    # Plot the center and radius boundaries
    circle1 = plt.Circle((center_x, center_y), min_radius, color='red', fill=False, linestyle='--')
    circle2 = plt.Circle((center_x, center_y), max_radius, color='red', fill=False, linestyle='--')
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    
    plt.title('Wedge Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



def calculate_3d_wedge_intensities(array, num_azimuthal_wedges, num_polar_wedges, min_radius, max_radius):
    """
    Calculate the intensity of discrete wedges on a 3D array within a radius range.
    
    Args:
        array (np.ndarray): 3D array representing the data.
        num_azimuthal_wedges (int): Number of azimuthal wedges (phi).
        num_polar_wedges (int): Number of polar wedges (theta).
        min_radius (float): Minimum radius for the wedge calculation.
        max_radius (float): Maximum radius for the wedge calculation.
    
    Returns:
        np.ndarray: Intensities of each wedge.
    """
    # Get the center of the array
    center_z, center_y, center_x = np.array(array.shape) // 2
    
    # Create coordinate grids
    z_indices, y_indices, x_indices = np.indices(array.shape)
    x_indices = x_indices - center_x
    y_indices = y_indices - center_y
    z_indices = z_indices - center_z
    
    # Calculate spherical coordinates
    distances = np.sqrt(x_indices**2 + y_indices**2 + z_indices**2)
    azimuthal_angles = np.arctan2(y_indices, x_indices)  # phi
    polar_angles = np.arccos(z_indices / (distances + 1e-10))  # theta
    
    # Normalize angles
    azimuthal_angles = (azimuthal_angles + 2 * np.pi) % (2 * np.pi)
    
    # Calculate wedge boundaries
    azimuthal_boundaries = np.linspace(0, 2 * np.pi, num_azimuthal_wedges + 1)
    polar_boundaries = np.linspace(0, np.pi, num_polar_wedges + 1)
    
    # Calculate intensities for each wedge
    wedge_intensities = np.zeros((num_azimuthal_wedges, num_polar_wedges))
    for i in range(num_azimuthal_wedges):
        for j in range(num_polar_wedges):
            # Create a mask for the current wedge
            mask = ((azimuthal_angles >= azimuthal_boundaries[i]) & (azimuthal_angles < azimuthal_boundaries[i + 1]) &
                    (polar_angles >= polar_boundaries[j]) & (polar_angles < polar_boundaries[j + 1]) &
                    (distances >= min_radius) & (distances <= max_radius))
            
            # Sum the intensities within the wedge
            wedge_intensity = np.sum(array[mask])
            wedge_intensities[i, j] = wedge_intensity
    
    return wedge_intensities


def extract_3d_data_from_figure(fig):
    """
    Extract 3D data from a Plotly figure.
    
    Args:
        fig (go.Figure): Plotly 3D figure.
    
    Returns:
        np.ndarray: 3D array of intensities.
    """
    # Assuming the data is in the first trace
    x = fig.data[0]['x']
    y = fig.data[0]['y']
    z = fig.data[0]['z']
    intensity = fig.data[0]['marker']['color']
    
    # Create a 3D grid based on the unique x, y, z values
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    z_unique = np.unique(z)
    
    # Initialize a 3D array
    array_3d = np.zeros((len(z_unique), len(y_unique), len(x_unique)))
    
    # Fill the 3D array with intensity values
    for xi, yi, zi, inten in zip(x, y, z, intensity):
        x_idx = np.where(x_unique == xi)[0][0]
        y_idx = np.where(y_unique == yi)[0][0]
        z_idx = np.where(z_unique == zi)[0][0]
        array_3d[z_idx, y_idx, x_idx] = inten
    
    return array_3d


def compute_orientation_tensor(magnitude, KX, KY, KZ):
    """
    Compute the orientation tensor for a voxel based on its 3D FFT magnitude.

    Args:
        magnitude (np.ndarray): 3D FFT magnitude.
        KX, KY, KZ (np.ndarray): Frequency coordinates.

    Returns:
        np.ndarray: 3x3 orientation tensor.
    """
    magnitude_normalized = magnitude / np.max(magnitude)
    
    # Calculate the center of mass in Fourier space
    total_intensity = np.sum(magnitude_normalized)
    if total_intensity == 0:
        return np.zeros((3, 3))


    
    kx_com = np.sum(KX * magnitude_normalized) / total_intensity
    ky_com = np.sum(KY * magnitude_normalized) / total_intensity
    kz_com = np.sum(KZ * magnitude_normalized) / total_intensity

    # Construct the orientation tensor
    orientation_tensor = np.array([
        [kx_com**2, kx_com*ky_com, kx_com*kz_com],
        [ky_com*kx_com, ky_com**2, ky_com*kz_com],
        [kz_com*kx_com, kz_com*ky_com, kz_com**2]
    ])

    return orientation_tensor

def generate_voxel_indices(x_range, y_range, z_range):
    """
    Generate voxel indices for a cubic range.

    Parameters:
    - x_range: tuple of (start, end) for the x dimension
    - y_range: tuple of (start, end) for the y dimension
    - z_range: tuple of (start, end) for the z dimension

    Returns:
    - List of tuples representing voxel indices within the specified range.
    """
    voxel_indices = [
        (z, y, x)
        for z in range(x_range[0], x_range[1])
        for y in range(y_range[0], y_range[1])
        for x in range(z_range[0], z_range[1])
    ]
    return voxel_indices



def initialize_combined_figure(nrows):
    """
    Initialize a combined figure with multiple rows of 3D subplots.
    
    Args:
        nrows (int): Number of rows in the combined figure
        
    Returns:
        plotly.graph_objects.Figure: Initialized figure with proper layout
    """
    # Create subplot titles for each row
    subplot_titles = []
    for i in range(nrows):
        subplot_titles.extend([f"Voxel Region {i+1}", f"FFT Peaks {i+1}"])
    
    # Create specs for 3D scenes
    specs = [[{'type': 'scene'}, {'type': 'scene'}] for _ in range(nrows)]
    
    # Initialize figure
    fig_combined = make_subplots(
        rows=nrows, 
        cols=2,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.02,
        horizontal_spacing=0.05
    )
    
    # Update overall layout
    fig_combined.update_layout(
        title="Voxel Analysis Along Z-Axis",
        width=1200,
        height=400*nrows,
        showlegend=False
    )
    
    # Update each subplot's layout
    for i in range(nrows):
        # Calculate vertical position for this row
        y_max = 1.0 - (i/nrows)
        y_min = 1.0 - ((i+1)/nrows)
        
        # Update scene for left subplot (Voxel Region)
        scene_name = f'scene{i*2 + 1}' if i > 0 else 'scene'
        fig_combined.update_layout(**{
            scene_name: dict(
                aspectmode='cube',
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                domain=dict(x=[0, 0.45], y=[y_min, y_max]),
                camera=dict(
                    eye=dict(x=0, y=2.5, z=0),  # front view
                    up=dict(x=0, y=0, z=1)
                )
            )
        })
        
        # Update scene for right subplot (FFT Peaks)
        scene_name = f'scene{i*2 + 2}'
        fig_combined.update_layout(**{
            scene_name: dict(
                aspectmode='cube',
                xaxis_title="KX",
                yaxis_title="KY",
                zaxis_title="KZ",
                domain=dict(x=[0.55, 1.0], y=[y_min, y_max]),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),  # isometric view
                    up=dict(x=0, y=0, z=1)
                )
            )
        })
    
    return fig_combined

#%%
# Load the data
#tomogram = "/net/micdata/data2/12IDC/2024_Dec/misc/JM02_3D/ROI2_Ndp512_MLc_p10_gInf_Iter1000/recons/tomogram_alignment_recon_cropped_14nm_2.tif"
tomogram = "/net/micdata/data2/12IDC//2021_Nov/results/tomography/Sample6_tomo6_SIRT_tomogram.tif"
tomo_data = tifffile.imread(tomogram)

#Rotate the tomogram
axis='y'
angle=0
if axis == 'x':
    rotated_data = rotate(tomo_data, angle, axes=(1, 2), reshape=False)
elif axis == 'y':
    rotated_data = rotate(tomo_data, angle, axes=(0, 2), reshape=False)
elif axis == 'z':
    rotated_data = rotate(tomo_data, angle, axes=(0, 1), reshape=False)
tomo_data = rotated_data

axis='x'
angle=0
if axis == 'x':
    rotated_data = rotate(tomo_data, angle, axes=(1, 2), reshape=False)
elif axis == 'y':
    rotated_data = rotate(tomo_data, angle, axes=(0, 2), reshape=False)
elif axis == 'z':
    rotated_data = rotate(tomo_data, angle, axes=(0, 1), reshape=False)
tomo_data = rotated_data

# Create and display the plot
fig = plot_3D_tomogram(tomo_data, intensity_threshold=0.8)
fig.show()
# Print dimensions
print(f"Tomogram shape: {tomo_data.shape}")
voxel_size = (10*8//6,10*8//6,10*8//6)  # Reduced from (32, 32, 32)
voxel_results = analyze_tomogram_voxels(tomo_data, voxel_size=voxel_size)

# Print number of voxels in each dimension
print(f"Number of voxels (z, y, x): {voxel_results['n_voxels']}")

show_plots = False





#%%

'''
TEST FOR SINGLE VOXEL
'''
vz, vy, vx = voxel_results['voxel_size']
figE = go.Figure()

# Process only the first voxel
z_idx, y_idx, x_idx = 5, 7, 5
intensity_threshold = 0.8

# Compute orientation tensor and eigenvalues/eigenvectors
region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
magnitude, KX, KY, KZ = compute_fft(region, use_vignette=True)
fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold)
fig.show()
# Define a threshold for the magnitude
threshold = 0.02*4 * np.max(magnitude)  # Example: 10% of the max magnitude

# Flatten the arrays
kx_flat = KX.flatten()
ky_flat = KY.flatten()
kz_flat = KZ.flatten()
magnitude_flat = magnitude.flatten()

# Apply the threshold
mask = magnitude_flat > threshold
kx_filtered = kx_flat[mask]
ky_filtered = ky_flat[mask]
kz_filtered = kz_flat[mask]
magnitude_filtered = magnitude_flat[mask]


#Find peaks in 3D
#peak_positions, peak_values = find_peaks_3d(magnitude,threshold = 1e-9 * np.max(magnitude) ,sigma=0.5)
peak_positions, peak_values = find_peaks_3d(magnitude,threshold = 0.04 ,sigma=0.5)

for pos, val in zip(peak_positions, peak_values):
    print(f"Peak at position {pos} with value {val}")

# Create a 3D scatter plot of the FFT magnitude
fig_fft = go.Figure(data=go.Scatter3d(
    x=kx_filtered,
    y=ky_filtered,
    z=kz_filtered,
    mode='markers',
    marker=dict(
        size=10,
        color=magnitude_filtered,
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title='Magnitude')
    )
))

fig_fft.update_layout(
    title="3D FFT Magnitude with Threshold",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

fig_fft.show()


















#%%

'''
TEST PEAK ANALYSISFOR multiple voxels
'''
vz, vy, vx = voxel_results['voxel_size']
figE = go.Figure()

# Process only the first voxel
z_idx, y_idx, x_idx = 5, 7, 5
intensity_threshold = 0.8

# Compute orientation tensor and eigenvalues/eigenvectors
region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
magnitude, KX, KY, KZ = compute_fft(region, use_vignette=True)
fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold)
fig.show()
# Define a threshold for the magnitude
threshold = 0.02*4 * np.max(magnitude)  # Example: 10% of the max magnitude

# Flatten the arrays
kx_flat = KX.flatten()
ky_flat = KY.flatten()
kz_flat = KZ.flatten()
magnitude_flat = magnitude.flatten()

# Apply the threshold
mask = magnitude_flat > threshold
kx_filtered = kx_flat[mask]
ky_filtered = ky_flat[mask]
kz_filtered = kz_flat[mask]
magnitude_filtered = magnitude_flat[mask]


#Find peaks in 3D
peak_positions, peak_values = find_peaks_3d(magnitude,threshold = 1e-9 * np.max(magnitude) ,sigma=0.5)
peak_positions, peak_values = find_peaks_3d(magnitude,threshold = 0.04 ,sigma=0.5)

for pos, val in zip(peak_positions, peak_values):
    print(f"Peak at position {pos} with value {val}")

# Create a 3D scatter plot of the FFT magnitude
fig_fft = go.Figure(data=go.Scatter3d(
    x=kx_filtered,
    y=ky_filtered,
    z=kz_filtered,
    mode='markers',
    marker=dict(
        size=10,
        color=magnitude_filtered,
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title='Magnitude')
    )
))

fig_fft.update_layout(
    title="3D FFT Magnitude with Threshold",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

fig_fft.show()


# Assuming peak_positions and peak_values are already obtained
peak_kx = KX[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]
peak_ky = KY[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]
peak_kz = KZ[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]

# Create a 3D scatter plot of the peaks
fig_peaks = go.Figure(data=go.Scatter3d(
    x=peak_kx,
    y=peak_ky,
    z=peak_kz,
    mode='markers',
    marker=dict(
        size=10,
        color=peak_values,
        colorscale='Plasma',
        opacity=0.8,
        #colorbar=dict(title='Peak Magnitude')
        showscale=False
    )
))

fig_peaks.update_layout(
    title="3D FFT Peaks",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

fig_peaks.show()









#%%
# Define the range of z indices to analyze
z_indices = range(3,7)# Example range along the z-axis

# Store peak data for each voxel
voxel_peaks = {}

# Intialize combined figure
fig_combined=initialize_combined_figure(len(z_indices))

for plot_idx, z_idx in enumerate(z_indices):
    # Extract the voxel region
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft(region, use_vignette=True)
    fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold)
    fig_combined.add_trace(fig.data[0], row=plot_idx+1, col=1)
    fig_combined.add_trace(fig.data[1], row=plot_idx+1, col=1)
    fig_combined.add_trace(fig.data[2], row=plot_idx+1, col=1)
    # Find peaks in the 3D FFT magnitude
    peak_positions, peak_values = find_peaks_3d(magnitude, threshold=0.04, sigma=0.5)
    
    # Extract peak coordinates
    peak_kx = KX[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]
    peak_ky = KY[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]
    peak_kz = KZ[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]
    
    # Store peaks for this voxel
    voxel_peaks[z_idx] = {
        'positions': np.column_stack((peak_kx, peak_ky, peak_kz)),
        'values': peak_values
    }
    # Create a 3D scatter plot of the peaks
    fig_peaks = go.Figure(data=go.Scatter3d(
        x=peak_kx,
        y=peak_ky,
        z=peak_kz,
        mode='markers',
        marker=dict(
            size=10,
            color=peak_values,
            colorscale='Plasma',
            opacity=0.8,
            colorbar=dict(title='Peak Magnitude'),
            showscale=False
            )
        )
    )
    fig_combined.add_trace(fig_peaks.data[0], row=plot_idx+1, col=2)

# Compare peaks between neighboring voxels
for z_idx in z_indices[:-1]:
    current_peaks = voxel_peaks[z_idx]
    next_peaks = voxel_peaks[z_idx + 1]
    
    # Compare positions and intensities
    for i, (pos, val) in enumerate(zip(current_peaks['positions'], current_peaks['values'])):
        # Find matching peaks in the next voxel
        distances = np.linalg.norm(next_peaks['positions'] - pos, axis=1)
        match_idx = np.argmin(distances)
        if distances[match_idx] < 1:#some_threshold:  # Define a suitable threshold
            matched_pos = next_peaks['positions'][match_idx]
            matched_val = next_peaks['values'][match_idx]
            
            print(f"Voxel {z_idx} Peak {i} at {pos} with value {val}")
            print(f"Matches Voxel {z_idx + 1} Peak at {matched_pos} with value {matched_val}")
            print(f"Position Difference: {distances[match_idx]}")
            print(f"Intensity Difference: {abs(val - matched_val)}\n")


fig_combined.show()














#%%
orientation_tensor = compute_orientation_tensor(magnitude, KX, KY, KZ)
epsilon = 1e-8
orientation_tensor += np.eye(3) * epsilon
eigenvalues, eigenvectors = np.linalg.eigh(orientation_tensor)

# Scale eigenvectors by eigenvalues for arrow lengths
axes_lengths = np.sqrt(eigenvalues)

# Normalize axes_lengths for better visualization
max_length = np.max(axes_lengths)
if max_length > 0:
    axes_lengths = axes_lengths / max_length


# Use the center of the frequency domain for the voxel
kx_center = 0#KX.mean()
ky_center = 0#KY.mean()
kz_center = 0#KZ.mean()

# Add arrows to plot in Fourier space
for i in range(3):
    figE.add_trace(go.Cone(
        x=[kx_center], y=[ky_center], z=[kz_center],
        u=[eigenvectors[0, i] * axes_lengths[i]], 
        v=[eigenvectors[1, i] * axes_lengths[i]], 
        w=[eigenvectors[2, i] * axes_lengths[i]],
        sizemode="absolute",
        sizeref=0,  # Adjust this value as needed
        anchor="tail",
        colorscale='Viridis',
        showscale=False
    ))

figE.update_layout(
    title="Orientation Vectors in Fourier Space for First Voxel",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
    
)

figE.show()

print(f"eigenvalues: {eigenvalues}\nand\neigenvectors: {eigenvectors}")





#%%
#figE = go.Figure()
figE = plot_3D_tomogram(tomo_data, intensity_threshold=0.8)
vz, vy, vx = voxel_results['voxel_size']
# Process multiple voxels
# voxel_indices = [(6//2, 10//2, 6//2-1), (6//2, 10//2, 6//2), (6//2, 10//2, 6//2+1),
#                  (6//2, 10//2-1, 6//2), (6//2, 10//2+1, 6//2), (6//2-1, 10//2, 6//2),
#                  (6//2+1, 10//2+1, 6//2), (6//2-1, 10//2+1, 6//2), (6//2, 10//2+1, 6//2+1),
#                  (6//2-1, 10//2-1, 6//2), (6//2+1, 10//2-1, 6//2), (6//2, 10//2-1, 6//2-1)]  # Example indices

voxel_indices = generate_voxel_indices(
                                        (2//2,voxel_results['n_voxels'][0]-3//2),
                                        (4//2,voxel_results['n_voxels'][1]-6//2),
                                        (8//2,voxel_results['n_voxels'][2]-1//2)
                                       )


thetas = []
for z_idx, y_idx, x_idx in voxel_indices:
    # Compute orientation tensor and eigenvalues/eigenvectors
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft(region, use_vignette=True)
    orientation_tensor = np.nan_to_num(compute_orientation_tensor(magnitude, KX, KY, KZ))
    epsilon = 1e-8
    orientation_tensor += np.eye(3) * epsilon
    eigenvalues, eigenvectors = np.linalg.eigh(orientation_tensor)
    print(eigenvalues)

    # Calculate the angle of the first eigenvector
    theta = np.rad2deg(np.arccos(eigenvectors[2, 2] / np.linalg.norm(eigenvectors[2, :], axis=0)))
    thetas.append(theta)
    

# Normalize thetas for color mapping
theta_min = min(thetas)
theta_max = max(thetas)
normalized_thetas = [(theta - theta_min) / (theta_max - theta_min) for theta in thetas]


for idx, (z_idx, y_idx, x_idx) in enumerate(voxel_indices):
    # Compute orientation tensor and eigenvalues/eigenvectors
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft(region, use_vignette=True)
    orientation_tensor = np.nan_to_num(compute_orientation_tensor(magnitude, KX, KY, KZ))
    epsilon = 1e-8
    orientation_tensor += np.eye(3) * epsilon
    eigenvalues, eigenvectors = np.linalg.eigh(orientation_tensor)

    # Scale eigenvectors by eigenvalues for arrow lengths
    axes_lengths = np.sqrt(eigenvalues)

    # Normalize axes_lengths for better visualization
    max_length = np.max(axes_lengths)
    if max_length > 0:
        axes_lengths = (axes_lengths / max_length) * 20

    # Calculate the center of the voxel in real space
    voxel_center = np.array([x_idx * vx, y_idx * vy, z_idx * vz])

    # Use the normalized theta for color
    color_value = normalized_thetas[idx]

#     # Create a custom HSV colorscale
#     hsv_colorscale = [
#         [0.0, 'rgb(255, 0, 0)'],   # Red
#         [0.1667, 'rgb(255, 255, 0)'],  # Yellow
#         [0.3333, 'rgb(0, 255, 0)'],   # Green
#         [0.5, 'rgb(0, 255, 255)'],   # Cyan
#         [0.6667, 'rgb(0, 0, 255)'],   # Blue
#         [0.8333, 'rgb(255, 0, 255)'],  # Magenta
#         [1.0, 'rgb(255, 0, 0)']    # Red (wraps around)
# ]

    for i in range(3):
        figE.add_trace(go.Cone(
            x=[voxel_center[0]], y=[voxel_center[1]], z=[voxel_center[2]],
            u=[eigenvectors[0, i] * axes_lengths[i]], 
            v=[eigenvectors[1, i] * axes_lengths[i]], 
            w=[eigenvectors[2, i] * axes_lengths[i]],
            sizemode="absolute",
            sizeref=0,
            anchor="tail",
            showscale=(i == idx+i),  # Show scale only once
            opacity=color_value,
            colorbar=dict(title='Theta Angle'),
            colorscale='hsv',
        ))

figE.update_layout(
    title="Orientation Vectors in Real Space",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode='cube'
    ),
    width=800, height=800
)

figE.show()


'''
TEST FOR SINGLE VOXEL FINISHED
'''









#%%
total_fig_xy = 0
total_fig_yz = 0
total_fig_xz = 0
figE = go.Figure()
for x_idx in range(0, voxel_results['n_voxels'][2] - 1):
    z_idx, y_idx, x_idx = 6, 10, x_idx
    intensity_threshold = 0.5

    # Example usage:
    fft_threshold = 0

    # Plot voxel region
    fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold)
    if show_plots:
        fig_local.show()
        fig.show()
    # Print voxel statistics
    print(f"\nVoxel Statistics:")
    print(f"Mean intensity: {voxel_results['means'][z_idx, y_idx, x_idx]:.2f}")
    print(f"Max intensity: {voxel_results['maxes'][z_idx, y_idx, x_idx]:.2f}")
    print(f"Standard deviation: {voxel_results['stds'][z_idx, y_idx, x_idx]:.2f}")

    # Analyze voxel Fourier with projections
    fig_3d_fft, fig_xy, fig_yz, fig_xz = analyze_voxel_fourier(
        tomo_data, voxel_results, z_idx, y_idx, x_idx, fft_threshold=fft_threshold, use_vignette=True, overlay_octants=False
    )
    if show_plots:
        fig_3d_fft.show()
        fig_xy.show()
        fig_yz.show()
        fig_xz.show()
    total_fig_xy += fig_xy.data[0]['z']
    total_fig_yz += fig_yz.data[0]['z']
    total_fig_xz += fig_xz.data[0]['z']

    # Compute orientation tensor and eigenvalues/eigenvectors
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft(region, use_vignette=True)
    orientation_tensor = compute_orientation_tensor(magnitude, KX, KY, KZ)
    eigenvalues, eigenvectors = np.linalg.eigh(orientation_tensor)

    # Scale eigenvectors by eigenvalues for arrow lengths
    axes_lengths = np.sqrt(eigenvalues)
    ellipsoid_center = np.array([x_idx * vx, y_idx * vy, z_idx * vz])

    # Normalize axes_lengths for better visualization
    max_length = np.max(axes_lengths)
    if max_length > 0:
        axes_lengths = axes_lengths / max_length

    # Add arrows to plot
    for i in range(3):
        figE.add_trace(go.Cone(
            x=[ellipsoid_center[0]], y=[ellipsoid_center[1]], z=[ellipsoid_center[2]],
            u=[eigenvectors[0, i] * axes_lengths[i]], 
            v=[eigenvectors[1, i] * axes_lengths[i]], 
            w=[eigenvectors[2, i] * axes_lengths[i]],
            sizemode="absolute",
            sizeref=0.05,  # Adjust this value as needed
            anchor="tail",
            colorscale='Viridis',
            showscale=False
        ))

figE.update_layout(
    title="Orientation Vectors for Voxels",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode='cube'
    ),
    width=800, height=800
)

figE.show()

    

    
    
    
    
    
    
    
#%%
n_wedges=16
min_r=1.5
max_r=4
wedge_intensities = calculate_wedge_intensities_with_radius(total_fig_xz, num_wedges=n_wedges, min_radius=min_r, max_radius=max_r)    
visualize_wedges(total_fig_xz, num_wedges=n_wedges, min_radius=min_r, max_radius=max_r, show_first_wedge=False)    
plt.figure()
plt.plot(wedge_intensities)
plt.show()


#%%
# Extract 3D data from fig_3d_fft
array_3d = extract_3d_data_from_figure(fig_3d_fft)
num_azimuthal_wedges = 8
num_polar_wedges = 4
min_radius = 5
max_radius = 20
intensities = calculate_3d_wedge_intensities(array_3d, num_azimuthal_wedges, num_polar_wedges, min_radius, max_radius)


















#%%
# Calculate and plot octant intensities
# Extract voxel size from voxel_results
vz, vy, vx = voxel_results['voxel_size']

# Ensure you have the magnitude and frequency coordinates from the FFT
fft_3d = np.fft.fftn(tomo_data[z_idx*vz:(z_idx+1)*vz, y_idx*vy:(y_idx+1)*vy, x_idx*vx:(x_idx+1)*vx])
fft_3d_shifted = np.fft.fftshift(fft_3d)
magnitude = np.abs(fft_3d_shifted)

# Create frequency coordinates
kz = np.fft.fftshift(np.fft.fftfreq(vz))
ky = np.fft.fftshift(np.fft.fftfreq(vy))
kx = np.fft.fftshift(np.fft.fftfreq(vx))

# Create meshgrid
KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')

# Calculate octant intensities
octant_intensities = calculate_octant_intensities(magnitude, KX, KY, KZ)
# Calculate orientation vector based on octants
kx_vector, ky_vector, kz_vector = calculate_orientation_vector_from_octants(octant_intensities, KX, KY, KZ)

print(f"k_vector: {kx_vector}, {ky_vector}, {kz_vector}")

# Plot the orientation vector on the 3D FFT plot
plot_orientation_vector_on_fft(fig_3d_fft, kx_vector, ky_vector, kz_vector)

fig_3d_fft.show()
# %%
