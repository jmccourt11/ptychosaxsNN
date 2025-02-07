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

def plot_3D_tomogram(tomo_data, intensity_threshold=0.1):
    """
    Create efficient 3D visualization of tomogram
    
    Args:
        tomo_data (np.ndarray): 3D tomographic data
        intensity_threshold (float): Threshold relative to max intensity (0-1)
    """
    # Create coordinate arrays
    nx, ny, nz = tomo_data.shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
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
            colorscale='Viridis',
            opacity=0.6,
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
            size=5,
            color=intensities_plot,
            colorscale='Viridis',
            opacity=0.6#,
            #colorbar=dict(title='Intensity', x=0.45)
        ),
        name='Voxel Region'
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
        name='Full Tomogram'
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
        name='Region Box'
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
            size=5,
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

def analyze_voxel_fourier(tomo_data, voxel_results, z_idx, y_idx, x_idx, fft_threshold=0.1):
    """
    Compute and visualize 3D Fourier transform of a voxel region
    
    Args:
        tomo_data (np.ndarray): 3D tomographic data
        voxel_results (dict): Results from analyze_tomogram_voxels
        z_idx, y_idx, x_idx (int): Indices of voxel to analyze
        fft_threshold (float): Threshold relative to max FFT magnitude (0-1)
    
    Returns:
        tuple: (Figure of 3D FFT, magnitude of FFT)
    """
    vz, vy, vx = voxel_results['voxel_size']
    
    # Extract voxel region
    region = tomo_data[z_idx*vz:(z_idx+1)*vz, 
                      y_idx*vy:(y_idx+1)*vy, 
                      x_idx*vx:(x_idx+1)*vx]
    
    # Compute 3D FFT
    fft_3d = np.fft.fftn(region)
    fft_3d_shifted = np.fft.fftshift(fft_3d)
    magnitude = np.abs(fft_3d_shifted)
    
    # Apply threshold
    max_magnitude = np.max(magnitude)
    threshold = max_magnitude * fft_threshold
    mask = magnitude > threshold
    
    # Create frequency coordinates
    kz = np.fft.fftshift(np.fft.fftfreq(region.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(region.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(region.shape[2]))
    
    # Create meshgrid for plotting
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    
    # Apply mask to coordinates and magnitudes
    KX_plot = KX[mask]
    KY_plot = KY[mask]
    KZ_plot = KZ[mask]
    magnitude_plot = magnitude[mask]
    
    # Create figure for 3D FFT
    fig_fft = go.Figure(data=[go.Scatter3d(
        x=KX_plot,
        y=KY_plot,
        z=KZ_plot,
        mode='markers',
        marker=dict(
            size=5,
            color=np.log10(magnitude_plot + 1),  # Log scale for better visualization
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title='Log Magnitude')
        )
    )])
    
    fig_fft.update_layout(
        title=f"3D FFT of Voxel (z={z_idx}, y={y_idx}, x={x_idx})",
        scene=dict(
            xaxis_title="kx",
            yaxis_title="ky",
            zaxis_title="kz",
            aspectmode='cube',
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        width=800, height=800
    )
    
    return fig_fft, magnitude

def analyze_voxel_fourier_with_projections(tomo_data, voxel_results, z_idx, y_idx, x_idx, fft_threshold=1e-3, use_vignette=False, overlay_octants=False):
    """
    Compute and visualize 3D Fourier transform of a voxel region with 2D projections
    
    Args:
        tomo_data (np.ndarray): 3D tomographic data
        voxel_results (dict): Results from analyze_tomogram_voxels
        z_idx, y_idx, x_idx (int): Indices of voxel to analyze
        fft_threshold (float): Threshold for FFT visualization
        use_vignette (bool): Whether to apply cosine window vignette before FFT
        overlay_octants (bool): Whether to overlay octant cuboids on the 3D FFT plot
    """
    mask_radius=voxel_results['voxel_size'][0]//10
    points_above_threshold=0#1e-5
    
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
        image=np.nan_to_num(image,nan=0.0)
        total_mass = np.sum(image)
        
        if total_mass == 0:
            raise ValueError("The total mass of the image is zero, cannot compute center of mass.")
        
        # Calculate center of mass
        x_com = np.sum(x_indices * image) / total_mass
        y_com = np.sum(y_indices * image) / total_mass
        
        return x_com, y_com
    
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
        threshold_mask = projection > (max_val * points_above_threshold)  # Using your threshold
        
        # Create circular mask to exclude the central region
        radius = mask_radius  # Equivalent to 12x12 square
        r = np.sqrt(x_coords**2 + y_coords**2)
        central_mask = r > radius
        mask = threshold_mask & central_mask
        
        if np.sum(mask) < 2:  # Need at least 2 points
            return None, None, None
        
        # Calculate center of mass
        x_com, y_com = calculate_center_of_mass(projection * mask)
        
        pdb.set_trace()
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
    
    def visualize_masked_projection(projection, kx, ky):
        """Create masked projection with circular mask"""
        center_x = len(kx) // 2
        center_y = len(ky) // 2
        
        # Create coordinate grids relative to center
        y_coords, x_coords = np.indices(projection.shape)
        x_coords = x_coords - center_x
        y_coords = y_coords - center_y
        
        # Create circular mask
        radius = mask_radius  # Same radius as in calculate_orientation
        r = np.sqrt(x_coords**2 + y_coords**2)
        mask = r > radius
        
        # Apply mask to show masked region in different color
        masked_proj = np.copy(projection)
        masked_proj[~mask] = np.nan  # Will show as different color
        
        return masked_proj
    
    def plot_center_of_mass(fig, angle, magnitude, freq_coords, image_size):
        """Add center of mass point and direction arrow to a figure"""
        if angle is not None:
            # Use the actual frequency coordinates
            x_com, y_com = freq_coords
            
            # Add scatter point for center of mass
            # fig.add_trace(
            #     go.Scatter(
            #         x=[x_com],
            #         y=[y_com],
            #         mode='markers',
            #         marker=dict(
            #             color='red',
            #             size=15,
            #             symbol='x'
            #         ),
            #         showlegend=False
            #     )
            # )
            
            # Calculate arrow length as 0.25 * size of image
            arrow_length = 0.25 * image_size/50
            
            # Calculate arrow end point based on direction
            x_end = x_com + arrow_length * np.cos(angle)
            y_end = y_com + arrow_length * np.sin(angle)
            
            # Add arrow to the plot
            fig.add_trace(
                go.Scatter(
                    #x=[x_com, x_end],
                    #y=[y_com, y_end],
                    x=[-x_end, x_end],
                    y=[-y_end, y_end],
                    mode='lines',
                    line=dict(color='red', width=4),
                    marker=dict(size=16, color='red', symbol='arrow'),
                    showlegend=False
                )
            )
            
            # Print debug info
            print(f"Center of Mass: ({x_com:.7f}, {y_com:.7f})")
            print(f"Angle: {np.degrees(angle):.1f}°")
    
    # Extract voxel region
    vz, vy, vx = voxel_results['voxel_size']
    region = tomo_data[z_idx*vz:(z_idx+1)*vz, 
                      y_idx*vy:(y_idx+1)*vy, 
                      x_idx*vx:(x_idx+1)*vx]
    
    # Apply vignette if requested
    if use_vignette:
        vignette = create_3d_vignette(region.shape)
        region_to_fft = region * vignette
        print("Applied 3D cosine window vignette")
    else:
        region_to_fft = region
    
    # Compute 3D FFT
    fft_3d = np.fft.fftn(region_to_fft)
    fft_3d_shifted = np.fft.fftshift(fft_3d)
    magnitude = np.abs(fft_3d_shifted)
    
    # Create frequency coordinates
    kz = np.fft.fftshift(np.fft.fftfreq(region.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(region.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(region.shape[2]))
    
    # Create meshgrid
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    
    # Apply threshold for 3D plot
    max_magnitude = np.max(magnitude)
    threshold = max_magnitude * fft_threshold
    mask = magnitude > threshold
    
    # 3D FFT plot
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
        title=f"3D FFT of Voxel (z={z_idx}, y={y_idx}, x={x_idx})",
        scene=dict(
            xaxis_title="kx",
            yaxis_title="ky",
            zaxis_title="kz",
            aspectmode='cube',
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        width=800, height=800
    )
    
    if overlay_octants:
        # Calculate octant intensities
        octant_intensities = calculate_octant_intensities(magnitude, KX, KY, KZ)

        # Overlay octant cuboids on the 3D FFT plot
        kx_center, ky_center, kz_center = 0, 0, 0  # Center in frequency space
        kx_max, ky_max, kz_max = KX.max(), KY.max(), KZ.max()
        kx_min, ky_min, kz_min = KX.min(), KY.min(), KZ.min()

        boundaries = {
            '+++': ((kx_center, ky_center, kz_center), (kx_max, ky_max, kz_max)),
            '++-': ((kx_center, ky_center, kz_min), (kx_max, ky_max, kz_center)),
            '+-+': ((kx_center, ky_min, kz_center), (kx_max, ky_center, kz_max)),
            '+--': ((kx_center, ky_min, kz_min), (kx_max, ky_center, kz_center)),
            '-++': ((kx_min, ky_center, kz_center), (kx_center, ky_max, kz_max)),
            '-+-': ((kx_min, ky_center, kz_min), (kx_center, ky_max, kz_center)),
            '--+': ((kx_min, ky_min, kz_center), (kx_center, ky_center, kz_max)),
            '---': ((kx_min, ky_min, kz_min), (kx_center, ky_center, kz_center))
        }

        intensities_norm = [(i - np.min(list(octant_intensities.values()))) / 
                            (np.max(list(octant_intensities.values())) - np.min(list(octant_intensities.values()))) 
                            for i in octant_intensities.values()]

        for idx, (key, intensity) in enumerate(octant_intensities.items()):
            (kx_start, ky_start, kz_start), (kx_end, ky_end, kz_end) = boundaries[key]
            x = [kx_start, kx_end, kx_end, kx_start, kx_start, kx_end, kx_end, kx_start]
            y = [ky_start, ky_start, ky_end, ky_end, ky_start, ky_start, ky_end, ky_end]
            z = [kz_start, kz_start, kz_start, kz_start, kz_end, kz_end, kz_end, kz_end]
            
            fig_3d.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                color='red',
                opacity=intensities_norm[idx] / 2,
                alphahull=0,
                name=f'Octant {key}'
            ))
    
    # Create 2D projections
    proj_xy = np.max(magnitude, axis=0)  # Project along z
    proj_xz = np.max(magnitude, axis=1)  # Project along y
    proj_yz = np.max(magnitude, axis=2)  # Project along x
    
    # Calculate orientations
    angle_xy, mag_xy, freq_xy = calculate_orientation(proj_xy, kx, ky)
    angle_xz, mag_xz, freq_xz = calculate_orientation(proj_xz, kx, kz)
    angle_yz, mag_yz, freq_yz = calculate_orientation(proj_yz, ky, kz)
    
    # Create separate figures for each projection
    fig_xy = go.Figure()
    fig_yz = go.Figure()
    fig_xz = go.Figure()
    
    # Add XY projection
    fig_xy.add_trace(
        go.Heatmap(z=visualize_masked_projection(proj_xy, kx, ky),
                   x=kx, y=ky,
                   colorscale='Viridis',
                   showscale=False)
    )
    plot_center_of_mass(fig_xy, angle_xy, mag_xy, freq_xy, proj_xy.shape[0])
    
    # Add YZ projection
    fig_yz.add_trace(
        go.Heatmap(z=visualize_masked_projection(proj_yz, ky, kz),
                   x=ky, y=kz,
                   colorscale='Viridis',
                   showscale=False)
    )
    plot_center_of_mass(fig_yz, angle_yz, mag_yz, freq_yz, proj_yz.shape[0])
    
    # Add XZ projection
    fig_xz.add_trace(
        go.Heatmap(z=visualize_masked_projection(proj_xz, kx, kz),
                   x=kx, y=kz,
                   colorscale='Viridis',
                   showscale=False)
    )
    plot_center_of_mass(fig_xz, angle_xz, mag_xz, freq_xz, proj_xz.shape[0])
    
    # Set consistent ranges for all plots
    max_range = max([max(abs(kx)), max(abs(ky)), max(abs(kz))])
    range_limit = [-max_range, max_range]
    
    # Update layouts
    fig_xy.update_layout(
        title="XY Projection",
        xaxis_title="kx",
        yaxis_title="ky",
        width=600, height=600,
        xaxis_range=range_limit,
        yaxis_range=range_limit
    )
    
    fig_yz.update_layout(
        title="YZ Projection",
        xaxis_title="ky",
        yaxis_title="kz",
        width=600, height=600,
        xaxis_range=range_limit,
        yaxis_range=range_limit
    )
    
    fig_xz.update_layout(
        title="XZ Projection",
        xaxis_title="kx",
        yaxis_title="kz",
        width=600, height=600,
        xaxis_range=range_limit,
        yaxis_range=range_limit
    )
    
    # Add zero lines to all plots
    for fig in [fig_xy, fig_yz, fig_xz]:
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='white')
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='white')
    
    # Print orientation information
    print("\nOrientation Analysis:")
    if angle_xy is not None:
        print(f"XY projection: {np.degrees(angle_xy):.1f}°, freq: ({freq_xy[0]:.7f}, {freq_xy[1]:.7f})")
    if angle_yz is not None:
        print(f"YZ projection: {np.degrees(angle_yz):.1f}°, freq: ({freq_yz[0]:.7f}, {freq_yz[1]:.7f})")
    if angle_xz is not None:
        print(f"XZ projection: {np.degrees(angle_xz):.1f}°, freq: ({freq_xz[0]:.7f}, {freq_xz[1]:.7f})")
    
    
    return fig_3d, fig_xy, fig_yz, fig_xz

def plot_orientation_projection(fig, kx_com, ky_com, kz_com, plane):
    """
    Add a line to the 2D plot to represent the projected orientation.
    
    Args:
        fig (go.Figure): Plotly figure.
        kx_com, ky_com, kz_com (float): Coordinates of the principal orientation.
        plane (str): The plane of projection ('xy', 'yz', 'xz').
    """
    if plane == 'xy':
        x_proj, y_proj = kx_com, ky_com
    elif plane == 'yz':
        x_proj, y_proj = ky_com, kz_com
    elif plane == 'xz':
        x_proj, y_proj = kx_com, kz_com
    else:
        raise ValueError("Invalid plane specified. Choose from 'xy', 'yz', 'xz'.")

    # Add a line to represent the orientation projection
    fig.add_trace(go.Scatter(
        x=[0, x_proj],
        y=[0, y_proj],
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=8, color='red', symbol='arrow'),
        showlegend=False
    ))

    # Add a cone to represent the orientation
    fig.add_trace(go.Cone(
        x=[0], y=[0], z=[0],
        u=[x_proj], v=[y_proj], w=[0],  # w=0 for 2D projection
        sizemode="absolute",
        sizeref=0.05,  # Adjust this value to change the size
        anchor="tail",
        colorscale='Reds',
        showscale=False
    ))

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

# Load the data
#tomogram = "/net/micdata/data2/12IDC/2024_Dec/misc/JM02_3D/ROI2_Ndp512_MLc_p10_gInf_Iter1000/recons/tomogram_alignment_recon_cropped_14nm_2.tif"
tomogram = "/net/micdata/data2/12IDC//2021_Nov/results/tomography/Sample6_tomo6_SIRT_tomogram.tif"
tomo_data = tifffile.imread(tomogram)

# Create and display the plot
fig = plot_3D_tomogram(tomo_data, intensity_threshold=0.8)
fig.show()

# # If you want the interactive version:
# @interact(intensity_threshold=FloatSlider(
#     min=0.01, max=1.0, step=0.01, value=0.1,
#     description='Threshold:'
# ))
# def update_plot(intensity_threshold):
#     fig = plot_3D_tomogram(tomo_data, intensity_threshold)
#     fig.show()
    
# Print dimensions
print(f"Tomogram shape: {tomo_data.shape}")

# Analyze voxels with smaller voxel size
#unit_cell_size=300
#pixel_size=14
#n_unit_cells=8
#voxel_size = (unit_cell_size//pixel_size*n_unit_cells, unit_cell_size//pixel_size*n_unit_cells, unit_cell_size//pixel_size*n_unit_cells)  # Reduced from (32, 32, 32)

voxel_size = (10,10,10)  # Reduced from (32, 32, 32)
voxel_results = analyze_tomogram_voxels(tomo_data, voxel_size=voxel_size)

# Print number of voxels in each dimension
print(f"Number of voxels (z, y, x): {voxel_results['n_voxels']}")

# Example: Plot a specific voxel region
#z_idx, y_idx, x_idx = 8//8, 12//8, 20//8  # Now z_idx can go higher
#intensity_threshold = 0.78

z_idx, y_idx, x_idx = 6, 10, 7  # Now z_idx can go higher
intensity_threshold = 0.5

# # Compute and plot FFT
# fig_fft, fft_magnitude = analyze_voxel_fourier(tomo_data, voxel_results, z_idx, y_idx, x_idx, fft_threshold=1e-4)#1e-3)
# fig_fft.show()

# # Print FFT statistics
# print(f"\nFFT Statistics:")
# print(f"Max magnitude: {np.max(fft_magnitude):.2f}")
# print(f"Mean magnitude: {np.mean(fft_magnitude):.2f}")
# print(f"Median magnitude: {np.median(fft_magnitude):.2f}")
# print(f"Number of points above threshold: {np.sum(fft_magnitude > np.max(fft_magnitude) * 0.3)}")

axis='z'
angle=0
if axis == 'x':
    rotated_data = rotate(tomo_data, angle, axes=(1, 2), reshape=False)
elif axis == 'y':
    rotated_data = rotate(tomo_data, angle, axes=(0, 2), reshape=False)
elif axis == 'z':
    rotated_data = rotate(tomo_data, angle, axes=(0, 1), reshape=False)

tomo_data = rotated_data

# Example usage:
fft_threshold = 1e-2

# Plot voxel region
fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold)
fig.show()
fig_local.show()

# Print voxel statistics
print(f"\nVoxel Statistics:")
print(f"Mean intensity: {voxel_results['means'][z_idx, y_idx, x_idx]:.2f}")
print(f"Max intensity: {voxel_results['maxes'][z_idx, y_idx, x_idx]:.2f}")
print(f"Standard deviation: {voxel_results['stds'][z_idx, y_idx, x_idx]:.2f}")

# Analyze voxel Fourier with projections
fig_3d_fft, fig_xy, fig_yz, fig_xz = analyze_voxel_fourier_with_projections(
    tomo_data, voxel_results, z_idx, y_idx, x_idx, fft_threshold, use_vignette=True, overlay_octants=False
)

fig_3d_fft.show()
fig_xy.show()
fig_yz.show()
fig_xz.show()

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
# # Print FFT statistics
# print(f"\nFFT Statistics:")
# print(f"Max magnitude: {np.max(fft_magnitude):.2f}")
# print(f"Mean magnitude: {np.mean(fft_magnitude):.2f}")
# print(f"Median magnitude: {np.median(fft_magnitude):.2f}")
# print(f"Number of points above threshold: {np.sum(fft_magnitude > np.max(fft_magnitude) * fft_threshold)}")

# Calculate orientation vector based on octants
kx_vector, ky_vector, kz_vector = calculate_orientation_vector_from_octants(octant_intensities, KX, KY, KZ)

print(f"k_vector: {kx_vector}, {ky_vector}, {kz_vector}")

# Plot the orientation vector on the 3D FFT plot
plot_orientation_vector_on_fft(fig_3d_fft, kx_vector, ky_vector, kz_vector)

fig_3d_fft.show()

# %%
