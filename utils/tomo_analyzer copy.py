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
from tqdm import tqdm
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import colorsys

def create_hsv_colorscale(n_colors=100):
    colors = []
    for i in range(n_colors + 1):  # +1 to include both endpoints
        # Convert to HSV color (hue cycles from 0 to 1)
        hue = i / n_colors
        # Full saturation and value for vibrant colors
        hsv = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # Convert to RGB string format
        rgb = f'rgb({int(hsv[0]*255)},{int(hsv[1]*255)},{int(hsv[2]*255)})'
        colors.append([i/n_colors, rgb])
    # Add the first color again at the end to make it cyclic
    colors.append([1.0, colors[0][1]])
    return colors

def get_orientation_matrix(peak_positions, peak_values):
    """
    Convert peak positions to orientation matrix using weighted peaks
    """
    # Center the peaks around origin
    center = np.mean(peak_positions, axis=0)
    centered_peaks = peak_positions - center
    
    # Calculate covariance matrix with peak value weights
    weights = peak_values / np.sum(peak_values)
    cov_matrix = np.zeros((3, 3))
    for peak, weight in zip(centered_peaks, weights):
        cov_matrix += weight * np.outer(peak, peak)
    
    return cov_matrix

def get_axis_angle(voxel_peaks, voxel_values, ref_peaks, ref_values):
    """
    Calculate axis-angle representation between voxel orientation and reference orientation
    with improved eigenvector handling
    """
    # Get orientation matrices
    voxel_orient = get_orientation_matrix(voxel_peaks, voxel_values)
    ref_orient = get_orientation_matrix(ref_peaks, ref_values)
    
    # Get principal directions (eigenvectors) and sort by eigenvalues
    voxel_eigvals, voxel_eigvecs = np.linalg.eigh(voxel_orient)
    ref_eigvals, ref_eigvecs = np.linalg.eigh(ref_orient)
    
    # Sort eigenvectors by eigenvalue magnitude
    voxel_order = np.argsort(-np.abs(voxel_eigvals))
    ref_order = np.argsort(-np.abs(ref_eigvals))
    
    voxel_eigvecs = voxel_eigvecs[:, voxel_order]
    ref_eigvecs = ref_eigvecs[:, ref_order]
    
    # Try both orientations of each eigenvector to find best alignment
    best_rmsd = float('inf')
    best_R = None
    
    for flip_x in [-1, 1]:
        for flip_y in [-1, 1]:
            for flip_z in [-1, 1]:
                test_voxel_eigvecs = voxel_eigvecs.copy()
                test_voxel_eigvecs[:, 0] *= flip_x
                test_voxel_eigvecs[:, 1] *= flip_y
                test_voxel_eigvecs[:, 2] *= flip_z
                
                R = Rotation.align_vectors(ref_eigvecs.T, test_voxel_eigvecs.T)[0]
                rotated = R.apply(voxel_peaks)
                rmsd = np.sqrt(np.mean(np.sum((rotated - ref_peaks)**2, axis=1)))
                
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_R = R
    
    # Convert best rotation to axis-angle representation
    axis, angle = best_R.as_rotvec(degrees=True), np.linalg.norm(best_R.as_rotvec(degrees=True))
    
    return axis, angle, best_rmsd


def get_axis_angle_simple(voxel_peaks, voxel_values, ref_peaks, ref_values):
    """
    Calculate rotation (axis and angle) needed to align voxel peaks with reference peaks
    
    Args:
        voxel_peaks: peak positions for voxel FFT
        voxel_values: peak intensities for voxel FFT
        ref_peaks: reference peak positions
        ref_values: reference peak intensities
    
    Returns:
        axis: unit vector representing rotation axis
        angle: rotation angle in degrees (0-180)
        rmsd: root mean square deviation after alignment
    """
    # Sort peaks by intensity and use top N peaks
    N = min(6, len(voxel_peaks), len(ref_peaks))
    voxel_order = np.argsort(-voxel_values)[:N]
    ref_order = np.argsort(-ref_values)[:N]
    
    voxel_peaks = voxel_peaks[voxel_order]
    ref_peaks = ref_peaks[ref_order]
    voxel_values = voxel_values[voxel_order]
    ref_values = ref_values[ref_order]
    
    # #only take the top 2 peaks
    # voxel_peaks=voxel_peaks[:2]
    # ref_peaks=ref_peaks[:2]
    # voxel_values=voxel_values[:2]
    # ref_values=ref_values[:2]
    
    # Get principal directions for both sets of peaks
    voxel_eigvals, voxel_eigvecs = np.linalg.eigh(get_orientation_matrix(voxel_peaks, voxel_values))
    ref_eigvals, ref_eigvecs = np.linalg.eigh(get_orientation_matrix(ref_peaks, ref_values))
    
    # Sort eigenvectors by eigenvalue magnitude
    voxel_order = np.argsort(-np.abs(voxel_eigvals))
    ref_order = np.argsort(-np.abs(ref_eigvals))
    
    voxel_eigvecs = voxel_eigvecs[:, voxel_order]
    ref_eigvecs = ref_eigvecs[:, ref_order]
    
    # Try both possible alignments
    R1 = Rotation.align_vectors(ref_eigvecs.T, voxel_eigvecs.T)[0]
    R2 = Rotation.align_vectors(-ref_eigvecs.T, voxel_eigvecs.T)[0]
    
    rotated1 = R1.apply(voxel_peaks)
    rotated2 = R2.apply(voxel_peaks)
    
    rmsd1 = np.sqrt(np.mean(np.sum((rotated1 - ref_peaks)**2, axis=1)))
    rmsd2 = np.sqrt(np.mean(np.sum((rotated2 - ref_peaks)**2, axis=1)))
    
    # Choose the better alignment
    if rmsd1 < rmsd2:
        R = R1
        rmsd = rmsd1
    else:
        R = R2
        rmsd = rmsd2
    
    # Convert to axis-angle representation
    rotvec = R.as_rotvec(degrees=True)
    angle = np.linalg.norm(rotvec)
    
    # Normalize angle to 0-180 range and adjust axis accordingly
    if angle > 180:
        angle = 360 - angle
        axis = -rotvec / np.linalg.norm(rotvec)
    else:
        axis = rotvec / np.linalg.norm(rotvec)
    
    # Force consistent axis direction (z component should be positive for z-axis rotation)
    if axis[2] < 0:
        axis = -axis
        
    return axis, angle, rmsd


def load_cellinfo_data(file_path):
    """
    Load and extract arrays from the 'cellinfo' structure in the given .mat file.
    
    Args:
        file_path (str): Path to the .mat file.
        
    Returns:
        dict: A dictionary where keys are field names and values are the corresponding arrays.
    """
    
    # Load the .mat file
    mat_data = loadmat(file_path)
    
    # Extract the 'cellinfo' data
    cellinfo_data = mat_data.get('cellinfo')
    
    if cellinfo_data is None:
        raise ValueError("'cellinfo' key not found in the .mat file.")
    
    # Initialize a dictionary to store the extracted data
    data_dict = {}
    
    # Iterate through each field and extract its content
    for field_name in cellinfo_data.dtype.names:
        data_dict[field_name] = cellinfo_data[field_name][0, 0]
    
    return data_dict


# Add VTK saving functionality
def save_peaks_to_vtk(fig_RS, filename="fft_peaks.vtp"):
    """
    Save peaks from a Plotly figure to VTK format.
    
    Args:
        fig_RS: Plotly figure containing 3D scatter traces
        filename: output filename (should end in .vtp)
    """
    import vtk
    from vtk.util import numpy_support
    
    # Extract all points and their values from the Plotly figure
    all_points = []
    all_values = []
    all_voxel_coords = []
    
    for trace in fig_RS.data:
        # Extract coordinates and values
        x = np.array(trace.x)
        y = np.array(trace.y)
        z = np.array(trace.z)
        values = np.array(trace.marker.color)
        
        # Extract voxel coordinates from the trace name
        name_parts = trace.name.split(',')
        voxel_x = int(name_parts[0].split('=')[1])
        voxel_y = int(name_parts[1].split('=')[1])
        voxel_z = int(name_parts[2].split('=')[1])
        
        # Store points and values
        points = np.column_stack((x, y, z))
        all_points.append(points)
        all_values.append(values)
        all_voxel_coords.append(np.array([[voxel_x, voxel_y, voxel_z]] * len(x)))
    
    # Combine all data
    all_points = np.vstack(all_points)
    all_values = np.concatenate(all_values)
    all_voxel_coords = np.vstack(all_voxel_coords)
    
    # Create vtkPoints object
    vtk_points = vtk.vtkPoints()
    for point in all_points:
        vtk_points.InsertNextPoint(point)
    
    # Create vtkPolyData object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    
    # Create vertex cells
    vertices = vtk.vtkCellArray()
    for i in range(len(all_points)):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)
    polydata.SetVerts(vertices)
    
    # Add peak values as point data
    vtk_values = numpy_support.numpy_to_vtk(all_values)
    vtk_values.SetName("peak_intensity")
    polydata.GetPointData().AddArray(vtk_values)
    
    # Add normalized values
    normalized_values = all_values / np.max(all_values)
    vtk_normalized = numpy_support.numpy_to_vtk(normalized_values)
    vtk_normalized.SetName("normalized_intensity")
    polydata.GetPointData().AddArray(vtk_normalized)
    
    # Add voxel coordinates as point data
    for i, name in enumerate(['voxel_x', 'voxel_y', 'voxel_z']):
        vtk_coord = numpy_support.numpy_to_vtk(all_voxel_coords[:, i])
        vtk_coord.SetName(name)
        polydata.GetPointData().AddArray(vtk_coord)
    
    # Write to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"Saved {len(all_points)} peaks to {filename}")
    print("Data fields saved:")
    print("- peak_intensity: raw peak values")
    print("- normalized_intensity: values normalized to [0,1]")
    print("- voxel_x, voxel_y, voxel_z: voxel coordinates")


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

def find_peaks_3d_cutoff(magnitude, threshold=0.1, sigma=1, center_cutoff_radius=5):
    """
    Find peaks in 3D data with central region exclusion
    
    Args:
        magnitude: 3D numpy array
        threshold: relative threshold value (0-1)
        sigma: smoothing parameter for Gaussian filter
        center_cutoff_radius: radius (in pixels) around center to exclude
    
    Returns:
        peak_positions: array of peak coordinates
        peak_values: array of peak intensities
    """
    # Apply Gaussian filter to smooth the data
    smoothed = gaussian_filter(magnitude, sigma=sigma)
    
    # Create central cutoff mask
    center_z, center_y, center_x = np.array(magnitude.shape) // 2
    z, y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1], :magnitude.shape[2]]
    central_mask = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 > center_cutoff_radius**2
    
    # Apply threshold
    max_intensity = smoothed.max()
    threshold_value = max_intensity * threshold
    threshold_mask = smoothed > threshold_value
    
    # Use maximum filter to find local maxima
    local_max = maximum_filter(smoothed, size=3) == smoothed
    
    # Combine masks: threshold, local maxima, and central cutoff
    peaks = threshold_mask & local_max & central_mask
    
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


def calculate_peaks_com(peaks_x, peaks_y, peaks_z, peak_values):
    """Calculate center of mass of peaks weighted by their values"""
    if len(peaks_x) == 0:
        return None, None, None
    
    total_weight = np.sum(peak_values)
    if total_weight == 0:
        return None, None, None
    
    com_x = np.sum(peaks_x * peak_values) / total_weight
    com_y = np.sum(peaks_y * peak_values) / total_weight
    com_z = np.sum(peaks_z * peak_values) / total_weight
    
    return com_x, com_y, com_z


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
    phase = np.angle(fft_3d_shifted)
    
    
    kz = np.fft.fftshift(np.fft.fftfreq(region.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(region.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(region.shape[2]))
    
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    
    return magnitude, KX, KY, KZ
    #return phase, KX, KY, KZ


def compute_fft_q(region, use_vignette=False, pixel_size=1.0):
    """
    Compute 3D FFT and return magnitude and q-vectors in reciprocal space
    
    Args:
        region: 3D numpy array of real space data
        use_vignette: Boolean to apply vignette filter
        pixel_size: Real space pixel size in nanometers
    
    Returns:
        magnitude: FFT magnitude
        QX, QY, QZ: Q-vectors in reciprocal space (Å⁻¹)
    """
    if use_vignette:
        vignette = create_3d_vignette(region.shape)
        region_to_fft = region * vignette
    else:
        region_to_fft = region
    
    fft_3d = np.fft.fftn(region_to_fft)
    fft_3d_shifted = np.fft.fftshift(fft_3d)
    magnitude = np.abs(fft_3d_shifted)
    
    # Calculate reciprocal space frequencies
    kz = np.fft.fftshift(np.fft.fftfreq(region.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(region.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(region.shape[2]))
    
    # Convert to q-space (Å⁻¹)
    # q = 4π*sin(θ)/λ = 2π/d, where d is real space distance
    # For small angles: q ≈ 2π*θ/λ
    pixel_size_A = pixel_size  # Convert nm to Å
    qz = 2 * np.pi * kz / (pixel_size_A)
    qy = 2 * np.pi * ky / (pixel_size_A)
    qx = 2 * np.pi * kx / (pixel_size_A)
    #print('Q pixel size of FFT:', qx[1]-qx[0], qy[1]-qy[0], qz[1]-qz[0])
    
    QZ, QY, QX = np.meshgrid(qz, qy, qx, indexing='ij')
    
    return magnitude, QX, QY, QZ


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

def visualize_single_voxel_orientation(tomo_data, voxel_results, crystal_peaks, h, k, l, 
                                     z_idx, y_idx, x_idx,threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18):
    """
    Analyze orientation for a single voxel and overlay on tomogram
    Parameters:
        z_idx, y_idx, x_idx: indices of the voxel to analyze
    """
    # Analyze single voxel
    voxel_data = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    _, angle, rmsd = test_orientation_analysis(voxel_data, crystal_peaks, 
                                             z_idx, y_idx, x_idx, 
                                             h, k, l,threshold=threshold, sigma=sigma, cutoff=cutoff, pixel_size=pixel_size, 
                                             visualize=False)
    
    # Create coordinate arrays for full tomogram
    Z, Y, X = np.meshgrid(np.arange(tomo_data.shape[0]), 
                         np.arange(tomo_data.shape[1]), 
                         np.arange(tomo_data.shape[2]), 
                         indexing='ij')
    
    # Apply threshold to full tomogram
    full_max = tomo_data.max()
    full_threshold = full_max * 0.8
    full_mask = tomo_data > full_threshold
    q
    # Create figure
    fig = go.Figure()
    
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
        name='Background',
        showlegend=False
    ))
    
    # Add analyzed voxel
    vz, vy, vx = voxel_results['voxel_size']
    z_start, z_end = z_idx*vz, (z_idx+1)*vz
    y_start, y_end = y_idx*vy, (y_idx+1)*vy
    x_start, x_end = x_idx*vx, (x_idx+1)*vx
    
    # Extract voxel region directly using array indexing
    voxel_region = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]
    voxel_mask = voxel_region > full_threshold
    
    # Create coordinate arrays for the voxel
    z_coords, y_coords, x_coords = np.where(voxel_mask)
    
    # Adjust coordinates to global position
    z_coords += z_start
    y_coords += y_start
    x_coords += x_start
    
    if len(z_coords) > 0:
        # Create cyclic colorscale over 120 degrees
        cyclic_angle = angle % 120  # Make angle cyclic over 120 degrees
        normalized_angle = cyclic_angle / 120  # Normalize to [0,1]
        
        # Custom colorscale that's continuous from start to end
        colors = [
            [0, 'rgb(68,1,84)'],       # Viridis start
            [0.25, 'rgb(59,82,139)'],
            [0.5, 'rgb(33,144,141)'],
            [0.75, 'rgb(94,201,98)'],
            [1, 'rgb(68,1,84)']        # Back to start for continuity
        ]
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=[normalized_angle] * len(z_coords),  # Same angle for all points
                colorscale=colors,
                opacity=0.3,
                colorbar=dict(
                    title='Orientation Angle (°)',
                    tickmode='array',
                    ticktext=['0°', '30°', '60°', '90°', '120°'],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0]
                )
            ),
            name=f'Voxel ({z_idx},{y_idx},{x_idx})',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Orientation = {angle:.1f}°, RMSD = {rmsd:.2e}",
            y=0.95
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=2, z=2)),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000, height=800,
        showlegend=True
    )
    
    return fig


def visualize_line_orientation(tomo_data, voxel_results, crystal_peaks, h, k, l, 
                             z_idx, y_range, x_idx,threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18):
    """
    Analyze orientation for a line of voxels along y-axis
    Parameters:
        z_idx, x_idx: fixed indices for z and x
        y_range: range of y indices to analyze
    """
    # Create coordinate arrays for full tomogram
    Z, Y, X = np.meshgrid(np.arange(tomo_data.shape[0]), 
                         np.arange(tomo_data.shape[1]), 
                         np.arange(tomo_data.shape[2]), 
                         indexing='ij')
    
    # Apply threshold to full tomogram
    full_max = tomo_data.max()
    full_threshold = full_max * 0.7
    full_mask = tomo_data > full_threshold
    
    # Create figure
    fig = go.Figure()
    
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
        name='Background',
        showlegend=False
    ))
    
    # Process each voxel in the y-range
    vz, vy, vx = voxel_results['voxel_size']
    all_angles = []
    all_coords = []
    
    for y_idx in y_range:
        # Extract and analyze voxel
        voxel_data = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
        try:
            _, angle, rmsd = test_orientation_analysis(voxel_data, crystal_peaks, 
                                                     z_idx, y_idx, x_idx, 
                                                     h, k, l,threshold=threshold, sigma=sigma, cutoff=cutoff, pixel_size=pixel_size, 
                                                     visualize=False)
            
            # Get voxel coordinates
            z_start, z_end = z_idx*vz, (z_idx+1)*vz
            y_start, y_end = y_idx*vy, (y_idx+1)*vy
            x_start, x_end = x_idx*vx, (x_idx+1)*vx
            
            voxel_region = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]
            voxel_mask = voxel_region > full_threshold
            
            z_coords, y_coords, x_coords = np.where(voxel_mask)
            
            # Adjust coordinates to global position
            z_coords += z_start
            y_coords += y_start
            x_coords += x_start
            
            # Store results
            cyclic_angle = angle % 120
            normalized_angle = cyclic_angle / 120
            
            all_angles.extend([normalized_angle] * len(z_coords))
            all_coords.extend(zip(x_coords, y_coords, z_coords))
            
        except Exception as e:
            print(f"Error processing voxel (z={z_idx},y={y_idx},x={x_idx}): {e}")
            continue
    
    if all_coords:
        # Convert coordinates to arrays
        x_coords, y_coords, z_coords = zip(*all_coords)
        
        # Custom colorscale
        colors = [
            [0, 'rgb(68,1,84)'],
            [0.25, 'rgb(59,82,139)'],
            [0.5, 'rgb(33,144,141)'],
            [0.75, 'rgb(94,201,98)'],
            [1, 'rgb(68,1,84)']
        ]
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=all_angles,
                colorscale=colors,
                opacity=0.3,
                colorbar=dict(
                    title='Orientation Angle (°)',
                    tickmode='array',
                    ticktext=['0°', '30°', '60°', '90°', '120°'],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0]
                )
            ),
            name='Analyzed Voxels',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Orientation Analysis Along Y-axis (z={z_idx}, y={y_range[0]}-{y_range[-1]}, x={x_idx})",
            y=0.95
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=2, z=2)),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000, height=800,
        showlegend=True
    )
    
    return fig


def visualize_section_orientation(tomo_data, voxel_results, crystal_peaks, h, k, l, 
                                z_range, y_range, x_range, threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18,cyclic_period=None):
    """
    Analyze orientation for a 3D section of voxels
    Parameters:
        z_range, y_range, x_range: ranges of indices to analyze
        cyclic_period: if set, angles will be made cyclic with this period (in degrees)
                      if None, raw angles will be used
    """
    # Create coordinate arrays for full tomogram
    Z, Y, X = np.meshgrid(np.arange(tomo_data.shape[0]), 
                         np.arange(tomo_data.shape[1]), 
                         np.arange(tomo_data.shape[2]), 
                         indexing='ij')
    
    # Apply threshold to full tomogram
    full_max = tomo_data.max()
    full_threshold = full_max * 0.75
    full_mask = tomo_data > full_threshold
    
    # Create figure
    fig = go.Figure()
    
    # Add full tomogram points with low opacity
    fig.add_trace(go.Scatter3d(
        x=X[full_mask],
        y=Y[full_mask],
        z=Z[full_mask],
        mode='markers',
        marker=dict(
            size=2,
            color='gray',
            opacity=0.05,
        ),
        name='Background',
        showlegend=False
    ))
    
    # Process each voxel in the section
    vz, vy, vx = voxel_results['voxel_size']
    all_angles = []
    all_coords = []
    all_rmsds = []
    
    # Progress tracking
    total_voxels = len(z_range) * len(y_range) * len(x_range)
    processed = 0
    
    # Track angle range for normalization
    min_angle = float('inf')
    max_angle = float('-inf')
    
    for z_idx in z_range:
        for y_idx in y_range:
            for x_idx in x_range:
                processed += 1
                if processed % 10 == 0:
                    print(f"Processing voxel {processed}/{total_voxels}")
                
                voxel_data = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
                try:
                    _, angle, rmsd = test_orientation_analysis(voxel_data, crystal_peaks, 
                                                             z_idx, y_idx, x_idx, 
                                                             h, k, l,threshold=threshold, sigma=sigma, cutoff=cutoff, pixel_size=pixel_size, 
                                                             visualize=False)
                    
                    # Process angle based on cyclic_period
                    if cyclic_period is not None:
                        angle = angle % cyclic_period
                    
                    # Update angle range
                    min_angle = min(min_angle, angle)
                    max_angle = max(max_angle, angle)
                    
                    # Get voxel coordinates
                    z_start, z_end = z_idx*vz, (z_idx+1)*vz
                    y_start, y_end = y_idx*vy, (y_idx+1)*vy
                    x_start, x_end = x_idx*vx, (x_idx+1)*vx
                    
                    voxel_region = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]
                    voxel_mask = voxel_region > full_threshold
                    
                    z_coords, y_coords, x_coords = np.where(voxel_mask)
                    
                    # Adjust coordinates to global position
                    z_coords += z_start
                    y_coords += y_start
                    x_coords += x_start
                    
                    all_angles.extend([angle] * len(z_coords))
                    all_coords.extend(zip(x_coords, y_coords, z_coords))
                    all_rmsds.extend([rmsd] * len(z_coords))
                    
                except Exception as e:
                    print(f"Error processing voxel (z={z_idx},y={y_idx},x={x_idx}): {e}")
                    continue
    
    if all_coords:
        # Convert coordinates to arrays
        x_coords, y_coords, z_coords = zip(*all_coords)
        
        # Normalize angles to [0,1] for colorscale
        angle_range = max_angle - min_angle
        if angle_range > 0:
            normalized_angles = [(a - min_angle) / angle_range for a in all_angles]
        else:
            normalized_angles = [0.5] * len(all_angles)
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=normalized_angles,
                colorscale=create_hsv_colorscale(5),  # Creates 5 color stops plus the cyclic endpoint,
                opacity=0.05,
                colorbar=dict(
                    title='Orientation Angle (°)',
                    tickmode='array',
                    ticktext=[f'{min_angle:.0f}°', 
                             f'{(min_angle + angle_range/4):.0f}°',
                             f'{(min_angle + angle_range/2):.0f}°',
                             f'{(min_angle + 3*angle_range/4):.0f}°',
                             f'{max_angle:.0f}°'],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0]
                )
            ),
            name='Analyzed Voxels',
            showlegend=False
        ))
    
    # Update layout
    period_text = f" (Cyclic {cyclic_period}°)" if cyclic_period is not None else ""
    fig.update_layout(
        title=dict(
            text=f"Orientation Analysis{period_text} for Section:<br>z={z_range[0]}-{z_range[-1]}, y={y_range[0]}-{y_range[-1]}, x={x_range[0]}-{x_range[-1]}",
            y=0.95
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=2, z=2)),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000, height=800,
        showlegend=True
    )
    
    return fig

# Test case with visualization
def test_orientation_analysis(voxel_data, crystal_peaks, z_idx, y_idx, x_idx, h, k, l, threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18, visualize=True):
    """
    Test and visualize all three steps:
    1. Original peaks
    2. After (110) alignment
    3. After rotation around (110)
    """
    # Get voxel FFT peaks and setup (same as before)
    magnitude, KX, KY, KZ = compute_fft_q(voxel_data, use_vignette=True, pixel_size=pixel_size)
    voxel_peaks, voxel_values = find_peaks_3d_cutoff(magnitude, 
                                                    threshold=threshold,
                                                    sigma=sigma, 
                                                    center_cutoff_radius=cutoff)

    voxel_peaks_q = np.array([
        KX[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]],
        KY[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]],
        KZ[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]]
    ]).T

    
    # Get (110) direction and first rotation (same as before)
    crystal_110_idx = np.where((h == 1) & (k == 1) & (l == 0))[0][0]
    v2 = crystal_peaks[crystal_110_idx]
    v2_norm = v2 / np.linalg.norm(v2)
    
    strongest_idx = np.argmax(voxel_values)
    v1 = voxel_peaks_q[strongest_idx]
    v1_norm = v1 / np.linalg.norm(v1)
    
    # First rotation
    rot_axis1 = np.cross(v1_norm, v2_norm)
    if np.all(rot_axis1 == 0):
        R1 = Rotation.from_rotvec([0, 0, 0])
    else:
        rot_axis1 = rot_axis1 / np.linalg.norm(rot_axis1)
        cos_angle = np.dot(v1_norm, v2_norm)
        initial_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        R1 = Rotation.from_rotvec(rot_axis1 * initial_angle)
    
    # Apply first rotation
    aligned_peaks = R1.apply(voxel_peaks_q)
    
    # Second rotation around (110)
    best_angle = 0
    best_rmsd = float('inf')
    final_rotated = None
    
    for angle in np.arange(0, 360, 1):
        R2 = Rotation.from_rotvec(v2_norm * np.deg2rad(angle))
        rotated = R2.apply(aligned_peaks)
        
        total_rmsd = 0
        for rot_peak in rotated:
            dists = np.linalg.norm(crystal_peaks - rot_peak, axis=1)
            total_rmsd += np.min(dists)**2
        rmsd = np.sqrt(total_rmsd / len(rotated))
        
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_angle = angle
            final_rotated = rotated
    
    if visualize:
        # Create three subplots
        fig = make_subplots(rows=1, cols=3, 
                           subplot_titles=('Original Peaks', 
                                        'After (110) Alignment',
                                        f'After {best_angle:.1f}° Rotation'),
                           specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]])
        
        # Plot settings
        scale = 0.15
        sizes = 5 + (voxel_values - voxel_values.min()) / (voxel_values.max() - voxel_values.min()) * 15
        
        # Plot 1: Original peaks (same as before)
        fig.add_trace(
            go.Scatter3d(x=crystal_peaks[:,0], y=crystal_peaks[:,1], z=crystal_peaks[:,2],
                         mode='markers', marker=dict(size=8, color='blue', opacity=0.1),
                         name='Crystal Structure'), row=1, col=1)
        
        fig.add_trace(
            go.Scatter3d(x=voxel_peaks_q[:,0], y=voxel_peaks_q[:,1], z=voxel_peaks_q[:,2],
                         mode='markers', marker=dict(size=sizes, color='red', opacity=0.6),
                         name='Original Voxel Peaks'), row=1, col=1)
        
        # Plot 2: After (110) alignment (same as before)
        fig.add_trace(
            go.Scatter3d(x=crystal_peaks[:,0], y=crystal_peaks[:,1], z=crystal_peaks[:,2],
                         mode='markers', marker=dict(size=8, color='blue', opacity=0.1),
                         showlegend=False), row=1, col=2)
        
        fig.add_trace(
            go.Scatter3d(x=aligned_peaks[:,0], y=aligned_peaks[:,1], z=aligned_peaks[:,2],
                         mode='markers', marker=dict(size=sizes, color='orange', opacity=0.6),
                         name='(110) Aligned Peaks'), row=1, col=2)
        
        # Plot 3: After rotation around (110)
        fig.add_trace(
            go.Scatter3d(x=crystal_peaks[:,0], y=crystal_peaks[:,1], z=crystal_peaks[:,2],
                         mode='markers', marker=dict(size=8, color='blue', opacity=0.1),
                         showlegend=False), row=1, col=3)
        
        fig.add_trace(
            go.Scatter3d(x=final_rotated[:,0], y=final_rotated[:,1], z=final_rotated[:,2],
                         mode='markers', marker=dict(size=sizes, color='green', opacity=0.6),
                         name='Final Aligned Peaks'), row=1, col=3)
        
        # Add (110) axis to all plots
        for col in [1, 2, 3]:
            fig.add_trace(
                go.Scatter3d(x=[0, v2_norm[0] * scale], y=[0, v2_norm[1] * scale], z=[0, v2_norm[2] * scale],
                            mode='lines', line=dict(color='yellow', width=5),
                            name='(110) Axis' if col==1 else None,
                            showlegend=col==1), row=1, col=col)
        
        # Highlight strongest peak in all plots
        fig.add_trace(
            go.Scatter3d(x=[voxel_peaks_q[strongest_idx,0]], y=[voxel_peaks_q[strongest_idx,1]], 
                         z=[voxel_peaks_q[strongest_idx,2]], mode='markers',
                         marker=dict(size=15, color='purple', symbol='diamond'),
                         name='Strongest Peak'), row=1, col=1)
        
        fig.add_trace(
            go.Scatter3d(x=[aligned_peaks[strongest_idx,0]], y=[aligned_peaks[strongest_idx,1]], 
                         z=[aligned_peaks[strongest_idx,2]], mode='markers',
                         marker=dict(size=15, color='purple', symbol='diamond'),
                         showlegend=False), row=1, col=2)
        
        fig.add_trace(
            go.Scatter3d(x=[final_rotated[strongest_idx,0]], y=[final_rotated[strongest_idx,1]], 
                         z=[final_rotated[strongest_idx,2]], mode='markers',
                         marker=dict(size=15, color='purple', symbol='diamond'),
                         showlegend=False), row=1, col=3)
        
        fig.update_layout(
            title=f"Peak Alignment Steps (Voxel {z_idx},{y_idx},{x_idx})<br>RMSD: {best_rmsd:.3e} nm⁻¹",
            scene1=dict(aspectmode='cube'),
            scene2=dict(aspectmode='cube'),
            scene3=dict(aspectmode='cube'),
            width=1800
        )
        
          
        return fig, best_angle, best_rmsd
    else:
        return None, best_angle, best_rmsd



#%%
# Load the data
#tomogram = "/net/micdata/data2/12IDC/2024_Dec/misc/JM02_3D/ROI2_Ndp512_MLc_p10_gInf_Iter1000/recons/tomogram_alignment_recon_cropped_14nm_2.tif"
tomogram = "/net/micdata/data2/12IDC//2021_Nov/results/tomography/Sample6_tomo6_SIRT_tomogram.tif"
tomo_data = tifffile.imread(tomogram).swapaxes(1,2) #need to swap x and y axes for cell info to match the tomogram

#Rotate the tomogram
axis='z'
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

#magnitude, KX, KY, KZ = compute_fft(tomo_data, use_vignette=True)
magnitude, KX, KY, KZ=compute_fft_q(tomo_data, use_vignette=True, pixel_size=18)
   

# Define a threshold for the magnitude
threshold = 0.01 * np.max(magnitude)  # Example: 10% of the max magnitude

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


# Load and plot cell info
cellinfo_data = load_cellinfo_data("/home/beams/PTYCHOSAXS/cellinfo.mat")
h=np.array([1,1,0,0,-1,-1,0,0,1,-1,1,-1])
k=np.array([1,-1,1,1,-1,1,-1,-1,0,0,0,0])
l=np.array([0,0,1,-1,0,0,-1,1,0,1,-1,-1])
vs=[]

for i,h in enumerate(h):
    v=h*cellinfo_data['recilatticevectors'][0]+k[i]*cellinfo_data['recilatticevectors'][1]+l[i]*cellinfo_data['recilatticevectors'][2]
    vs.append(v)
    
vs=np.array(vs)
fig_fft.add_trace(go.Scatter3d(
    x=vs.T[0],
    y=vs.T[1],
    z=vs.T[2],
    mode='markers',
    marker=dict(size=10, color='red', opacity=0.8),
    name='Cell Info'
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




# Voxel size
# Cindy: pixel size is ~18nm
# Tomogram.shape = (179,185,162) -> (z,y,x)
pixel_size=18 #nm
limiting_axes=np.min(tomo_data.shape) #pixels
tomo_nm_size=pixel_size*limiting_axes #nm
n_unit_cells=tomo_nm_size//(cellinfo_data['Vol'][0][0]**(1/3)) #n unit cells / tomogram

print(f'~n unit cells per tomogram: {n_unit_cells}')

voxel_size = (12,12,12)  # cubic voxel size, pixels

print(f'~m unit cells per voxel: {n_unit_cells*voxel_size[0]/limiting_axes}')

voxel_results = analyze_tomogram_voxels(tomo_data, voxel_size=voxel_size)

# Print number of voxels in each dimension
print(f"Number of voxels (z, y, x): {voxel_results['n_voxels']}")

show_plots = False

# Define a threshold for the magnitude
threshold = 0.05 # Example: 5% of the max magnitude

# Peak finding threshold and sigma
peak_threshold=4e-2
sigma=.5





#%%

# Run test
# Load and plot cell info
cellinfo_data = load_cellinfo_data("/home/beams/PTYCHOSAXS/cellinfo.mat")
hs=np.array([1,1,0,0,-1,-1,0,0,1,-1,1,-1])
ks=np.array([1,-1,1,1,-1,1,-1,-1,0,0,0,0])
ls=np.array([0,0,1,-1,0,0,-1,1,0,1,-1,-1])
vs=[]

for i,h in enumerate(hs):
    v=hs[i]*cellinfo_data['recilatticevectors'][0]+ks[i]*cellinfo_data['recilatticevectors'][1]+ls[i]*cellinfo_data['recilatticevectors'][2]
    vs.append(v)
    
crystal_peaks = np.array(vs)  # Your existing vs array from cellinfo
z_idx, y_idx, x_idx=voxel_results['n_voxels'][0]//2, voxel_results['n_voxels'][1]//2, voxel_results['n_voxels'][2]//2
voxel_data = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, 0.8)
fig.show()

fig, angle, rmsd = test_orientation_analysis(rotate(voxel_data, 0, axes=(1,2), reshape=False), 
                                          crystal_peaks, z_idx, y_idx, x_idx, hs, ks, ls,threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18, visualize=True)
print(f"\nResults:")
print(f"Rotation angle around (110): {angle:.1f}°")
print(f"Final RMSD: {rmsd:.3e} nm⁻¹")

# Calculate angles between reciprocal lattice vectors
a_star = cellinfo_data['recilatticevectors'][0]
b_star = cellinfo_data['recilatticevectors'][1] 
c_star = cellinfo_data['recilatticevectors'][2]

# Calculate magnitudes
a_mag = np.linalg.norm(a_star)
b_mag = np.linalg.norm(b_star)
c_mag = np.linalg.norm(c_star)

# Calculate angles (in degrees)
ab_angle = np.arccos(np.dot(a_star, b_star)/(a_mag * b_mag)) * 180/np.pi
bc_angle = np.arccos(np.dot(b_star, c_star)/(b_mag * c_mag)) * 180/np.pi
ac_angle = np.arccos(np.dot(a_star, c_star)/(a_mag * c_mag)) * 180/np.pi

print("\nReciprocal Lattice Vector Magnitudes:")
print(f"||a*|| = {a_mag:.3f} nm⁻¹")
print(f"||b*|| = {b_mag:.3f} nm⁻¹")
print(f"||c*|| = {c_mag:.3f} nm⁻¹")

print("\nAngles between Reciprocal Lattice Vectors:")
print(f"a*^b* = {ab_angle:.1f}°")
print(f"b*^c* = {bc_angle:.1f}°")
print(f"a*^c* = {ac_angle:.1f}°")

# Plot reciprocal lattice vectors
scale = 1  # Scale factor for better visualization
fig.add_trace(go.Scatter3d(
    x=[0, a_star[0] * scale],
    y=[0, a_star[1] * scale],
    z=[0, a_star[2] * scale],
    mode='lines+text',
    line=dict(color='red', width=5),
    text=['', 'a*'],
    name='a* vector'
))

fig.add_trace(go.Scatter3d(
    x=[0, b_star[0] * scale],
    y=[0, b_star[1] * scale],
    z=[0, b_star[2] * scale],
    mode='lines+text',
    line=dict(color='green', width=5),
    text=['', 'b*'],
    name='b* vector'
))

fig.add_trace(go.Scatter3d(
    x=[0, c_star[0] * scale],
    y=[0, c_star[1] * scale],
    z=[0, c_star[2] * scale],
    mode='lines+text',
    line=dict(color='blue', width=5),
    text=['', 'c*'],
    name='c* vector'
))

fig.show()





#%%


# # Example usage:
# z_idx, y_idx, x_idx = 3, 3, 4  # Example voxel coordinates
# fig = visualize_single_voxel_orientation(tomo_data, voxel_results, crystal_peaks, 
#                                        hs, ks, ls, z_idx, y_idx, x_idx)
# fig.show()

# # Example usage:
# z_idx = 2
# x_idx = 4
# y_range = np.arange(0, 9)  # Analyze voxels y=1 through y=4
# fig = visualize_line_orientation(tomo_data, voxel_results, crystal_peaks, 
#                                hs, ks, ls, z_idx, y_range, x_idx)
# fig.show()



# Example usage:

# #12x12x12 pixel voxels
# z_range = np.arange(2, 10)
# y_range = np.arange(2, 10)
# x_range = np.arange(5, 12)

#15x15x15 pixel voxels
z_range = np.arange(0, voxel_results['n_voxels'][0]-1)
y_range = np.arange(0, voxel_results['n_voxels'][1]-1)
x_range = np.arange(0, voxel_results['n_voxels'][2]-1)


# With cyclic period
cyclic_period=120
fig = visualize_section_orientation(tomo_data, voxel_results, crystal_peaks, 
                                  hs, ks, ls, z_range, y_range, x_range,threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18, 
                                  cyclic_period=cyclic_period)
# Save as HTML file
fig.write_html(f"/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/orientation_analysis_{voxel_size[0]}x{voxel_size[1]}x{voxel_size[2]}_cyclic_period_{cyclic_period}.html")
#fig.show()

# # Without cyclic period (raw angles)
# fig = visualize_section_orientation(tomo_data, voxel_results, crystal_peaks, 
#                                   hs, ks, ls, z_range, y_range, x_range,
#                                   cyclic_period=None)
# fig.show()
























#%%
'''
TEST PEAK ANALYSISFOR SINGLE VOXEL
'''
vz, vy, vx = voxel_results['voxel_size']
figE = go.Figure()

# Process only the first voxel
z_idx, y_idx, x_idx = voxel_results['n_voxels'][0]//2, voxel_results['n_voxels'][1]//2, voxel_results['n_voxels'][2]//2
intensity_threshold_tomo = 0.5

# Compute orientation tensor and eigenvalues/eigenvectors
region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True, pixel_size=18)
fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold_tomo)
fig.show()

# Flatten the arrays
kx_flat = KX.flatten()
ky_flat = KY.flatten()
kz_flat = KZ.flatten()
magnitude_flat = magnitude.flatten()

# Apply the threshold
mask = magnitude_flat > threshold*np.max(magnitude)
kx_filtered = kx_flat[mask]
ky_filtered = ky_flat[mask]
kz_filtered = kz_flat[mask]
magnitude_filtered = magnitude_flat[mask]

#Find peaks in 3D
peak_positions, peak_values = find_peaks_3d_cutoff(magnitude,threshold = peak_threshold ,sigma=sigma, center_cutoff_radius=3)
voxel_peaks,voxel_values=peak_positions,peak_values

for pos, val in zip(peak_positions, peak_values):
    print(f"Peak at position {pos} with value {val}")

# Extract peak coordinates
voxel_peak_kx = KX[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]]
voxel_peak_ky = KY[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]]
voxel_peak_kz = KZ[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]]

# Create a 3D scatter plot of the FFT magnitude
fig_fft_ex = go.Figure(data=go.Scatter3d(
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

fig_fft_ex.update_layout(
    title="3D FFT Magnitude with Threshold",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

# Assuming peak_positions and peak_values are already obtained
peak_kx = KX[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]
peak_ky = KY[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]
peak_kz = KZ[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]

# Create a 3D scatter plot of the peaks
fig_peaks_ex = go.Figure(data=go.Scatter3d(
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

fig_peaks_ex.add_trace(go.Scatter3d(
    x=crystal_peaks.T[0],
    y=crystal_peaks.T[1],
    z=crystal_peaks.T[2],
    mode='markers',
    marker=dict(size=5, color='red', opacity=0.5),
    name='Cell Info'
))


fig_peaks_ex.update_layout(
    title="3D FFT Peaks",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

fig_peaks_ex.show()






















#%%
'''
VISUALIZING TOMOGRAM WITH RECIPROCAL SPACE PEAKS FOR MULTIPLE VOXELS
'''
# Calculate maximum magnitudes for each voxel for normalization
z_indices_all = range(0,10)
all_magnitudes = []
for plot_idx, z_idx in enumerate(z_indices_all):
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True, pixel_size=18)
    all_magnitudes.append(np.max(magnitude))

# Define the range of z indices to analyze
# Plots can only handle so many subplots, have to break up for memory sake

#z_indices = z_indices_all[:len(z_indices_all)//2]# Example range along the z-axis
z_indices = z_indices_all[len(z_indices_all)//2:]# Example range along the z-axis

# Store peak data for each voxel
voxel_peaks = {}

# Intialize combined figure
fig_combined=initialize_combined_figure(len(z_indices))

    
for plot_idx, z_idx in enumerate(z_indices):
    # Extract the voxel region
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True,pixel_size=18)
    fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold_tomo)
    fig_combined.add_trace(fig.data[0], row=plot_idx+1, col=1)
    fig_combined.add_trace(fig.data[1], row=plot_idx+1, col=1)
    fig_combined.add_trace(fig.data[2], row=plot_idx+1, col=1)
        
    # Find peaks in the 3D FFT magnitude
    peak_positions, peak_values = find_peaks_3d(magnitude, threshold=peak_threshold, sigma=sigma)
    
    # Extract peak coordinates
    peak_kx = KX[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]# + x_idx
    peak_ky = KY[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]# + y_idx
    peak_kz = KZ[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]# + z_idx
    
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
            size=5,
            color=peak_values,
            colorscale='Plasma',
            opacity=np.max(peak_values)/np.max(all_magnitudes),
            colorbar=dict(title='Peak Magnitude'),
            showscale=False
            )
        )
    )
    fig_combined.add_trace(fig_peaks.data[0], row=plot_idx+1, col=2)


    # Create a 3D scatter plot of the FFT magnitude

    # Flatten the arrays
    kx_flat = KX.flatten()
    ky_flat = KY.flatten()
    kz_flat = KZ.flatten()
    magnitude_flat = magnitude.flatten()

    # Apply the threshold
    mask = magnitude_flat > threshold*np.max(all_magnitudes)
    kx_filtered = kx_flat[mask]
    ky_filtered = ky_flat[mask]
    kz_filtered = kz_flat[mask]
    magnitude_filtered = magnitude_flat[mask]

    # Create a 3D scatter plot of the FFT magnitude
    fig_fft = go.Figure(data=go.Scatter3d(
        x=kx_filtered,
        y=ky_filtered,
        z=kz_filtered,
        mode='markers',
        marker=dict(
            size=2,
            color=magnitude_filtered,
            colorscale='Viridis',
            opacity=0.3,
            showscale=False
            #colorbar=dict(title='Magnitude')
        )
    ))
    fig_combined.add_trace(fig_fft.data[0], row=plot_idx+1, col=2)


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

'''
Plot FFT peaks for all voxels in the tomogram
'''
fig_RS = go.Figure()
voxel_results['n_voxels'][0]
# Define the range for all dimensions
z_indices = range(3, voxel_results['n_voxels'][0]-5)  # Adjust range as needed
y_indices = range(3, voxel_results['n_voxels'][1]-5)  # Adjust range as needed
x_indices = range(3, voxel_results['n_voxels'][2]-5)  # Adjust range as needed



# First pass to find maximum peak value across all voxels
max_peak_value = 0
for z_idx in tqdm(z_indices):
    for y_idx in y_indices:
        for x_idx in x_indices:
            region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
            magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True,pixel_size=18)
            peak_positions, peak_values = find_peaks_3d(magnitude, threshold=peak_threshold, sigma=sigma)
            
            if len(peak_values) > 0:
                # Filter out central peaks
                non_central_peaks = []
                for i, pos in enumerate(peak_positions):
                    peak_kx = KX[pos[0], pos[1], pos[2]]
                    peak_ky = KY[pos[0], pos[1], pos[2]]
                    peak_kz = KZ[pos[0], pos[1], pos[2]]
                    
                    # Check if peak is not at center (allowing for small numerical errors)
                    if not (abs(peak_kx) < 0.02 and abs(peak_ky) < 0.02 and abs(peak_kz) < 0.02):
                        non_central_peaks.append(i)
                
                if non_central_peaks:  # If there are non-central peaks
                    max_peak_value = max(max_peak_value, np.max(peak_values[non_central_peaks]))

# Second pass to plot peaks
for z_idx in tqdm(z_indices):
    for y_idx in y_indices:
        for x_idx in x_indices:
            # Extract the voxel region
            region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
            magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True,pixel_size=18)
            
            # Find peaks in the 3D FFT magnitude
            peak_threshold = 0.2
            peak_positions, peak_values = find_peaks_3d(magnitude, threshold=peak_threshold, sigma=0.5)
            
            if len(peak_positions) > 0:  # Only add traces if peaks were found
                for i in range(len(peak_positions)):
                    # Extract single peak coordinates
                    peak_kx = KX[peak_positions[i, 0], peak_positions[i, 1], peak_positions[i, 2]]
                    peak_ky = KY[peak_positions[i, 0], peak_positions[i, 1], peak_positions[i, 2]]
                    peak_kz = KZ[peak_positions[i, 0], peak_positions[i, 1], peak_positions[i, 2]]
                    
                    # Skip central peaks
                    if abs(peak_kx) < 0.01 and abs(peak_ky) < 0.01 and abs(peak_kz) < 0.01:
                        continue
                        
                    # Shift to voxel position
                    peak_kx += x_idx
                    peak_ky += y_idx
                    peak_kz += z_idx
                    
                    # Normalize peak value for opacity
                    normalized_opacity = peak_values[i] / max_peak_value
                    
                    # Add individual peak to the figure
                    fig_RS.add_trace(
                        go.Scatter3d(
                            x=[peak_kx],
                            y=[peak_ky],
                            z=[peak_kz],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=[peak_values[i]],
                                colorscale='Plasma',
                                opacity=normalized_opacity,
                                showscale=False
                            ),
                            name=f'x={x_idx},y={y_idx},z={z_idx},p={i}',
                            showlegend=False
                        )
                    )

# Update layout
fig_RS.update_layout(
    title=f"FFT Peaks in All Voxel Positions (excluding central peaks, max value: {max_peak_value:.2f})",
    scene=dict(
        aspectmode='cube',
        xaxis_title="KX + voxel_x",
        yaxis_title="KY + voxel_y",
        zaxis_title="KZ + voxel_z",
        camera=dict(
            eye=dict(x=2, y=2, z=2)
        )
    ),
    width=1000,
    height=1000
)

fig_RS.show()

# # Plot peak-based orientations with magnitude-based scaling
# for i, y_pos in enumerate(y_positions):
#     fig_all.add_trace(
#         go.Scatter3d(
#             x=[fixed_x, fixed_x + all_orientations_peak[i,0] * norm_magnitudes_peak[i]],
#             y=[y_pos, y_pos + all_orientations_peak[i,1] * norm_magnitudes_peak[i]],
#             z=[fixed_z, fixed_z + all_orientations_peak[i,2] * norm_magnitudes_peak[i]],
#             mode='lines',
#             line=dict(
#                 color='blue', 
#                 width=3,
#                 #opacity=0.7
#             ),
#             name='Peak orientation'
#         )
#     )

# Update layout
fig_all.update_layout(
    title=f"Orientation Vectors Along Y-axis (x={fixed_x}, z={fixed_z})<br>Arrow length scaled by intensity",
    scene=dict(
        aspectmode='cube',
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        camera=dict(eye=dict(x=2, y=2, z=2))
    ),
    width=1000,
    height=1000,
    showlegend=True
)

fig_all.show()

# Print average orientations and magnitudes
print("\nFFT Magnitude Method:")
print(f"Average orientation: ({np.mean(all_orientations_fft[:,0]):.3f}, {np.mean(all_orientations_fft[:,1]):.3f}, {np.mean(all_orientations_fft[:,2]):.3f})")
print(f"Average magnitude: {np.mean(all_magnitudes_fft):.3f}")

print("\nPeak-based Method:")
print(f"Average orientation: ({np.mean(all_orientations_peak[:,0]):.3f}, {np.mean(all_orientations_peak[:,1]):.3f}, {np.mean(all_orientations_peak[:,2]):.3f})")
print(f"Average magnitude: {np.mean(all_magnitudes_peak):.3f}")






#%%
'''
Plot orientation vectors for all voxels in the 3D volume
'''
# Create figure for all orientation vectors
fig_all = go.Figure()

# Store orientations and magnitudes for comparison
all_orientations = []
all_magnitudes = []
x_positions = []
y_positions = []
z_positions = []

# Analyze each voxel in the volume
for x_idx in range(0, voxel_results['n_voxels'][2]):
    for y_idx in range(0, voxel_results['n_voxels'][1]):
        for z_idx in range(0, voxel_results['n_voxels'][0]):
            # Extract and process voxel
            region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
            magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True,pixel_size=18)
            
            # ---- FFT Magnitude Method ----
            center_x = magnitude.shape[0] // 2
            center_y = magnitude.shape[1] // 2
            center_z = magnitude.shape[2] // 2
            
            z_coords, y_coords, x_coords = np.meshgrid(
                np.arange(magnitude.shape[2]) - center_z,
                np.arange(magnitude.shape[1]) - center_y,
                np.arange(magnitude.shape[0]) - center_x,
                indexing='ij'
            )
            
            # Create masks
            radius = 5
            r = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
            central_mask = r > radius
            upper_mask = KZ > 0
            
            max_val = np.max(magnitude)
            threshold = max_val * 0
            threshold_mask = magnitude > threshold
            combined_mask = central_mask & threshold_mask & upper_mask
            
            total_intensity = np.sum(magnitude * combined_mask)
            
            if total_intensity > 0:
                com_x = np.sum(KX * magnitude * combined_mask) / total_intensity
                com_y = np.sum(KY * magnitude * combined_mask) / total_intensity
                com_z = np.sum(KZ * magnitude * combined_mask) / total_intensity
                
                # Store orientation and magnitude
                all_orientations.append([com_x, com_y, com_z])
                all_magnitudes.append(total_intensity)
                x_positions.append(x_idx)
                y_positions.append(y_idx)
                z_positions.append(z_idx)

# Convert to numpy arrays
all_orientations = np.array(all_orientations)
all_magnitudes = np.array(all_magnitudes)
x_positions = np.array(x_positions)
y_positions = np.array(y_positions)
z_positions = np.array(z_positions)

# Normalize magnitudes for scaling
max_mag = np.max(all_magnitudes)
norm_magnitudes = all_magnitudes / max_mag * 2.0  # Reduced scale factor for better visibility

# Plot orientation vectors
for i in range(len(x_positions)):
    # Add arrow
    fig_all.add_trace(
        go.Scatter3d(
            x=[x_positions[i], x_positions[i] + all_orientations[i,0] * norm_magnitudes[i]],
            y=[y_positions[i], y_positions[i] + all_orientations[i,1] * norm_magnitudes[i]],
            z=[z_positions[i], z_positions[i] + all_orientations[i,2] * norm_magnitudes[i]],
            mode='lines',
            line=dict(
                color='red', 
                width=1  # Reduced width for better visibility
                #opacity=0.6  # Added opacity for better visualization of overlapping arrows
            ),
            name='FFT orientation',
            showlegend=False
        )
    )

# Update layout
fig_all.update_layout(
    title="3D Orientation Vectors (All Voxels)<br>Arrow length scaled by intensity",
    scene=dict(
        aspectmode='cube',
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    width=1000,
    height=1000,
    showlegend=False
)

fig_all.show()

# Print average orientation and magnitude
print("\nFFT Magnitude Method:")
print(f"Average orientation: ({np.mean(all_orientations[:,0]):.3f}, {np.mean(all_orientations[:,1]):.3f}, {np.mean(all_orientations[:,2]):.3f})")
print(f"Average magnitude: {np.mean(all_magnitudes):.3f}")

# Optional: Save orientations and positions to file
np.savez('orientation_data.npz', 
         orientations=all_orientations,
         magnitudes=all_magnitudes,
         x_pos=x_positions,
         y_pos=y_positions,
         z_pos=z_positions)





























