import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from scipy.signal import convolve2d as conv2np
from skimage.filters import window
import cupy as cp
from cupyx.scipy.signal import convolve2d as conv2
from cupyx.scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy.io import loadmat
from skimage.transform import resize
plt.rcParams['image.cmap'] = 'jet'
import inspect
import pdb
import random

def print_arg_types(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        print("Positional argument types:")
        for arg, param in zip(args, inspect.signature(func).parameters.values()):
            print(f"  {param.name}: {type(arg)}")
        print("Keyword argument types:")
        for key, value in kwargs.items():
            print(f"  {key}: {type(value)}")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} returned type: {type(result)}")
        return result
    return wrapper
    
#@print_arg_types
def create_lattice(lattice_type, num_points_x, num_points_y, num_points_z, spacing,zero_pad_factor=1,apply_window=False):
    """
    Create a lattice of points in a 3D space.

    Parameters:
    lattice_type (str): Type of the lattice ('SC', 'BCC', 'FCC').
    num_points_x (int): Number of points along the x-axis.
    num_points_y (int): Number of points along the y-axis.
    num_points_z (int): Number of points along the z-axis.
    spacing (float): Distance between adjacent points.

    Returns:
    cp.ndarray: Array of shape (N, 3) containing the lattice points, where N is the total number of points.
    """
    x_coords = cp.linspace(0, (num_points_x - 1) * spacing, num_points_x)
    y_coords = cp.linspace(0, (num_points_y - 1) * spacing, num_points_y)
    z_coords = cp.linspace(0, (num_points_z - 1) * spacing, num_points_z)
    
    xv, yv, zv = cp.meshgrid(x_coords, y_coords, z_coords)
    
    if lattice_type == 'SC':
        lattice_points = cp.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
    elif lattice_type == 'BCC':
        lattice_points = cp.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
        xv_bcc = xv + spacing / 2
        yv_bcc = yv + spacing / 2
        zv_bcc = zv + spacing / 2
        bcc_points = cp.column_stack([xv_bcc.ravel(), yv_bcc.ravel(), zv_bcc.ravel()])
        lattice_points = cp.vstack([lattice_points, bcc_points])
    elif lattice_type == 'FCC':
        lattice_points = cp.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
        fcc_points = cp.vstack([
            cp.column_stack([xv.ravel() + spacing / 2, yv.ravel() + spacing / 2, zv.ravel()]),
            cp.column_stack([xv.ravel() + spacing / 2, yv.ravel(), zv.ravel() + spacing / 2]),
            cp.column_stack([xv.ravel(), yv.ravel() + spacing / 2, zv.ravel() + spacing / 2])
        ])
        lattice_points = cp.vstack([lattice_points, fcc_points])
    else:
        raise ValueError("Unsupported lattice type. Supported types are: 'SC', 'BCC', 'FCC'")
        
        
#    # Apply vignette effect to the lattice
#    lattice_center = cp.array([
#        (num_points_x - 1) * spacing / 2,
#        (num_points_y - 1) * spacing / 2,
#        (num_points_z - 1) * spacing / 2,
#    ])
#    distances = cp.linalg.norm(lattice_points - lattice_center, axis=1)
#    max_distance = cp.linalg.norm(lattice_center)
#    
#    # Compute vignette weights
#    weights = cp.exp(-distances**2 / (0.5* max_distance**2))
#    vignette_points = lattice_points[weights > 0.25]  # Threshold to include points with sufficient weight
#    #pdb.set_trace()
#    
#    lattice_points=gaussian_filter(vignette_points,sigma=0.5)


    # Apply a window function to reduce edge effects
    if apply_window:
        lattice_center = cp.array([
            (num_points_x - 1) * spacing / 2,
            (num_points_y - 1) * spacing / 2,
            (num_points_z - 1) * spacing / 2,
        ])
        distances = cp.linalg.norm(lattice_points - lattice_center, axis=1)
        max_distance = cp.linalg.norm(lattice_center)
        
        # Apply a Hanning window
        weights = 0.5 * (1 + cp.cos(cp.pi * distances / max_distance))
        weights[distances > max_distance] = 0  # Set weights to zero outside the lattice bounds
        lattice_points = lattice_points[weights > 0.2]
    
    # Add zero-padding to the lattice
    if zero_pad_factor > 1:
        grid_size_x = num_points_x * zero_pad_factor
        grid_size_y = num_points_y * zero_pad_factor
        grid_size_z = num_points_z * zero_pad_factor
        
        padded_grid = cp.zeros((grid_size_x, grid_size_y, grid_size_z))
        for point in lattice_points:
            x, y, z = (point / spacing).astype(int)
            padded_grid[x % grid_size_x, y % grid_size_y, z % grid_size_z] += 1
        
        # Convert back to lattice points
        lattice_points = cp.argwhere(padded_grid > 0) * spacing
   
    
    return lattice_points
    

def form_factor_sphere(radius, q):
    """
    Form factor for a spherical particle.

    Parameters:
    radius (float): Radius of the sphere.
    q (cp.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    q_norm = cp.linalg.norm(q)
    if q_norm == 0:
        return 1.0
    return 3 * (cp.sin(q_norm * radius) - q_norm * radius * cp.cos(q_norm * radius)) / (q_norm**3 * radius**3)
    
def form_factor_cube(side_length, q):
    """
    Form factor for a cubic particle.

    Parameters:
    side_length (float): Side length of the cube.
    q (cp.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    q_norm = cp.linalg.norm(q)
    if q_norm == 0:
        return 1.0
    half_side = side_length / 2
    return cp.sinc(q[0] * half_side / cp.pi) * cp.sinc(q[1] * half_side / cp.pi) * cp.sinc(q[2] * half_side / cp.pi)


def form_factor_icosahedron(edge_length, q):
    """
    Form factor for an icosahedron particle.

    Parameters:
    edge_length (float): Edge length of the icosahedron.
    q (cp.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    # Volume of an icosahedron: V = (5/12) * (3 + sqrt(5)) * a^3
    volume = (5/12) * (3 + cp.sqrt(5)) * edge_length**3
    q_norm = cp.linalg.norm(q)
    if q_norm == 0:
        return volume
    # Approximate form factor calculation for an icosahedron
    f_q = volume * (cp.sin(q_norm * edge_length) / (q_norm * edge_length))
    return f_q
    
def form_factor_octahedron(edge_length, q):
    """
    Form factor for an octahedron particle.

    Parameters:
    edge_length (float): Edge length of the octahedron.
    q (cp.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    # Volume of an octahedron: V = (1/3) * sqrt(2) * a^3
    volume = (1/3) * cp.sqrt(2) * edge_length**3
    q_norm = cp.linalg.norm(q)
    if q_norm == 0:
        return volume
    # Approximate form factor calculation for an octahedron
    f_q = volume * (cp.sin(q_norm * edge_length) / (q_norm * edge_length))
    return f_q
    
def form_factor_tetrahedron(edge_length, q):
    """
    Form factor for a tetrahedron particle.

    Parameters:
    edge_length (float): Edge length of the tetrahedron.
    q (cp.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    # Volume of a tetrahedron: V = (1/12) * sqrt(2) * a^3
    volume = (1/12) * cp.sqrt(2) * edge_length**3
    q_norm = cp.linalg.norm(q)
    if q_norm == 0:
        return volume
    # Approximate form factor calculation for a tetrahedron
    f_q = volume * (cp.sin(q_norm * edge_length) / (q_norm * edge_length))
    return f_q

#@print_arg_types
def structure_factor(lattice_points, q):
    """
    Compute the structure factor for a given scattering vector.

    Parameters:
    lattice_points (cp.ndarray): Array containing the lattice points.
    q (cp.ndarray): The scattering vector.

    Returns:
    complex: The structure factor for the given scattering vector.
    """
    #pdb.set_trace()
    return cp.sum(cp.exp(1j * cp.dot(lattice_points, q)))
    #return cp.sum(cp.exp(1j * cp.dot(q,lattice_points)))

#@print_arg_types
def simulate_diffraction(lattice_points, q_values, form_factor_func, form_factor_params):
    """
    Simulate the X-ray diffraction pattern.

    Parameters:
    lattice_points (cp.ndarray): Array containing the lattice points.
    q_values (cp.ndarray): Array of scattering vectors.
    form_factor_func (function): The form factor function.
    form_factor_params (tuple): Parameters to pass to the form factor function.

    Returns:
    cp.ndarray: Array of intensity values corresponding to the scattering vectors.
    """
    intensities = cp.zeros(q_values.shape[0])
    for i, q in enumerate(tqdm(q_values,desc='Simulating diffraction')):
        #F_q = structure_factor(lattice_points, q)
        F_q = structure_factor(lattice_points.T, q)
        f_q = form_factor_func(*form_factor_params, q)
        intensities[i] = cp.abs(F_q)**2 * cp.abs(f_q)**2
            
    return intensities

def simulate_diffraction_optimized(lattice_points, q_values, form_factor_func, form_factor_params):
    """
    Optimized simulation of the X-ray diffraction pattern.

    Parameters:
    lattice_points (cp.ndarray): Array containing the lattice points.
    q_values (cp.ndarray): Array of scattering vectors.
    form_factor_func (function): The form factor function.
    form_factor_params (tuple): Parameters to pass to the form factor function.

    Returns:
    cp.ndarray: Array of intensity values corresponding to the scattering vectors.
    """
    # Axis mismathc on lattice
    lattice_points=lattice_points.T
    q_values=q_values.T
    
    # Precompute the form factors for all q_values
    f_q = form_factor_func(*form_factor_params, q_values)
    
    # Vectorized computation of the structure factor for all q_values
    def batch_structure_factor(lattice_points, q_values):
        """
        Compute the structure factor for all q_values in a vectorized manner.

        Parameters:
        lattice_points (cp.ndarray): Lattice points of the structure.
        q_values (cp.ndarray): Scattering vectors.

        Returns:
        cp.ndarray: Structure factors for each q.
        """
        phase_factors = cp.exp(1j * cp.dot(lattice_points,q_values))
        return cp.sum(phase_factors, axis=0)
    
    #pdb.set_trace()
    F_q = batch_structure_factor(lattice_points,q_values)
    
    # Compute intensities
    intensities = cp.abs(F_q)**2 * cp.abs(f_q)**2
    
    return intensities

def generate_q_values_2d(num_q, q_max):
    """
    Generate a grid of q values for the simulation in 2D.

    Parameters:
    num_q (int): Number of q values along each axis.
    q_max (float): Maximum value of q.

    Returns:
    cp.ndarray: Array of shape (num_q**2, 2) containing the q values.
    """
    q_coords = cp.linspace(-q_max, q_max, num_q)
    qx, qy = cp.meshgrid(q_coords, q_coords)
    q_values = cp.column_stack([qx.ravel(), qy.ravel()])
    return q_values

def project_to_2d_plane(q_values, q_z):
    """
    Project 3D q values to a 2D plane by keeping qz constant.

    Parameters:
    q_values (cp.ndarray): Array of 3D q values.
    q_z (float): The z-component of the scattering vector to keep constant.

    Returns:
    cp.ndarray: Array of 2D q values.
    """
    return cp.column_stack([q_values[:, 0], q_values[:, 1], cp.full(q_values.shape[0], q_z)])
    
#@print_arg_types
#def euler_rotation_matrix(phi, theta, psi):
#    """
#    Create a rotation matrix from Euler angles.

#    Parameters:
#    phi (float): Rotation angle around the z-axis.
#    theta (float): Rotation angle around the y-axis.
#    psi (float): Rotation angle around the x-axis.

#    Returns:
#    cp.ndarray: Rotation matrix of shape (3, 3).
#    """

#    pdb.set_trace()
#    Rz = cp.array([
#        [cp.cos(phi), -cp.sin(phi), 0],
#        [cp.sin(phi), cp.cos(phi), 0],
#        [0, 0, 1]
#    ], dtype=cp.float64)

#    Ry = cp.array([
#        [cp.cos(theta), 0, cp.sin(theta)],
#        [0, 1, 0],
#        [-cp.sin(theta), 0, cp.cos(theta)]
#    ], dtype=cp.float64)

#    Rx = cp.array([
#        [1, 0, 0],
#        [0, cp.cos(psi), -cp.sin(psi)],
#        [0, cp.sin(psi), cp.cos(psi)]
#    ], dtype=cp.float64)
#    
#    return Rz @ Ry @ Rx
    
def euler_rotation_matrix(phi, theta, psi):
    """
    Create a rotation matrix from Euler angles.

    Parameters:
    phi (float): Rotation angle around the z-axis.
    theta (float): Rotation angle around the y-axis.
    psi (float): Rotation angle around the x-axis.

    Returns:
    cp.ndarray: Rotation matrix of shape (3, 3).
    """

    a=float(cp.asnumpy(cp.cos(phi)))
    b=float(cp.asnumpy(cp.sin(phi)))
    aa=float(cp.asnumpy(cp.cos(theta)))
    bb=float(cp.asnumpy(cp.sin(theta)))
    aaa=float(cp.asnumpy(cp.cos(psi)))
    bbb=float(cp.asnumpy(cp.sin(psi)))
    
    Rz = cp.array([
        [a, -b, 0],
        [b, a, 0],
        [0, 0, 1]
    ], dtype=cp.float64)

    Ry = cp.array([
        [aa, 0, bb],
        [0, 1, 0],
        [-bb, 0, aa]
    ], dtype=cp.float64)

    Rx = cp.array([
        [1, 0, 0],
        [0, aaa, -bbb],
        [0, bbb, aaa]
    ], dtype=cp.float64)
    
    return Rz @ Ry @ Rx

#@print_arg_types
def rotate_lattice(lattice_points, phi, theta, psi):
    """
    Rotate the lattice points using Euler angles.

    Parameters:
    lattice_points (cp.ndarray): Array containing the lattice points.
    phi (float): Rotation angle around the z-axis.
    theta (float): Rotation angle around the y-axis.
    psi (float): Rotation angle around the x-axis.

    Returns:
    cp.ndarray: Rotated lattice points.
    """
    phi = float(cp.asnumpy(phi))
    theta = float(cp.asnumpy(theta))
    psi = float(cp.asnumpy(psi))
    rotation_matrix = euler_rotation_matrix(phi, theta, psi)
    rotated_points = cp.dot(lattice_points, rotation_matrix).T
    return rotated_points

def plot_side_by_side(lattice_points, q_values_2d, intensities, q_max):
    """
    Plot the 3D lattice and the 2D diffraction pattern side by side.

    Parameters:
    lattice_points (cp.ndarray): Array containing the lattice points.
    q_values_2d (cp.ndarray): Array of 2D scattering vectors.
    intensities (cp.ndarray): Array of intensity values.
    q_max (float): Maximum value of q.
    """
    fig = plt.figure(figsize=(12, 6))

    # Plot the 3D lattice
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(lattice_points[:, 0], lattice_points[:, 1], lattice_points[:, 2], c='b', marker='o')
    ax1.set_title('3D Lattice')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=90, azim=-90, roll=0)

    # Plot the 2D diffraction pattern
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(q_values_2d[:, 0], q_values_2d[:, 1], c=intensities, cmap='viridis', marker='.',norm=colors.LogNorm())
    plt.colorbar(scatter, ax=ax2, label='Intensity')
    ax2.set_title('2D X-ray Diffraction Pattern')
    ax2.set_xlabel('Qx')
    ax2.set_ylabel('Qy')
    ax2.set_xlim([-q_max, q_max])
    ax2.set_ylim([-q_max, q_max])
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
    

def hanning(image):
    #circular mask of radius=radius over image 
    xs=cp.hanning(image.shape[0])
    ys=cp.hanning(image.shape[1])
    temp=cp.outer(xs,ys)
    return temp
    
    
def rec_hanning_window(image,iterations):
    if iterations==1:
        return image * window('hann', image.shape)
    else:
        return rec_hanning_window(image * window('hann', image.shape),iterations-1)




def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def resize_cp(cupy_array,size):
    numpy_array = cp.asnumpy(cupy_array)
    resized_numpy = resize(numpy_array, (size,size), preserve_range=True,anti_aliasing=True)
    resized_cupy = cp.asarray(resized_numpy)
    return resized_cupy
    
    
    
if __name__=="__main__":
    # load probe
    # probe = cp.abs(cp.load('/home/beams/B304014/ptychosaxs/NN/probe_FT.npy'))
    #scan=312
    #probe = cp.abs(cp.fft.fftshift(cp.fft.fft2(loadmat(f'/mnt/micdata2/12IDC/2024_Dec/results/RC_01_/fly{scan}/roi0_Ndp512/MLc_L1_p10_gInf_Ndp256_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/MLs_L1_p10_g100_Ndp512_pc100_model_scale_asymmetry_rotation_shear_maxPosError200nm_noModelCon_bg0.1_vi_mm/Niter1000.mat')['probe'].T[0].T)))
    probe=loadmat("/net/micdata/data2/12IDC/2024_Dec/results/JM02_3D_/fly482/roi2_Ndp1024/MLc_L1_p10_gInf_Ndp256_mom0.5_pc100_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g400_Ndp512_mom0.5_pc400_noModelCon_bg0.1_vp4_vi_mm/Niter1000.mat")['probe'].T[0][0].T 
    #probe=loadmat('/mnt/micdata2/12IDC/2024_Dec/results/RC_01_/fly315/roi0_Ndp1024/MLc_L1_p10_g200_Ndp512_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/Niter2000.mat')['probe'].T[0].T
    
    #cupy probe
    probe=cp.asarray(probe)
    
    # resize probe
    dpsize=1408
    probe = resize_cp(probe,dpsize)
    
    # create ideal probe
    #image=hanning(probe)
    s=(48,48)
    image=hanning(cp.zeros(s))
    image=cp.pad(image,104,pad_with)

    # repeat hanning window to make a smaller and smaller probe
    # THIS CREATES ARTIFACTS IN THE DIFFRACTION PATTERN
    # hanning_probe=rec_hanning_window(hanning(probe),iterations=10)



    # plot hanning probes of different sizes
#    fig,ax=plt.subplots(2,2,layout='constrained')
#    ax[0][0].imshow(cp.abs(probe))
#    ax[0][1].imshow(cp.abs(image))
    #hanning_FT_image=cp.abs(cp.fft.fftshift(cp.fft.fft2(probe)))
    #hanning_FT_probe1=cp.abs(cp.fft.fftshift(cp.fft.fft2(image)))
    hanning_FT_image=cp.fft.fftshift(cp.fft.fft2(probe))
    hanning_FT_probe1=cp.fft.fftshift(cp.fft.fft2(image))
    #hanning_FT_probe=cp.abs(cp.fft.fftshift(cp.fft.fft2(hanning_probe)))
#    ax[1][0].imshow(cp.abs(hanning_FT_image)**2,norm=colors.LogNorm())
#    ax[1][1].imshow(cp.abs(hanning_FT_probe1)**2,norm=colors.LogNorm())
#    plt.show()


    # Parameters for the q values
    num_q = 1024  # Number of q values along each axis
    q_max = 4.0  # Maximum value of q

    #probeFT and resized
    #probeFT=cp.abs(cp.fft.fftshift(cp.fft.fft2(probe)))
    probeFT=cp.fft.fftshift(cp.fft.fft2(probe))
    probeFT=resize_cp(probeFT,num_q)
    
#    fig,ax=plt.subplots(1,2)
#    ax[0].imshow(cp.abs(probeFT)**2,norm=colors.LogNorm())
#    ax[1].imshow(cp.abs(probe))
#    plt.show()
    
    #pinhole
    #psf_pinhole=cp.abs(cp.load('/home/beams/B304014/ptychosaxs/NN/probe_pinhole.npy'))
    psf_pinhole=cp.load('/home/beams/B304014/ptychosaxs/NN/probe_pinhole.npy')
    psf_pinhole=resize_cp(psf_pinhole,num_q)

#    plt.figure()
#    plt.imshow(cp.abs(psf_pinhole)**2,norm=colors.LogNorm())
#    plt.show()
 
 
 
    plot=False
    dr=16
    save=True

    for i in tqdm(range(0,5000)):
        
        # Parameters for the lattice
        lattice_type = 'FCC'  # Type of lattice: 'SC', 'BCC', 'FCC'
        num_points_x =  cp.int64(12)    # Number of points along the x-axis
        num_points_y = cp.int64(12)   # Number of points along the y-axis
        num_points_z = cp.int64(12)     # Number of points along the z-axis
        spacing = cp.float64(8.0)         # Distance between adjacent points
        
        # Create the lattice
        #print('creating lattice...')
        lattice_points = create_lattice(lattice_type, num_points_x, num_points_y, num_points_z, spacing,zero_pad_factor=2,apply_window=True)
        

        # Rotate the lattice
        phi = cp.radians(random.randint(0,90))     # Rotation around the z-axis
        theta = cp.radians(random.randint(0,90))  # Rotation around the y-axis
        psi = cp.radians(random.randint(0,90))  # Rotation around the x-axis

        #print('rotating lattice...')
        rotated_lattice_points = rotate_lattice(lattice_points, phi, theta, psi)

        # Parameters for the form factor
        radius = spacing/2

        # Generate 2D q values
        #print('generating q values...')
        q_values_2d = generate_q_values_2d(num_q, q_max)

        # Project to 2D plane with a constant qz value (qz=0, projection approximation)
        q_z = 0.0
        q_values_3d = project_to_2d_plane(q_values_2d, q_z)

        # Simulate the diffraction pattern with specified form factor
        #print('simulating diffraction...')
        intensities = simulate_diffraction_optimized(rotated_lattice_points, q_values_3d, form_factor_sphere, (radius,))

        # CONVOLUTING THE DIFFRACTION PATTERNS WITH A FOCUSED PROBE
        # convert to pixel image
        intensities_image=intensities.reshape(num_q,num_q)
        
        
        # convolute probe diffraction pattern and simulated diffraction pattern
        probe=cp.asarray(probeFT)
        intensities_image=cp.asarray(intensities_image)
        hanning_FT_probe1=cp.asarray(hanning_FT_probe1)
        psf_pinhole=cp.asarray(psf_pinhole)
        
        #convDP = conv2(intensities_image,cp.abs(probe)**2,'same',boundary='symm')
        #convDP_ideal = conv2(intensities_image,cp.abs(psf_pinhole)**2,'same',boundary='symm')
        convDP_ideal = conv2(intensities_image,cp.abs(hanning_FT_probe1)**2,'same',boundary='symm')
        convDP_ideal = conv2(convDP_ideal,cp.abs(psf_pinhole)**2,'same',boundary='symm')
        convDP = conv2(convDP_ideal,cp.abs(probe)**2,'same',boundary='symm')
        
        ideal_DP=intensities_image.get()
        hanning_DP=convDP_ideal.get()
        convDP=convDP.get()
        psf_pinhole=psf_pinhole.get()
        probe=probe.get()
        hanning_FT_probe1=hanning_FT_probe1.get()

        # plot
        if plot:
            fig,ax=plt.subplots(2,3,layout='constrained');
            im1=ax[0][0].imshow(ideal_DP,norm=colors.LogNorm());
            im2=ax[0][1].imshow(convDP,norm=colors.LogNorm());
            im3=ax[0][2].imshow(hanning_DP,norm=colors.LogNorm());
            plt.colorbar(im1)
            plt.colorbar(im2)
            plt.colorbar(im3)
            ax[1][0].imshow(np.abs(psf_pinhole)**2,norm=colors.LogNorm())
            ax[1][1].imshow(np.abs(probe)**2,norm=colors.LogNorm())
            ax[1][2].imshow(np.abs(hanning_FT_probe1)**2,norm=colors.LogNorm())
            plt.show()
      
        if save:
            np.savez('/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/{}/output_hanning_conv_{:05d}.npz'.format(dr,i+1),pinholeDP=hanning_DP,convDP=convDP)
         



        # plot different form factors
    #    fig,ax=plt.subplots(1,5,layout='constrained')
    #    form_factor_list=[form_factor_sphere,form_factor_cube,form_factor_tetrahedron,form_factor_octahedron,form_factor_icosahedron]
    #    
    #    for i in tqdm(range(0,len(form_factor_list))):
    #        intensities = simulate_diffraction_optimized(rotated_lattice_points, q_values_3d, form_factor_list[i], (radius,))
    #        ax[i].scatter(q_values_2d[:, 0].get(), q_values_2d[:, 1].get(), c=intensities.get(), marker='.',norm=colors.LogNorm())

    #    plt.show()

