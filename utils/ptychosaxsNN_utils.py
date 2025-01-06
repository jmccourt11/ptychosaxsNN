import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.transform import resize
import torch
from tqdm import tqdm
import os
import h5py
import hdf5plugin
from scipy.ndimage import maximum_filter, label, find_objects
import scipy.fft as spf
import scipy.io as sio
from typing import List, Tuple
import pyFAI
from pathlib import Path
import re

def log10_custom(arr):
    # Create a mask for positive values
    positive_mask = arr > 0
    
    # Initialize result array
    result = np.zeros_like(arr, dtype=float)
    
    # Calculate log10 only on positive values
    log10_positive = np.log10(arr[positive_mask])
    
    # Find the minimum log10 value from the positive entries
    min_log10_value = log10_positive.min() if log10_positive.size > 0 else 0
    
    # Set positive entries to their log10 values
    result[positive_mask] = log10_positive
    
    # Set non-positive entries to the minimum log10 value
    result[~positive_mask] = min_log10_value
    
    return result

def set_path(path):
    return Path(path)
   
def create_circular_mask(image, center_x=0,center_y=0,radius=48):
    """
    Creates a circular mask at the center of the image.
    """
    # Get the dimensions of the image
    h, w = image.shape[:2]
    
    # Calculate the center of the image
    if center_x == 0 and center_y==0:
        center_x, center_y = w // 2, h // 2
    
    # Create a grid of x and y coordinates
    y, x = np.ogrid[:h, :w]
    
    # Calculate the distance from each pixel to the center
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Create the circular mask
    mask = distance_from_center >= radius
    
    return mask.astype(np.uint8)  # Return as uint8 (0s and 1s)

def replace_2d_array_values_by_row_indices(array, start, end):
    """
    Replaces values in a 2D numpy array with 0 if their row indices fall within the specified range.
    Used for masking diffraction patterns at edges where padding is used in ptycho recon
    
    Returns:
    np.ndarray: 2D numpy array with specified rows replaced with 0.

    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("The input must be a 2D numpy array")
    
    if not (0 <= start <= end < array.shape[0]):
        raise ValueError("Invalid range specified")

    array[start:end+1, :] = np.min(array)
    return array

def replace_2d_array_values_by_column_indices(array, start, end):
    """
    Replaces values in a 2D numpy array with 0 if their row indices fall within the specified range.
    Used for masking diffraction patterns at edges where padding is used in ptycho recon

    Returns:
    np.ndarray: 2D numpy array with specified rows replaced with 0.

    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("The input must be a 2D numpy array")
    
    if not (0 <= start <= end < array.shape[0]):
        raise ValueError("Invalid range specified")

    array[:,start:end+1] = np.min(array)
    return array
    
def preprocess_cindy(dp):#,probe):
    size=256
    dp_pp=dp
    #probe_sub=abs(spf.fftshift(spf.fft2(probe)))**2
    #dp_pp=dp-probe_sub
    dp_pp=np.asarray(replace_2d_array_values_by_column_indices(replace_2d_array_values_by_column_indices(replace_2d_array_values_by_row_indices(replace_2d_array_values_by_row_indices(dp_pp,0,16),495,511),0,16),495,511))
    dp_pp=log10_custom(dp_pp)
    #dp_pp[np.isnan(dp_pp)] = 0
    #dp_pp[dp_pp <= 0] = np.min(dp_pp[dp_pp > 0])# small positive value
    dp_pp=np.asarray(resize(dp_pp[:,:],(size,size),preserve_range=True,anti_aliasing=True))
    #dp_pp=np.log10(dp_pp)

    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))
    return dp_pp,sf,bkg

def preprocess_zhihua2(dp,mask,center_decay=2):#,probe):
    size=512
    dp_pp=dp
    #probe_sub=abs(spf.fftshift(spf.fft2(probe)))**2
    #dp_pp=dp-probe_sub
    #mask=mask#+1e-9
    dp_pp=np.asarray(dp_pp*mask)
    #mask=resize(mask,(size,size),preserve_range=True,anti_aliasing=True)+1e-3

    #dp_pp[np.isnan(dp_pp)] = 0
    #dp_pp[dp_pp <= 0] = np.min(dp_pp[dp_pp > 0])# small positive value
    dp_pp=np.asarray(resize(dp_pp,(size,size),preserve_range=True,anti_aliasing=True))
    #dp_pp=np.asarray(dp_pp*mask)
    dp_pp=log10_custom(dp_pp)
    
    #dp_pp=np.log10(dp_pp)

    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))
    #print(dp_pp.shape)
    #dp_pp=vignette_transform(dp_pp, center_decay=center_decay)
    #print(dp_pp.shape)
    return dp_pp,sf,bkg


def preprocess_zhihua(dp,mask,center_decay=2):#,probe):
    size=512
    dp_pp=dp
    dp_pp=np.asarray(dp_pp*mask)

    dp_pp=np.asarray(resize(dp_pp,(size,size),preserve_range=True,anti_aliasing=True))
    dp_pp=log10_custom(dp_pp)

    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))

    return dp_pp,sf,bkg

def generate_weight_mask(shape, center_decay):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    weight_mask = (distance / max_distance) ** center_decay
    return weight_mask

def vignette_transform(image, center_decay=2):
    h, w = image.shape[-2:]
    weight_mask = generate_weight_mask((h, w), center_decay)
    weight_mask = torch.tensor(weight_mask, dtype=image.dtype, device=image.device).unsqueeze(0)
    return image * weight_mask

def preprocess_chansong(dp,probe):
    lbound,ubound=(23,38),(235,250)
    size=256
    dp_pp=dp
    #dp_pp=np.asarray(replace_2d_array_values_by_row_indices(replace_2d_array_values_by_row_indices(dp_pp,ubound[0],ubound[1]),lbound[0],lbound[1]))
    dp_pp=log10_custom(dp_pp)
    dp_pp=np.asarray(resize(dp_pp[:,:],(size,size),preserve_range=True,anti_aliasing=True))
    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))
    probe=resize_dp(probe)
    dp_probe=torch.tensor(probe.reshape(1,1,size,size))
    dp_pp=torch.cat([dp_pp, dp_probe], dim=1)
    return dp_pp,sf,bkg

def invert_preprocess_cindy(dp,sf,bkg):
    dp_rec=dp*sf + bkg
    dp_rec=10**(dp_rec)
    return dp_rec
        
def plot_and_save_scan(dps,ptycho_scan,scanx=20,scany=15):
    
    fig, axs = plt.subplots(scany,scanx, sharex=True,sharey=True,figsize=(scanx,scany))

    # Remove vertical space between Axes
    fig.subplots_adjust(hspace=0,wspace=0)
    count=0
    inputs=[]
    outputs=[]
    sfs=[]
    bkgs=[]
    for i in tqdm(range(0,scany)):
        for j in range(0,scanx):
            dp_count=np.asarray(dps[count][1:513,259:771])
          
            dp_count_copy=dp_count.copy()
            result,sf,bkg=preprocess_cindy(dp_count_copy)
            resulta=result.to(device=ptycho_scan.device, dtype=torch.float)

            result=ptycho_scan.model(resulta).detach().to("cpu").numpy()[0][0]
            im=axs[i][j].imshow(result)
            axs[i][j].imshow(result)
            axs[i][j].axis("off")

            outputs.append(result)
            sfs.append(sf)
            bkgs.append(bkg)
            inputs.append(resulta.detach().to("cpu").numpy()[0][0])
         
            count+=1
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    return inputs,outputs,sfs,bkgs
        
def resize_dp(dp):
    return resize(dp,(256,256),preserve_range=True,anti_aliasing=True)

def norm_0to1(image):
    return np.asarray((image-np.min(image))/(np.max(image)-np.min(image)))
    
def find_peaks2d(dp, center_cut=100, n=25, threshold=0.3, plot=True):
    peaks = []
    
    # Define the shape of the image
    rows, cols = dp.shape
    
    # Calculate half of the neighborhood size
    half_n = n // 2
    
    # Iterate over each pixel, excluding border pixels based on neighborhood size
    for i in range(half_n, rows - half_n):
        for j in range(half_n, cols - half_n):
            # Extract the nxn neighborhood of the current pixel
            neighborhood = dp[i-half_n:i+half_n+1, j-half_n:j+half_n+1]
            
            # Coordinates of the center of the image
            center_x, center_y = rows // 2, cols // 2
            
            # Check if the center pixel is greater than all its neighbors, above the threshold, and unique
            if dp[i, j] > threshold and dp[i, j] == np.max(neighborhood) and np.count_nonzero(dp[i, j] == neighborhood) == 1:
                if (i-center_x)**2 + (j-center_y)**2 > center_cut**2:
                    peaks.append((i, j))
    if plot:
        fig,ax=plt.subplots()
        im=ax.imshow(dp, cmap='jet', interpolation='nearest')
        peak_y, peak_x = zip(*peaks)
        ax.scatter(peak_x, peak_y, color='red', marker='x', s=100, label='Peaks')
        plt.colorbar(im)
        plt.show()

    return peaks

def find_peaks_2d_filter(diffraction_pattern, center_cut=25,n=25, threshold=0.1,plot=True):
    """
    Find peaks in a 2D diffraction pattern.

    Parameters:
    diffraction_pattern (ndarray): The 2D diffraction pattern as a NumPy array.
    n(int): The size of the neighborhood for local maximum detection. Default is 5.
    threshold (float): Optional intensity threshold. Peaks below this value will be ignored.

    Returns:
    peaks (list of tuples): A list of (row, col) indices of the detected peaks.
    """
    # Apply a maximum filter to identify local maxima
    local_max = maximum_filter(diffraction_pattern, size=n) == diffraction_pattern
    
    # Apply an optional intensity threshold to remove low-intensity peaks
    if threshold is not None:
        local_max &= diffraction_pattern > threshold
    
    # Label the peaks
    labeled, num_objects = label(local_max)
    
    # Extract the coordinates of the peaks
    slices = find_objects(labeled)
    peaks = [(int((s[0].start + s[0].stop - 1) / 2), int((s[1].start + s[1].stop - 1) / 2)) for s in slices]
    
    # Find the center of the diffraction pattern
    center_row, center_col = np.array(diffraction_pattern.shape) // 2
    
    # Filter out peaks within the specified radius from the center
    filtered_peaks = []
    for peak in peaks:
        distance_from_center = np.sqrt((peak[0] - center_row) ** 2 + (peak[1] - center_col) ** 2)
        if distance_from_center > center_cut:
            filtered_peaks.append(peak)

    return filtered_peaks

def neighborhood_intensity(image, x, y, radius=5):
    """
    Calculate the sum of pixel intensities in a square neighborhood around (x, y).
    The neighborhood is a square of side 2*radius+1 centered on (x, y).
    
    :param image: 2D NumPy array representing the image
    :param x: X-coordinate of the peak
    :param y: Y-coordinate of the peak
    :param radius: The radius around the peak to define the neighborhood
    :return: Integrated intensity within the neighborhood
    """
    # Define the neighborhood bounds, ensuring they stay within image limits
    x_min = max(0, x - radius)
    x_max = min(image.shape[0], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(image.shape[1], y + radius + 1)
    
    # Extract the neighborhood and calculate the sum of intensities
    neighborhood = image[x_min:x_max, y_min:y_max]
    return np.sum(neighborhood)

def circular_neighborhood_intensity(image, x, y, radius=5,plot=True):
    x_min = max(0, x - radius)
    x_max = min(image.shape[0], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(image.shape[1], y + radius + 1)
    

    # Create a meshgrid of coordinates in the neighborhood
    X, Y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
    
    # Calculate the Euclidean distance from the peak (x, y)
    distances = np.sqrt((X - x)**2 + (Y - y)**2)
    
    # Mask for points within the radius
    mask = distances <= radius
    
    # Sum the pixel intensities within the mask
    neighborhood = image[x_min:x_max, y_min:y_max]
    if plot:    
        # Plot the original image with the circular neighborhood overlay
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='jet',norm=colors.LogNorm(),clim=(1,1000))
        
        # Create a circle patch to show the circular neighborhood on the image
        circle = plt.Circle((y, x), radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # Mark the center point
        ax.plot(y, x, 'ro')
        
        # Set labels and title
        ax.set_title(f'Circular Neighborhood (Center: [{x}, {y}], Radius: {radius})')
        plt.show()
    
    return np.sum(neighborhood[mask])
    
def read_hdf5_file(file_path):
    """
    Reads an HDF5 file and returns its contents.

    Parameters:
    file_path (str): The path to the HDF5 file.

    Returns:
    dict: A dictionary with dataset names as keys and their data as values.
    """
    data_dict = {}

    try:
        with h5py.File(file_path, 'r') as hdf_file:
            def extract_data(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data_dict[name] = obj[()]

            hdf_file.visititems(extract_data)
    except Exception as e:
        print(f"An error occurred: {e}")

    return data_dict

def find_directories_with_number(base_path, number):
    """
    Finds immediate subdirectories containing a specific number in their name,
    allowing for flexible number formatting.

    Args:
    - base_path (str): The path to the directory to search.
    - number (int): The number to search for in subdirectory names.

    Returns:
    - list: A list of matching directory paths.
    """
    matching_dirs = []
    # Create a regex pattern to match the number with optional leading zeros anywhere in the name
    #number_pattern = rf"0*{number}\b"
    number_pattern = rf"(^|[^0-9])0*{number}([^0-9]|$)"

    try:
        # List only directories in the base path
        for entry in os.listdir(base_path):
            full_path = os.path.join(base_path, entry)
            # Check if the entry is a directory and matches the pattern
            if os.path.isdir(full_path) and re.search(number_pattern, entry):
                matching_dirs.append(full_path)
    except FileNotFoundError:
        print(f"The path '{base_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access '{base_path}'.")

    return [Path(m) for m in matching_dirs]
    
def load_h5_scan_to_npy(file_path,scan,plot=True):
    # For loading cindy ptycho scan data
    # file_path = '/net/micdata/data2/12IDC/2021_Nov/ptycho/'
    # scan = 1125 (e.g.)
    dps=[]
    file_path_new=find_directories_with_number(file_path,scan)[0]
    for filename in os.listdir(file_path_new)[:-1]:
        filename = file_path_new / filename
        data = read_hdf5_file(filename)['entry/data/data']
        print(filename)
        for j in range(0,len(data)):
            dps.append(data[j])
            if plot:
                plt.figure()
                plt.imshow(data[j],norm=colors.LogNorm())
                plt.show()
    dps=np.asarray(dps)
    return dps

def load_hdf5_scan_to_npy(file_path,scan,plot=True):
    # For loading cindy ptycho scan data
    # file_path = '/net/micdata/data2/12IDC/2021_Nov/results/ML_recon/'
    # scan = 1125 (e.g.)
    dps=[]
    file_path_new=find_directories_with_number(file_path,scan)[0]
    for filename in os.listdir(file_path_new)[:-1]:
        filename = file_path_new / filename
        read_hdf5_file(file_path_new / filename).keys()
        if 'dp' not in read_hdf5_file(file_path_new / filename).keys(): #skip parameter file
            continue
        else:
            data = read_hdf5_file(file_path_new / filename)['dp']
            for j in range(0,len(data)):
                dps.append(data[j])
                if plot:
                    plt.figure()
                    plt.imshow(data[j],norm=colors.LogNorm())
                    plt.show()
    dps=np.asarray(dps)
    return dps
    
def create_annular_mask(shape, peak_x,peak_y, r_outer):
    """Create an annular mask between r_inner and r_outer centered at 'center'."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - peak_x)**2 + (y - peak_y)**2)
    mask = (dist_from_center <= r_outer)
    return mask
    
def ensure_inverse_peaks(peaks: List[Tuple[int, int]], tolerance: int = 4) -> List[Tuple[int, int]]:
    """
    Ensures that each peak in the list has an inverse peak, within a given tolerance.
    
    Parameters:
    - peaks: List of tuples, each representing (p1, p2) coordinates of peaks.
    - tolerance: Tolerance in pixels to check if an inverse peak is present.
    
    Returns:
    - List of peaks with ensured inverse peaks.
    """
        
    ## Example usage:
    #peaks = [(10, 20), (-8, -18), (15, 25)]
    #updated_peaks = ensure_inverse_peaks(peaks)
    #print(updated_peaks)
    #ensure_inverse_peaks(peaks_shifted)
    
    # Function to check if a point is within tolerance of another point
    def within_tolerance(point1, point2, tol):
        return abs(point1[0] - point2[0]) <= tol and abs(point1[1] - point2[1]) <= tol

    # Result list starting with original peaks
    result_peaks = peaks.copy()
    
    # Iterate over each peak and ensure its inverse is present
    for p1, p2 in peaks:
        # Calculate the inverse coordinates
        inv_p1, inv_p2 = -p1, -p2
        
        # Check if the inverse peak is within tolerance in the current list
        if not any(within_tolerance((inv_p1, inv_p2), (px, py), tolerance) for px, py in result_peaks):
            # If no peak within tolerance, add the inverse peak
            result_peaks.append((inv_p1, inv_p2))
    
    return result_peaks
