#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cmasher as cmr
from pathlib import Path
import torch
import sys
import os
import importlib
import pandas as pd
import glob
# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)
import tifffile
#%%
class DiffractionAnalyzer:
    def __init__(self, base_path, scan_number, dp_size=512, center_offset_y=100, center_offset_x=0):
        """
        Initialize the analyzer with scan parameters
        
        Args:
            base_path (str or Path): Path to the data directory
            scan_number (int): Scan number to analyze
            dp_size (int): Size to crop diffraction patterns to
            center_offset_y (int): Vertical offset from center for cropping
            center_offset_x (int): Horizontal offset from center for cropping
        """
        self.base_path = Path(base_path)
        self.scan_number = scan_number
        self.dp_size = dp_size
        self.center_offset_y = center_offset_y
        self.center_offset_x = center_offset_x
        
        # Initialize attributes
        self.dps = None
        self.dps_sum = None
        self.model = None
        self.deconvolved_patterns = None
        self.positions = None
        
        # Set plotting defaults
        plt.rcParams['image.cmap'] = 'jet'
        self.cmap = cmr.get_sub_cmap('jet', 0., 0.5)

    def load_and_crop_data(self):
        """Load H5 data and crop diffraction patterns"""
        # Load the H5 data
        self.dps = ptNN_U.load_h5_scan_to_npy(self.base_path, self.scan_number, plot=False, point_data=True)
        
        # Crop the diffraction patterns
        dps_size = self.dps[0].shape
        offset_y = self.center_offset_y
        offset_x = self.center_offset_x
        dpsize = self.dp_size
        
        self.dps = self.dps[:, 
            dps_size[0]//2-offset_y - dpsize//2:dps_size[0]//2-offset_y + dpsize//2,
            dps_size[1]//2-offset_x - dpsize//2:dps_size[1]//2-offset_x + dpsize//2
        ]
        
        # Remove hot pixels
        for i, dp in enumerate(self.dps):
            dp[dp >= 2**16-1] = np.min(dp)
        
        self.dps_sum = np.sum(self.dps, axis=0)
        return self

    def load_model(self, model_path):
        """Load the neural network model"""
        self.model = ptNN.ptychosaxsNN()
        self.model.load_model(state_dict_pth=model_path)
        self.model.set_device()
        self.model.model.to(self.model.device)
        self.model.model.eval()
        return self

    def perform_deconvolution(self):
        """Perform deconvolution on each diffraction pattern using the loaded model"""
        if self.model is None:
            raise ValueError("Model must be loaded before deconvolution")
            
        mask = np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy')
        
        # Initialize list to store deconvolved patterns
        self.deconvolved_patterns = []
        
        # Process each diffraction pattern
        for dp in self.dps:
            # Preprocess individual pattern
            resultT, sfT, bkgT = ptNN_U.preprocess_ZCB_9(dp, mask)
            resultTa = resultT.to(device=self.model.device, dtype=torch.float)
            
            # Perform deconvolution
            deconvolved = self.model.model(resultTa).detach().to("cpu").numpy()[0][0]
            self.deconvolved_patterns.append(deconvolved)
        
        # Convert to numpy array for easier handling
        self.deconvolved_patterns = np.array(self.deconvolved_patterns)
        return self

    def load_positions(self):
        """Load position data for the scan"""
        scan_dir = os.path.join('/mnt/micdata2/12IDC/2025_Feb/positions', f'{self.scan_number:03d}')
        
        if not os.path.exists(scan_dir):
            raise ValueError(f"Position directory not found for scan {self.scan_number}")
        
        # Get all position files for the scan
        files = glob.glob(os.path.join(scan_dir, f'*{self.scan_number:03d}_*.dat'))
        if not files:
            raise ValueError(f"No position files found for scan {self.scan_number}")
        
        # Extract line numbers and find maximum
        line_numbers = [int(os.path.basename(f).split(f'{self.scan_number:03d}_')[1].split('_')[0]) for f in files]
        max_line = max(line_numbers)
        
        # Process each line and point to get positions
        positions_list = []
        
        for line in range(1, max_line + 1):
            point_files = glob.glob(os.path.join(scan_dir, f'*{self.scan_number:03d}_{line:05d}_*.dat'))
            point_files = sorted(point_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            for point_file in point_files:
                try:
                    pos_arr = np.genfromtxt(point_file, delimiter='')
                    avg_pos = np.mean(pos_arr, axis=0)  # Average position for this point
                    positions_list.append(avg_pos)
                except Exception as e:
                    print(f"Error processing {point_file}: {str(e)}")
                    continue
        
        self.positions = np.array(positions_list)
        # Subtract mean to center positions
        self.positions -= np.mean(self.positions, axis=0)
        return self

    def plot_full_scan(self, use_deconvolved=False, shift_y=0, shift_x=0, scale_factor=2.0):
        """
        Plot full scan of either original or deconvolved patterns using actual position data.
        Places each diffraction pattern at its corresponding position in the grid.
        
        Args:
            use_deconvolved (bool): Whether to plot deconvolved patterns instead of raw data
            shift_y (float): Vertical shift in projection pixels (56nm/pixel)
            shift_x (float): Horizontal shift in projection pixels (56nm/pixel)
            scale_factor (float): Factor to scale the size of displayed patterns (default: 2.0)
        """
        if self.positions is None:
            self.load_positions()
            
        data_to_plot = self.deconvolved_patterns if use_deconvolved else self.dps
        if data_to_plot is None:
            raise ValueError("No data available to plot")
        
        # Convert shifts from pixels (56nm/pixel) to nm
        shift_y_nm = shift_y * 56  # nm
        shift_x_nm = shift_x * 56  # nm
        
        # Apply shifts to positions
        shifted_positions = self.positions.copy()
        shifted_positions[:, 1] += shift_y_nm  # Y1 position
        shifted_positions[:, 2] += shift_x_nm  # X position
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot each diffraction pattern at its position
        pattern_size = data_to_plot[0].shape[0] * scale_factor  # Scale the pattern size
        
        for idx, pos in enumerate(shifted_positions):
            if idx >= len(data_to_plot):
                break
                
            # Create a small axis for this pattern
            pattern_extent = [
                pos[2] - pattern_size/2,  # left
                pos[2] + pattern_size/2,  # right
                pos[1] - pattern_size/2,  # bottom
                pos[1] + pattern_size/2   # top
            ]
            
            if use_deconvolved:
                ax1.imshow(data_to_plot[idx], extent=pattern_extent)
            else:
                ax1.imshow(data_to_plot[idx], extent=pattern_extent, norm=colors.LogNorm(), cmap=self.cmap)
        
        # Load and process the reconstructed object
        obj_path = f"/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly{self.scan_number}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/O_phase_roi/O_phase_roi_Niter1000.tiff"
        obj = tifffile.imread(obj_path)
        
        # Flip object horizontally and vertically
        obj = np.flipud(np.fliplr(obj))
        
        # Calculate object extent in nm (56nm per pixel)
        obj_height, obj_width = obj.shape
        obj_extent = [
            np.min(shifted_positions[:, 2]),# - shift_x_nm,  # left
            np.max(shifted_positions[:, 2]),# - shift_x_nm,  # right
            np.min(shifted_positions[:, 1]),# - shift_y_nm,  # bottom
            np.max(shifted_positions[:, 1])# - shift_y_nm   # top
        ]
        
        # Plot the object first
        ax1.imshow(obj, extent=obj_extent, cmap='gray', alpha=0.5)
        
        ax1.set_title(f'Full Scan - {("Deconvolved" if use_deconvolved else "Original")}\n'
                     f'Shifts: ({shift_x:.2f}, {shift_y:.2f}) pixels ({shift_x_nm:.1f}, {shift_y_nm:.1f}) nm')
        ax1.axis('equal')
        
        
        
        # Plot the positions
        ax2.scatter(self.positions[:, 2], self.positions[:, 1], c='blue', label='Original', alpha=0.5)
        ax2.scatter(shifted_positions[:, 2], shifted_positions[:, 1], c='red', label='Shifted', alpha=0.5)
        ax2.set_xlabel('X Position (nm)')
        ax2.set_ylabel('Y Position (nm)')
        ax2.set_title('Scan Positions')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_shifted_tomogram_scan(self, df, use_deconvolved=False):
        """
        Plot the scan with shifts applied from tomographic alignment data.
        
        Args:
            df (pd.DataFrame): DataFrame containing alignment shifts
            use_deconvolved (bool): Whether to plot deconvolved patterns
        """
        # Get shifts for current scan
        scan_data = df[df['scanNo'] == self.scan_number]
        if len(scan_data) == 0:
            raise ValueError(f"No alignment data found for scan {self.scan_number}")
            
        shift_y = scan_data['y_shift'].iloc[0]
        shift_x = scan_data['x_shift'].iloc[0]
        
        # Plot with shifts
        self.plot_full_scan(
            use_deconvolved=use_deconvolved,
            shift_y=shift_y,
            shift_x=shift_x
        )

#%%
# Read the file, skipping the first row (which starts with #) and using the second row as headers
df = pd.read_csv('/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/ZCB9_3D_alignment_shifts_28nm.txt', 
                 comment='#',  # Skip lines starting with #
                 names=['Angle', 'y_shift', 'x_shift', 'scanNo'])  # Specify column names

# Convert scanNo to integer if needed
df['scanNo'] = df['scanNo'].astype(int)

# Sort by angle and find scans at roughly 45-degree increments
target_angles = np.arange(8, 9, 1)
selected_scans = []
for target in target_angles:
    # Find the scan with angle closest to target
    closest_scan = df.iloc[(df['Angle'] - target).abs().argsort()[:1]]
    selected_scans.append({
        'scan': int(closest_scan['scanNo'].iloc[0]),
        'angle': closest_scan['Angle'].iloc[0],
        'shift_y': closest_scan['y_shift'].iloc[0],
        'shift_x': closest_scan['x_shift'].iloc[0]
    })

# Print selected scans for reference
print("Selected scans for analysis:")
for scan in selected_scans:
    print(f"Angle: {scan['angle']:.1f}°, Scan: {scan['scan']}, Shifts (x,y): ({scan['shift_x']:.2f}, {scan['shift_y']:.2f})")

ncols=36
nrows=29
center=(517,575)

center_offset_y=1043//2-center[0]
center_offset_x=981//2-center[1]

model_path = Path('/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/best_model_ZCB_9_Unet_epoch_500_pearson_loss.pth')

obj=tifffile.imread(f"/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly{selected_scans[0]['scan']}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/O_phase_roi/O_phase_roi_Niter1000.tiff")
plt.imshow(obj,cmap='gray')
plt.show()


#%%
# Process each selected scan
for scan_info in selected_scans:
    print(f"\nProcessing scan {scan_info['scan']} at angle {scan_info['angle']:.1f}°")
    
    analyzer = DiffractionAnalyzer(
    base_path='/net/micdata/data2/12IDC/2025_Feb/ptycho/',
        scan_number=scan_info['scan'],
        dp_size=256,
        center_offset_y=center_offset_y,
        center_offset_x=center_offset_x
    )

    # Load and process data
    analyzer.load_and_crop_data()
    analyzer.load_model(model_path=model_path)
    analyzer.perform_deconvolution()

    # Larger patterns (2.5x)
    analyzer.plot_full_scan(
        use_deconvolved=False,
        shift_y=scan_info['shift_y'],
        shift_x=scan_info['shift_x'],
        scale_factor=2.5
    )
    # Larger patterns (2.5x)
    analyzer.plot_full_scan(
        use_deconvolved=True,
        shift_y=scan_info['shift_y'],
        shift_x=scan_info['shift_x'],
        scale_factor=2.5
    )

#%%