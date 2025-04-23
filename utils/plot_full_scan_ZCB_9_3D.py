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
import scipy.io as sio
import scipy.fft as spf
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
        # Subtract mean to center positions and flip y-coordinates
        self.positions -= np.mean(self.positions, axis=0)
        # Flip y-coordinates (positions[:, 1]) to match correct scan direction
        self.positions[:, 1] = -self.positions[:, 1]
        return self

    def load_probe(self, probe_array):
        """
        Load a probe array and store it for multiplication with object regions
        
        Args:
            probe_array (np.ndarray): The probe array to use
        """
        self.probe = probe_array
        return self

    def visualize_probe_positions(self, obj, positions, pixel_size=28):
        """
        Visualize probe positions on the padded object to verify scanning positions
        
        Args:
            obj (np.ndarray): The object array
            positions (np.ndarray): Array of scan positions
            pixel_size (float): Size of pixels in nm (default: 28)
        """
        probe_size = self.probe.shape[0]
        half_probe = probe_size // 2
        
        # Calculate object extent in nm (same as in plot_full_scan)
        obj_extent = [
            np.min(positions[:, 2]),
            np.max(positions[:, 2]),
            np.min(positions[:, 1]),
            np.max(positions[:, 1])
        ]
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        
        # Plot object with extent to match coordinate system
        ax1.imshow(obj, extent=obj_extent, cmap='gray')
        
        # Create color array for scan direction
        num_points = len(positions)
        colors_viridis = plt.cm.viridis(np.linspace(0, 1, num_points))
        
        # Plot scan positions with color gradient (using nm coordinates directly)
        x_positions = positions[:, 2]
        y_positions = positions[:, 1]
        
        # Plot points with color gradient
        scatter = ax1.scatter(x_positions, y_positions, c=np.arange(num_points), 
                            cmap='viridis', alpha=0.6, s=50)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax1, label='Scan Direction')
        
        # Add arrows to show direction
        for i in range(0, num_points-1, max(1, num_points//20)):  # Add arrows every few points
            ax1.arrow(x_positions[i], y_positions[i],
                     x_positions[i+1] - x_positions[i],
                     y_positions[i+1] - y_positions[i],
                     head_width=100, head_length=100, fc='white', ec='white', alpha=0.3)
        
        # Mark start and end points
        ax1.plot(x_positions[0], y_positions[0], 'go', label='Start', markersize=10)
        ax1.plot(x_positions[-1], y_positions[-1], 'ro', label='End', markersize=10)
        
        ax1.set_title('Scan Pattern on Object')
        ax1.legend()
        ax1.axis('equal')
        
        # For probe overlay, convert middle position to pixels
        example_pos = positions[len(positions)//2]  # Take middle position
        pos_y_px = int((example_pos[1] - obj_extent[2]) * obj.shape[0] / (obj_extent[3] - obj_extent[2]))
        pos_x_px = int((example_pos[2] - obj_extent[0]) * obj.shape[1] / (obj_extent[1] - obj_extent[0]))
        
        # Extract region around middle position
        padded_obj = np.pad(obj, ((half_probe, half_probe), (half_probe, half_probe)), mode='constant')
        region = padded_obj[
            pos_y_px:pos_y_px + probe_size,
            pos_x_px:pos_x_px + probe_size
        ]
        
        # Plot object region
        ax2.imshow(region, cmap='gray')
        # Overlay probe with transparency
        ax2.imshow(np.abs(self.probe), cmap='hot', alpha=0.5)
        ax2.set_title('Example Probe Overlay (Middle Position)')
        
        ax3.imshow(np.abs(spf.fftshift(spf.fft2(self.probe*region))), cmap='jet',norm=colors.LogNorm())
        ax3.set_title('Diffraction Pattern')
        
        plt.tight_layout()
        plt.show()

    def load_corrected_positions(self, corrected_positions, pixel_size=28):
        """
        Load corrected probe positions from ptychography reconstruction
        
        Args:
            corrected_positions (np.ndarray): Array of corrected positions [N, 2] where N is number of scan points
            pixel_size (float): Size of pixels in nm (default: 28)
        """
        if corrected_positions.shape[1] != 2:
            raise ValueError("Corrected positions should have shape [N, 2]")
            
        # Create positions array in the same format as self.positions
        self.corrected_positions = np.zeros((len(corrected_positions), 3))
        # Note: we flip the y-coordinates to match the coordinate system
        self.corrected_positions[:, 1] = -corrected_positions[:, 1]*pixel_size  # y-positions (flipped)
        self.corrected_positions[:, 2] = -corrected_positions[:, 0]*pixel_size  # x-positions
        
        # Center the positions
        self.corrected_positions -= np.mean(self.corrected_positions, axis=0)
        return self

    def multiply_probe_with_object(self, obj, positions, pixel_size=28, use_corrected_positions=False):
        """
        Multiply probe with object at each scan position
        
        Args:
            obj (np.ndarray): The object array
            positions (np.ndarray): Array of scan positions
            pixel_size (float): Size of pixels in nm (default: 28)
            use_corrected_positions (bool): Whether to use corrected positions from ptychography reconstruction
            
        Returns:
            list: List of probe-object multiplications at each position
        """
        probe_size = self.probe.shape[0]
        half_probe = probe_size // 2
        
        # Determine which positions to use
        if use_corrected_positions and hasattr(self, 'corrected_positions'):
            actual_positions = self.corrected_positions
        else:
            actual_positions = positions
        
        # Pad object to handle probe at edge positions
        padded_obj = np.pad(obj, ((half_probe, half_probe), (half_probe, half_probe)), mode='constant')
        
        # Calculate scaling factors to convert from nm to pixels
        obj_extent = [
            np.min(actual_positions[:, 2]),
            np.max(actual_positions[:, 2]),
            np.min(actual_positions[:, 1]),
            np.max(actual_positions[:, 1])
        ]
        
        # Calculate scale factors (nm to pixels)
        scale_x = obj.shape[1] / (obj_extent[1] - obj_extent[0])
        scale_y = obj.shape[0] / (obj_extent[3] - obj_extent[2])
        
        results = []
        for pos in actual_positions:
            # Convert position from nm to pixels relative to object
            x_px = int((pos[2] - obj_extent[0]) * scale_x)
            y_px = int((pos[1] - obj_extent[2]) * scale_y)
            
            # Add padding offset
            x_px += half_probe
            y_px += half_probe
            
            # Extract region and multiply with probe
            obj_region = padded_obj[
                y_px - half_probe:y_px + half_probe,
                x_px - half_probe:x_px + half_probe
            ]
            
            # Multiply probe with object region and flip vertically to match diffraction pattern orientation
            # result = np.flipud(self.probe * obj_region)
            result=self.probe*np.flipud(np.fliplr(obj_region))
            #result = self.probe * obj_region
            results.append(result)
            
        return results

    def plot_full_scan(self, use_deconvolved=False, shift_y=0, shift_x=0, scale_factor=2.0, use_corrected_positions=False, highlight_index=None):
        """
        Plot full scan of either original or deconvolved patterns using actual position data.
        Places each diffraction pattern at its corresponding position in the grid.
        
        Args:
            use_deconvolved (bool): Whether to plot deconvolved patterns instead of raw data
            shift_y (float): Vertical shift in projection pixels (28nm/pixel)
            shift_x (float): Horizontal shift in projection pixels (28nm/pixel)
            scale_factor (float): Factor to scale the size of displayed patterns (default: 2.0)
            use_corrected_positions (bool): Whether to use corrected positions from ptychography reconstruction
                                          (only affects probe-object multiplication)
            highlight_index (int, optional): Index of the pattern to highlight
        """
        if self.positions is None:
            self.load_positions()
            
        data_to_plot = self.deconvolved_patterns if use_deconvolved else self.dps
        if data_to_plot is None:
            raise ValueError("No data available to plot")
        
        # Convert shifts from pixels (28nm/pixel) to nm
        shift_y_nm = shift_y * 28  # nm
        shift_x_nm = shift_x * 28  # nm
        
        # Always use original positions for displaying experimental data
        shifted_positions_exp = self.positions.copy()
        shifted_positions_exp[:, 1] += shift_y_nm  # Y1 position
        shifted_positions_exp[:, 2] += shift_x_nm  # X position
        
        # For probe-object multiplication, use corrected positions if requested
        if use_corrected_positions and hasattr(self, 'corrected_positions'):
            shifted_positions_calc = self.corrected_positions.copy()
            shifted_positions_calc[:, 1] += shift_y_nm
            shifted_positions_calc[:, 2] += shift_x_nm
        else:
            shifted_positions_calc = shifted_positions_exp.copy()
        
        # Create figure with three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        
        # Load and process the reconstructed object
        #obj_path = f"/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly{self.scan_number}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/O_phase_roi/O_phase_roi_Niter1000.tiff"
        #obj = tifffile.imread(obj_path)
        obj_path = f"/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly{self.scan_number}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/Niter1000.mat"
        obj = sio.loadmat(obj_path)["object_roi"]
        
        # Only flip horizontally to match coordinate system
        obj = np.flipud(np.fliplr(obj))
        
        # Calculate object extent using UNSHIFTED positions for proper centering
        obj_extent = [
            np.min(self.positions[:, 2]),
            np.max(self.positions[:, 2]),
            np.min(self.positions[:, 1]),
            np.max(self.positions[:, 1])
        ]
        
        # Plot the object
        ax1.imshow(np.angle(obj), extent=obj_extent, cmap='gray', alpha=0.8, origin='lower')
        
        # Plot diffraction patterns at experimental positions
        pattern_size = data_to_plot[0].shape[0] * scale_factor
        
        for idx, pos in enumerate(shifted_positions_exp):
            if idx >= len(data_to_plot):
                break
                
            pattern_extent = [
                pos[2] - pattern_size/2,
                pos[2] + pattern_size/2,
                pos[1] - pattern_size/2,
                pos[1] + pattern_size/2
            ]
            
            if use_deconvolved:
                ax1.imshow(data_to_plot[idx], extent=pattern_extent)
            else:
                ax1.imshow(data_to_plot[idx], extent=pattern_extent, norm=colors.LogNorm(), cmap=self.cmap)
                
            # Highlight the selected pattern if specified
            if highlight_index is not None and idx == highlight_index:
                rect = plt.Rectangle((pattern_extent[0], pattern_extent[2]), 
                                   pattern_size, pattern_size,
                                   fill=False, color='red', linewidth=2)
                ax1.add_patch(rect)
                # Add text annotation
                ax1.text(pos[2], pos[1] + pattern_size/1.5, f'Index: {idx}', 
                        color='red', fontsize=10, ha='center', va='bottom')
        
        ax1.set_title(f'Full Scan - {("Deconvolved" if use_deconvolved else "Original")}\n'
                     f'Shifts: ({shift_x:.2f}, {shift_y:.2f}) pixels ({shift_x_nm:.1f}, {shift_y_nm:.1f}) nm')
        ax1.axis('equal')
        
        # Plot both sets of positions
        ax2.scatter(self.positions[:, 2], self.positions[:, 1], c='blue', label='Original', alpha=0.5)
        if use_corrected_positions and hasattr(self, 'corrected_positions'):
            ax2.scatter(shifted_positions_calc[:, 2], shifted_positions_calc[:, 1], c='red', label='Corrected', alpha=0.5)
        else:
            ax2.scatter(shifted_positions_exp[:, 2], shifted_positions_exp[:, 1], c='red', label='Shifted', alpha=0.5)
            
        # Highlight the selected position if specified
        if highlight_index is not None:
            pos = shifted_positions_exp[highlight_index]
            ax2.plot(pos[2], pos[1], 'r*', markersize=15, label=f'Selected (Index: {highlight_index})')
            
        ax2.set_xlabel('X Position (nm)')
        ax2.set_ylabel('Y Position (nm)')
        ax2.set_title('Scan Positions')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True)
        
        # Calculate and plot probe-object multiplications if probe is available
        if hasattr(self, 'probe'):
            probe_obj_results = self.multiply_probe_with_object(obj, shifted_positions_calc, use_corrected_positions=use_corrected_positions)
            
            for idx, (result, pos) in enumerate(zip(probe_obj_results, shifted_positions_calc)):
                if idx >= len(data_to_plot):
                    break
                    
                # Calculate FFT and shift for this individual pattern
                fft_result = spf.fftshift(spf.fft2(result))
                
                pattern_extent = [
                    pos[2] - pattern_size/2,
                    pos[2] + pattern_size/2,
                    pos[1] - pattern_size/2,
                    pos[1] + pattern_size/2
                ]
                
                ax3.imshow(np.abs(fft_result), extent=pattern_extent, cmap='jet', norm=colors.LogNorm())
                
                # Highlight the selected pattern if specified
                if highlight_index is not None and idx == highlight_index:
                    rect = plt.Rectangle((pattern_extent[0], pattern_extent[2]), 
                                       pattern_size, pattern_size,
                                       fill=False, color='red', linewidth=2)
                    ax3.add_patch(rect)
                    # Add text annotation
                    ax3.text(pos[2], pos[1] + pattern_size/1.5, f'Index: {idx}', 
                            color='red', fontsize=10, ha='center', va='bottom')
            
            ax3.set_title('Probe-Object Multiplication\n' + ('(Using Corrected Positions)' if use_corrected_positions else '(Using Original Positions)'))
            ax3.axis('equal')
        
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

    def plot_difference_map(self, scan_index, shift_y=0, shift_x=0, use_deconvolved=False, use_corrected_positions=False,scale_factor=2.5):
        """
        Plot difference map between experimental and calculated diffraction patterns for a specific scan position
        
        Args:
            scan_index (int): Index of the scan position to analyze
            shift_y (float): Vertical shift in projection pixels (28nm/pixel)
            shift_x (float): Horizontal shift in projection pixels (28nm/pixel)
            use_corrected_positions (bool): Whether to use corrected positions from ptychography reconstruction
        """
        # First show the full scan with highlighted position
        print("Full scan view with highlighted position:")
        self.plot_full_scan(
            use_deconvolved=use_deconvolved,
            shift_y=shift_y,
            shift_x=shift_x,
            scale_factor=scale_factor,
            use_corrected_positions=use_corrected_positions,
            highlight_index=scan_index
        )
        
        # Then proceed with the difference map
        if self.positions is None:
            self.load_positions()
            
        if self.dps is None:
            raise ValueError("No experimental data available")
            
        if not hasattr(self, 'probe'):
            raise ValueError("Probe not loaded")
            
        # Convert shifts from pixels to nm
        shift_y_nm = shift_y * 28
        shift_x_nm = shift_x * 28
        
        # Determine which positions to use as base
        base_positions = self.corrected_positions if use_corrected_positions and hasattr(self, 'corrected_positions') else self.positions
        
        # Apply shifts to positions
        shifted_positions = base_positions.copy()
        shifted_positions[:, 1] += shift_y_nm
        shifted_positions[:, 2] += shift_x_nm
        
        # Load object
        obj_path = f"/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly{self.scan_number}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/O_phase_roi/O_phase_roi_Niter1000.tiff"
        obj = tifffile.imread(obj_path)
        obj = np.flipud(np.fliplr(obj))
        
        # Calculate probe-object multiplication
        probe_obj_results = self.multiply_probe_with_object(obj, shifted_positions, use_corrected_positions=use_corrected_positions)
        
        # Get experimental and calculated patterns for the specified index
        exp_pattern = self.dps[scan_index]
        calc_pattern = np.abs(spf.fftshift(spf.fft2(probe_obj_results[scan_index])))
        
        # Create figure
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(50, 10))
        
        # Plot experimental pattern
        im1 = ax1.imshow(exp_pattern, norm=colors.LogNorm(), cmap='jet')
        ax1.set_title('Experimental Pattern')
        plt.colorbar(im1, ax=ax1)
        
        # Plot calculated pattern
        im2 = ax2.imshow(calc_pattern, norm=colors.LogNorm(), cmap='jet')
        ax2.set_title('Calculated Pattern')
        plt.colorbar(im2, ax=ax2)
        
        # Calculate and plot difference
        difference = exp_pattern - calc_pattern
        im3 = ax3.imshow(difference, cmap='RdBu_r')
        ax3.set_title('Difference (Exp - Calc)')
        plt.colorbar(im3, ax=ax3)
        
        # Plot relative difference
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = difference / calc_pattern
            relative_diff = np.nan_to_num(relative_diff)  # Replace inf/nan with 0
        
        im4 = ax4.imshow(relative_diff, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_title('Relative Difference\n(Exp - Calc)/Calc')
        plt.colorbar(im4, ax=ax4)
        
        # Plot deconvolved pattern if available
        if hasattr(self, 'deconvolved_patterns') and self.deconvolved_patterns is not None:
            deconv_pattern = self.deconvolved_patterns[scan_index]
            im5 = ax5.imshow(deconv_pattern, norm=colors.LogNorm(), cmap='jet')
            ax5.set_title('Deconvolved Pattern')
            plt.colorbar(im5, ax=ax5)
        else:
            ax5.text(0.5, 0.5, 'No deconvolved pattern available', 
                    horizontalalignment='center', verticalalignment='center')
            ax5.set_title('Deconvolved Pattern')
        
        # Add scan position information to the title
        pos = shifted_positions[scan_index]
        plt.suptitle(f'Scan Position {scan_index} at ({pos[2]:.1f}, {pos[1]:.1f}) nm\n'
                    f'Shifts: ({shift_x:.2f}, {shift_y:.2f}) pixels ({shift_x_nm:.1f}, {shift_y_nm:.1f}) nm')
        
        plt.tight_layout()
        plt.show()
        
        # Return statistics
        stats = {
            'mean_difference': np.mean(difference),
            'std_difference': np.std(difference),
            'max_difference': np.max(difference),
            'min_difference': np.min(difference),
            'mean_relative_diff': np.mean(relative_diff),
            'std_relative_diff': np.std(relative_diff)
        }
        
        print("\nDifference Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.3f}")
            
        return stats
    
    def calculate_probe_radius(self, show_fit=False):
        """
        Calculate the radius of the probe using a Gaussian fit
        
        Args:
            show_fit (bool): Whether to show the radial profile fit and probe visualization
        
        Returns:
            float: Radius of the probe in pixels
        """
        if not hasattr(self, 'probe'):
            raise ValueError("Probe not loaded")
            
        # Get the amplitude of the probe
        probe_amp = np.abs(self.probe)
        
        # Get the center coordinates
        center_y, center_x = np.array(probe_amp.shape) // 2
        y, x = np.indices(probe_amp.shape)
        
        # Calculate radial distance from center
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Get the radial profile
        r_unique = np.unique(r.ravel())
        intensity_radial = np.array([probe_amp[r == r_val].mean() for r_val in r_unique])
        
        # Normalize
        intensity_radial = intensity_radial / intensity_radial.max()
        
        # Find the radius where intensity falls to 1/e
        radius_pixels = r_unique[np.abs(intensity_radial - 1/np.e).argmin()]
        
        if show_fit:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Plot radial profile
            ax1.plot(r_unique, intensity_radial, 'b-', label='Radial Profile')
            ax1.axvline(x=radius_pixels, color='r', linestyle='--', 
                       label=f'1/e radius = {radius_pixels:.1f} px\n({radius_pixels*28:.0f} nm)')
            ax1.axhline(y=1/np.e, color='g', linestyle='--', label='1/e intensity')
            ax1.set_xlabel('Radius (pixels)', fontsize=14)
            ax1.set_ylabel('Normalized Intensity', fontsize=14)
            ax1.set_title('Radial Intensity Profile', fontsize=16)
            ax1.grid(True)
            ax1.legend(fontsize=12)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            
            # Plot probe with radius circle
            im = ax2.imshow(probe_amp, cmap='viridis')
            circle = plt.Circle((center_x, center_y), radius_pixels,
                              fill=False, color='r', linestyle='--', linewidth=2)
            ax2.add_patch(circle)
            ax2.set_title('Probe Amplitude with 1/e Radius', fontsize=16)
            plt.colorbar(im, ax=ax2)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            plt.show()
        
        return radius_pixels
        
    def plot_scan_overlay(self, use_deconvolved=False, shift_y=0, shift_x=0, scale_factor=2.0, use_corrected_positions=False, highlight_index=None, show_probe_size=True, save_path=None):
        """
        Plot full scan of either original or deconvolved patterns using actual position data.
        Places each diffraction pattern at its corresponding position in the grid.
        
        Args:
            use_deconvolved (bool): Whether to plot deconvolved patterns instead of raw data
            shift_y (float): Vertical shift in projection pixels (28nm/pixel)
            shift_x (float): Horizontal shift in projection pixels (28nm/pixel)
            scale_factor (float): Factor to scale the size of displayed patterns (default: 2.0)
            use_corrected_positions (bool): Whether to use corrected positions from ptychography reconstruction
            highlight_index (int, optional): Index of the pattern to highlight
            show_probe_size (bool): Whether to show the probe size as a dashed circle
        """
        if self.positions is None:
            self.load_positions()
            
        data_to_plot = self.deconvolved_patterns if use_deconvolved else self.dps
        if data_to_plot is None:
            raise ValueError("No data available to plot")
        
        # Convert shifts from pixels (28nm/pixel) to nm
        shift_y_nm = shift_y * 28  # nm
        shift_x_nm = shift_x * 28  # nm
        
        # Always use original positions for displaying experimental data
        shifted_positions_exp = self.positions.copy()
        shifted_positions_exp[:, 1] += shift_y_nm  # Y1 position
        shifted_positions_exp[:, 2] += shift_x_nm  # X position
        
        # For probe-object multiplication, use corrected positions if requested
        if use_corrected_positions and hasattr(self, 'corrected_positions'):
            shifted_positions_calc = self.corrected_positions.copy()
            shifted_positions_calc[:, 1] += shift_y_nm
            shifted_positions_calc[:, 2] += shift_x_nm
        else:
            shifted_positions_calc = shifted_positions_exp.copy()
        
        # Create figure with three subplots side by side
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
        
        # Load and process the reconstructed object
        obj_path = f"/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly{self.scan_number}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/Niter1000.mat"
        obj = sio.loadmat(obj_path)["object_roi"]
        
        # Only flip horizontally to match coordinate system
        obj = np.flipud(np.fliplr(obj))
        
        # Calculate object extent using UNSHIFTED positions for proper centering
        obj_extent = [
            np.min(self.positions[:, 2]),
            np.max(self.positions[:, 2]),
            np.min(self.positions[:, 1]),
            np.max(self.positions[:, 1])
        ]
        
        # Plot the object
        ax1.imshow(np.angle(obj), extent=obj_extent, cmap='gray', alpha=0.8, origin='lower')
        
        # Plot diffraction patterns at experimental positions
        pattern_size = data_to_plot[0].shape[0] * scale_factor
        
        for idx, pos in enumerate(shifted_positions_exp):
            if idx >= len(data_to_plot):
                break
                
            pattern_extent = [
                pos[2] - pattern_size/2,
                pos[2] + pattern_size/2,
                pos[1] - pattern_size/2,
                pos[1] + pattern_size/2
            ]
            
            if use_deconvolved:
                ax1.imshow(data_to_plot[idx], extent=pattern_extent)
            else:
                ax1.imshow(data_to_plot[idx], extent=pattern_extent, norm=colors.LogNorm())
                
            # Highlight the selected pattern if specified
            if highlight_index is not None and idx == highlight_index:
                rect = plt.Rectangle((pattern_extent[0], pattern_extent[2]), 
                                   pattern_size, pattern_size,
                                   fill=False, color='red', linewidth=2,
                                   label=f'Selected Pattern (Index: {idx})')
                ax1.add_patch(rect)
                # Add text annotation
                #ax1.text(pos[2], pos[1] + pattern_size/1.5, f'Index: {idx}', 
                #        color='red', fontsize=14, ha='center', va='bottom')
                
                # Add probe size circle if requested
                if show_probe_size and hasattr(self, 'probe'):
                    probe_radius_px = self.calculate_probe_radius()
                    # Convert probe radius from pixels to nm
                    probe_radius_nm = probe_radius_px * 28  # 28 nm/pixel
                    probe_radius_um = probe_radius_nm / 1000
                    circle = plt.Circle((pos[2], pos[1]), probe_radius_nm,
                                     fill=False, linestyle='--', color='#fcd83f', linewidth=4,
                                     label=f'Probe Radius ({probe_radius_um:.1f} µm)')
                    ax1.add_patch(circle)
                    # Add text annotation for probe size
                    # ax1.text(pos[2], pos[1] + 3*pattern_size, 
                    #         f'Probe radius: {probe_radius_um:.1f} um',
                    #         color='#fcd83f', fontsize=16, ha='center', va='top')
        
        ax1.set_title(f'Full Scan - {("Deconvolved" if use_deconvolved else "Original")}\n', fontsize=24)
        ax1.axis('equal')
        ax1.set_xlabel('X Position (nm)', fontsize=22)
        ax1.set_ylabel('Y Position (nm)', fontsize=22)
        
        # Increase tick label sizes
        ax1.tick_params(axis='both', which='major', labelsize=18)
        
        # Add legend with larger font size
        ax1.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()

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

obj_path = f"/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly{selected_scans[0]['scan']}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/Niter1000.mat"
obj = sio.loadmat(obj_path)["object_roi"]
plt.imshow(np.angle(obj),cmap='gray')
plt.show()

probe_positions=sio.loadmat(f"/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly{selected_scans[0]['scan']}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/Niter1000.mat")["outputs"]["probe_positions"][0][0]
plt.plot(-probe_positions[:,0],probe_positions[:,1],'o')
plt.show()

#%%
ob=sio.loadmat("/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly5102/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/Niter1000.mat")
pb=ob['probe']
# Only first mode
pb1=pb[:,:,0,0]
plt.imshow(np.abs(pb1),cmap='gray')
plt.show()


# #%%
# # Process each selected scan
# for scan_info in selected_scans:
#     print(f"\nProcessing scan {scan_info['scan']} at angle {scan_info['angle']:.1f}°")
#     analyzer = DiffractionAnalyzer(
#                 base_path='/net/micdata/data2/12IDC/2025_Feb/ptycho/',
#                 scan_number=scan_info['scan'],
#                 dp_size=256,
#                 center_offset_y=center_offset_y,
#                 center_offset_x=center_offset_x
#         )

#     # Load and process data
#     analyzer.load_and_crop_data()
#     analyzer.load_model(model_path=model_path)
#     analyzer.perform_deconvolution()

#     # Load probe
#     analyzer.load_probe(pb1)
    
#     analyzer.plot_difference_map(scan_index=661-36*6+5,shift_y=scan_info['shift_y'],
#         shift_x=scan_info['shift_x'],use_corrected_positions=False)  # Or any other index

#     # Larger patterns (2.5x)
#     # analyzer.plot_full_scan(
#     #     use_deconvolved=False,
#     #     shift_y=scan_info['shift_y'],
#     #     shift_x=scan_info['shift_x'],
#     #     scale_factor=2.5
#     # )

# #%%

# analyzer.load_corrected_positions(probe_positions)
# # Then when plotting, specify use_corrected_positions=True

# analyzer.plot_difference_map(scan_index=661-36*6+5, shift_y=scan_info['shift_y'],
#         shift_x=scan_info['shift_x'],use_corrected_positions=True)
# # analyzer.plot_full_scan(use_corrected_positions=True,shift_y=scan_info['shift_y'],
# #         shift_x=scan_info['shift_x'], scale_factor=2.5)
# # %%
# analyzer.plot_difference_map(scan_index=664,use_corrected_positions=False,use_deconvolved=True,scale_factor=2)









# %%

analyzer = DiffractionAnalyzer(
        base_path='/net/micdata/data2/12IDC/2025_Feb/ptycho/',
        scan_number=5065,
        dp_size=256,
        center_offset_y=center_offset_y,
        center_offset_x=center_offset_x
)

# Load and process data
analyzer.load_and_crop_data()
analyzer.load_model(model_path=model_path)
analyzer.perform_deconvolution()

# Load probe
analyzer.load_probe(pb1)
# analyzer.calculate_probe_radius(show_fit=True)
analyzer.plot_scan_overlay(highlight_index=664,use_corrected_positions=False,use_deconvolved=False,scale_factor=2,show_probe_size=True,save_path=f"ZCB_9_3D_scan_overlay_original_{analyzer.scan_number}.pdf")
analyzer.plot_scan_overlay(highlight_index=664,use_corrected_positions=False,use_deconvolved=True,scale_factor=2,show_probe_size=False,save_path=f"ZCB_9_3D_scan_overlay_deconvolved_{analyzer.scan_number}.pdf")
# %%
