#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cmasher as cmr
from pathlib import Path
import torch
from tqdm import tqdm
import sys
import os
import importlib
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)


class DiffractionAnalyzer:
    def __init__(self, base_path, scan_number, dp_size=512, center_offset_y=100, center_offset_x=0,
                 center_cut=75, n_peaks=10, peak_threshold=0.25):
        """
        Initialize the analyzer with scan parameters
        
        Args:
            base_path (str or Path): Path to the data directory
            scan_number (int): Scan number to analyze
            dp_size (int): Size to crop diffraction patterns to
            center_offset_y (int): Vertical offset from center for cropping
            center_offset_x (int): Horizontal offset from center for cropping
            center_cut (int): Size of central region to exclude in peak finding
            n_peaks (int): Number of peaks to find
            peak_threshold (float): Threshold for peak detection
        """
        self.base_path = Path(base_path)
        self.scan_number = scan_number
        self.dp_size = dp_size
        self.center_offset_y = center_offset_y
        self.center_offset_x = center_offset_x
        self.center_cut = center_cut
        self.n_peaks = n_peaks
        self.peak_threshold = peak_threshold
        
        # Initialize attributes
        self.dps = None
        self.dps_sum = None
        self.model = None
        self.deconvolved = None
        self.peaks = None
        self.input=None
        # Set plotting defaults
        plt.rcParams['image.cmap'] = 'jet'
        self.cmap = cmr.get_sub_cmap('jet', 0., 0.5)

    def load_and_crop_data(self):
        """Load H5 data and crop diffraction patterns"""
        # Load the H5 data
        self.dps = ptNN_U.load_h5_scan_to_npy(self.base_path, self.scan_number, plot=False,point_data=False)
        
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
        """Perform deconvolution using the loaded model and find peaks"""
        if self.model is None:
            raise ValueError("Model must be loaded before deconvolution")
            
        mask = np.where(self.dps_sum <= 0, 0, 1)
        #resultT, sfT, bkgT = ptNN_U.preprocess_zhihua(self.dps_sum, mask)
        resultT, sfT, bkgT = ptNN_U.preprocess_cindy(self.dps_sum)
        self.input=resultT[0][0]
        resultTa = resultT.to(device=self.model.device, dtype=torch.float)
        self.deconvolved = self.model.model(resultTa).detach().to("cpu").numpy()[0][0]
        
        # Find peaks in deconvolved image
        self.peaks = ptNN_U.find_peaks2d(
            self.deconvolved,
            center_cut=self.center_cut,
            n=self.n_peaks,
            threshold=self.peak_threshold,
            plot=False
        )
        
        # Adjust peaks if needed (similar to your original script)
        center_x, center_y = 134, 124  # You might want to make these parameters
        peaks_shifted = [(p[0]-center_x, p[1]-center_y) for p in self.peaks]
        updated_peaks = ptNN_U.ensure_inverse_peaks(peaks_shifted)
        self.peaks = [(p[0]+center_x, p[1]+center_y) for p in updated_peaks]
        
        return self

    def calculate_frame_intensities_and_orientations(self, image, ref_image):
        """Helper function to calculate intensities and orientations for a single frame"""
        sub = np.subtract(image, ref_image, dtype=float)
        frame_intensities = []
        frame_orientations = []
        
        for peak in self.peaks:
            x, y = peak
            x = x*2  # Scale for 512x512
            y = y*2  # Scale for 512x512
            
            # Calculate intensity using circular neighborhood
            intensity = ptNN_U.circular_neighborhood_intensity(
                sub, x, y, radius=self.radius, plot=False
            )
            frame_intensities.append(intensity)
            
            # Calculate orientation
            x_min = max(0, x - self.radius)
            x_max = min(sub.shape[0], x + self.radius + 1)
            y_min = max(0, y - self.radius)
            y_max = min(sub.shape[1], y + self.radius + 1)
            
            region = sub[x_min:x_max, y_min:y_max]
            y_coords, x_coords = np.mgrid[0:region.shape[0], 0:region.shape[1]]
            total_intensity = np.sum(region)
            
            if total_intensity > 0:
                center_x = np.sum(x_coords * region) / total_intensity
                center_y = np.sum(y_coords * region) / total_intensity
                angle = np.arctan2(center_y - region.shape[0]/2, 
                                 center_x - region.shape[1]/2)
                angle_deg = (np.degrees(angle) + 360) % 360
            else:
                angle_deg = 0
                
            frame_orientations.append(angle_deg)
        
        return frame_intensities, frame_orientations

    def analyze_peaks(self, radius=56):
        """Analyze peaks for both intensity and orientation in the summed dataset"""
        if self.peaks is None:
            raise ValueError("Peaks must be detected before analysis. Run perform_deconvolution first.")
        
        self.radius = radius
        
        # Calculate reference-subtracted sum
        dps_sum_sub = []
        for dp in self.dps:
            test = np.subtract(dp, self.dps[0], dtype=float)
            dps_sum_sub.append(test)
        dps_sum_sub = np.sum(dps_sum_sub, axis=0)
        
        # Calculate intensities and orientations for summed pattern
        self.intensities_sum, self.orientations_sum = self.calculate_frame_intensities_and_orientations(
            dps_sum_sub, np.zeros_like(dps_sum_sub)
        )
        
        # Print results
        for idx, peak in enumerate(self.peaks):
            x, y = peak
            print(f"Peak at ({x*2}, {y*2}) has intensity: {self.intensities_sum[idx]}, "
                  f"orientation: {self.orientations_sum[idx]:.1f}°")
        
        return self

    def plot_deconvolution_results(self):
        """Plot original, preprocessed, and deconvolved data"""
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = ax[0].imshow(self.dps_sum, norm=colors.LogNorm())
        im2 = ax[1].imshow(self.deconvolved)
        
        # Plot peaks on deconvolved image
        peak_y, peak_x = zip(*self.peaks)
        ax[1].scatter(peak_x, peak_y, color='red', marker='x', s=100)
        
        # Plot summed pattern with weighted peaks
        #im3 = ax[2].imshow(self.dps_sum, norm=colors.LogNorm())
        im3 = ax[2].imshow(self.input)
        max_intensity = max(max(self.intensities_sum), 1e-10)  # Avoid division by zero
        alphas = [max(0, i/max_intensity) for i in self.intensities_sum]  # Ensure non-negative alphas
        ax[2].scatter(peak_x, peak_y, color='red', s=100, alpha=alphas)
        
        plt.colorbar(im1, ax=ax[0])
        plt.colorbar(im2, ax=ax[1])
        plt.colorbar(im3, ax=ax[2])
        
        ax[0].set_title('Original Sum')
        ax[1].set_title('Deconvolved with Peaks')
        ax[2].set_title('Sum with Weighted Peaks')
        plt.tight_layout()
        plt.show()

    def plot_full_scan(self, grid_size_row, grid_size_col):
        """Plot full scan with weighted peaks colored by orientation"""
        image_size = self.dps[0].shape
        grid_image = np.zeros((grid_size_row * image_size[0], 
                              grid_size_col * image_size[1]))
        
        # Create grid image
        for j in range(grid_size_row):
            for i in range(grid_size_col):
                image_idx = j * grid_size_col + i
                if image_idx < len(self.dps):
                    grid_image[
                        j * image_size[0]:(j + 1) * image_size[0],
                        i * image_size[1]:(i + 1) * image_size[1]
                    ] = self.dps[image_idx]
        
        # Calculate intensities and orientations for each frame
        ref = self.dps[0]
        ss = []
        orientations = []
        for image in tqdm(self.dps, desc="Calculating intensities"):
            frame_intensities, frame_orientations = self.calculate_frame_intensities_and_orientations(
                image, ref
            )
            s = np.array([frame_intensities[i]/self.intensities_sum[i]/np.max(self.intensities_sum) 
                         for i in range(len(frame_intensities))])
            ss.append(s)
            orientations.append(frame_orientations)
        
        test_ss = np.array([(s-np.min(ss))/(np.max(ss)-np.min(ss)) for s in ss])
        orientations = np.array(orientations)
        
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(grid_image, norm=colors.LogNorm(), cmap=self.cmap)
        
        # Create colormap for orientations
        orientation_cmap = plt.cm.hsv
        
        # Plot grid image with orientation-colored peaks
        peak_y, peak_x = zip(*self.peaks)
        for i in tqdm(range(grid_size_row), desc="Plotting overlay"):
            for j in range(grid_size_col):
                idx = i * grid_size_col + j
                if idx < len(self.dps):
                    x_offset = j * image_size[1]
                    y_offset = i * image_size[0]
                    colors_frame = [orientation_cmap(angle/360) for angle in orientations[idx]]
                    
                    # Plot scatter points
                    ax[0].scatter(
                        np.array(peak_x)*2 + x_offset,
                        np.array(peak_y)*2 + y_offset,
                        c=colors_frame, s=50,  # Increased size from 10 to 50
                        alpha=[max(0, alpha) for alpha in test_ss[idx]]
                    )
                    
                    # Add circles for each peak
                    for px, py, alpha in zip(peak_x, peak_y, test_ss[idx]):
                        circle = plt.Circle(
                            (px*2 + x_offset, py*2 + y_offset), 
                            self.radius, 
                            color='white',
                            fill=False,
                            alpha=max(0, alpha),
                            linewidth=0.5
                        )
                        ax[0].add_patch(circle)
        
        ax[0].axis('off')
        
        # Plot summed pattern with regular red dots
        ax[1].imshow(self.dps_sum, norm=colors.LogNorm())
        
        # Plot scatter points and circles for summed pattern
        ax[1].scatter(
            np.array(peak_x)*2, 
            np.array(peak_y)*2, 
            color='red', 
            s=200,  # Increased size from 100 to 200
            alpha=[i/max(self.intensities_sum) for i in self.intensities_sum]
        )
        
        # Add circles for summed pattern
        for px, py, alpha in zip(peak_x, peak_y, [i/max(self.intensities_sum) for i in self.intensities_sum]):
            circle = plt.Circle(
                (px*2, py*2), 
                self.radius, 
                color='red',
                fill=False,
                alpha=alpha,
                linewidth=1
            )
            ax[1].add_patch(circle)
        
        # Add colorwheel legend - adjust position to be closer to ax[0]
        ax_wheel = fig.add_axes([0.2, 0.88, 0.2, 0.02])  # [left, bottom, width, height]
        norm = plt.Normalize(0, 360)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=orientation_cmap), 
                         cax=ax_wheel, orientation='horizontal')
        cb.set_label('Orientation (degrees)', labelpad=2)
        
        plt.show()

    def save_peak_analysis(self, save_path):
        """
        Save peak positions, intensities, and orientations for all frames
        
        Args:
            save_path (str or Path): Directory to save the analysis files
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Get frame data
        ref = self.dps[0]
        frame_data = []
        
        # Calculate for each frame
        for frame_idx, image in enumerate(tqdm(self.dps, desc="Analyzing frames")):
            intensities, orientations = self.calculate_frame_intensities_and_orientations(image, ref)
            
            # Calculate normalized intensities
            norm_intensities = [i/self.intensities_sum[idx]/np.max(self.intensities_sum) 
                              for idx, i in enumerate(intensities)]
            
            # Store data for each peak in this frame
            for peak_idx, peak in enumerate(self.peaks):
                x, y = peak
                frame_data.append({
                    'frame': frame_idx,
                    'peak_idx': peak_idx,
                    'x': x*2,  # Scale for 512x512
                    'y': y*2,
                    'intensity': intensities[peak_idx],
                    'norm_intensity': norm_intensities[peak_idx],
                    'orientation': orientations[peak_idx]
                })
        
        # Save as CSV
        df = pd.DataFrame(frame_data)
        csv_path = save_path / f'peak_analysis_scan_{self.scan_number}.csv'
        df.to_csv(csv_path, index=False)
        
        # Save as NPY
        # Reshape data into more convenient arrays
        n_frames = len(self.dps)
        n_peaks = len(self.peaks)
        
        peak_positions = np.array([(x*2, y*2) for x, y in self.peaks])
        intensities_array = np.zeros((n_frames, n_peaks))
        norm_intensities_array = np.zeros((n_frames, n_peaks))
        orientations_array = np.zeros((n_frames, n_peaks))
        
        for frame_idx in range(n_frames):
            frame_data_subset = df[df['frame'] == frame_idx]
            intensities_array[frame_idx] = frame_data_subset['intensity'].values
            norm_intensities_array[frame_idx] = frame_data_subset['norm_intensity'].values
            orientations_array[frame_idx] = frame_data_subset['orientation'].values
        
        npy_data = {
            'scan_number': self.scan_number,
            'peak_positions': peak_positions,
            'intensities': intensities_array,
            'normalized_intensities': norm_intensities_array,
            'orientations': orientations_array,
            'summed_intensities': np.array(self.intensities_sum),
            'summed_orientations': np.array(self.orientations_sum)
        }
        
        npy_path = save_path / f'peak_analysis_scan_{self.scan_number}.npy'
        np.save(npy_path, npy_data)
        
        print(f"Analysis saved to:")
        print(f"CSV: {csv_path}")
        print(f"NPY: {npy_path}")
        
        return self


# Example usage
#scans=np.arange(1670,1685,1) 
#scans=np.arange(1537,1789,1) #Box/12IDC_3D/Sample6
scans=[1635]
#scans=np.arange(1053,1167,1) #Box/12IDC_3D/Sample4/tomo4
for scan in scans:
    try:
        analyzer = DiffractionAnalyzer(
            base_path='/net/micdata/data2/12IDC/2021_Nov/ptycho/',
            scan_number=scan,
            dp_size=512,
            center_offset_y=0,  # Vertical offset
            center_offset_x=0,    # Horizontal offset
            center_cut=64,#75,
            n_peaks=10,
            peak_threshold=0.25
        )

        # Load and process data
        analyzer.load_and_crop_data()


        # Load model and perform deconvolution (which now includes peak finding)
        #model_path = Path('/net/micdata/data2/12IDC/ptychosaxs/models/best_model_diff_sim15_zhihua_JM02_3D.pth')
        model_path = Path('/net/micdata/data2/12IDC/ptychosaxs/models/best_model_Unet_cindy.pth')
        analyzer.load_model(model_path)
        analyzer.perform_deconvolution()

        # Analyze the automatically detected peaks
        analyzer.analyze_peaks(radius=64)

        # Generate plots
        analyzer.plot_deconvolution_results()
        #analyzer.plot_full_scan(grid_size_row=15, grid_size_col=20) 
        analyzer.plot_full_scan(grid_size_row=12, grid_size_col=11) 

        # Save peak analysis
        analyzer.save_peak_analysis(save_path='/net/micdata/data2/12IDC/ptychosaxs/peak_analysis/')
    except:
        print(f"Error for scan {scan}")
# %%
