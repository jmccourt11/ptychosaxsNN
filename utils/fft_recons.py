# %%
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.widgets import RectangleSelector, Button
%matplotlib widget

def read_and_fft_tiff(
    tiff_path,
    roi=None,  # roi can be a tuple (rect) or dict (circle)
    frame=0,   # for multi-frame tiffs, select which frame to use
    vignette=True  # apply a 2D Hann window if True
):
    """
    Reads a TIFF file, selects an ROI (rectangular or circular), applies a vignette, and computes the FFT.

    Parameters:
        tiff_path (str): Path to the TIFF file.
        roi (tuple, dict, or None): Rectangle (start_row, end_row, start_col, end_col) or circle {'type': 'circle', 'center': (row, col), 'radius': r}. If None, use full image.
        frame (int): Frame index for multi-frame TIFFs.
        vignette (bool): Whether to apply a 2D Hann window before FFT.

    Returns:
        fft_result (np.ndarray): The complex FFT result.
        fft_magnitude (np.ndarray): The magnitude spectrum (log-scaled).
        image (np.ndarray): The image or ROI used (after vignette if applied).
    """
    # Read the TIFF file
    img = tifffile.imread(tiff_path)
    if img.ndim == 3:  # multi-frame
        img = img[frame]
    
    # Select ROI if specified
    if roi is not None:
        if isinstance(roi, tuple):
            # Rectangle
            start_row, end_row, start_col, end_col = roi
            img = img[start_row:end_row, start_col:end_col]
        elif isinstance(roi, dict) and roi.get('type') == 'circle':
            cy, cx = roi['center']
            r = roi['radius']
            y1, y2 = int(cy - r), int(cy + r)
            x1, x2 = int(cx - r), int(cx + r)
            img = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            # Apply circular mask
            h, w = img.shape
            Y, X = np.ogrid[:h, :w]
            mask = (Y - (cy - y1))**2 + (X - (cx - x1))**2 <= r**2
            img = img * mask
        else:
            raise ValueError('ROI must be a tuple (rectangle) or dict with type "circle"')
    
    # Apply vignette (2D Hann window)
    if vignette:
        h, w = img.shape
        win_row = np.hanning(h)
        win_col = np.hanning(w)
        window = np.outer(win_row, win_col)
        img = img * window

    # Compute FFT
    fft_result = np.fft.fftshift(np.fft.fft2(img))
    fft_magnitude = np.log1p(np.abs(fft_result))

    return fft_result, fft_magnitude, img

def scan_fft_over_image(
    img,
    roi_height,
    roi_width,
    step_y,
    step_x,
    vignette=True,
    show_progress=True
):
    """
    Scan the image with a fixed-size ROI and compute FFTs at each position.
    Returns a list of (y, x, fft_magnitude) for each ROI.
    """
    h, w = img.shape
    results = []
    y_positions = range(0, h - roi_height + 1, step_y)
    x_positions = range(0, w - roi_width + 1, step_x)
    for i, y in enumerate(y_positions):
        # Snake pattern: reverse x direction every other row
        if i % 2 == 0:
            xs = x_positions
        else:
            xs = reversed(list(x_positions))
        for x in xs:
            roi = img[y:y+roi_height, x:x+roi_width]
            if vignette:
                win_row = np.hanning(roi_height)
                win_col = np.hanning(roi_width)
                window = np.outer(win_row, win_col)
                roi = roi * window
            fft_result = np.fft.fftshift(np.fft.fft2(roi))
            fft_mag = np.log1p(np.abs(fft_result))
            results.append({'y': y, 'x': x, 'fft_mag': fft_mag})
        if show_progress:
            print(f'Scanned row {i+1}/{len(list(y_positions))}')
    return np.array(results)

#%%
#base_dir = '/net/micdata/data2/12IDC/'
base_dir='/scratch/'
exp_dir = '2025_Feb/'
recon_dir = 'results/'
sample_dir = 'ZCB_9_3D_/'
scan_num = 5065
recon_path = 'roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/'
iterations=1000
vignette=True

h5_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:04d}/data_roi0_Ndp256_para.hdf5'
with h5py.File(h5_path, 'r') as f:
    sdd=f['detector_distance'][()] #m
    angle=f['angle'][()] #deg
    energy=f['energy'][()] #keV
Ndp=256
#wavelength=1.239842e-10 # m
wavelength = (12.3984/energy)*10**(-10) # nm
delta_p=172e-6 # m
pixel_size=wavelength*sdd/(Ndp*delta_p)
print(f'pixel size: {pixel_size*1e9} nm')
if Ndp==256:
    recon_path+='MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/'
tiff_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:04d}/{recon_path}/O_phase_roi/O_phase_roi_Niter{iterations}.tiff'


# Full image FFT
fft_result, fft_mag, img = read_and_fft_tiff(tiff_path,vignette=vignette)

# ROI FFT (e.g., rows 100:200, cols 150:250)
if Ndp==256:
    roi = (300, 400, 325, 425) #256
else:
    roi = (300//2, 400//2, 250//2, 350//2) #128
fft_result_roi, fft_mag_roi, img_roi = read_and_fft_tiff(tiff_path, roi=roi,vignette=vignette)

plt.figure()
plt.subplot(1,2,1)
plt.title('Image with ROI')
plt.imshow(img, cmap='gray')

# Overlay ROI rectangle
roi_rect = patches.Rectangle(
    (roi[2], roi[0]),  # (x, y) = (start_col, start_row)
    roi[3] - roi[2],   # width
    roi[1] - roi[0],   # height
    linewidth=2, edgecolor='r', facecolor='none'
)
plt.gca().add_patch(roi_rect)

plt.subplot(1,2,2)
plt.title('FFT Magnitude')
plt.imshow(fft_mag, cmap='jet')
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.title('ROI')
plt.imshow(img_roi, cmap='gray')
plt.subplot(1,2,2)
plt.title('FFT Magnitude')
plt.imshow(fft_mag_roi, cmap='jet')
plt.show()


#%%
# Example usage:
img = tifffile.imread(tiff_path)
if img.ndim == 3:
    img = img[0]
roi_height, roi_width = 200, 200
step_y, step_x = 50, 50
fft_results = scan_fft_over_image(img, roi_height, roi_width, step_y, step_x, vignette=True)
print(f"Total FFTs computed: {len(fft_results)}")
fft_total=np.zeros(fft_results[0]['fft_mag'].shape) 
for i in range(0,len(fft_results)):
    fft_total+=fft_results[i]['fft_mag']
plt.figure()
plt.imshow(np.log1p(fft_total), cmap='jet')
plt.show()







#%%

# Interactive ROI selection and FFT calculation
class InteractiveFFT:
    def __init__(self, tiff_path, vignette=True):
        self.tiff_path = tiff_path
        self.vignette = vignette
        self.roi = None
        self.roi_type = 'rectangle'  # 'rectangle' or 'circle'
        self.circle_selector_active = False
        self.circle_artist = None
        self.circle_params = None  # (center_y, center_x, radius)
        # Load the original image (no vignette)
        img = tifffile.imread(tiff_path)
        if img.ndim == 3:
            img = img[0]
        self.img_orig = img
        # For display, optionally apply vignette
        if vignette:
            h, w = img.shape
            win_row = np.hanning(h)
            win_col = np.hanning(w)
            window = np.outer(win_row, win_col)
            self.img = img * window
        else:
            self.img = img
        self.fig, (self.ax_img, self.ax_fft) = plt.subplots(1, 2, figsize=(10, 5))
        self.rect_selector = RectangleSelector(
            self.ax_img, self.onselect_rectangle, useblit=True,
            button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )
        self.rect_selector.set_active(True)
        self.cid_circle = self.fig.canvas.mpl_connect('button_press_event', self.on_circle_press)
        self.cid_circle_release = self.fig.canvas.mpl_connect('button_release_event', self.on_circle_release)
        self.cid_circle_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_circle_motion)
        self.circle_press = None
        self.roi_rect_patch = None
        self.img_disp = self.ax_img.imshow(self.img, cmap='gray')
        self.ax_img.set_title('Select ROI')
        self.fft_disp = self.ax_fft.imshow(np.zeros_like(self.img), cmap='jet')
        self.ax_fft.set_title('FFT Magnitude (ROI)')
        # Add a close button
        close_ax = self.fig.add_axes([0.85, 0.01, 0.1, 0.05])
        self.close_button = Button(close_ax, 'Close')
        self.close_button.on_clicked(self.close_figure)
        # Add a toggle button for ROI type
        toggle_ax = self.fig.add_axes([0.7, 0.01, 0.13, 0.05])
        self.toggle_button = Button(toggle_ax, 'Toggle ROI (Rect/Circle)')
        self.toggle_button.on_clicked(self.toggle_roi_type)
        plt.tight_layout()
        plt.show()

    def toggle_roi_type(self, event):
        if self.roi_type == 'rectangle':
            self.roi_type = 'circle'
            self.rect_selector.set_active(False)
            self.circle_selector_active = True
            self.ax_img.set_title('Select ROI (Circle: click center, drag to edge)')
        else:
            self.roi_type = 'rectangle'
            self.rect_selector.set_active(True)
            self.circle_selector_active = False
            self.ax_img.set_title('Select ROI (Rectangle)')
        self.fig.canvas.draw_idle()

    def close_figure(self, event):
        plt.close(self.fig)

    def onselect_rectangle(self, eclick, erelease):
        if self.roi_type != 'rectangle':
            return
        x1, y1 = int(np.floor(eclick.xdata)), int(np.floor(eclick.ydata))
        x2, y2 = int(np.floor(erelease.xdata)), int(np.floor(erelease.ydata))
        # Ensure proper order
        start_row, end_row = sorted([y1, y2])
        start_col, end_col = sorted([x1, x2])
        # Make end exclusive (add 1)
        end_row += 1
        end_col += 1
        # Clamp to image bounds
        start_row = max(0, start_row)
        end_row = min(self.img_orig.shape[0], end_row)
        start_col = max(0, start_col)
        end_col = min(self.img_orig.shape[1], end_col)
        # Extract ROI from original image, then apply vignette and FFT
        img_roi = self.img_orig[start_row:end_row, start_col:end_col]
        self.roi = img_roi
        if self.vignette:
            h, w = img_roi.shape
            win_row = np.hanning(h)
            win_col = np.hanning(w)
            window = np.outer(win_row, win_col)
            img_roi = img_roi * window
        fft_result = np.fft.fftshift(np.fft.fft2(img_roi))
        fft_mag_roi = np.log1p(np.abs(fft_result))
        # Update ROI rectangle
        if self.roi_rect_patch:
            self.roi_rect_patch.remove()
        self.roi_rect_patch = patches.Rectangle(
            (start_col, start_row), end_col - start_col, end_row - start_row,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        self.ax_img.add_patch(self.roi_rect_patch)
        # Remove circle if present
        if self.circle_artist:
            self.circle_artist.remove()
            self.circle_artist = None
        # Update FFT plot
        self.fft_disp.remove()  # Remove the old image
        self.fft_disp = self.ax_fft.imshow(
            fft_mag_roi, cmap='jet', origin='upper',
            vmin=np.min(fft_mag_roi), vmax=np.max(fft_mag_roi)
        )
        self.ax_fft.set_title('FFT Magnitude (ROI)')
        self.ax_fft.set_xlim(0, fft_mag_roi.shape[1])
        self.ax_fft.set_ylim(fft_mag_roi.shape[0], 0)
        self.fig.canvas.draw_idle()

    def on_circle_press(self, event):
        if not self.circle_selector_active or event.inaxes != self.ax_img:
            return
        self.circle_press = (event.xdata, event.ydata)
        if self.circle_artist:
            self.circle_artist.remove()
            self.circle_artist = None

    def on_circle_release(self, event):
        if not self.circle_selector_active or self.circle_press is None or event.inaxes != self.ax_img:
            return
        x0, y0 = self.circle_press
        x1, y1 = event.xdata, event.ydata
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return
        # Calculate center and radius
        cx, cy = x0, y0
        radius = np.hypot(x1 - x0, y1 - y0)
        self.circle_params = (cy, cx, radius)
        # Extract circular ROI from original image
        y1b = int(max(0, cy - radius))
        y2b = int(min(self.img_orig.shape[0], cy + radius))
        x1b = int(max(0, cx - radius))
        x2b = int(min(self.img_orig.shape[1], cx + radius))
        img_roi = self.img_orig[y1b:y2b, x1b:x2b]
        h, w = img_roi.shape
        Y, X = np.ogrid[:h, :w]
        mask = (Y - (cy - y1b))**2 + (X - (cx - x1b))**2 <= radius**2
        img_roi = img_roi * mask
        self.roi = img_roi
        if self.vignette:
            win_row = np.hanning(h)
            win_col = np.hanning(w)
            window = np.outer(win_row, win_col)
            img_roi = img_roi * window
        fft_result = np.fft.fftshift(np.fft.fft2(img_roi))
        fft_mag_roi = np.log1p(np.abs(fft_result))
        # Remove rectangle if present
        if self.roi_rect_patch:
            self.roi_rect_patch.remove()
            self.roi_rect_patch = None
        # Draw circle
        if self.circle_artist:
            self.circle_artist.remove()
        self.circle_artist = patches.Circle((cx, cy), radius, linewidth=2, edgecolor='r', facecolor='none')
        self.ax_img.add_patch(self.circle_artist)
        # Update FFT plot
        self.fft_disp.remove()
        self.fft_disp = self.ax_fft.imshow(
            fft_mag_roi, cmap='jet', origin='upper',
            vmin=np.min(fft_mag_roi), vmax=np.max(fft_mag_roi)
        )
        self.ax_fft.set_title('FFT Magnitude (ROI)')
        self.ax_fft.set_xlim(0, fft_mag_roi.shape[1])
        self.ax_fft.set_ylim(fft_mag_roi.shape[0], 0)
        self.fig.canvas.draw_idle()
        self.circle_press = None

    def on_circle_motion(self, event):
        if not self.circle_selector_active or self.circle_press is None or event.inaxes != self.ax_img:
            return
        x0, y0 = self.circle_press
        x1, y1 = event.xdata, event.ydata
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return
        cx, cy = x0, y0
        radius = np.hypot(x1 - x0, y1 - y0)
        # Remove previous circle
        if self.circle_artist:
            self.circle_artist.remove()
        self.circle_artist = patches.Circle((cx, cy), radius, linewidth=2, edgecolor='r', facecolor='none', alpha=0.5)
        self.ax_img.add_patch(self.circle_artist)
        self.fig.canvas.draw_idle()

# To use the interactive plotter, uncomment the following line:
InteractiveFFT(tiff_path, vignette=vignette)

# %%
