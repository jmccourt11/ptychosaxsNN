#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.transform import resize
import cmasher as cmr
from scipy.ndimage import gaussian_filter
from pathlib import Path
import importlib
import os
import sys
from tqdm import tqdm
import torch
plt.rcParams['image.cmap'] = 'jet'
cmap = cmr.get_sub_cmap('jet', 0.,0.5)

#%%
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../'))) 
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)

#%%
# basepath
#basepath=Path('Y:/2024_Dec/misc/dps/RC02_3D/')
basepath=Path('/net/micdata/data2/12IDC/2024_Dec/misc/dps/RC02_3D')


# load dps
filepath=basepath / 'full_scan_888.npy'
print(f'file: {filepath}')
dps = np.load(filepath)
dps_sum = np.sum(dps,axis=0)
full_dps=dps

# dps = ptNN_U.load_h5_scan_to_npy(Path(f'/net/micdata/data2/12IDC/2024_Dec/ptycho/'),888,plot=False)
# dps_sum=np.sum(dps,axis=0)
# full_dps=dps


# plot frames
count=0

#peaks=ptNN_U.find_peaks2d(final,center_cut=center_cut,n=n,threshold=threshold,plot=False)
# peaks=[(591,299),(369,281),(1237,479),
#        (1306,575),(1328,802),(154,665),
#        (183,889),(257,983),(908,1165),
#        (1110,1249),(401,1042),(1085,410),
#        (987,258),(481,1206)]
peaks=[(299,591),(281,369),(479,1237),
       (575,1306),(802,1328),(665,154),
       (889,183),(983,257),(1165,908),
       (1249,1110),(1042,401),(410,1085),
       (258,987),(1206,481)]






#FOR CINDY
#%%
# Load model
m=ptNN.ptychosaxsNN()
#path='/net/micdata/data2/12IDC/ptychosaxs/'
path=Path('Y:/ptychosaxs/')
model_path='models/best_model_Unet_cindy.pth'
# peak_params
center_cut=75
n=10
threshold=0.25
m.load_model(state_dict_pth=path / model_path)
m.set_device()
m.model.to(m.device)
m.model.eval()

# For radius of circular averages
r=28

plot=True
save=False
all_peaks=[]
save_peaks=False
scan=1115
print(scan)
# Load h5 data
full_dps_orig=ptNN_U.load_h5_scan_to_npy(Path('Y:/2021_Nov/ptycho/'),scan=scan,plot=False)


# Crop original data to 512x512 and remove hot pixel
full_dps=full_dps_orig.copy()[:,1:513,259:771]
for i,dp_pp in enumerate(full_dps):
    dp_pp[dp_pp >= 2**16-1] = np.min(dp_pp) #get rid of hot pixel

# Summed scan 
dps_copy=np.sum(full_dps,axis=0)

# Preprocess and run data through NN
resultT,sfT,bkgT=ptNN_U.preprocess_cindy(dps_copy) # preprocess
resultTa=resultT.to(device=m.device, dtype=torch.float) #convert to tensor and send to device
final=m.model(resultTa).detach().to("cpu").numpy()[0][0] #pass through model and convert to np.array

# Plot
if plot:
    fig,ax=plt.subplots(1,3)
    im1=ax[0].imshow(dps_copy,norm=colors.LogNorm())
    im2=ax[1].imshow(resultT[0][0])
    im3=ax[2].imshow(final)
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.colorbar(im3)
    plt.show()
    
    


# Find and plot peaks in NN result deconvolution, plot over frame
# peaks=find_peaks_2d_filter(final,center_cut=center_cut,n=n,threshold=threshold,plot=False)
peaks=ptNN_U.find_peaks2d(final,center_cut=center_cut,n=n,threshold=threshold,plot=False)
center_x,center_y=134,124
adjusted=[]
peaks_shifted=[(p[0]-center_x,p[1]-center_y) for p in peaks] 
updated_peaks=ptNN_U.ensure_inverse_peaks(peaks_shifted)
updated_peaks_unshifted=[(p[0]+center_x,p[1]+center_y) for p in updated_peaks]
peaks=updated_peaks_unshifted
all_peaks.append({"scan":scan,"peaks":peaks})


peak_y,peak_x=zip(*peaks)



dps_sub=full_dps.copy()
dps_sum_sub=[]
for i in range(0,len(dps_sub)):
    test=np.subtract(dps_sub[i],full_dps[0],dtype=float)
    dps_sum_sub.append(test)
dps_sum_sub=np.array(dps_sum_sub)
dps_sum_sub=np.sum(dps_sum_sub,axis=0)


radius = 28#56   # Define the neighborhood radius
intensities_sum=[]
for peak in peaks:
    x, y = peak
    #intensity = ptNN_U.circular_neighborhood_intensity(dps_sum_sub, x, y, radius=radius,plot=False)#-bkg
    intensity = ptNN_U.circular_neighborhood_intensity(ptNN_U.resize_dp(dps_sum_sub), x, y, radius=radius,plot=False)#-bkg
    
    #intensity = ptNN_U.circular_neighborhood_intensity(dps_sum, x, y, radius=radius,plot=False)#-bkg
    intensities_sum.append(intensity)
    print(f"Peak at ({x}, {y}) has neighborhood integrated intensity: {intensity}")




#CINDY
fig,ax=plt.subplots(1,2)
ax[0].imshow(ptNN_U.resize_dp(dps_copy),norm=colors.LogNorm(),cmap='jet')
ax[0].scatter(peak_x,peak_y,color='red',marker='x',s=100)
ax[1].imshow(final,cmap='jet')
ax[1].scatter(peak_x,peak_y,color='red',marker='x',s=100)
plt.show()

dps=full_dps.copy()
dps_sum=np.sum(full_dps,axis=0)









image_size = ptNN_U.resize_dp(dps[0]).shape
images = np.asarray([ptNN_U.resize_dp(dp) for dp in dps[:]])

# image_size = dps[0].shape
# images = dps[:]

# Create a grid of images (31x31 for 961 images)
#grid_size = int(np.sqrt(len(images)))
grid_size_row=15#37
grid_size_col=20#26


# #grid_image = np.zeros((grid_size_row * image_size[0], grid_size_col * image_size[1]))
# grid_image = np.zeros((grid_size_col * image_size[1], grid_size_row * image_size[0]))

# for j in tqdm(range(grid_size_row)):
#     for i in range(grid_size_col):
#         try:
#             grid_image[
#                 i * image_size[0]:(i + 1) * image_size[0],
#                 j * image_size[1]:(j + 1) * image_size[1]
#             ] = images[i * grid_size_row + j]
#         except:
#             grid_image[
#                 i * image_size[0]:(i + 1) * image_size[0],
#                 j * image_size[1]:(j + 1) * image_size[1]
#             ] = np.zeros(image_size)
            
            

# Initialize the grid image with correct dimensions
grid_image = np.zeros((grid_size_row * image_size[0], grid_size_col * image_size[1]))

# Iterate over rows and columns to place images
for j in tqdm(range(grid_size_row)):  # Row index
    for i in range(grid_size_col):   # Column index
        # Calculate the linear index of the current image
        image_idx = j * grid_size_col + i
        
        # Check if the index is within bounds
        if image_idx < len(images):
            grid_image[
                j * image_size[0]:(j + 1) * image_size[0],
                i * image_size[1]:(i + 1) * image_size[1]
            ] = images[image_idx]
        else:
            # If out of bounds, fill with zeros (optional)
            grid_image[
                j * image_size[0]:(j + 1) * image_size[0],
                i * image_size[1]:(i + 1) * image_size[1]
            ] = np.zeros(image_size)






#%%
# Plot the combined image
# Plot the combined grid image
fig, ax = plt.subplots(1,2,figsize=(12, 12))
ax[0].imshow(grid_image, norm=colors.LogNorm(),cmap=cmap)
ax[0].axis('off')


# overlay_data = [
#     (np.random.randint(0, image_size[1], 10),  # X coordinates
#      np.random.randint(0, image_size[0], 10))  # Y coordinates
#     for _ in range(len(images))
# ]

overlay_data=[list(zip(*peaks)) for _ in range(len(images))]
overlay_data = [
    (
        np.array([item for item in data[1]]),  # Extract x-coordinates
        np.array([item for item in data[0]])   # Extract y-coordinates 
    )
    for data in overlay_data
]

ref=images[0]
r=radius
ss=[]
for image in images:
    # Pick specific frame
    dps_index=image

    # Calculate normalized difference map between summed and frame
    ref_index=ref.copy()
    test_index=dps_index.copy()

    # Subtracted from reference value
    sub=np.subtract(test_index,ref_index,dtype=float)

    # Example: Calculate the integrated intensity for each peak's neighborhood from the normalized difference map
    radius = r  # Define the neighborhood radius
    intensities=[]
        
    for peak in peaks:
        x, y = peak
        #intensity = ptNN_U.circular_neighborhood_intensity(sub, x, y, radius=radius,plot=False)
        intensity = ptNN_U.circular_neighborhood_intensity(ptNN_U.resize_dp(sub), x, y, radius=radius,plot=False)#-bkg
    
        #intensity = ptNN_U.circular_neighborhood_intensity(test_index, x, y, radius=radius,plot=False)
        intensities.append(intensity)

    # Normalize intensities
    s=np.array([intensities[i]/intensities_sum[i]/np.max(intensities_sum) for i in range(0,len(intensities))])
    alphas=[(i-np.min(s))/(np.max(s)-np.min(s)) for i in s]
    ss.append(s)
ss=np.asarray(ss)
test_ss=np.array([(s-np.min(ss))/(np.max(ss)-np.min(ss)) for s in ss])



#%%
fig,ax=plt.subplots(1,2)
# Overlay the scatter plots
for j in range(grid_size_row):
    for i in range(grid_size_col):
        x_offset = i * image_size[1]
        y_offset = j * image_size[0]
        print(x_offset,y_offset)
        
        # Get the scatter data for this image
        try:
            #y_scatter,x_scatter = overlay_data[j * grid_size_row + i]
            x_scatter,y_scatter=overlay_data[0]
            #adjusted_y_scatter = image_size[1] - y_scatter
        #Overlay scatter points (adjust for grid offsets)
            ax[0].imshow(np.zeros((grid_size_row,grid_size_col)))
            ax[0].scatter(
                x_scatter + x_offset,  # Adjust Y by row offset
                y_scatter + y_offset,  # Adjust X by column offset
                color='red', s=10,alpha=test_ss[j * grid_size_row + i]
            )
            #ax[0].invert_yaxis()

        except:
            continue
        
        
ax[1].imshow(ptNN_U.resize_dp(dps_sum),norm=colors.LogNorm())
#ax[1].imshow(dps_sum,norm=colors.LogNorm())
x_scatter, y_scatter = overlay_data[0]
ax[1].scatter(x_scatter,y_scatter,color='red', s=100, alpha=np.array([intensities_sum[i]/np.max(intensities_sum) for i in range(0,len(intensities_sum))]))
plt.show()
# %%
