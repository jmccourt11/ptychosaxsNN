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
#basepath=Path('Y:/2024_Dec/ptycho/')


# load dps

# filepath=basepath / 'full_scan_888.npy'
# print(f'file: {filepath}')
# dps = np.load(filepath)
# dps_sum = np.sum(dps,axis=0)
# full_dps=dps[:]

dps = ptNN_U.load_h5_scan_to_npy(Path(f'/net/micdata/data2/12IDC/2024_Dec/ptycho/'),455,plot=False)
dps_sum=np.sum(dps,axis=0)
full_dps=dps

dps_size=dps[0].shape
offset=100
dpsize=512#1280
full_dps=full_dps.copy()[:,dps_size[0]//2-offset - dpsize//2:dps_size[0]//2-offset + dpsize//2,dps_size[1]//2 - dpsize//2:dps_size[1]//2 + dpsize//2]
for i,dp_pp in enumerate(full_dps):
    dp_pp[dp_pp >= 2**16-1] = np.min(dp_pp) #get rid of hot pixel


# Convert row/col indices to linear indices in the DPS array
col_start = 0
col_end = 41 #37  # Total width is 37
row_start = 0
row_end = 39 #26  # Total height is 26

grid_size_col=col_end-col_start
grid_size_row=row_end-row_start

# Calculate the indices for the subset
selected_indices = []
for row in range(row_start, row_end):  # Remove the +1
    for col in range(col_start, col_end):  # Remove the +1
        idx = row * col_end + col  # Use col_end as the width
        if idx < len(dps):  # Make sure we don't exceed array bounds
            selected_indices.append(idx)

# Extract the subset of DPS data
dp_test = full_dps[selected_indices, :, :]


# Optional: Verify the shape
print(f"Original DPS shape: {full_dps.shape}")
print(f"Selected DPS shape: {dp_test.shape}")

# Optional: Visualize the sum of the selected region
plt.figure()
plt.imshow(np.sum(dp_test, axis=0), norm=colors.LogNorm(), cmap='jet')
plt.title('Sum of Selected Region')
plt.show()





full_dps=dp_test
dps_sum=np.sum(full_dps,axis=0)
# plot frames
count=0



#%%
# Load model
m=ptNN.ptychosaxsNN()
path=Path('/net/micdata/data2/12IDC/ptychosaxs/')
model_path='models/best_model_diff_sim15_zhihua_JM02_3D.pth'
# peak_params
center_cut=75
n=10
threshold=0.25
m.load_model(state_dict_pth=path / model_path)
m.set_device()
m.model.to(m.device)
m.model.eval()




# Preprocess and run data through NN
mask = np.where(dps_sum <= 0, 0, 1)
resultT,sfT,bkgT=ptNN_U.preprocess_zhihua(dps_sum,mask) # preprocess
resultTa=resultT.to(device=m.device, dtype=torch.float) #convert to tensor and send to device
final=m.model(resultTa).detach().to("cpu").numpy()[0][0] #pass through model and convert to np.array



fig,ax=plt.subplots(1,3)
im1=ax[0].imshow(dps_sum,norm=colors.LogNorm())
im2=ax[1].imshow(resultT[0][0])
im3=ax[2].imshow(final)
plt.colorbar(im1)
plt.colorbar(im2)
plt.colorbar(im3)
plt.show()

#%%


# peaks=ptNN_U.find_peaks2d(final,center_cut=center_cut,n=n,threshold=threshold,plot=False)

# scan 888
# full ~1400x1600 dps
# peaks=[(591,299),(369,281),(1237,479),
#        (1306,575),(1328,802),(154,665),
#        (183,889),(257,983),(908,1165),
#        (1110,1249),(401,1042),(1085,410),
#        (987,258),(481,1206)]
# peaks=[(299,591),(281,369),(479,1237),
#        (575,1306),(802,1328),(665,154),
#        (889,183),(983,257),(1165,908),
#        (1249,1110),(1042,401),(410,1085),
#        (258,987),(1206,481)]
#1280x1280 dps
# peaks=[(205,300),(195,491),(565,65),
#        (805,96),(905,169),(986,288),
#        (1109,376),(1071,809),(1113,989),
#        (714,1211),(475,1211),(378,1142),
#        (313,993),(150,892)]
#to flip or not to flip
#peaks=[(300,205),(491,195),(65,565),
#       (96,805),(169,905),(288,986),
#       (376,1109),(809,1071),(989,1113),
#       (1211,714),(1211,475),(1142,378),
#       (993,313),(892,150)]

#scan 911
peaks=[(804,58),(1043,175),(1161,297),
       (164,1020),(240,1131),(468,1245),
       (946,96),(316,1211),(1140,1023)]#,(141,333)]


peak_y,peak_x=zip(*peaks)


dps_sub=full_dps.copy()
dps_sum_sub=[]
for i in range(0,len(dps_sub)):
    test=np.subtract(dps_sub[i],full_dps[0],dtype=float)
    dps_sum_sub.append(test)
dps_sum_sub=np.array(dps_sum_sub)
dps_sum_sub=np.sum(dps_sum_sub,axis=0)


radius =56 # Define the neighborhood radius
intensities_sum=[]
for peak in peaks:
    x, y = peak
    intensity = ptNN_U.circular_neighborhood_intensity(dps_sum_sub, x, y, radius=radius,plot=False)#-bkg
    #intensity = ptNN_U.circular_neighborhood_intensity(dps_sum, x, y, radius=radius,plot=False)#-bkg
    intensities_sum.append(intensity)
    print(f"Peak at ({x}, {y}) has neighborhood integrated intensity: {intensity}")


dps_copy=full_dps.copy()
fig,ax=plt.subplots()
ax.imshow(np.sum(dps_copy,axis=0),norm=colors.LogNorm(),cmap='jet')
ax.scatter(peak_x,peak_y,color='red',marker='x',s=100)
plt.show()



#%%
dps=full_dps.copy()
dps_sum=np.sum(full_dps,axis=0)


image_size = dps[0].shape
images = dps[:]

# Create a grid of images (31x31 for 961 images)
#grid_size = int(np.sqrt(len(images)))

#grid_size_row=26#15 #37
#grid_size_col=37#20 #26   

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




# Plot the combined image
# Plot the combined grid image
fig, ax = plt.subplots(1,2,figsize=(12,12))
ax[0].imshow(grid_image, norm=colors.LogNorm(),cmap=cmap,clim=(1,2))
ax[0].axis('off')


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
print("Calculating intensities at peaks...")
for image in tqdm(images):
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
        intensity = ptNN_U.circular_neighborhood_intensity(sub, x, y, radius=radius,plot=False)
        #intensity = ptNN_U.circular_neighborhood_intensity(test_index, x, y, radius=radius,plot=False)
        intensities.append(intensity)

    # Normalize intensities
    s=np.array([intensities[i]/intensities_sum[i]/np.max(intensities_sum) for i in range(0,len(intensities))])
    alphas=[(i-np.min(s))/(np.max(s)-np.min(s)) for i in s]
    ss.append(s)
    
#ss=np.asarray(ss)
test_ss=np.array([(s-np.min(ss))/(np.max(ss)-np.min(ss)) for s in ss])

#fig,ax=plt.subplots(1,2)
# Overlay the scatter plots
print('Plotting overlay...')
for i in tqdm(range(grid_size_row)):
    for j in range(grid_size_col):
        x_offset = j * image_size[0]
        y_offset = i * image_size[1]
        #print(x_offset,y_offset)
        
        # Get the scatter data for this image
        try:
            #y_scatter,x_scatter = overlay_data[j * grid_size_row + i]
            x_scatter,y_scatter=overlay_data[0]
            #adjusted_y_scatter = image_size[1] - y_scatter
            #Overlay scatter points (adjust for grid offsets)
            #ax[0].imshow(np.zeros((grid_size_row,grid_size_col)))
            
            ax[0].scatter(
                x_scatter + x_offset,  # Adjust Y by row offset
                y_scatter + y_offset,  # Adjust X by column offset
                color='red', s=1,alpha=test_ss[i* grid_size_col + j]
            )
            
            #ax[0].invert_yaxis()

        except:
            continue

ax[1].imshow(dps_sum,norm=colors.LogNorm())
x_scatter, y_scatter = overlay_data[0]
ax[1].scatter(x_scatter,y_scatter,color='red', s=100, alpha=np.array([intensities_sum[i]/np.max(intensities_sum) for i in range(0,len(intensities_sum))]))
plt.show()
# %%
