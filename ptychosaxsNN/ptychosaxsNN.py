#%%
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../'))) 
from utils.ptychosaxsNN_utils import *
from models.UNet import recon_model
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../deconvolutionNN/'))) 
import deconvolutionNN
#%%
class ptychosaxsNN:
    def __init__(self):
        self.model=None
        self.nconv=64 #same number for convolutions for all networks so far
        self.probe=None
        self.device=None
        self.full_data=None
        self.sum_data=None
        self.ptycho_hdf5=None
        self.scan=None
        
    def __repr__(self):
        return f'ptychosaxsNN (model: {self.model!r}, probe: {self.probe!r}, device: {self.device!r})'
           
    #def load_probe(self,probefile):
    #    self.probe=np.load(probefile)
    
    def load_probe(self,probe_file,file_format='mat'):
        if file_format=='mat':
            # Load reconstructed probe from ptychoshelves recon and take only the first mode
            probe = sio.loadmat(probe_file)['probe'][:,:,0,0]
            self.probe = probe
        else:
            print('Need *.mat formatted probe')
            
    def load_model(self,state_dict_pth=None,load_state_dict=True):
        self.model= recon_model(self.nconv)

        if load_state_dict and state_dict_pth!=None:
            # Load the state_dict on the CPU first to avoid memory issues
            state_dict=torch.load(state_dict_pth, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
        
    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            #self.model = nn.parallel.DistributedDataParallel(self.model)
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)
        
    def model_size_in_megabytes(self):
        param_size = 0
        for param in x.model.parameters():
            param_size += param.numel() * param.element_size()  # numel() gives number of elements, element_size() gives size in bytes
        param_size /= (1024 ** 2)  # Convert bytes to megabytes
        print(f"Model size: {param_size:.2f} MB")
        
    def load_h5(self,exp_dir,scan):
        #e.g. 2024_Dec , 630
        file_path=Path(f'/net/micdata/data2/12IDC/{exp_dir}/ptycho/')
        res=load_h5_scan_to_npy(file_path=file_path,scan=scan,plot=False)
        self.full_data=res
        self.sum_data=np.sum(res,axis=0)
        self.scan=scan

    def load_hdf5(self,exp_dir,sample_name,scan):
        # e.g. 2024_Dec , JM02_3D_ , 630
        file_path=Path(f'/net/micdata/data2/12IDC/{exp_dir}/results/{sample_name}/')
        self.ptycho_hdf5=load_hdf5_scan_to_npy(file_path=file_path,scan=scan,plot=False)
        self.scan=scan
        
                
#%%
if __name__ == "__main__":
    x=ptychosaxsNN()
    #path = os.path.abspath(os.path.join(os.getcwd(), '../'))
    local=True
    if local:
        path = Path("Y:/ptychosaxs")
    else:
        path = Path('/net/micdata/data2/12IDC/ptychosaxs/')
        
    x.load_model(state_dict_pth=path / 'models/best_model_Unet_cindy.pth')   
    x.set_device()
    x.model.to(x.device)
    x.model.eval()
    #%%
    


    date_dir = '2024_Dec'
    exp_dir='JM03_3D_'
    scans=np.arange(706,720,1)

    directory='results' #'test'
    Ndp=512
    filepath=Path("Y:/") / f'{date_dir}/{directory}/{exp_dir}/fly{scans[0]:03d}/data_roi0_Ndp{Ndp}_dp.hdf5'
    data=read_hdf5_file(filepath)
    dps_copy = np.sum(data['dp'],axis=0)
    
      
    # # Load data
    # scan=1125 #1115,1083,1098
    # filename = path / f'data/cindy_scan{scan}_diffraction_patterns.npy'
    # full_dps_orig=np.load(filename)
    # full_dps=full_dps_orig.copy()
    # for dp_pp in full_dps:
    #     dp_pp[dp_pp >= 2**16-1] = np.min(dp_pp) #get rid of hot pixel
    
    # # Plot and return a full scan
    # # inputs,outputs,sfs,bkgs=plot_and_save_scan(full_dps,x,scanx=20,scany=15)
    
    # # Summed scan 
    # dps_copy=np.sum(full_dps[:,1:513,259:771],axis=0)
    
#%%    
    # # Specific frame
    # index=230
    # dps_copy=full_dps[index,1:513,259:771]
    
    # Preprocess and run data through NN
    resultT,sfT,bkgT=preprocess_cindy(dps_copy) # preprocess
    resultTa=resultT.to(device=x.device, dtype=torch.float) #convert to tensor and send to device
    final=x.model(resultTa).detach().to("cpu").numpy()[0][0] #pass through model and convert to np.array
    
    # Plot
    fig,ax=plt.subplots(1,3)
    im1=ax[0].imshow(dps_copy,norm=colors.LogNorm())
    im2=ax[1].imshow(resultT[0][0])
    im3=ax[2].imshow(final)
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.colorbar(im3)
    plt.show()

# %%
