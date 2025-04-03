import torch
import torch.nn as nn

class recon_model(nn.Module):

  def __init__(self,nconv=64):
      super(recon_model, self).__init__()
      self.nconv=nconv
      #convoluted diffraction pattern encoder
      self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
        nn.Conv2d(in_channels=2, out_channels=self.nconv, kernel_size=3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv),
        nn.ReLU(),
        nn.Conv2d(self.nconv, self.nconv, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Dropout(0.5),
        
        nn.Conv2d(self.nconv, self.nconv*2, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*2),
        nn.ReLU(),
        nn.Conv2d(self.nconv*2, self.nconv*2, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*2),          
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Dropout(0.5),
          
        nn.Conv2d(self.nconv*2, self.nconv*4, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*4),
        nn.ReLU(),
        nn.Conv2d(self.nconv*4, self.nconv*4, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*4),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Dropout(0.5),
          
        )

      #ideal diffraction pattern decoder
      self.decoder1 = nn.Sequential(
      
        nn.Conv2d(self.nconv*4, self.nconv*4, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*4),
        nn.ReLU(),
        nn.Conv2d(self.nconv*4, self.nconv*4, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*4),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear'),

        nn.Conv2d(self.nconv*4, self.nconv*2, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*2),
        nn.ReLU(),
        nn.Conv2d(self.nconv*2, self.nconv*2, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*2),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear'),
          
        nn.Conv2d(self.nconv*2, self.nconv*2, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*2),
        nn.ReLU(),
        nn.Conv2d(self.nconv*2, self.nconv*2, 3, stride=1, padding=(1,1)),
        nn.BatchNorm2d(self.nconv*2),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear'),

        nn.Conv2d(self.nconv*2, 1, 3, stride=1, padding=(1,1)),
        nn.Sigmoid()
        #nn.ReLU()
        #nn.Tanh()
        )

  
  def forward(self,x):
      x1 = self.encoder(x)
      amp = self.decoder1(x1)

      return amp
