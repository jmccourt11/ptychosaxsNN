import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        
        # Channel-wise max pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        
        # Concatenate along the channel dimension
        concat_out = torch.cat([avg_out, max_out], dim=1)  # Shape: (B, 2, H, W)
        
        # Apply a convolution layer followed by a sigmoid activation
        attention_map = self.conv(concat_out)  # Shape: (B, 1, H, W)
        attention_map = self.sigmoid(attention_map)  # Shape: (B, 1, H, W)
        
        # Multiply attention map with the original input feature map
        return x * attention_map  # Shape: (B, C, H, W)

class recon_model(nn.Module):
    def __init__(self,nconv=64):
        super(recon_model, self).__init__()
        self.nconv=nconv
        
        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=(1,1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=(1,1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            return block

           
        def up_conv(in_channels, out_channels):
           return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

           
        def conv_last(in_channels,out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, 3, stride=1, padding=(1,1)),
                nn.Sigmoid()
            )
            return block
           
        #convoluted diffraction pattern encoder
        self.encoder1 = conv_block(1,self.nconv)
        self.encoder2 = conv_block(self.nconv,self.nconv*2)
        self.encoder3 = conv_block(self.nconv*2,self.nconv*4)

        self.pool = nn.MaxPool2d((2,2))
        self.drop = nn.Dropout(0.5)

        self.bottleneck = conv_block(self.nconv*4, self.nconv*4*2)

        #convoluted diffraction pattern decoder blocks
        self.decoder4=conv_block(self.nconv*4*2,self.nconv*4)
        self.decoder3=conv_block(self.nconv*4,self.nconv*2)
        self.decoder2=conv_block(self.nconv*2,self.nconv)

        #self.up_conv=nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_conv4=up_conv(self.nconv*4*2,self.nconv*4)
        self.up_conv3=up_conv(self.nconv*4,self.nconv*2)
        self.up_conv2=up_conv(self.nconv*2,self.nconv)

        self.conv_last=conv_last(self.nconv,1)



        # # Spatial Attention blocks
#        self.spatial_attention1 = SpatialAttention()
#        self.spatial_attention2 = SpatialAttention()
#        self.spatial_attention3 = SpatialAttention()
       
    def forward(self,x):#,p):
        x1 = self.encoder1(x)
        #x1 = self.spatial_attention1(x1)
        x2 = self.encoder2(self.drop(self.pool(x1)))
        #x2 = self.spatial_attention1(x2)
        x3 = self.encoder3(self.drop(self.pool(x2)))
        #x3 = self.spatial_attention1(x3)

        b = self.bottleneck(self.drop(self.pool(x3)))

        d3 = self.up_conv4(b)
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.decoder4(d3)

        d2 = self.up_conv3(d3)
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.decoder3(d2)


        d1 = self.up_conv2(d2)
        d1 = torch.cat((d1, x1), dim=1)
        d1 = self.decoder2(d1)

        d0 = self.conv_last(d1)

        out=d0
        return out
#        
        
        

#class recon_model(nn.Module):

#   def __init__(self,nconv=64):
#       super(recon_model, self).__init__()
#       self.nconv=nconv
#       #convoluted diffraction pattern encoder
#       self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
#         nn.Conv2d(in_channels=2, out_channels=self.nconv, kernel_size=3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv),
#         nn.ReLU(),
#         nn.Conv2d(self.nconv, self.nconv, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv),
#         nn.ReLU(),
#         nn.MaxPool2d((2,2)),
#         nn.Dropout(0.5),
#         
#         nn.Conv2d(self.nconv, self.nconv*2, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*2),
#         nn.ReLU(),
#         nn.Conv2d(self.nconv*2, self.nconv*2, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*2),          
#         nn.ReLU(),
#         nn.MaxPool2d((2,2)),
#         nn.Dropout(0.5),
#           
#         nn.Conv2d(self.nconv*2, self.nconv*4, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*4),
#         nn.ReLU(),
#         nn.Conv2d(self.nconv*4, self.nconv*4, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*4),
#         nn.ReLU(),
#         nn.MaxPool2d((2,2)),
#         nn.Dropout(0.5),
#           
#         )

#       #ideal diffraction pattern decoder
#       self.decoder1 = nn.Sequential(
#       
#         nn.Conv2d(self.nconv*4, self.nconv*4, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*4),
#         nn.ReLU(),
#         nn.Conv2d(self.nconv*4, self.nconv*4, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*4),
#         nn.ReLU(),
#         nn.Upsample(scale_factor=2, mode='bilinear'),

#         nn.Conv2d(self.nconv*4, self.nconv*2, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*2),
#         nn.ReLU(),
#         nn.Conv2d(self.nconv*2, self.nconv*2, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*2),
#         nn.ReLU(),
#         nn.Upsample(scale_factor=2, mode='bilinear'),
#           
#         nn.Conv2d(self.nconv*2, self.nconv*2, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*2),
#         nn.ReLU(),
#         nn.Conv2d(self.nconv*2, self.nconv*2, 3, stride=1, padding=(1,1)),
#         nn.BatchNorm2d(self.nconv*2),
#         nn.ReLU(),
#         nn.Upsample(scale_factor=2, mode='bilinear'),

#         nn.Conv2d(self.nconv*2, 1, 3, stride=1, padding=(1,1)),
#         nn.Sigmoid()
#         #nn.ReLU()
#         #nn.Tanh()
#         )

#   
#   def forward(self,x):
#       x1 = self.encoder(x)
#       amp = self.decoder1(x1)

#       return amp
