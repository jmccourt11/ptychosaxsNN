import torch
import torch.nn as nn

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

        # Spatial Attention blocks
        # self.spatial_attention1 = SpatialAttention()
        # self.spatial_attention2 = SpatialAttention()
        # self.spatial_attention3 = SpatialAttention()
       
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