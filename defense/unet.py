import torch.nn as nn
import torch


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, dim_out, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, dim_out, padding_type, norm_layer, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim_out, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim_out),
                       nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim_out)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


def double_conv(in_channels, out_channels, bn=1, bias=False):
    if bn == 'bn':
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    elif bn == 'in':
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
        )   
    else:
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias),
            nn.LeakyReLU(0.2,inplace=True),
        ) 
    return block


class UNet(nn.Module):
    def __init__(self,bn=True,repeat_num=5):
        super().__init__()
        indim = 3
        self.conv_first = double_conv(indim, 64, bn)
        self.dconv_down1 = double_conv(64, 128, bn)
        self.dconv_down2 = double_conv(128, 256, bn)
        self.maxpool = nn.MaxPool2d(2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.dconv_up2 = double_conv(256 + 256, 512, bn)
        self.dconv_up1 = double_conv(128 + 512, 256, bn)        
        self.dconv_up0 = double_conv(256 + 64, 128, bn)

        self.conv_last = nn.Conv2d(128, 3, 1)
        self.tanh = nn.Sigmoid()
        
        norm_layer=nn.BatchNorm2d

        bottle_neck_lis = []
        for _ in range(repeat_num):
            bottle_neck_lis.append(ResnetBlock(256,256,padding_type='zero',norm_layer=norm_layer))

        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        
    def forward(self, img, label = None,cat_img = False):
        x_ = img
        conv0 = self.conv_first(x_)
        conv1 = self.dconv_down1(conv0)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        x = self.bottle_neck(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)     
        x = self.dconv_up1(x)
        
        x = torch.cat([x, conv0], dim=1)
        x = self.dconv_up0(x)
        out = self.conv_last(x)

        if cat_img:
            out = self.tanh(out + img)
        else:
            out = self.tanh(out)   
        return out
        

class UNet_Res_2(nn.Module):
    def __init__(self,bn=True,embedding_dim=-1,repeat_num=5):
        super().__init__()
        
        self.embedding_dim = embedding_dim

        if self.embedding_dim != -1:
            self.embeddings = nn.Embedding(num_embeddings=74,embedding_dim=embedding_dim)
            indim = 8*self.embedding_dim
        else:
            indim = 0

        self.conv_first = double_conv(3, 64, bn)

        self.dconv_down1 = double_conv(64, 128, bn)
        self.dconv_down2 = double_conv(128, 256, bn)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up2 = double_conv(256 + 512, 512, bn)
        self.dconv_up1 = double_conv(128 + 512, 256, bn)        
        self.dconv_up0 = double_conv(256 + 64, 128, bn)

        self.conv_last = nn.Conv2d(128, 3, 1)
        self.tanh = nn.Tanh()
        
        
        if bn == 'in':
            norm_layer=nn.InstanceNorm2d
        else:
            norm_layer=nn.BatchNorm2d
            
        bottle_neck_lis = []
        bottle_neck_lis.append(double_conv(256+indim,512,bn=bn,bias=False))
        for _ in range(repeat_num-1):
            bottle_neck_lis.append(ResnetBlock(512,512,padding_type='zero',norm_layer=norm_layer))
            # bottle_neck_lis.append(ResidualBlock(256+indim,)

        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        
    def forward(self, img, label = None,cat_img = False):
        conv0 = self.conv_first(img)
        
        conv1 = self.dconv_down1(conv0)
        x = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        if self.embedding_dim != -1:
            label_embeding = self.embeddings(label).view(x.size(0),-1,1,1)
            label_embeding = label_embeding.repeat(1,1,x.size(2),x.size(3))
            x = torch.cat((x, label_embeding), dim=1)
        else:
            x = x
        x = self.bottle_neck(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)     
        x = self.dconv_up1(x)
        
        x = torch.cat([x, conv0], dim=1)
        x = self.dconv_up0(x)
        
        out = self.conv_last(x)

        if cat_img:
            out = self.tanh(out + img)
        else:
            out = self.tanh(out)   
        return out
        


class Generator_StarGan(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=128, embedding_dim=-1, repeat_num=6):
        super(Generator_StarGan, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        if self.embedding_dim != -1:
            self.embeddings = nn.Embedding(num_embeddings=74,embedding_dim=embedding_dim)
            indim = 3 + 8*self.embedding_dim
        else:
            indim = 3

        layers = []
        layers.append(nn.Conv2d(indim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self,img, label=None,cat_img=False):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        if self.embedding_dim != -1:
            label_embeding = self.embeddings(label).view(img.size(0),-1,1,1)
            label_embeding = label_embeding.repeat(1,1,img.size(2),img.size(3))
            x_ = torch.cat((img,label_embeding),dim=1)
        else:
            x_ = img
        y = self.main(x_)
        return y



