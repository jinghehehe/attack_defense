import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, dim_out,padding_type, norm_layer, use_bias)

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


class Discriminator(nn.Module):
    def __init__(self, image_nc,expand_ratio=1):
        super(Discriminator, self).__init__()
        # MNIST: 1*32*100   (32-4)/2 + 1  
        
        block1 = [
            nn.Conv2d(image_nc, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 16 50

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 16 50
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 8 25

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 4 12

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 2 6
        ]
        
        self.block1 = nn.Sequential(*block1)

        self.fc = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,1)

    def forward(self, x):
        
        output = self.block1(x)
        # print(output.shape)

        output = F.adaptive_max_pool2d(output,1).squeeze()
        output = self.fc2(F.relu(self.fc(output)))
        # output = F.sigmoid(output)
        
        return output.squeeze()


class Discriminator_NOBN(nn.Module):
    def __init__(self, image_nc,expand_ratio=1):
        super(Discriminator_NOBN, self).__init__()
        # MNIST: 1*32*100   (32-4)/2 + 1  
        
        block1 = [
            nn.Conv2d(image_nc, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 16 50

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 16 50
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 8 25

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 4 12

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2,2), # 2 6
        ]
        
        self.block1 = nn.Sequential(*block1)
        self.fc = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,1)

    def forward(self, x):
        
        output = self.block1(x)
        # print(output.shape)

        output = F.adaptive_max_pool2d(output,1).squeeze()
        output = self.fc2(F.relu(self.fc(output)))
        
        return output.squeeze()

class better_upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super(better_upsampling, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',align_corners=True)

    def forward(self, x):
        x = self.up(x)
        x = F.pad(x, (3 // 2, int(3 / 2),3 // 2, int(3 / 2)))
        x = self.conv(x)
        return x

class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()
        ex = 4

        encoder_lis = [
            nn.Conv2d(gen_input_nc, 8*ex, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(8*ex),
            nn.ReLU(),
            nn.Conv2d(8*ex, 16*ex, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(16*ex),
            nn.ReLU(),
            nn.Conv2d(16*ex, 32*ex, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(32*ex),
            nn.ReLU(),
        ]
        
        norm_layer = nn.InstanceNorm2d
        bottle_neck_lis = [ResnetBlock(32*ex,norm_layer=norm_layer),
                       ResnetBlock(32*ex,norm_layer=norm_layer),
                       ResnetBlock(32*ex,norm_layer=norm_layer),
                       ResnetBlock(32*ex,norm_layer=norm_layer),]
        
        decoder_lis = [
            # nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            better_upsampling(32*ex,16*ex,2),
            nn.InstanceNorm2d(16*ex),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            better_upsampling(16*ex,8*ex,2),
            # nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8*ex),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            better_upsampling(8*ex,image_nc,1),
            # nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        
        self.handlers = []
        self.layer_name = 'decoder.7'
        # self._register_hook()
        
    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]
        print('grad \t ',output_grad[0])

    def _register_hook(self):
        # from IPython import embed; embed()
        for (name, module) in self.named_modules():
            if name == self.layer_name:
                print(name)
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    import torch

    x = torch.rand(16,3,32,100)
    
    net = Discriminator(3)
    
    cnn_params = sum([p.nelement() for p in net.parameters()])
    print("------params------[{}]".format(cnn_params/1e6))

    out3 = net(x)   
    print(out3.shape)
