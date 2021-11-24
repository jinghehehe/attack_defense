import torch
import torch.nn as nn
import torch.nn.functional as F

# ***** VGG ******


def convRelu(nIn, nOut, K, S, P, bn=False, leakyRelu=False):
    cnn = []
    cnn.append( nn.Conv2d(nIn, nOut, K, S, P) )
    if bn:
        cnn.append(nn.BatchNorm2d(nOut))
    if leakyRelu:
        cnn.append(nn.LeakyReLU(0.2, inplace=True))
    else:
        cnn.append(nn.ReLU(inplace=True))
    return cnn


class VGG(nn.Module):
    def __init__(self, nclass=73):
        super(VGG, self).__init__()
        self.nclass = nclass
        cnn = []
        cnn.extend(convRelu(3,32,K=3,S=1,P=1,bn=True)) # b,32,32,100

        cnn.extend(convRelu(32,64,K=3,S=1,P=1,bn=True)) # b,32,32,100
        cnn.append(nn.MaxPool2d(2,2)) # 16,50

        cnn.extend(convRelu(64,128,K=3,S=1,P=1,bn=True))
        cnn.append(nn.MaxPool2d(2,2)) # 8,25

        cnn.extend(convRelu(128,256,K=3,S=1,P=1,bn=True)) 
        cnn.append(nn.MaxPool2d(2,2)) # 4,12

        cnn.extend(convRelu(256,256,K=3,S=1,P=1,bn=True)) # b,64,4,13
        cnn.append(nn.MaxPool2d((2,1),(2,1))) # 2,12
        
        cnn.extend(convRelu(256,512,K=2,S=1,P=(0,1),bn=True)) # b,64,1,12
        
        cnn.extend(convRelu(512,512,K=(1,3),S=(1,2),P=(0,2),bn=True)) # b,64,1,8
        
        self.cnn = nn.Sequential(*cnn)
        
        self.fc = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,nclass)

    def forward(self, input, feature=False):
        conv = self.cnn(input)  # conv features
        if feature:
            return conv
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # b c w
        
        a = conv.permute(0, 2, 1).reshape(-1, c)  # b w c
        
        a = self.fc2(F.relu(self.fc(a)))  # b w c
        output = a.reshape(b, w ,self.nclass)  #.permute(1,0,2)

        return output

# ***** RESNET ******
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_block=[3,4,6], num_classes=73):
        super().__init__()

        self.in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
    
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2) # b * 64 * 16 * 50
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)  # b * 128 * 8 * 25
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)  # b * 256 * 4 * 13
        self.conv5 = []
        self.conv5.extend(convRelu(256,512,K=3,S=(2,1),P=1, bn=True))         # b, 512, 2, 13 
        self.conv5.extend(convRelu(512,512,K=2,S=1,P=(0,1), bn=True))         # b, 512, 1, 13
        self.conv5.extend(convRelu(512,512,K=(1,3),S=(1,2),P=(0,2),bn=True))   # b, 512, 1, 8
        self.conv5_x = nn.Sequential(*self.conv5)
        
        self.fc = nn.Linear(512,256)
        self.fc2 = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        if feature:
            return output

        b, c, h, w = output.size()
        assert h == 1, "the height of conv must be 1"
        conv = output.squeeze(2)  # b 512 8
        
        a = conv.permute(0,2,1).reshape(-1,c) # b*8 , 512
        a = self.fc2(F.relu(self.fc(a)))  # b*8, 73
        
        output = a.reshape(b, w, 73)   
        return output


# ***** DENSENET ***** 

#Bottleneck layers. Although each layer only produces k output feature-maps, it typically has many more inputs. Ithas been noted in [37, 11] that a 1×1 convolution can be in- troduced as bottleneck layer before each 3×3 convolution to reduce the number of input feature-maps, and thus to improve computational efficiency."""

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#We refer to layers between blocks as transition layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The transition layers used in our experiments consist of a batch normalization layer and an 1×1 convolutional layer followed by a 2×2 average pooling layer.
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=24, reduction=0.5, num_classes=73):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks)):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        # in_ch = 24 * 24 /2 = 288, N * 288 * 4 * 13
        conv5 = []
        conv5.extend(convRelu(inner_channels, 1024,K=3,S=(2,1),P=1, bn=True))         # b, 512, 2, 13 
        conv5.extend(convRelu(1024,1024,K=2,S=1,P=(0,1), bn=True))         # b, 512, 1, 13
        conv5.extend(convRelu(1024,1024,K=(1,3),S=(1,2),P=(0,2),bn=True))   # b, 512, 1, 8
        self.conv5 = nn.Sequential(*conv5)
        
        self.fc = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, feature=False):
        output = self.conv1(x)
        output = self.features(output)
        output = self.conv5(output)
        if feature:
            return output
        b, c, h, w = output.size()
        #print(b, c, h, w)
        assert h == 1, "the height of conv must be 1"
        conv = output.squeeze(2)  # b 512 8
        a = conv.permute(0,2,1).reshape(-1,c) # b*8 , 512
        a = self.fc2(F.relu(self.fc(a)))  # b*8, 73
        output = a.reshape(b, w, 73)   
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def DenseNet121():
    # densenet 121
    #return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)
    return DenseNet(Bottleneck, [6,12,24], growth_rate=32)

# ***** TRANSFORMER *****
# Paper Link: https://arxiv.org/abs/2101.11605, Google&Berkley
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]),  requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        #print(self.rel_w.shape, self.rel_h.shape, content_position.shape)
        content_position = torch.matmul(content_position, q)
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out


class BNeck(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(BNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BotNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=73, resolution=(100, 32), heads=4):
        super(BotNet, self).__init__()
        self.in_planes = 32
        self.resolution = list(resolution)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads, mhsa=True)  # b,512,4,13
        self.layer5 = []
        self.layer5.extend(convRelu(512,768,K=3,S=(2,1),P=1, bn=True))           # b, 1024, 2, 13
        self.layer5.extend(convRelu(768,768,K=2,S=1,P=(0,1), bn=True))          # b, 1024, 1, 13
        self.layer5.extend(convRelu(768,768,K=(1,3),S=(1,2),P=(0,2),bn=True))   # b, 1024, 1, 8
        self.layer5 = nn.Sequential(*self.layer5)

        self.fc = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution[0] = (self.resolution[0]+1)/2
                self.resolution[1] = (self.resolution[1]+1)/2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        b, c, h, w = out.size()
        assert h == 1, "the height of conv must be 1"
        conv = out.squeeze(2)  # b 512 8
        a = conv.permute(0, 2, 1).reshape(-1, c)  # b*8 , 512
        a = self.fc2(F.relu(self.fc(a)))  # b*8, 73
        out = a.reshape(b, w, 73)
        return out


def BotNet50(num_classes=73, resolution=(100, 32), heads=4):
    return BotNet(BNeck, [3, 4, 6, 3], num_classes=num_classes, resolution=resolution, heads=heads)




