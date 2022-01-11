import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class TextureRefinementStage(nn.Module):
    # initializers
    def __init__(self, input_channels=3, output_channels=3, d=64):
        super(TextureRefinementStage, self).__init__()

        # Unet encoder
        ## outermost
        self.enc1 = nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=4, stride=2, padding=1)
        ## middle
        self.enc2_act = nn.LeakyReLU(0.2)
        self.enc2 = nn.Conv2d(in_channels=d, out_channels=2*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc2_bn = nn.BatchNorm2d(num_features=2*d)
        
        self.enc3_act = nn.LeakyReLU(0.2)
        self.enc3 = nn.Conv2d(in_channels=2*d, out_channels=4*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc3_bn = nn.BatchNorm2d(num_features=4*d)

        self.enc4_act = nn.LeakyReLU(0.2)
        self.enc4 = nn.Conv2d(in_channels=4*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc4_bn = nn.BatchNorm2d(num_features=8*d)
        ## bottleneck ## Add one more bottleneck to handle 512x512 inputs.
        self.enc5_act = nn.LeakyReLU(0.2)
        self.enc5 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc5_bn = nn.BatchNorm2d(num_features=8*d)

        self.enc6_act = nn.LeakyReLU(0.2)
        self.enc6 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc6_bn = nn.BatchNorm2d(num_features=8*d)

        self.enc7_act = nn.LeakyReLU(0.2)
        self.enc7 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc7_bn = nn.BatchNorm2d(num_features=8*d)

        self.enc8_act = nn.LeakyReLU(0.2)
        self.enc8 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc8_bn = nn.BatchNorm2d(num_features=8*d)

        ## innermost
        self.enc9_act = nn.LeakyReLU(0.2)
        self.enc9 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)


        # Unet decoder
        ## innermost
        self.dec1_act = nn.ReLU()
        self.dec1 = nn.ConvTranspose2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec1_bn = nn.BatchNorm2d(num_features=8*d)
        ## bottleneck
        self.dec2_act = nn.ReLU()
        self.dec2 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec2_bn = nn.BatchNorm2d(num_features=8*d)
        self.dec2_dp = nn.Dropout(0.5)

        self.dec3_act = nn.ReLU()
        self.dec3 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec3_bn = nn.BatchNorm2d(num_features=8*d)
        self.dec3_dp = nn.Dropout(0.5)

        self.dec4_act = nn.ReLU()
        self.dec4 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec4_bn = nn.BatchNorm2d(num_features=8*d)
        self.dec4_dp = nn.Dropout(0.5)

        self.dec5_act = nn.ReLU()
        self.dec5 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec5_bn = nn.BatchNorm2d(num_features=8*d)
        self.dec5_dp = nn.Dropout(0.5)

        ## middle
        self.dec6_act = nn.ReLU()
        self.dec6 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=4*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec6_bn = nn.BatchNorm2d(num_features=4*d)

        self.dec7_act = nn.ReLU()
        self.dec7 = nn.ConvTranspose2d(in_channels=4*d*2, out_channels=2*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec7_bn = nn.BatchNorm2d(num_features=2*d)

        self.dec8_act = nn.ReLU()
        self.dec8 = nn.ConvTranspose2d(in_channels=2*d*2, out_channels=d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec8_bn = nn.BatchNorm2d(num_features=d)
        ## outermost
        self.dec9_act = nn.ReLU()
        self.dec9 = nn.ConvTranspose2d(in_channels=d*2, out_channels=output_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):

        ## Unet encoder
        ## outermost
        e1 = self.enc1(input.permute(0, 3, 1, 2))
        ## middle
        e2 = self.enc2_bn(self.enc2(self.enc2_act(e1)))
        e3 = self.enc3_bn(self.enc3(self.enc3_act(e2)))
        e4 = self.enc4_bn(self.enc4(self.enc4_act(e3)))
        ## bottleneck
        e5 = self.enc5_bn(self.enc5(self.enc5_act(e4)))
        e6 = self.enc6_bn(self.enc6(self.enc6_act(e5)))
        e7 = self.enc7_bn(self.enc7(self.enc7_act(e6)))
        e8 = self.enc8_bn(self.enc8(self.enc8_act(e7)))
        ## innermost
        e9 = self.enc9(self.enc9_act(e8))

        ## Unet decoder
        ## innermost
        d1 = self.dec1_bn(self.dec1(self.dec1_act(e9)))
        d1 = torch.cat([d1, e8], 1)

        ## bottleneck
        d2 = self.dec2_dp(self.dec2_bn(self.dec2(self.dec2_act(d1))))
        d2 = torch.cat([d2, e7], 1)

        d3 = self.dec3_dp(self.dec3_bn(self.dec3(self.dec3_act(d2))))
        d3 = torch.cat([d3, e6], 1)

        d4 = self.dec4_dp(self.dec4_bn(self.dec4(self.dec4_act(d3))))
        d4 = torch.cat([d4, e5], 1)

        d5 = self.dec5_dp(self.dec5_bn(self.dec5(self.dec5_act(d4))))
        d5 = torch.cat([d5, e4], 1)

        ## middle
        d6 = self.dec6_bn(self.dec6(self.dec6_act(d5)))
        d6 = torch.cat([d6, e3], 1)

        d7 = self.dec7_bn(self.dec7(self.dec7_act(d6)))
        d7 = torch.cat([d7, e2], 1)

        d8 = self.dec8_bn(self.dec8(self.dec8_act(d7)))
        d8 = torch.cat([d8, e1], 1)

        ## outermost
        out = self.tanh(self.dec9(self.dec9_act(d8))).permute(0, 2, 3, 1)

        return out

class discriminator_mesh(nn.Module):
    # initializers
    def __init__(self, input_channels=4, d=64):
        super(discriminator_mesh, self).__init__()

        ## CONV 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=4, stride=2, padding=1)
        self.conv1_act = nn.LeakyReLU(0.2)

        ## CONV 2
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=2*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=2*d)
        self.conv2_act = nn.LeakyReLU(0.2)

        ## CONV 3
        self.conv3 = nn.Conv2d(in_channels=2*d, out_channels=4*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=4*d)
        self.conv3_act = nn.LeakyReLU(0.2)

        ## CONV 4
        self.conv4 = nn.Conv2d(in_channels=4*d, out_channels=8*d, kernel_size=4, stride=1, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=8*d)
        self.conv4_act = nn.LeakyReLU(0.2)

        ## CONV 5
        self.conv5 = nn.Conv2d(in_channels=8*d, out_channels=16*d, kernel_size=4, stride=1, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=16*d)
        self.conv5_act = nn.LeakyReLU(0.2)

        ## CONV 6 (OUT)
        self.conv6 = nn.Conv2d(in_channels=16*d, out_channels=1, kernel_size=4, stride=1, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        c1 = self.conv1_act(self.conv1(input))
        c2 = self.conv2_act(self.conv2_bn(self.conv2(c1)))
        c3 = self.conv3_act(self.conv3_bn(self.conv3(c2)))
        c4 = self.conv4_act(self.conv4_bn(self.conv4(c3)))
        c5 = self.conv5_act(self.conv5_bn(self.conv5(c4)))

        out = self.conv6(c5)

        return out

class face_discriminator(nn.Module):
    # initializers
    def __init__(self, input_channels=3, d=64):
        super(face_discriminator, self).__init__()

        ## CONV 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=2*d, kernel_size=4, stride=2, padding=1)
        self.conv1_act = nn.LeakyReLU(0.2)

        ## CONV 2
        self.conv2 = nn.Conv2d(in_channels=2*d, out_channels=4*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=4*d)
        self.conv2_act = nn.LeakyReLU(0.2)

        ## CONV 3
        self.conv3 = nn.Conv2d(in_channels=4*d, out_channels=8*d, kernel_size=4, stride=1, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=8*d)
        self.conv3_act = nn.LeakyReLU(0.2)

        ## CONV 4
        self.conv4 = nn.Conv2d(in_channels=8*d, out_channels=1, kernel_size=4, stride=1, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        c1 = self.conv1_act(self.conv1(input))
        c2 = self.conv2_act(self.conv2_bn(self.conv2(c1)))
        c3 = self.conv3_act(self.conv3_bn(self.conv3(c2)))

        out = self.conv4(c3)

        return out

class discriminator(nn.Module):
    # initializers
    def __init__(self, input_channels=3, d=64):
        super(discriminator, self).__init__()

        ## CONV 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=4, stride=2, padding=1)
        self.conv1_act = nn.LeakyReLU(0.2)

        ## CONV 2
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=2*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=2*d)
        self.conv2_act = nn.LeakyReLU(0.2)

        ## CONV 3
        self.conv3 = nn.Conv2d(in_channels=2*d, out_channels=4*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=4*d)
        self.conv3_act = nn.LeakyReLU(0.2)

        ## CONV 4
        self.conv4 = nn.Conv2d(in_channels=4*d, out_channels=8*d, kernel_size=4, stride=1, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=8*d)
        self.conv4_act = nn.LeakyReLU(0.2)

        ## CONV 5
        self.conv5 = nn.Conv2d(in_channels=8*d, out_channels=16*d, kernel_size=4, stride=1, padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=16*d)
        self.conv5_act = nn.LeakyReLU(0.2)

        ## CONV 6 (OUT)
        self.conv6 = nn.Conv2d(in_channels=16*d, out_channels=1, kernel_size=4, stride=1, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        c1 = self.conv1_act(self.conv1(input))
        c2 = self.conv2_act(self.conv2_bn(self.conv2(c1)))
        c3 = self.conv3_act(self.conv3_bn(self.conv3(c2)))
        c4 = self.conv4_act(self.conv4_bn(self.conv4(c3)))
        c5 = self.conv5_act(self.conv5_bn(self.conv5(c4)))

        out = self.conv6(c5)

        return out

class TextureResidualStage(nn.Module):
    # initializers
    def __init__(self, input_channels=4, output_channels=3, d=64):
        super(TextureResidualStage, self).__init__()

        # Unet encoder
        ## outermost
        self.enc1 = nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=4, stride=2, padding=1)
        ## middle
        self.enc2_act = nn.LeakyReLU(0.2)
        self.enc2 = nn.Conv2d(in_channels=d, out_channels=2*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc2_bn = nn.BatchNorm2d(num_features=2*d)
        
        self.enc3_act = nn.LeakyReLU(0.2)
        self.enc3 = nn.Conv2d(in_channels=2*d, out_channels=4*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc3_bn = nn.BatchNorm2d(num_features=4*d)

        self.enc4_act = nn.LeakyReLU(0.2)
        self.enc4 = nn.Conv2d(in_channels=4*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc4_bn = nn.BatchNorm2d(num_features=8*d)
        ## bottleneck ## Add one more bottleneck to handle 512x512 inputs.
        self.enc5_act = nn.LeakyReLU(0.2)
        self.enc5 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc5_bn = nn.BatchNorm2d(num_features=8*d)

        self.enc6_act = nn.LeakyReLU(0.2)
        self.enc6 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc6_bn = nn.BatchNorm2d(num_features=8*d)

        self.enc7_act = nn.LeakyReLU(0.2)
        self.enc7 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc7_bn = nn.BatchNorm2d(num_features=8*d)

        self.enc8_act = nn.LeakyReLU(0.2)
        self.enc8 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc8_bn = nn.BatchNorm2d(num_features=8*d)

        ## innermost
        self.enc9_act = nn.LeakyReLU(0.2)
        self.enc9 = nn.Conv2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)


        # Unet decoder
        ## innermost
        self.dec1_act = nn.ReLU()
        self.dec1 = nn.ConvTranspose2d(in_channels=8*d, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec1_bn = nn.BatchNorm2d(num_features=8*d)
        ## bottleneck
        self.dec2_act = nn.ReLU()
        self.dec2 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec2_bn = nn.BatchNorm2d(num_features=8*d)
        self.dec2_dp = nn.Dropout(0.5)

        self.dec3_act = nn.ReLU()
        self.dec3 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec3_bn = nn.BatchNorm2d(num_features=8*d)
        self.dec3_dp = nn.Dropout(0.5)

        self.dec4_act = nn.ReLU()
        self.dec4 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec4_bn = nn.BatchNorm2d(num_features=8*d)
        self.dec4_dp = nn.Dropout(0.5)

        self.dec5_act = nn.ReLU()
        self.dec5 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=8*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec5_bn = nn.BatchNorm2d(num_features=8*d)
        self.dec5_dp = nn.Dropout(0.5)

        ## middle
        self.dec6_act = nn.ReLU()
        self.dec6 = nn.ConvTranspose2d(in_channels=8*d*2, out_channels=4*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec6_bn = nn.BatchNorm2d(num_features=4*d)

        self.dec7_act = nn.ReLU()
        self.dec7 = nn.ConvTranspose2d(in_channels=4*d*2, out_channels=2*d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec7_bn = nn.BatchNorm2d(num_features=2*d)

        self.dec8_act = nn.ReLU()
        self.dec8 = nn.ConvTranspose2d(in_channels=2*d*2, out_channels=d, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec8_bn = nn.BatchNorm2d(num_features=d)
        ## outermost
        self.dec9_act = nn.ReLU()
        self.dec9 = nn.ConvTranspose2d(in_channels=d*2, out_channels=output_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):

        ## Unet encoder
        ## outermost
        e1 = self.enc1(input.permute(0, 3, 1, 2))
        ## middle
        e2 = self.enc2_bn(self.enc2(self.enc2_act(e1)))
        e3 = self.enc3_bn(self.enc3(self.enc3_act(e2)))
        e4 = self.enc4_bn(self.enc4(self.enc4_act(e3)))
        ## bottleneck
        e5 = self.enc5_bn(self.enc5(self.enc5_act(e4)))
        e6 = self.enc6_bn(self.enc6(self.enc6_act(e5)))
        e7 = self.enc7_bn(self.enc7(self.enc7_act(e6)))
        e8 = self.enc8_bn(self.enc8(self.enc8_act(e7)))
        ## innermost
        e9 = self.enc9(self.enc9_act(e8))

        ## Unet decoder
        ## innermost
        d1 = self.dec1_bn(self.dec1(self.dec1_act(e9)))
        d1 = torch.cat([d1, e8], 1)

        ## bottleneck
        d2 = self.dec2_dp(self.dec2_bn(self.dec2(self.dec2_act(d1))))
        d2 = torch.cat([d2, e7], 1)

        d3 = self.dec3_dp(self.dec3_bn(self.dec3(self.dec3_act(d2))))
        d3 = torch.cat([d3, e6], 1)

        d4 = self.dec4_dp(self.dec4_bn(self.dec4(self.dec4_act(d3))))
        d4 = torch.cat([d4, e5], 1)

        d5 = self.dec5_dp(self.dec5_bn(self.dec5(self.dec5_act(d4))))
        d5 = torch.cat([d5, e4], 1)

        ## middle
        d6 = self.dec6_bn(self.dec6(self.dec6_act(d5)))
        d6 = torch.cat([d6, e3], 1)

        d7 = self.dec7_bn(self.dec7(self.dec7_act(d6)))
        d7 = torch.cat([d7, e2], 1)

        d8 = self.dec8_bn(self.dec8(self.dec8_act(d7)))
        d8 = torch.cat([d8, e1], 1)

        ## outermost
        out = self.tanh(self.dec9(self.dec9_act(d8))).permute(0, 2, 3, 1)

        return input[..., :3] + out, out

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, std)
        m.bias.data.zero_()
