import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SubSpectralNorm(nn.Module):
    def __init__(self,
        num_features: int,
        num_subspecs: int = 2,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super(SubSpectralNorm, self).__init__()
        self.eps = eps
        self.subpecs = num_subspecs
        self.gamma = nn.Parameter(torch.ones(1,num_features * num_subspecs,1,1),
            requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros(1,num_features * num_subspecs,1,1),
            requires_grad=affine)
    
    def forward(self, x: Tensor):
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, num_channels*self.subpecs, height//self.subpecs,
            width)

        x_mean = x.mean([0, 2, 3]).view(1, num_channels * self.subpecs, 1, 1)
        x_var = x.var([0, 2, 3]).view(1, num_channels * self.subpecs, 1, 1)

        x = (x - x_mean) / (x_var + self.eps).sqrt() * self.gamma + self.beta

        return x.view(batch_size, num_channels, height, width)

class CausalConv2d(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (1, 1),
        stride: tuple = (1, 1),
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super(CausalConv2d, self).__init__()
        self.pad = nn.ZeroPad2d(((kernel_size[1] - 1) * dilation, 0,
            kernel_size[0] // 2, kernel_size[0] // 2))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
            dilation=(1, dilation), groups=groups, bias=bias)
        
    def forward(self, x: Tensor):
        x = self.pad(x)
        x = self.conv(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        dilation: int = 1,
        groups: int = 1,
        num_subspecs: int = 1,
        affine: bool = True
    ):
        super(ConvLayer, self).__init__()

        self.conv = CausalConv2d(in_channels, out_channels, kernel_size, stride,
            dilation, groups=groups, bias=False)
        self.swish = nn.Hardswish(inplace=True)
        self.norm = SubSpectralNorm(out_channels, num_subspecs, affine=affine)

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.swish(x)
        x = self.norm(x)
        return x

class ConvSubBlock(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        num_subspecs: int = 1,
        affine: bool = True
    ):
        super(ConvSubBlock, self).__init__()
        # Depthwise
        self.conv_sub_block1 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride = stride,
            dilation = 1,
            groups = in_channels,
            num_subspecs = num_subspecs,
            affine = affine
        )
        # Pointwise
        self.conv_sub_block2 = ConvLayer(
            out_channels,
            out_channels,
            kernel_size = (1, 1),
            stride = (1, 1),
            dilation = 1,
            groups = 1,
            num_subspecs = num_subspecs,
            affine = affine
        )

    def forward(self, x: Tensor):
        x = self.conv_sub_block1(x)
        x = self.conv_sub_block2(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        num_subspecs: int = 1,
        affine: bool = True
    ):
        super(ConvBlock, self).__init__()
        self.conv_block1 = ConvSubBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride = (1, 1),
            num_subspecs = num_subspecs,
            affine = affine
        )
        self.conv_block2 = ConvSubBlock(
            out_channels,
            out_channels,
            kernel_size,
            stride = (2, 2),
            num_subspecs = num_subspecs,
            affine = affine
        )
    def forward(self, x: Tensor):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x

class EnvironNet(nn.Module):
    def __init__(self,
        in_channels: int,
        n_classes: int,
        kernel_size: tuple = (3, 3),
        num_subspecs: int = 1,
        affine: bool = True,
        dropout: float = 0.5
    ):
        super(EnvironNet, self).__init__()
        self.dropout = dropout
        self.conv0 = ConvLayer(
            in_channels,
            32,
            kernel_size = (7, 7),
            stride = (2, 2),
            dilation = 1,
            num_subspecs = num_subspecs,
            affine = affine
        )
        self.conv_block1 = ConvBlock(
            32,
            64,
            kernel_size,
            num_subspecs = num_subspecs,
            affine = affine
        )
        self.conv_block2 = ConvBlock(
            64,
            128,
            kernel_size,
            num_subspecs = num_subspecs,
            affine = affine
        )
        self.conv_block3 = ConvBlock(
            128,
            256,
            kernel_size,
            num_subspecs = num_subspecs,
            affine = affine
        )
        self.conv_block4 = ConvBlock(
            256,
            512,
            kernel_size,
            num_subspecs = num_subspecs,
            affine = affine
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear1 = nn.Linear(512, n_classes)
        self.initialize_weights()

    def forward(self, x: Tensor):
        x = self.conv0(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, self.dropout, self.training)
        x = self.linear1(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    from utils import num_params
    # import torch.optim as optim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = EnvironNet(
        in_channels = 1,
        n_classes = 50,
        kernel_size = (3, 3),
        num_subspecs = 2,
        affine = True,
        dropout = 0.5
    ).to(device)
    
    print(f"Number of trainable parameters: {num_params(model)}")

    # model = nn.DataParallel(model)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr = 0.0005,
    #     momentum = 0.9,
    #     weight_decay = 0.001,
    #     nesterov = True
    # )