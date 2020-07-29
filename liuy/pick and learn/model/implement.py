import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchviz import make_dot
from torchsummary import summary


class inConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inConv, self).__init__()
        self.inConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.inConv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.downConv = nn.Sequential(
            nn.MaxPool2d(2),
            inConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downConv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = inConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#     __constants__ = ['downsample']
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, 1, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.fc = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)


        # print(f"x shape {x.shape}")
        x = self.fc(self.relu(x))
        res_avg = self.avg_pool(x)
        res_max = self.max_pool(x)
        res = res_avg + res_max
        return res


class QAM(nn.Module):
    def __init__(self):
        super(QAM, self).__init__()

        self.conv = resnet18(pretrained=True)
        self.conv.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
        num_ftrs = self.conv.fc.in_features
        self.channel_attention = ChannelAttention(in_planes=num_ftrs)
        self.spatial_attention = SpatialAttention(kernel_size=3)
        self.conv.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x, y):
        input = torch.cat((x, y), 1)
        res = self.conv(input)
        return res


class OCM(nn.Module):
    def __init__(self):
        super(OCM, self).__init__()
        self.sigmod = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        res = self.sigmod(x)
        res = self.softmax(res)
        return res


class QAM_cbam(nn.Module):
    def __init__(self):
        super(QAM_cbam, self).__init__()

        model = resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        # self.conv = model
        # self.conv.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
        #                             bias=False)
        # self.conv.fc = None

        self.conv = nn.Sequential(*list(model.children())[:-2])
        self.conv[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)

        # print(self.conv)
        # print(num_ftrs)
        self.channel_attention = ChannelAttention(in_planes=num_ftrs)
        self.spatial_attention = SpatialAttention(kernel_size=3)


    def forward(self, x, y):
        input = torch.cat((x, y), 1)
        res_resnet = self.conv(input)
        # print(f' res_resnet shape is {res_resnet.shape}')
        res_channel = self.channel_attention(res_resnet)
        # print(f' res_channel shape is {res_channel.shape}')
        res_spatial = self.spatial_attention(res_resnet)
        # print(f' res_spatial shape is {res_spatial.shape}')
        res = res_channel + res_spatial
        return res


class branch(nn.Module):
    def __init__(self):
        super(branch, self).__init__()
        self.qam = QAM_cbam()
        self.ocm = OCM()

    def forward(self, x, y):
        res = self.qam(x, y)
        res = self.ocm(res)
        return res


if __name__ == '__main__':
    test_module = branch()
    x = torch.randn(1, 3, 321, 321)
    y = torch.randn(1, 1, 321, 321)
    # res2 = test_module(x, y)
    # net_plot = make_dot(res2, params=dict(list(qam.named_parameters()) + [('x', x)]))
    # net_plot.view()

    summary(test_module, input_size=[(3, 321, 321), (1, 321, 321)], device="cpu")
