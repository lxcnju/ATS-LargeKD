import torch.nn as nn
import torch.nn.functional as F


def get_resnet_cfg(n_layer):
    cfgs = {
        8: [16, 16, 32, 64],
        14: [16, 16, 32, 64],
        20: [16, 16, 32, 64],
        32: [16, 16, 32, 64],
        44: [16, 16, 32, 64],
        56: [16, 16, 32, 64],
        110: [16, 16, 32, 64],
    }
    assert n_layer in cfgs, "n_layer=8,14,20,32,44,56,110"

    return cfgs[n_layer]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, groups=32, base_width=4,
            stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        width = int(planes * groups * base_width / 64.0)

        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNeXt(nn.Module):

    def __init__(self, n_layer, groups, base_width, n_classes=10):
        super(ResNeXt, self).__init__()
        assert (n_layer - 2) % 9 == 0, '9n+2, e.g. 29'
        n = (n_layer - 2) // 9

        block = Bottleneck

        num_filters = [64, 64, 128, 256]

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(
            3, num_filters[0], kernel_size=3, padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, groups, base_width, num_filters[1], n
        )
        self.layer2 = self._make_layer(
            block, groups, base_width, num_filters[2], n, stride=2
        )
        self.layer3 = self._make_layer(
            block, groups, base_width, num_filters[3], n, stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            num_filters[3] * block.expansion, n_classes
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, groups, base_width, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                groups=groups,
                base_width=base_width,
                stride=stride,
                downsample=downsample,
                is_last=(blocks == 1)
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=groups,
                    base_width=base_width,
                    is_last=(i == blocks - 1)
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False, kd_layers=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x, f1_pre = self.layer1(x)  # 32x32
        f1 = x
        x, f2_pre = self.layer2(x)  # 16x16
        f2 = x
        x, f3_pre = self.layer3(x)  # 8x8
        f3 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        x = self.fc(x)

        if is_feat:
            if preact:
                fs = [f0, f1_pre, f2_pre, f3_pre, f4]
            else:
                fs = [f0, f1, f2, f3, f4]

            if kd_layers is not None:
                fs = [fs[k] for k in kd_layers]
            return fs, x
        else:
            return x


if __name__ == '__main__':
    import torch

    for size in [32]:
        for n_layer, groups, base_width in [(29, 32, 4), (29, 8, 4)]:
            x = torch.randn(2, 3, size, size)
            model = ResNeXt(
                n_layer=n_layer, groups=groups,
                base_width=base_width, n_classes=100
            )
            feats, logit = model(x, is_feat=True, preact=True)

            print([name for name, _ in model.named_parameters()])
            n_params = sum([
                param.numel() for param in model.parameters()
            ])
            print("Total number of parameters : {:.2f}M".format(
                n_params / 1000000.0
            ))

            for f in feats:
                print(f.shape, f.min().item())
            print(logit.shape)
