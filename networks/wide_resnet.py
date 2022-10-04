import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False
        ) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(
            self, nb_layers, in_planes, out_planes,
            block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
            self, block, in_planes, out_planes,
            nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(
                i == 0 and in_planes or out_planes,
                out_planes,
                i == 0 and stride or 1,
                dropRate
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, n_layer, widen_factor, n_classes, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [
            16, 16 * widen_factor,
            32 * widen_factor, 64 * widen_factor
        ]
        assert (n_layer - 4) % 6 == 0, 'depth should be 6n+4'
        n = (n_layer - 4) // 6

        block = BasicBlock

        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1,
            padding=1, bias=False
        )

        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate
        )

        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate
        )

        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate
        )

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(nChannels[3], n_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, is_feat=False, preact=False, kd_layers=None):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        f3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        f4 = out
        out = self.fc(out)

        if is_feat:
            if preact:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)

            fs = [f0, f1, f2, f3, f4]

            if kd_layers is not None:
                fs = [fs[k] for k in kd_layers]
            return fs, out
        else:
            return out


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)

    for n_layer in [16, 40]:
        for widen_factor in [1, 2]:
            model = WideResNet(
                n_layer=n_layer, widen_factor=widen_factor, n_classes=100
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
