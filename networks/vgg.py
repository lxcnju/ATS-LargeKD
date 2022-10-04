import torch.nn as nn
import torch.nn.functional as F
import math


def get_vgg_cfg(n_layer):
    if n_layer == 8:
        cfg = [[64], [128], [256], [512], [512]]
    elif n_layer == 11:
        cfg = [[64], [128], [256, 256], [512, 512], [512, 512]]
    elif n_layer == 13:
        cfg = [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]]
    elif n_layer == 16:
        cfg = [
            [64, 64], [128, 128], [256, 256, 256],
            [512, 512, 512], [512, 512, 512]
        ]
    elif n_layer == 19:
        cfg = [
            [64, 64], [128, 128], [256, 256, 256, 256],
            [512, 512, 512, 512], [512, 512, 512, 512]
        ]
    else:
        raise ValueError("No such n_layer: {}".format(n_layer))

    return cfg


class VGG(nn.Module):

    def __init__(self, n_layer, n_classes, use_bn=True):
        super(VGG, self).__init__()
        cfg = get_vgg_cfg(n_layer)

        self.block0 = self._make_layers(cfg[0], use_bn, 3)
        self.block1 = self._make_layers(cfg[1], use_bn, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], use_bn, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], use_bn, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], use_bn, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, n_classes)
        self._initialize_weights()

    def forward(self, x, is_feat=False, preact=False, kd_layers=None):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)

        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        x = self.pool1(x)

        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        x = self.pool2(x)

        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x

        if h == 64:
            x = self.pool3(x)

        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        f5 = x
        x = self.classifier(x)

        if is_feat:
            if preact:
                fs = [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5]
            else:
                fs = [f0, f1, f2, f3, f4, f5]

            if kd_layers is not None:
                fs = [fs[k] for k in kd_layers]
            return fs, x
        else:
            return x

    @staticmethod
    def _make_layers(cfg, use_bn=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if use_bn:
                    layers += [
                        conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    import torch

    for size in [32]:
        for n_layer in [8, 11, 13, 16, 19]:
            x = torch.randn(10, 3, size, size)
            model = VGG(n_layer=n_layer, n_classes=100)
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
