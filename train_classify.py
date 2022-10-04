import os
import random
from collections import namedtuple

import torch

from datasets.data import load_data

from networks.resnet import ResNet
from networks.wide_resnet import WideResNet

from classify import Classify
from paths import save_dir
from paths import ckpt_dir


def main_classify(para_dict):
    from utils import set_gpu
    torch.backends.cudnn.benchmark = True
    set_gpu("1")

    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # data
    train_set, test_set = load_data(args.dataset)

    # load model
    net_name = args.net_name
    if net_name == "ResNet14":
        model = ResNet(
            n_layer=14,
            n_classes=100,
        )
    elif net_name == "ResNet110":
        model = ResNet(
            n_layer=110,
            n_classes=100,
        )
    elif net_name == "WRN28-1":
        model = WideResNet(
            n_layer=28,
            widen_factor=1,
            n_classes=100,
        )
    elif net_name == "WRN28-8":
        model = WideResNet(
            n_layer=28,
            widen_factor=8,
            n_classes=100,
        )
    else:
        raise ValueError("No such net: {}".format(net_name))

    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    # classify
    algo = Classify(
        train_set=train_set,
        test_set=test_set,
        model=model,
        args=args
    )
    algo.main(save_ckpts=[args.epoches], ext=None)

    fpath = os.path.join(
        save_dir, args.fname
    )
    algo.save_logs(fpath)
    print(algo.logs)


def main():
    candi_param_dict = {
        "dataset": ["cifar100"],
        "net": ["ResNet"],
        "n_layer": ["none"],
        "widen_factor": ["none"],
        "groups": ["none"],
        "base_width": ["none"],
        "pretrain": [False],
        "n_classes": [100],
        "epoches": [240],
        "batch_size": [128],
        "optimizer": ["SGD"],
        "momentum": [0.9],
        "lr": [0.03],
        "scheduler": ["none"],
        "ws_step": [5],
        "weight_decay": [5e-4],
        "cuda": [True],
        "save_ckpts": [False],
    }

    net_pairs = [
        ("ResNet", [14, 110]),
        ("WideResNet", [(28, 1), (28, 8)])
    ]

    for dataset in ["cifar100"]:
        for net, infos in net_pairs:
            for info in infos:
                for epoches in [240]:
                    lr = 0.05

                    if dataset == "cifar100":
                        n_classes = 100
                    else:
                        n_classes = 10

                    para_dict = {}
                    for k, vs in candi_param_dict.items():
                        para_dict[k] = random.choice(vs)

                    para_dict["dataset"] = dataset
                    para_dict["net"] = net

                    if net == "WideResNet":
                        para_dict["n_layer"] = info[0]
                        para_dict["widen_factor"] = info[1]
                        net_name = "WRN{}-{}".format(info[0], info[1])
                    elif net == "ResNeXt":
                        para_dict["n_layer"] = info[0]
                        para_dict["groups"] = info[1]
                        para_dict["base_width"] = info[2]
                        net_name = "ResNeXt{}-{}-{}d".format(
                            info[0], info[1], info[2]
                        )
                    elif net in ["VGG", "ResNet"]:
                        para_dict["n_layer"] = info
                        net_name = "{}{}".format(net, info)
                    elif net in ["ShuffleNetV1", "MobileNetV2"]:
                        net_name = "{}".format(net)
                    else:
                        raise ValueError("No such net:{}".format(net))

                    para_dict["net_name"] = net_name
                    para_dict["lr"] = lr
                    para_dict["n_classes"] = n_classes

                    para_dict["epoches"] = epoches
                    para_dict["scheduler"] = "WSKDLR"
                    para_dict["fname"] = "classify-{}-{}.log".format(
                        dataset, net_name
                    )

                    main_classify(para_dict)


if __name__ == "__main__":
    main()
