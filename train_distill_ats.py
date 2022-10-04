import os
import random
from collections import namedtuple

import torch

from datasets.data import load_data

from networks.resnet import ResNet
from networks.wide_resnet import WideResNet
from networks.vgg import VGG

from distill import Distill

from utils import set_gpu
from utils import weights_init
from paths import save_dir
from paths import ckpt_dir


def main_distill(para_dict):
    torch.backends.cudnn.benchmark = True

    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # data
    train_set, test_set = load_data(args.dataset)

    # load teacher model
    t_net_name = args.t_net_name
    if t_net_name == "ResNet14":
        t_model = ResNet(
            n_layer=14,
            n_classes=100,
        )
    elif t_net_name == "ResNet110":
        t_model = ResNet(
            n_layer=110,
            n_classes=100,
        )
    elif t_net_name == "WRN28-1":
        t_model = WideResNet(
            n_layer=28,
            widen_factor=1,
            n_classes=100,
        )
    elif t_net_name == "WRN28-8":
        t_model = WideResNet(
            n_layer=28,
            widen_factor=8,
            n_classes=100,
        )
    else:
        raise ValueError("No such net: {}".format(t_net_name))

    # load teacher model
    fpath = os.path.join(
        ckpt_dir, "{}-{}-E{}.pth".format(
            args.dataset, args.t_net_name, args.t_epoch
        )
    )
    t_model.load_state_dict(torch.load(fpath))
    print("Teacher model loaded from: {}".format(fpath))

    s_model = VGG(
        n_layer=8,
        n_classes=100,
        use_bn=True
    )
    s_model.apply(weights_init)

    for model in [t_model, s_model]:
        print(model)
        print([name for name, _ in model.named_parameters()])
        n_params = sum([
            param.numel() for param in model.parameters()
        ])
        print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        t_model = t_model.cuda()
        s_model = s_model.cuda()

    algo = Distill(
        train_set=train_set,
        test_set=test_set,
        t_model=t_model,
        model=s_model,
        args=args
    )
    algo.main()

    fpath = os.path.join(
        save_dir, args.fname
    )
    algo.save_logs(fpath)
    print(algo.logs)


def main_ats():
    candi_param_dict = {
        "dataset": ["none"],
        "net": ["none"],
        "t_n_layer": ["none"],
        "t_widen_factor": ["none"],
        "t_groups": ["none"],
        "t_base_width": ["none"],
        "t_pretrain": [False],
        "n_layer": ["none"],
        "widen_factor": ["none"],
        "groups": ["none"],
        "base_width": ["none"],
        "pretrain": [False],
        "n_classes": ["none"],
        "epoches": [240],
        "batch_size": [128],
        "optimizer": ["SGD"],
        "momentum": [0.9],
        "lr": ["none"],
        "scheduler": ["WSKDLR"],
        "ws_step": [5],
        "weight_decay": [5e-4],
        "cuda": [True],
        "save_ckpts": [False],
        "t_tau": ["none"],
        "lamb": ["none"],
    }

    t_epoch = 240
    dataset = "cifar100"
    t_net = "WideResNet"
    t_info = (28, 8)
    net = "VGG"
    info = 8
    s_tau = 1.0
    lamb = 0.5

    for tp_tau in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        for t_tau in [1.0, 2.0, 3.0, 4.0, 5.0]:
            lr = 0.05

            if dataset == "cifar100":
                n_classes = 100
            else:
                n_classes = 10

            para_dict = {}
            for k, vs in candi_param_dict.items():
                para_dict[k] = random.choice(vs)

            para_dict["dataset"] = dataset
            para_dict["t_net"] = t_net
            para_dict["net"] = net

            if t_net == "WideResNet":
                para_dict["t_n_layer"] = t_info[0]
                para_dict["t_widen_factor"] = t_info[1]
                t_net_name = "WRN{}-{}".format(
                    t_info[0], t_info[1]
                )
            elif t_net == "ResNeXt":
                para_dict["t_n_layer"] = t_info[0]
                para_dict["t_groups"] = t_info[1]
                para_dict["t_base_width"] = t_info[2]
                t_net_name = "ResNeXt{}-{}-{}d".format(
                    t_info[0], t_info[1], t_info[2]
                )
            elif t_net in ["VGG", "ResNet"]:
                para_dict["t_n_layer"] = t_info
                t_net_name = "{}{}".format(t_net, t_info)
            elif t_net in ["ShuffleNetV1", "ShuffleNetV2"]:
                t_net_name = "{}".format(t_net)
            elif t_net in ["MobileNetV2"]:
                t_net_name = "{}".format(t_net)
            else:
                raise ValueError("No such net:{}".format(t_net))

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
            elif net in ["ShuffleNetV1", "ShuffleNetV2"]:
                net_name = "{}".format(net)
            elif net in ["MobileNetV2"]:
                net_name = "{}".format(net)
            else:
                raise ValueError("No such net:{}".format(net))

            para_dict["t_net_name"] = t_net_name
            para_dict["net_name"] = net_name
            para_dict["lr"] = lr
            para_dict["lamb"] = lamb
            para_dict["s_tau"] = s_tau
            para_dict["t_tau"] = t_tau
            para_dict["tp_tau"] = tp_tau
            para_dict["t_epoch"] = t_epoch
            para_dict["n_classes"] = n_classes
            para_dict["fname"] = "distill-ats-{}-{}-{}.log".format(
                dataset, t_net_name, net_name
            )

            main_distill(para_dict)


if __name__ == "__main__":
    set_gpu("0")
    main_ats()
