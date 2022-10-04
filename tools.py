import torch
from torch.utils import data

import numpy as np


def construct_loaders(train_set, test_set, args):
    train_loader = data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, drop_last=False
    )

    test_loader = data.DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, drop_last=False
    )

    return train_loader, test_loader


def construct_fine_optimizer(model, args):
    param_groups = []

    for name, params in model.named_parameters():
        print(name)
        param_groups.append(
            {"params": params, "lr": args.lr, "name": name}
        )

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(args.optimizer))
    return optimizer


def construct_group_optimizer(model, args):
    lr_x = args.lr * args.lr_mu

    encoder_param_ids = list(map(id, model.encoder.parameters()))

    other_params = filter(
        lambda p: id(p) not in encoder_param_ids,
        model.parameters()
    )

    param_groups = [
        {"params": other_params, "lr": args.lr},
        {"params": model.encoder.parameters(), "lr": lr_x},
    ]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(args.optimizer))
    return optimizer


def construct_optimizer(model, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(args.optimizer))
    return optimizer


def construct_lr_scheduler(optimizer, args):
    if args.scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == "CosLR":
        CosLR = torch.optim.lr_scheduler.CosineAnnealingLR
        lr_scheduler = CosLR(
            optimizer, T_max=args.epoches, eta_min=1e-8
        )
    elif args.scheduler == "CosLRWR":
        CosLRWR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        lr_scheduler = CosLRWR(
            optimizer, T_0=args.step_size
        )
    elif args.scheduler == "CyclicLR":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=0.0,
            max_lr=args.lr,
            step_size_up=args.step_size,
        )
    elif args.scheduler == "WSKDLR":
        # LambdaLR: quadratic
        def lr_warm_start_kd_lr(
                t, T0=args.ws_step, Tm=120, step=30, T_max=args.epoches):
            # T0 = int(0.1 * T_max)
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            elif t <= Tm:
                return 1.0
            else:
                return 0.1 ** (int((t - Tm) / step))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_kd_lr,
        )
    elif args.scheduler == "WSQuadLR":
        # LambdaLR: quadratic
        def lr_warm_start_quad(t, T0=args.ws_step, T_max=args.epoches):
            # T0 = int(0.1 * T_max)
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            else:
                return (1.0 - 1.0 * (t - T0) / (T_max - T0)) ** 2

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_quad,
        )
    elif args.scheduler == "WSStepLR":
        # LambdaLR: step lr
        def lr_warm_start_step(
            t, T0=args.ws_step,
            step_size=args.step_size, gamma=args.gamma
        ):
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            else:
                return gamma ** int((t - T0) / step_size)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_step,
        )
    elif args.scheduler == "WSCosLR":
        # LambdaLR: coslr
        def lr_warm_start_cos(
            t, T0=args.ws_step, T_max=args.epoches
        ):
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            else:
                return (np.cos((t - T0) / (T_max - T0) * np.pi) + 1.0) / 2.0
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_cos,
        )
    else:
        raise ValueError("No such scheduler: {}".format(args.scheduler))
    return lr_scheduler


if __name__ == "__main__":
    from collections import namedtuple
    from matplotlib import pyplot as plt

    import torch.nn as nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 2)

        def forward(self, xs):
            return None

    model = Network()

    lr = 0.1
    epoches = 200

    para_dict = {
        "lr": lr,
        "ws_step": 10,
        "step_size": 60,
        "gamma": 0.1,
        "epoches": epoches,
    }

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr
    )

    # WS-StepLR
    para_dict["scheduler"] = "WSStepLR"
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    scheduler = construct_lr_scheduler(optimizer, args)

    lrs = []
    for _ in range(epoches):
        model.zero_grad()
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.figure()
    plt.plot(range(len(lrs)), lrs)
    plt.title("WSStepLR: ws_step=10, epoch=200, step_size=60, gamma=0.1")
    plt.show()

    # WS-CosLR
    para_dict["scheduler"] = "WSCosLR"
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    scheduler = construct_lr_scheduler(optimizer, args)

    lrs = []
    for _ in range(epoches):
        model.zero_grad()
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.figure()
    plt.plot(range(len(lrs)), lrs)
    plt.title("WSCosLR: ws_step=10, epoch=200")
    plt.show()

    # CosLR
    para_dict["scheduler"] = "CosLR"
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    scheduler = construct_lr_scheduler(optimizer, args)

    lrs = []
    for _ in range(epoches):
        model.zero_grad()
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.figure()
    plt.plot(range(len(lrs)), lrs)
    plt.title("CosLR: epoch=200")
    plt.show()

    # WS-QuadLR
    para_dict["scheduler"] = "WSQuadLR"
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    scheduler = construct_lr_scheduler(optimizer, args)

    lrs = []
    for _ in range(epoches):
        model.zero_grad()
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.figure()
    plt.plot(range(len(lrs)), lrs)
    plt.title("WSQuadLR: ws_step=10, epoch=200")
    plt.show()

