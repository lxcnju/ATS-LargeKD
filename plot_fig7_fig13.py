import os
import numpy as np
import torch
from matplotlib import pyplot as plt

import matplotlib.gridspec as gridspec

from datasets.data import load_data
from networks.resnet import ResNet
from networks.wide_resnet import WideResNet
from paths import ckpt_dir, save_dir


def get_ew_vw(ps):
    ew = ps.mean()
    vw = ps.std()
    return ew, vw


def my_softmax_class(xs, tau, c):
    taus = np.array([0.75 * tau] * len(xs))
    taus[c] = 1.25 * tau

    ps = torch.FloatTensor(xs / taus).softmax(dim=0).numpy()

    return ps


def split_values(values, labels):
    tvalues = []
    fvalues = []

    C = values.shape[-1]
    for vs, c in zip(values, labels):
        inds = np.array([i for i in range(C) if i != c])
        fvalues.append(vs[inds])
        tvalues.append(vs[c])
    fvalues = np.stack(fvalues, axis=0)
    tvalues = np.array(tvalues)
    return tvalues, fvalues


def extract(model, loader, cuda):
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for _, tx, ty in loader:
            if cuda:
                tx, ty = tx.cuda(), ty.cuda()

            logits = model(tx, is_feat=False)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(ty.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits, labels


def get_logits(dataset, net_name, cuda=True):
    train_set, _ = load_data(dataset)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=False
    )

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
        raise ValueError("No such net.")

    fpath = os.path.join(
        ckpt_dir,
        "{}-{}-E240.pth".format(dataset.lower(), net_name)
    )
    model.load_state_dict(torch.load(fpath))

    if cuda is True:
        model = model.cuda()

    logits, labels = extract(
        model, train_loader, cuda
    )

    return logits, labels


def plot_logits_taus(d, p):
    pairs = [
        ("ResNet", "RES", "14", "110"),
        ("WRN", "WRN", "28-1", "28-8"),
        ("ResNeXt", "RNX", "29-4-4d", "29-64-4d"),
    ]

    datasets = [
        ("cifar10", "CIFAR-10"),
        ("cifar100", "CIFAR-100"),
    ]

    names = [
        "{}:TS".format(pairs[p][1] + pairs[p][2]),
        "{}:TS".format(pairs[p][1] + pairs[p][3]),
        "{}:ATS".format(pairs[p][1] + pairs[p][3]),
    ]

    taus = np.linspace(0.0, 10.0, 21)[1:]
    taus = [0.05, 0.1, 0.25, 0.4] + list(taus)

    fig = plt.figure(figsize=(8, 2.5))
    gs = gridspec.GridSpec(1, 3)
    gs.update(hspace=0.0, wspace=0.1)

    all_ews = []
    all_sws = []
    for k in range(2):
        logits, labels = get_logits(
            datasets[d][0], pairs[p][0] + pairs[p][k + 2]
        )

        ews = []
        sws = []
        for tau in taus:
            print(tau)
            probs = torch.FloatTensor(
                logits / tau
            ).softmax(dim=1).numpy()
            _, wprobs = split_values(probs, labels)
            ew = wprobs.mean(axis=1).mean()
            sw = wprobs.std(axis=1).mean()
            ews.append(ew)
            sws.append(sw)

        all_ews.append(ews)
        all_sws.append(sws)

        if k == 1:
            ews = []
            sws = []
            for tau in taus:
                print(tau)
                probs = []
                for logis, c in zip(logits, labels):
                    ps = my_softmax_class(logis, tau, c)
                    probs.append(ps)

                probs = np.stack(probs, axis=0)
                _, wprobs = split_values(probs, labels)
                ew = wprobs.mean(axis=1).mean()
                sw = wprobs.std(axis=1).mean()
                ews.append(ew)
                sws.append(sw)

            all_ews.append(ews)
            all_sws.append(sws)

    all_ews = np.array(all_ews)
    all_sws = np.array(all_sws)

    for i in range(3):
        plt.subplot(gs[i])
        ax = plt.gca()
        tax = ax.twinx()

        xs = list(range(len(taus)))
        ax.plot(xs, all_ews[i], color="#660066", linestyle="solid")
        tax.plot(xs, all_sws[i], color="#663333", linestyle="dashed")

        ax.set_ylim(all_ews.min(), all_ews.max() * 1.2)
        tax.set_ylim(all_sws.min(), all_sws.max() * 1.2)

        ax.grid(False)
        tax.grid(False)

        ax.legend([r"$e(q)$"], loc="upper left", fontsize=10.5)
        tax.legend([r"$\sigma(q)$"], loc="upper right", fontsize=10.5)

        if i == 0:
            ax.set_ylabel(r"Derived Average: $e(q)$", fontsize=13)

        if i == 2:
            tax.set_ylabel(r"Derived Variance: $\sigma(q)$", fontsize=13)

        if i >= 1:
            ax.set_yticks([])

        if i <= 1:
            tax.set_yticks([])

        plt.setp(
            ax.get_yticklabels(), color="#660066", rotation=65,
            fontsize=8.5,
        )

        plt.setp(
            tax.get_yticklabels(), color="#663333", rotation=65,
            fontsize=8.5,
        )

        xticks = [0, 4, 9, 14, 19, len(taus) - 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(taus[x]) for x in xticks])
        ax.set_xlabel("Temperature", fontsize=14)

        ax.set_title("{}".format(names[i]), fontsize=18)

    fig.savefig(
        os.path.join(
            save_dir,
            "obser-tau-{}-{}.pdf".format(
                datasets[d][0], pairs[p][0]
            )
        ), dpi=300,
        bbox_inches='tight', format='pdf'
    )
    fig.savefig(
        os.path.join(
            save_dir,
            "obser-tau-{}-{}.jpg".format(
                datasets[d][0], pairs[p][0]
            )
        ), dpi=300,
        bbox_inches='tight',
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_logits_taus(d=1, p=0)
    plot_logits_taus(d=1, p=1)
