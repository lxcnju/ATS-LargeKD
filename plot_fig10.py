import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('ggplot')


def parse_logs(fpath):
    values = []

    with open(fpath, "r") as fr:
        for line in fr:
            line = line.strip()

            if len(line) <= 0:
                continue

            if line.startswith("[TeACCS]"):
                line = line.split(":")[1]
                vs = [float(v) for v in line.split()]
                values.append(vs[-1])

    values = np.array(values)
    return values


pairs = [
    ("cifar100", "WRN28-8-VGG8"),
]

titles = [
    r"CIFAR-100:WRN28-8$\rightarrow$VGG8",
]


fig = plt.figure(figsize=(8, 5.0))
gs = gridspec.GridSpec(1, 1)
gs.update(hspace=0.0, wspace=0.1)


for p, (dataset, pair) in enumerate(pairs):
    tp_taus = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t_taus = [1.0, 2.0, 3.0, 4.0, 5.0]

    fpath = "./logs/distill-ats-{}-{}.log".format(dataset, pair)

    values = parse_logs(fpath)
    mat = values.reshape((len(tp_taus), len(t_taus)))
    mat = mat * 100.0

    plt.subplot(gs[p])
    ax = plt.gca()

    cax = ax.imshow(mat, cmap=plt.get_cmap("RdPu"), alpha=0.5)
    plt.colorbar(cax)

    ax.grid(False)

    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax.text(
                j, i, "{:.2f}".format(v),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=6.5
            )

    xs = list(range(len(t_taus)))
    ax.set_xticks(xs)

    ys = list(range(len(tp_taus)))

    ax.set_xticklabels([
        r"$\tau_2$={}".format(t) for t in t_taus
    ])

    if p == 0:
        ax.set_yticks(ys)
        ax.set_yticklabels([
            r"$\tau_1$={}".format(t) for t in tp_taus
        ])
    else:
        ax.set_yticks([])

    plt.setp(
        ax.get_xticklabels(), rotation=30,
        fontsize=8.5,
    )

    plt.setp(
        ax.get_yticklabels(), rotation=45,
        fontsize=8.5,
    )

    ax.set_title(titles[p], fontsize=8.0)

plt.show()
plt.close()
