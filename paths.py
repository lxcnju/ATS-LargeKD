import os

cur_dir = "./"

cifar_fdir = "./data"
ckpt_dir = "./ckpts"

if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)


save_dir = os.path.join(cur_dir, "logs")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


cifar_fpaths = {
    "cifar100": {
        "train_fpaths": [
            os.path.join(cifar_fdir, "cifar100-train-part1.pkl"),
            os.path.join(cifar_fdir, "cifar100-train-part2.pkl"),
        ],
        "test_fpath": os.path.join(cifar_fdir, "cifar100-test.pkl")
    },
}
