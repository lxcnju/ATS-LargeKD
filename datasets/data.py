
def load_data(dset):
    if dset in ["cifar100"]:
        from datasets.cifar_data import load_cifar_data as load_data
        from datasets.cifar_data import CifarDataset as Dataset
    else:
        raise ValueError("No such dset: {}".format(dset))

    train_xs, train_ys, test_xs, test_ys = load_data(
        dset
    )

    train_set = Dataset(train_xs, train_ys, is_train=True)
    test_set = Dataset(test_xs, test_ys, is_train=False)
    return train_set, test_set
