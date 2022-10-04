import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from tqdm import tqdm

from utils import count_acc, Averager
from utils import append_to_logs
from utils import format_logs

from tools import construct_loaders
from tools import construct_optimizer
from tools import construct_lr_scheduler

from paths import ckpt_dir


class Classify():
    def __init__(
        self,
        train_set,
        test_set,
        model,
        args
    ):
        self.train_set = train_set
        self.test_set = test_set
        self.model = model
        self.args = args

        self.train_loader, self.test_loader = \
            construct_loaders(
                train_set, test_set, args
            )

        self.optimizer = construct_optimizer(
            self.model, args
        )
        self.lr_scheduler = construct_lr_scheduler(self.optimizer, args)

        self.logs = {
            "EPOCHS": [],
            "LOSSES": [],
            "TrACCS": [],
            "TeACCS": [],
        }

    def main(self, save_ckpts=False, ext=None):
        for epoch in range(1, self.args.epoches + 1):
            ce_loss, tr_acc = self.train(
                model=self.model,
                optimizer=self.optimizer,
                loader=self.train_loader,
                args=self.args
            )
            te_acc = self.test(
                model=self.model,
                loader=self.test_loader,
                args=self.args
            )
            print("[Epoch:{}] [Loss:{}] [TrAcc:{}] [TeAcc:{}]".format(
                epoch, ce_loss, tr_acc, te_acc
            ))

            # add to log
            self.logs["EPOCHS"].append(epoch)
            self.logs["LOSSES"].append(ce_loss)
            self.logs["TrACCS"].append(tr_acc)
            self.logs["TeACCS"].append(te_acc)

            self.lr_scheduler.step()

            if save_ckpts is not False:
                assert isinstance(save_ckpts, list)

                if epoch in save_ckpts:
                    if ext is None:
                        fpath = os.path.join(
                            ckpt_dir,
                            "{}-{}-E{}.pth".format(
                                self.args.dataset,
                                self.args.net_name,
                                epoch
                            )
                        )
                        self.save_ckpt(fpath)
                    else:
                        fpath = os.path.join(
                            ckpt_dir,
                            "{}-{}-E{}-{}.pth".format(
                                self.args.dataset,
                                self.args.net_name,
                                epoch,
                                ext
                            )
                        )
                        self.save_ckpt(fpath)

    def train(self, model, optimizer, loader, args):
        model.train()

        avg_ce_loss = Averager()
        acc_avg = Averager()
        for _, sx, sy in tqdm(loader):
            if args.cuda:
                sx, sy = sx.cuda(), sy.cuda()

            logits = model(sx, is_feat=False)
            loss = F.cross_entropy(logits, sy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_ce_loss.add(loss.item())

            acc = count_acc(logits, sy)
            acc_avg.add(acc)

        ce_loss = avg_ce_loss.item()
        acc = acc_avg.item()
        return ce_loss, acc

    def test(self, model, loader, args):
        model.eval()

        acc_avg = Averager()
        with torch.no_grad():
            for _, tx, ty in loader:
                if args.cuda:
                    tx, ty = tx.cuda(), ty.cuda()

                logits = model(tx, is_feat=False)

                acc = count_acc(logits, ty)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def save_ckpt(self, fpath):
        # save model
        torch.save(self.model.state_dict(), fpath)
        print("Model saved in: {}".format(fpath))

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
