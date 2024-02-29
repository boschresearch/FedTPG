"""
Modified from https://github.com/KaiyangZhou/deep-person-reid

Copyright (c) 2018 Kaiyang Zhou, licensed under the MIT License
cf. 3rd-party-licenses.txt file in the root directory of this source tree

"""
import warnings
import torch
import torch.nn as nn
import os
import shutil
from collections import defaultdict
from collections import OrderedDict, defaultdict
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x
class Classification:
    """Evaluator for classification."""

    def __init__(self, cfg):
        self.cfg = cfg
        # self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
        # if cfg.TEST.PER_CLASS_RESULT
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.3f}%\n"
            f"* error: {err:.3f}%\n"
            f"* macro_f1: {macro_f1:.3f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = os.path.join(os.path.join(self.cfg.OUTPUT_DIR,self.cfg.DATASET.NAME+'_'+str(self.cfg.DATASET.NUM_SHOTS)),
                                     "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results

def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def build_optimizer(model, optim_cfg):

    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
        param_groups: If provided, directly optimize param_groups and abandon model
    """
    AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "adamw"]
    optim = optim_cfg.NAME
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM


    if optim not in AVAI_OPTIMS:
        raise ValueError(
            f"optim must be one of {AVAI_OPTIMS}, but got {optim}"
        )


    if isinstance(model, nn.Module):
        param_groups = model.parameters()
    else:
        param_groups = model

    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

    return optimizer


def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """

    AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.STEPSIZE
    gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(stepsize)}"
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )


    return scheduler

def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    model_name=""
):
    r"""Save checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is True.
        model_name (str, optional): model name to save.
    """
    mkdir_if_missing(save_dir)

    # save model
    epoch = state["epoch"]
    if not model_name:
        model_name = "model.pth.tar-" + str(epoch)
    fpath = os.path.join(save_dir, model_name)
    torch.save(state, fpath)
    print(f"Checkpoint saved to {fpath}")

    # save current model name
    checkpoint_file = os.path.join(save_dir, "checkpoint")
    checkpoint = open(checkpoint_file, "w+")
    checkpoint.write("{}\n".format(os.path.basename(fpath)))
    checkpoint.close()

    if is_best:
        best_fpath = os.path.join(os.path.dirname(fpath), "model-best.pth.tar")
        shutil.copy(fpath, best_fpath)
        print('Best checkpoint saved to "{}"'.format(best_fpath))


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    """
    if fpath is None:
        raise ValueError("File path is None")

    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)


    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint


def resume_from_checkpoint(fdir, model, optimizer=None, scheduler=None):
    r"""Resume training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fdir (str): directory where the model was saved.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (Scheduler, optional): an Scheduler.

    Returns:
        int: start_epoch.
    """
    with open(os.path.join(fdir, "checkpoint"), "r") as checkpoint:
        model_name = checkpoint.readlines()[0].strip("\n")
        fpath = os.path.join(fdir, model_name)

    print('Loading checkpoint from "{}"'.format(fpath))
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint["state_dict"])
    print("Loaded model weights")

    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded optimizer")

    if scheduler is not None and "scheduler" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded scheduler")

    start_epoch = checkpoint["epoch"]
    print("Previous epoch: {}".format(start_epoch))

    return start_epoch


class AverageMeter:
    """Compute and store the average and current value.
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


class MetricMeter:
    """Store the average and current value for a set of metrics.
    """

    def __init__(self, delimiter=" "):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                "Input to MetricMeter.update() must be a dictionary"
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(f"{name} {meter.val:.4f} ({meter.avg:.4f})")
        return self.delimiter.join(output_str)
