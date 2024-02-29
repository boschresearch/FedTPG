""" Federated Text-driven Prompt Generation for Vision-Language Models (ICLR 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import torch
from config.defaults import _C as cfg_default
from config.utils import reset_cfg
from utils import setup_logger
from federated.server import Server

import os
def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def setup_cfg(args):
    cfg = cfg_default.clone()

    # 3. From input arguments
    reset_cfg(cfg, args)

    cfg.freeze()

    return cfg

def main(args):
    cfg = setup_cfg(args)

    setup_logger(os.path.join(cfg.OUTPUT_DIR,cfg.EXP_NAME,cfg.MODEL.NAME,str(cfg.TRAIN.NUM_CLASS_PER_CLIENT)+"_"+str(cfg.DATASET.NUM_SHOTS),str(cfg.SEED)))

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    fl_server = Server(cfg)


    if args.eval_only:
        if args.model_name!='clip':
            fl_server.load_model(args.model_dir, epoch=args.load_epoch)
            
        # fl_server.local_test()
        if fl_server.cfg.TEST.SPLIT== 'base&new':
            fl_server.test("base")
            fl_server.test("new")
        else:
            fl_server.test(fl_server.cfg.TEST.SPLIT)
        return

    if not args.no_train:
        fl_server.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./DATA", help="path to dataset")

    parser.add_argument(
        "--exp_name", type=str, default="cross_cls", help="cross_data, cross_data, cross_cls"
    )
    parser.add_argument(
        "--model_name", type=str, default="fedtpg", help="fedtpg, coop, vlp"
    )
    parser.add_argument(
        "--num_shots", type=int, default=8, help="number of samples each class"
    )
    parser.add_argument(
        "--depth_ctx", type=int, default=1, help="depth of ctx"
    )
    parser.add_argument(
        "--model_depth", type=int, default=0, help="number of self-attention modules in prompt net"
    )
    parser.add_argument(
        "--n_ctx", type=int,default=4,help="length of ctx"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=500, help="number of running epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=200, help="batch size"
    )
    parser.add_argument(
        "--num_cls_per_client", type=int, default=20, help="number of cls per client"
    )
    parser.add_argument(
        "--avail_percent", type=float, default=1.0, help="avail_percent"
    )
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=43, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--w", type=int, default=0, help="weight of regularization for KgCoOp"
    )
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="name of CNN backbone")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int,
        help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--per-class", action="store_true", help="do not call trainer.train()"
    )

    args = parser.parse_args()
    main(args)