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

def reset_cfg(cfg, args):
    cfg.EXP_NAME = args.exp_name
    if args.exp_name == "cross_data":
        cfg.DATASET.NAME_SPACE = ["imagenet"]
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE = ['caltech101', 'oxford_flowers', 'fgvc_aircraft', 'ucf101', 'oxford_pets', 'food101', 'dtd', 'stanford_cars', 'sun397', 'eurosat']
    elif args.exp_name == "cross_domain":
        cfg.DATASET.NAME_SPACE = ["imagenet"]
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE =['imagenet-v2','imagenet-s','imagenet-a','imagenet-r','imagenet']
    elif args.exp_name == "cross_cls":
        cfg.DATASET.NAME_SPACE = ['caltech101', 'oxford_flowers', 'fgvc_aircraft', 'ucf101', 'oxford_pets', 'food101', 'dtd', 'stanford_cars', 'sun397']
        cfg.TRAIN.SPLIT = 'base'
        cfg.TEST.SPLIT = 'base&new'
        cfg.DATASET.TESTNAME_SPACE = ['caltech101', 'oxford_flowers', 'fgvc_aircraft', 'ucf101', 'oxford_pets', 'food101', 'dtd', 'stanford_cars', 'sun397']


    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.OUTPUT_DIR = args.output_dir
    cfg.RESUME = args.resume
    cfg.SEED = args.seed
    cfg.MODEL.BACKBONE.NAME = args.backbone
    cfg.OPTIM.MAX_EPOCH = args.num_epoch


    cfg.MODEL.D_CTX = args.depth_ctx
    cfg.MODEL.N_CTX = args.n_ctx
    cfg.MODEL.DEPTH = args.model_depth
    cfg.MODEL.NAME = args.model_name

    cfg.DATALOADER.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_CLASS_PER_CLIENT = args.num_cls_per_client
    cfg.TRAIN.AVAIL_PERCENT = args.avail_percent
    cfg.TRAIN.W = args.w