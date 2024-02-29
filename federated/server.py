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
import time
import numpy as np
import datetime
from tqdm import tqdm
from model.FedTPG import FedTPG,load_clip_to_cpu
from model.custom_coop import CoOpCLIP
from model.custom_vlp import VLPCLIP
from dataloader.dm_federated import TestDataManager 
from federated.utils import *
import copy
from federated.client import Client
import math
import random
from federated.base_trainer import TrainerBase

class Server(TrainerBase):
    # expand the trainer to the FL scenarios

    def __init__(self, cfg):
        super().__init__()
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        seed = cfg.SEED
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
            # Set the random seed for PyTorch
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = os.path.join(cfg.OUTPUT_DIR,cfg.EXP_NAME,cfg.MODEL.NAME,str(cfg.TRAIN.NUM_CLASS_PER_CLIENT)+"_"+str(cfg.DATASET.NUM_SHOTS),str(cfg.SEED))
        self.cfg = cfg

        self.build_model()

        cfg.defrost()

        self.evaluator = Classification(cfg)
        self.clients = []
        self.init_server(cfg)

        self.cfg = cfg


    def build_model(self):
        cfg = self.cfg

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        self.model_name = cfg.MODEL.NAME
        print("Building custom CLIP")
        if cfg.MODEL.NAME == 'fedtpg':
            self.model = FedTPG(cfg, clip_model,device = self.device)
        elif cfg.MODEL.NAME == 'coop':
            self.model = CoOpCLIP(cfg, clip_model,device = self.device)
        elif cfg.MODEL.NAME == 'vlp':
            self.model = VLPCLIP(cfg, clip_model,device = self.device)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.register_model("prompt_learner", self.model.prompt_learner)

        self.clip_model = clip_model


    def init_server(self, cfg):

        num_class_per_client = cfg.TRAIN.NUM_CLASS_PER_CLIENT
        available_datasets = cfg.DATASET.NAME_SPACE

        dataset_classnum = {'imagenet': 1000, 'caltech101':100, 'oxford_flowers': 102,'eurosat':10, 'oxford_pets':37, 'fgvc_aircraft': 100,
                            'food101': 101, 'dtd': 47, 'ucf101':101,'stanford_cars':196,'sun397':397 }
        client_id,last_num_clients = 0,0

        for dataname in available_datasets:

            if cfg.TRAIN.SPLIT =='base':
                m = math.ceil(dataset_classnum[dataname] / 2)
            elif cfg.TRAIN.SPLIT =='all':
                m = dataset_classnum[dataname]
            all_cls_idx = np.arange(m)
            num_client_dataset = np.around(m/num_class_per_client).astype(int)
            if num_client_dataset==0:
                num_client_dataset = 1
            current_num_clients = last_num_clients+num_client_dataset
            while client_id< current_num_clients:

                if client_id==current_num_clients-1:
                    available_cls = all_cls_idx[(client_id - last_num_clients) * num_class_per_client:]
                else:
                    available_cls = all_cls_idx[(client_id-last_num_clients)*num_class_per_client:(client_id-last_num_clients+1)*num_class_per_client]

                client = Client(cfg, len(self.clients),dataname,available_cls,self.clip_model)

                self.clients.append(client)
                client_id+=1
            last_num_clients = current_num_clients

        self.num_clients = len(self.clients)

        print(f'total number of clients:{self.num_clients}')
    def distribute(self, idx):

        self.clients[idx].load_meta(self.meta_net_glob.state_dict())

    def train(self):
        self.before_train()
        self.meta_net_glob = copy.deepcopy(self.model.prompt_learner)
        for epoch in range(self.start_epoch, self.max_epoch):

            self.epoch = epoch

            num_selected = max(int(self.cfg.TRAIN.AVAIL_PERCENT * self.num_clients), 1)
            idxs_users = np.random.choice(range(len(self.clients)), num_selected, replace=False)

            w_glob = None
            # updates = []
            for idx in idxs_users:
                self.distribute(idx)
                w_local = self.clients[idx].train(epoch)
                if w_glob is None:
                    w_glob = copy.deepcopy(w_local)
                else:
                    for k in w_glob.keys():
                        w_glob[k] +=w_local[k]
            
            for k in w_glob.keys():
                # if "token_prefix" not in k and "token_suffix" not in k:
                w_glob[k] = torch.div(w_glob[k], num_selected)
            self.meta_net_glob.load_state_dict(w_glob, strict=False)

        self.model.prompt_learner.load_state_dict(self.meta_net_glob.state_dict())
        
        self.after_train()

    def model_inference(self, input,classnames, dataname):
        # return self.model(input,classnames, dataname)
        return self.model(input,classnames, dataname)[0]

    def parse_batch(self, batch):
        input = batch["img"]
        label = batch["label"]
        cname = batch["cname"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, cname

    @torch.no_grad()
    def test(self,split):
        """A generic testing pipeline."""
        self.set_model_mode("eval")

        dm = TestDataManager(self.cfg,split)
        data_loaders = dm.test_loaders
        datasets = dm.test_datasets
        acc_list = []
        for i, data_loader in enumerate(data_loaders):
            self.evaluator.reset()
            print(f"Evaluate on the *{split}* set of {self.cfg.DATASET.TESTNAME_SPACE[i]}")
            classnames = datasets[i].classnames
            dataname = datasets[i].data_name

            for batch_idx, batch in enumerate(tqdm(data_loader)):
                inputs, labels, cnames = self.parse_batch(batch)
                outputs = self.model_inference(inputs, classnames, dataname)
                self.evaluator.process(outputs, labels)

            results = self.evaluator.evaluate()

            acc_list.append(list(results.values())[0])
        acc_mean = np.mean(acc_list)
        print(f"avg accuracy: {acc_mean}")

    @torch.no_grad()
    def local_test(self):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        acc_dict = {}

        for i, client in enumerate(self.clients):
            self.evaluator.reset()
            print(f"Evaluate on the *{str(i)}th* client of {client.data_name}")
            classnames = client.available_classes
            dataname = client.data_name
            test_loader = client.test_loader
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                inputs, labels, cnames = self.parse_batch(batch)
                outputs = self.model_inference(inputs, classnames, dataname)
                self.evaluator.process(outputs, labels)

            results = self.evaluator.evaluate()
            acc= list(results.values())[0]

            if dataname not in acc_dict:
                acc_dict[dataname]= [acc]
            else:
                acc_dict[dataname].append(acc)
        acc_list = []
        for key in acc_dict.keys():
            acc_list.append(np.mean(acc_dict[key]))
            print(f"avg acc of {key}: {np.mean(acc_dict[key])}")
        print(f"avg local accuracy: {np.mean(acc_list)}")

    def before_train(self):
        directory = self.output_dir
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = os.path.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")
        last_epoch = (self.epoch + 1) == self.max_epoch
        if last_epoch:
            self.save_model(self.epoch, self.output_dir)
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            # eval_based on each dataset
            self.local_test()
            if self.cfg.TEST.SPLIT=='base&new':
                self.test('base')
                self.test('new')
            else:
                self.test(self.cfg.TEST.SPLIT)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))

        print(f"Elapsed: {elapsed}")




