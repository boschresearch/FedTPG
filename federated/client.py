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

from model.FedTPG import FedTPG
from model.custom_coop import CoOpCLIP
from model.custom_vlp import VLPCLIP
from dataloader.dm_federated import TrainDataManager
from federated.utils import *
import torch.nn.functional as F
from federated.base_trainer import TrainerBase

class Client(TrainerBase):
    """A local client with frozen clip and FL meta_net and private training data"""
    def __init__(self, cfg, client_id,dataname,available_cls,clip_model):
        super().__init__()
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.client_id = client_id

        # self.id = -1
        self.cfg = cfg
        self.build_data_loader(dataname,available_cls)
        self.build_model(clip_model)


    def build_data_loader(self,dataname,available_cls):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = TrainDataManager(self.cfg, dataname,available_cls)

        self.train_loader = dm.train_loader
        # self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.available_classes = dm.available_classes
        self.data_name = dm.data_name

    def build_model(self,clip_model):
        cfg = self.cfg

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        # clip_model = load_clip_to_cpu(cfg)
        self.model_name = cfg.MODEL.NAME
        print("Building custom CLIP")
        if cfg.MODEL.NAME == 'fedtpg':
            self.model = FedTPG(cfg, clip_model,device = self.device)
        elif cfg.MODEL.NAME == 'coop':
            self.model = CoOpCLIP(cfg, clip_model,device = self.device)
        elif cfg.MODEL.NAME == 'vlp':
            self.model = VLPCLIP(cfg, clip_model,device = self.device)

        self.w = cfg.TRAIN.W

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

        # params = ([p for p in self.model.prompt_learner.parameters()])
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

    def train(self,num_round):
        self.set_model_mode("train")
        losses = MetricMeter()

        # lab2cname= self.dataset.lab2cname
        dataname = self.data_name
        classnames = self.available_classes
        # batch = next(iter(self.train_loader))
        for batch in self.train_loader:
            loss,acc = self.forward_backward(batch,dataname,classnames)
            self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        losses.update(loss_summary)

        info = []
        info += [f"epoch [{num_round + 1}/{self.max_epoch}]"]
        info += [f"client_id [{self.client_id}]"]
        info += [f"{dataname}"]
        info += [f"{losses}"]
        info += [f"lr {self.get_current_lr():.4e}"]
        print(" ".join(info))

        self.update_lr()
        local_updates = self.model.prompt_learner.state_dict()
        return local_updates



    def load_meta(self, global_net):
        self.model.prompt_learner.load_state_dict(global_net)


    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError


    def forward_backward(self, batch, dataname,classnames):
        images, labels,cnames = self.parse_batch(batch)

        output, score = self.model(images,classnames, dataname)
        loss = F.cross_entropy(output, labels) + self.w*score
        return loss,compute_accuracy(output, labels)[0].item()

    def parse_batch(self, batch):
        input = batch["img"]
        label = batch["label"]
        cname = batch["cname"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, cname

    def get_current_lr(self, names=None):
        # current_lr = self.sched.get_last_lr()
        # return current_lr[0]
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]
    def model_inference(self, input, classnames, dataname):
        # return self.model(input,classnames, dataname)
        return self.model(input, classnames, dataname)[0]