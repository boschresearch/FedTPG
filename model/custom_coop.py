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

import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CoOpPromptLearner(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        n_ctx, ctx_depth = cfg.MODEL.N_CTX, cfg.MODEL.D_CTX
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, 512)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized


    def forward(self):

        ctx = self.ctx      

        return ctx

class CoOpCLIP(nn.Module):
    def __init__(self, cfg, clip_model,device='cuda'):
        super().__init__()
        self.cfg = cfg


        self.set_prompt_prefix()
        # ctx_dim = clip_model.ln_final.weight.shape[0]

        self.prompt_learner = CoOpPromptLearner(cfg)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.clip_model_ = clip_model
    def set_prompt_prefix(self):

        ctx_init = self.cfg.MODEL.CTX_INIT
        n_ctx = self.cfg.MODEL.N_CTX

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            self.n_ctx = len(ctx_init.split(" "))
            self.prompt_prefix = ctx_init
        else:
            # random initialization
            self.n_ctx = n_ctx
            self.prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")


    def get_tokenized_classnames(self, classnames):

        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)
        token_prefix = embedding[:, :1, :]  # SOS
        token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
        return token_prefix, token_suffix,tokenized_prompts

    def forward(self, image,classnames, dataname):


        classnames = [name.replace("_", " ") for name in classnames]
        text_features = self.encode_text(classnames, dataname)
        # [N,512]

        temp = CUSTOM_TEMPLATES[dataname]
        prompts_ = [temp.format(c) for c in classnames]
        # print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features_ = self.clip_model_.encode_text(prompts_)
            text_features_ = text_features_ / text_features_.norm(dim=-1, keepdim=True)

        image_features = self.encode_image(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)

        kg_score = cos(text_features,text_features_)
        kg_score = 1.0-torch.mean(kg_score)


        return logits, kg_score


    def encode_image(self,image):
        return self.image_encoder(image.type(self.dtype))

    def encode_text(self,classnames):

        token_prefix, token_suffix,tokenized_prompts = self.get_tokenized_classnames(classnames) 

        ctx = self.prompt_learner()
        ctx = ctx.to(self.dtype)
        ctx = ctx.unsqueeze(0).expand(token_prefix.shape[0], -1, -1)
        prompt_vectors = torch.cat(
            [
                token_prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                token_suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        text_features = self.text_encoder(prompt_vectors, tokenized_prompts)
        return text_features

