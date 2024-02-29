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
from model.prompt_net import PromptTranslator
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


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


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.conv1 = clip_model.conv1
        self.class_embedding = clip_model.class_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_pre = clip_model.ln_pre
        self.transformer = clip_model.transformer
        self.ln_post = clip_model.ln_post
        self.proj = clip_model.proj

    def forward(self, x, vis_ctx=[]):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]forwad
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, vis_ctx, False)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, text_ctx):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, text_ctx, True)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        n_ctx, ctx_depth = cfg.MODEL.N_CTX, cfg.MODEL.D_CTX
        self.meta_net = PromptTranslator(n_ctx, ctx_depth, depth=cfg.MODEL.DEPTH)
        self.meta_net.half()

        self.ctx_depth = ctx_depth
        self.n_ctx = n_ctx

    def forward(self, context_emb):
        text_ctx, vis_ctx = self.meta_net(context_emb.unsqueeze(0))  # (n_ctx, ctx_dim) # self.ctx

        return text_ctx, vis_ctx


class FedTPG(nn.Module):
    def __init__(self, cfg, clip_model,  device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.set_prompt_prefix()
        # ctx_dim = clip_model.ln_final.weight.shape[0]

        self.prompt_learner = PromptLearner(cfg)
        self.image_encoder = ImageEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)

        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.clip_model_ = clip_model


    def set_prompt_prefix(self):

        n_ctx = self.cfg.MODEL.N_CTX
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
        # token_prefix = embedding[:, :1, :]  # SOS
        # token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
        return embedding, tokenized_prompts

    def forward(self, image, classnames,dataname):

        classnames = [name.replace("_", " ") for name in classnames]
        prompts_ = classnames
        # print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features_ = self.clip_model_.encode_text(prompts_)
            text_features_ = text_features_ / text_features_.norm(dim=-1, keepdim=True)


        text_features, vis_ctx = self.encode_text(classnames, text_features_)
        image_features = self.encode_image(image, vis_ctx)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        return logits, 0

    def encode_image(self, image, vis_ctx):
        return self.image_encoder(image.type(self.dtype))

    def encode_text(self, classnames, text_features_):

        context_emb = text_features_
        prompt_vectors, tokenized_prompts = self.get_tokenized_classnames(classnames)

        text_ctx, vis_ctx = self.prompt_learner(context_emb)

        prompt_vectors = torch.cat(
            [
                prompt_vectors[:, :1],  # (dim0, 1, dim)
                text_ctx[0].unsqueeze(0).expand(prompt_vectors.shape[0], -1, -1),  # (dim0, n_ctx, dim)
                prompt_vectors[:, 1 + text_ctx.shape[1]:],  # (dim0, *, dim)
            ],
            dim=1,
        )
        if len(text_ctx) > 1:
            text_ctx = text_ctx[1:]
        else:
            text_ctx = []
        text_features = self.text_encoder(prompt_vectors, tokenized_prompts, text_ctx)
        return text_features, vis_ctx

