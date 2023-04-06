# dataset.py
#
# Created by Sharad Chandakacherla on 3/21/23
#
# Copyright © 2023 Sharad Chandakacherla
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import Type
import torch
import torch.nn as nn


class LiT(nn.Module):
    def __init__(self, embedding_dim: int, vision_model_path: str, text_model_path: str,
                 tokenizer_len: int, vision_encoder: Type[nn.Module], text_encoder: Type[nn.Module],
                 logit_scale_init_value=2.6592):
        super(LiT, self).__init__()
        self.vision_encoder = vision_encoder(embedding_dim=embedding_dim, model_path=vision_model_path)
        self.text_encoder = text_encoder(embedding_dim=embedding_dim, model_path=text_model_path,
                                         tokenizer_len=tokenizer_len)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

    def forward(self, text_data, image_data, img_samples: int):
        text_projection = self.text_encoder(text_data)
        normalized_text = nn.functional.normalize(text_projection, p=2, dim=-1)

        image_projection = self.vision_encoder(image_data)
        normalized_image = nn.functional.normalize(image_projection, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()

        text_batch_size = normalized_text.shape[0]

        # (text_batch, img_samples x image_batch); text_batch == image_batch
        text_per_image_logit = torch.matmul(normalized_text, normalized_image.t()) * logit_scale

        # text_batch, img_samples
        out = torch.zeros((text_batch_size, img_samples), device=text_per_image_logit.device)

        for text_idx in range(text_batch_size):
            img_idx = img_samples * text_idx
            out[text_idx, :] += text_per_image_logit[text_idx, img_idx: img_idx + img_samples]

        return out
