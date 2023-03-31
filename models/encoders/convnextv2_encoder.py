# convnextv2_encoder.py
#
# Created by Vaibhav Bhargava on 20-03-2023
# 
# Copyright © 2023 Vaibhav Bhargava
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import torch.nn as nn
from transformers import ConvNextV2Model


class ConvNextV2Encoder(nn.Module):
    def __init__(self, embedding_dim: int, model_path: str):
        super(ConvNextV2Encoder, self).__init__()

        self.conv_next_model = ConvNextV2Model.from_pretrained(model_path)

        # Freeze the encoder model.
        for param in self.conv_next_model.parameters():
            param.requires_grad = False

        in_dim = list(self.conv_next_model.modules())[-1].weight.shape[0]
        self.embedding_dim = nn.Linear(in_dim, embedding_dim)
        
    def forward(self, x):
        x = self.conv_next_model(x)['pooler_output']
        x = self.embedding_dim(x)

        return x
