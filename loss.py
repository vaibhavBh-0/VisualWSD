# loss.py
#
# Created by Vaibhav Bhargava on 31-03-2023
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
import torch
import torch.nn as nn


class LiTLoss(nn.Module):
    def __init__(self):
        super(LiTLoss, self).__init__()

    def forward(self, similarity):
        loss = self.clip_loss(similarity)
        return loss

    # clip loss as mentioned in "Learning Transferable Visual Models From Natural Language Supervision",
    # refactored version of the hugging face implementation of VisionTextDualEncoderModel's source
    @staticmethod
    def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    # clip loss as mentioned in "Learning Transferable Visual Models From Natural Language Supervision",
    # refactored version of the hugging face implementation of VisionTextDualEncoderModel's source
    def clip_loss(self, similarity) -> torch.Tensor:
        text_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (text_loss + image_loss) / 2.0


if __name__ == '__main__':
    loss_criterion = LiTLoss()
    x = torch.randn(10,10)
    loss = loss_criterion(x)
    print(f'the result {loss} and  x.shape {x.shape}')
