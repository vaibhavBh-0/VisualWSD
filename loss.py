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

    def forward(self, similarity: torch.Tensor, text_to_img_mapping: torch.Tensor):
        text_loss = nn.functional.cross_entropy(similarity, target=text_to_img_mapping)
        img_to_text_mapping_target_mask = torch.zeros_like(similarity, device=similarity.device).T

        # Not all images will have corresponding correct mapping to text.
        for idx, val in enumerate(text_to_img_mapping.tolist()):
            img_to_text_mapping_target_mask[val, idx] = 1.0

        visual_loss = nn.functional.cross_entropy(similarity.T, target=img_to_text_mapping_target_mask)

        loss = (text_loss + visual_loss) / 2.0

        return loss


if __name__ == '__main__':
    loss_criterion = LiTLoss()
    x = torch.randn(5, 10)
    size, img_shape = x.shape
    # gold_example for mapping text to image.
    y = torch.randint(img_shape, size=(size,))
    loss_val = loss_criterion(x, y)
    print(f'the result {loss_val} and  x.shape {x.shape}')
