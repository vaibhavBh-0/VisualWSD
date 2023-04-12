# metrics.py
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


class RuntimeMetrics:
    @staticmethod
    def reciprocal_rank_per_batch(logit_scores, gold_indices, top_k):
        rank_indices = (torch.topk(logit_scores, dim=1, k=top_k)[1])
        mask = gold_indices.repeat_interleave(top_k).reshape(-1, top_k)
        ranks = (mask == rank_indices).int().argmax(dim=1) + 1
        reciprocal_ranks = ranks.float() ** -1

        return torch.sum(reciprocal_ranks)

    @staticmethod
    def hit_rate_at1(logit_scores, gold_indices):
        hit_rate = logit_scores.argmax(dim=1) == gold_indices

        return torch.sum(hit_rate.float())


def test_reciprocal_rank():
    score = torch.randn(50, 10)
    gold_indices = torch.randint(10, (50,))
    rr = RuntimeMetrics.reciprocal_rank_per_batch(score, gold_indices, top_k=10)
    print(rr)


def test_hit_rate_at1():
    score = torch.randn(50, 10)
    gold_indices = torch.randint(0, 9, (50,))
    hr = RuntimeMetrics.hit_rate_at1(score, gold_indices)
    print(hr)


if __name__ == "__main__":
    test_reciprocal_rank()
    test_hit_rate_at1()
