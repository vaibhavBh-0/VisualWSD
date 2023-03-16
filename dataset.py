# dataset.py
#
# Created by Vaibhav Bhargava on 3/2/23
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


import os
import random
import torch
from enum import Enum
import pandas as pd
from torch.utils.data import Dataset
import torchvision


class DatasetConfig(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class VWSDDataset(Dataset):
    def __init__(self, img_data_path: str, wsd_data_path: str, gold_data_path: str, config: DatasetConfig, seed=42,
                 split=0.8):
        self.img_data_path = img_data_path
        self.wsd_data_path = wsd_data_path
        self.gold_data_path = gold_data_path
        self._config = config
        self.seed = seed
        self.split = split

        columns = ['word', 'context'] + [f'{a}{idx}' for idx, a in enumerate(['sample'] * 10, start=1)]
        df = pd.read_csv(self.wsd_data_path, sep='\t', names=columns)

        if self._config != DatasetConfig.TEST:
            df_gold = pd.read_csv(self.gold_data_path, sep='\t', names=['gold'])

            df = pd.concat([df, df_gold], axis=2)
            df = self.dehyphenate(df)

            sampling_idx = list(range(len(df)))
            random.seed(self.seed)
            random.shuffle(sampling_idx)

            ratio = int(split * len(sampling_idx))
            self.sampling_idx = sampling_idx[: ratio] if self._config == DatasetConfig.TRAIN else sampling_idx[ratio:]
        else:
            df = self.dehyphenate(df)
            self.sampling_idx = list(range(len(df)))

        self.df = df

    @staticmethod
    def dehyphenate(df):
        """
        The dataset uses hypens as a token to separate two words. The method therefore dehypenates
        :param df: The raw dataframe.
        :return: dehypenated
        """

        return df.replace('-', ' ', regex=True)

    def __len__(self):
        return len(self.sampling_idx)

    def __getitem__(self, item):
        # 1. Using purely transformers Vision Text Dual Encoder.
        # 2. Base classing Vision Text Dual Encoder so that we can override it's forward method to pass embeddings.
        # Assuming we have Images / ConNextv2 embeddings
        data_point = self.df.iloc[item]
        word, context = data_point[:2]


