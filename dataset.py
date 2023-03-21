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
from torchvision.io import read_image, ImageReadMode
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class DatasetConfig(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class VWSDDataset(Dataset):
    def __init__(self, img_data_path: str, wsd_data_path: str, gold_data_path: str, config: DatasetConfig,
                 image_processor: BaseImageProcessor, tokenizer: PreTrainedTokenizerFast, seed=42, split=0.8):
        self.img_data_path = img_data_path
        self.wsd_data_path = wsd_data_path
        self.gold_data_path = gold_data_path
        self._config = config
        self.seed = seed
        self.split = split
        self.image_processor = image_processor
        self.fast_tokenizer = tokenizer
        self.sep = '[CONDITIONED]'
        self.fast_tokenizer.add_tokens(new_tokens=self.sep, special_tokens=False)

        columns = ['word', 'context'] + [f'{a}{idx}' for idx, a in enumerate(['sample'] * 10, start=1)]
        df = pd.read_csv(self.wsd_data_path, sep='\t', names=columns)

        if self._config != DatasetConfig.TEST:
            df_gold = pd.read_csv(self.gold_data_path, sep='\t', names=['gold'])

            df = pd.concat([df, df_gold], axis=1)
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
        # 2. Base classing Vision Text Dual Encoder so that we can override its forward method to pass embeddings.
        # Assuming we have Images / ConNextv2 embeddings.
        sampled_idx = self.sampling_idx[item]

        data_point = self.df.iloc[sampled_idx]
        word, context = data_point[:2]

        # TODO: Need to see what shall be the separating token between the word and context if any.
        text = f'{word}{self.sep}{context}'

        # Tokenizer gives a dictionary of input_ids, token_type_ids, attention_mask.
        text_input = self.fast_tokenizer(text, return_tensors='pt')

        # TODO: Current Mining Scheme is only going as per the dataset.
        #  We can go for more negative image examples based on the "word" not conditioned on "context".

        # image processor gives a dictionary of pixel_values.
        # imgs_input = [self.image_processor(img, return_tensors='pt')['pixel_values'] for img in imgs]
        img_paths = data_point[2:] if self._config == DatasetConfig.TEST else data_point[2:-1]

        imgs = [
            self.image_processor(read_image(os.path.join(self.img_data_path, img_path), mode=ImageReadMode.RGB),
                                 return_tensors='pt')
            for img_path in img_paths
        ]

        gold_example = torch.zeros((1, len(imgs)))

        if self._config != DatasetConfig.TEST:
            gold_path = data_point[-1]
            for idx, img_path in enumerate(img_paths):
                if img_path == gold_path:
                    gold_example[:, idx] = 1.0
                    break

        # TODO: Testing whether this scales to batching in dataloader.

        return text_input, imgs, gold_example
