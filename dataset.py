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
import numpy as np
import cv2
from enum import Enum
import pandas as pd
from torch.utils.data import Dataset
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class DatasetConfig(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class VWSDDataset(Dataset):
    def __init__(self, img_data_path: str, wsd_data_path: str, gold_data_path: str, config: DatasetConfig,
                 image_processor: BaseImageProcessor, tokenizer: PreTrainedTokenizerFast, seed=42, split=0.8,
                 device=torch.device('cpu')):
        self.img_data_path = img_data_path
        self.wsd_data_path = wsd_data_path
        self.gold_data_path = gold_data_path
        self._config = config
        self.seed = seed
        self.split = split
        self.image_processor = image_processor
        self.fast_tokenizer = tokenizer
        self.sep = '[CONDITIONED]'
        self.fast_tokenizer.add_tokens(new_tokens=self.sep, special_tokens=True)
        self.model_max_length = self.fast_tokenizer.model_max_length
        self.device = device
        self.non_blocking = self.device.type != 'cpu'

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

        for col in ['word', 'context']:
            df[col] = df[col].apply(lambda x: x.replace('-', ' '))

        return df

    @staticmethod
    def load_image(path) -> torch.Tensor:
        """
        Load image using cv2 as a PyTorch Tensor.
        :param path: Image path
        :return: An image PyTorch Tensor.
        """
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.transpose(img, axes=[2, 0, 1]))

        return img

    def __len__(self):
        return len(self.sampling_idx)

    def __getitem__(self, item):
        # 1. Using purely transformers Vision Text Dual Encoder.
        # 2. Base classing Vision Text Dual Encoder so that we can override its forward method to pass embeddings.
        # Assuming we have Images / ConNextv2 embeddings.
        sampled_idx = self.sampling_idx[item]

        data_point = self.df.iloc[sampled_idx]
        word, context = data_point[:2]

        text = f'{word}{self.sep}{context}'

        # Tokenizer gives a dictionary of input_ids, token_type_ids, attention_mask.
        text_input = self.fast_tokenizer(text, return_tensors='pt', padding='max_length', truncation=True,
                                         max_length=self.model_max_length)

        text_input = {key: value.squeeze() for key, value in text_input.items()}

        # Future Work: Current Mining Scheme is only going as per the dataset.
        #  We can go for more negative image examples based on the "word" not conditioned on "context".

        img_paths = data_point[2:] if self._config == DatasetConfig.TEST else data_point[2:-1]

        imgs = self.image_processor.preprocess([
            self.load_image(os.path.join(self.img_data_path, img_path))
            for img_path in img_paths
        ], return_tensors='pt')

        gold_example = 0

        if self._config != DatasetConfig.TEST:
            gold_path = data_point[-1]

            for idx, img_path in enumerate(img_paths):
                if img_path == gold_path:
                    gold_example += idx
                    break

        return text_input, imgs, gold_example

    def collate_dataset(self, items: list):
        """
        Function to manually batch the tensors. More specifically the image tensors.
        :param items: List of data points from the __getitem__ method.
        :return: tuple of correctly batched tensors.
        """
        batch_size = len(items)

        text_inputs = {
            key: torch.stack([items[idx][0][key] for idx in range(batch_size)])
            .to(self.device, non_blocking=self.non_blocking)
            for key in items[0][0].keys()
        }

        imgs = torch.cat([items[idx][1]['pixel_values'] for idx in range(batch_size)])\
            .to(self.device, non_blocking=self.non_blocking)

        gold_examples = torch.Tensor([items[idx][2] for idx in range(batch_size)]).type(torch.LongTensor)\
            .to(self.device, non_blocking=self.non_blocking)

        return text_inputs, imgs, gold_examples


def test_dataset():
    # Testing dataset with DataLoader.
    base_path = '' #os.path.join('C:\\', 'Users', 'Vaibhav', 'Downloads', 'semeval-2023-task-1-V-WSD-train-v1')
    base_path= '/Users/sharadc/Documents/uic/semester2/CS521/project/semeval-2023-task-1-V-WSD-train-v1'
    img_path = os.path.join(base_path, 'train_v1', 'train_images_v1')
    wsd_path = os.path.join(base_path, 'train_v1', 'train.data.v1.txt')
    gold_path = os.path.join(base_path, 'train_v1', 'train.gold.v1.txt')

    from transformers import BertTokenizerFast, ConvNextImageProcessor

    img_processor = ConvNextImageProcessor()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = VWSDDataset(img_data_path=img_path, wsd_data_path=wsd_path, gold_data_path=gold_path,
                          config=DatasetConfig.TRAIN, image_processor=img_processor, tokenizer=tokenizer,
                          split=1.0, device=device)

    from torch.utils.data import DataLoader

    num_workers = 2

    train_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=False,
                              collate_fn=dataset.collate_dataset, num_workers=num_workers)

    for idx, (txt, imgs, gold_example) in enumerate(train_loader, start=1):
        print(f'Device {imgs.device.type}')
        print(f'Text shape {txt["input_ids"].shape} Image shape {imgs.shape} gold example shape {gold_example.shape}')
        break


if __name__ == '__main__':
    test_dataset()
