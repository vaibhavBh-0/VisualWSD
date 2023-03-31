# trainer.py
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

import os
from enum import Enum
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast, ConvNextImageProcessor

from dataset import VWSDDataset, DatasetConfig

from models.lit import LiT
from models.encoders.bert_encoder import BERTEncoder
from models.encoders.convnextv2_encoder import ConvNextV2Encoder


class TextConfig(Enum):
    BERT_ENCODER = 1


class VisionConfig(Enum):
    CONV_NEXT_V2 = 1


class Trainer:
    def __init__(self, text_config: TextConfig, vision_config: VisionConfig, **kwargs):
        base_path = kwargs['base_path']
        img_path = os.path.join(base_path, 'train_v1', 'train_images_v1')
        wsd_path = os.path.join(base_path, 'train_v1', 'train.data.v1.txt')
        gold_path = os.path.join(base_path, 'train_v1', 'train.gold.v1.txt')

        self.model_save_path = kwargs['model_save_path']
        os.makedirs(self.model_save_path, exist_ok=True)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        embedding_dim = kwargs.get('embedding_dim', 512)
        scale_factor = kwargs.get('scale_factor', 2.6592)
        train_split_ratio = kwargs.get('train_split_ratio', 0.8)
        splitting_seed = kwargs.get('splitting_seed', 42)
        self.epochs = kwargs.get('epochs', 10)
        self.current_epoch = kwargs.get('current_epoch', 1)

        self.train_batch_size = kwargs.get('train_batch_size', 5)
        self.val_batch_size = kwargs.get('val_batch_size', 5)

        text_enc, text_model_path, tokenizer = self._select_text_encoder(text_config)
        vision_enc, vision_model_path, img_processor = self._select_img_encoder(vision_config)

        train_dataset = VWSDDataset(img_data_path=img_path, wsd_data_path=wsd_path, gold_data_path=gold_path,
                                    config=DatasetConfig.TRAIN, image_processor=img_processor, tokenizer=tokenizer,
                                    split=train_split_ratio, seed=splitting_seed)

        val_dataset = VWSDDataset(img_data_path=img_path, wsd_data_path=wsd_path, gold_data_path=gold_path,
                                  config=DatasetConfig.VAL, image_processor=img_processor, tokenizer=tokenizer,
                                  split=train_split_ratio, seed=splitting_seed)

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.val_batch_size, shuffle=False)

        # TODO: Choose optimizer for LiT/CLIP - training.
        #  LiT uses modified AdaFactor. - https://github.com/google-research/big_vision/blob/47ac2fd075fcb66cadc0e39bd959c78a6080070d/big_vision/optax.py#L157
        #  CLIP uses Adam.

        self.model = LiT(embedding_dim=embedding_dim, vision_model_path=vision_model_path,
                         text_model_path=text_model_path,
                         tokenizer_len=len(tokenizer), vision_encoder=vision_enc, text_encoder=text_enc,
                         logit_scale_init_value=scale_factor).to(device=self.device, non_blocking=True)

        self._load_latest_weights()

    @staticmethod
    def _select_text_encoder(text_config: TextConfig):
        if text_config == TextConfig.BERT_ENCODER:
            text_model_path = "bert-base-uncased"
            tokenizer = BertTokenizerFast.from_pretrained(text_model_path)

            return BERTEncoder, text_model_path, tokenizer
        else:
            raise NotImplementedError('Text Encoder not implemented.')

    @staticmethod
    def _select_img_encoder(vision_config: VisionConfig):
        if vision_config == VisionConfig.CONV_NEXT_V2:
            vision_model_path = "facebook/convnextv2-tiny-1k-224"
            processor = ConvNextImageProcessor()

            return ConvNextV2Encoder, vision_model_path, processor
        else:
            raise NotImplementedError('Vision Encoder not implemented.')

    def _load_latest_weights(self):
        items = os.listdir(self.model_save_path)
        if items:
            self.current_epoch = 1 + len(items)
            latest_weights = sorted(items, key=os.path.getctime, reverse=True)[0]
            weight_path = os.path.join(self.model_save_path, latest_weights)
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device), strict=True)
            self.model.train()

    def train(self):
        running_loss, running_mrr, running_hit_rate = 0.0, 0.0, 0.0

        for epoch in range(self.current_epoch, self.epochs):
            with tqdm(total=len(self.train_dataloader), desc=f'Training {epoch}/{self.epochs}', colour='cyan') as bar:
                for idx, (txt, imgs, gold_example) in enumerate(self.train_dataloader, start=1):
                    # TODO: - Place tensors to devices.
                    # TODO: - Model takes text_data as a dict. image_data as a tensor.

                    # TODO: - Compute LiT/CLIP loss.
                    # TODO: - Develop Metrics - MRR and Hit Rate @ 1.

                    bar.update()

            running_loss, running_mrr, running_hit_rate = 0.0, 0.0, 0.0
