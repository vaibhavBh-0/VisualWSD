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
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import BertTokenizerFast, ConvNextImageProcessor, ConvNextFeatureExtractor
from transformers.optimization import Adafactor, AdafactorSchedule, get_cosine_schedule_with_warmup

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

        # https://github.com/google-research/big_vision/blob/47ac2fd075fcb66cadc0e39bd959c78a6080070d/big_vision/models/proj/image_text/two_towers.py#L33
        embedding_dim = kwargs.get('embedding_dim', 128)
        scale_factor = kwargs.get('scale_factor', 2.6592)
        train_split_ratio = kwargs.get('train_split_ratio', 0.8)
        splitting_seed = kwargs.get('splitting_seed', 42)

        self.epochs = kwargs.get('epochs', 10)
        self.current_epoch = kwargs.get('current_epoch', 1)
        self.lr = kwargs.get('lr', 0.001)

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

        self.optim = Adafactor(self.model.parameters(), warmup_init=True)

        self.optim_lr_scheduler = AdafactorSchedule(optimizer=self.optim, initial_lr=self.lr)

        # CosineAnnealingLR(self.optim, )

        # Cosine Annealing - As per the authors of LiT.
        warm_up_steps = 10000
        total_train_steps = len(self.train_dataloader)

        self.cosine_annealing_lr = get_cosine_schedule_with_warmup(self.optim, num_warmup_steps=warm_up_steps,
                                                                   num_training_steps=total_train_steps)

        self._load_model_state()

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
            processor = ConvNextImageProcessor().from_pretrained(vision_model_path)

            return ConvNextV2Encoder, vision_model_path, processor
        else:
            raise NotImplementedError('Vision Encoder not implemented.')

    def _load_model_state(self, index=-1):
        """
        Load the model state from checkpoints if any.
        :param index: Loads the checkpoint based on created date index. Defaults to -1 to load the latest checkpoint.
        """
        items = os.listdir(self.model_save_path)
        if items:
            latest_item = sorted(items, key=os.path.getctime)[index]
            checkpoint_path = os.path.join(self.model_save_path, latest_item)

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.model.train()

            self.optim.load_state_dict(checkpoint['optim'])
            self.optim_lr_scheduler.load_state_dict(checkpoint['ada_factor_scheduler'])
            self.cosine_annealing_lr.load_state_dict(checkpoint['cosine_annealing_scheduler'])

            self.current_epoch = 1 + checkpoint['epoch']

    def _save_model_state(self, current_epoch: int, **kwargs):
        path = os.path.join(self.model_save_path, f'{current_epoch}.pt')
        torch.save({
            'epoch': current_epoch,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'ada_factor_scheduler': self.optim_lr_scheduler.state_dict(),
            'cosine_annealing_scheduler': self.cosine_annealing_lr.state_dict(),
            **kwargs
        }, f=path)

    def train(self):
        running_loss, running_mrr, running_hit_rate = 0.0, 0.0, 0.0
        IMG_SAMPLES = 10

        for epoch in range(self.current_epoch, self.epochs):
            self.model.train()

            with tqdm(total=len(self.train_dataloader), colour='cyan', leave=True) as bar:
                for idx, (txt, imgs, gold_example) in enumerate(self.train_dataloader, start=1):
                    self.optim.zero_grad()

                    # txts = {key: val.to(self.device, non_blocking=True).repeat_interleave(repeats=IMG_SAMPLES, dim=0)[:2, :]
                    #         for key, val in txt.items()}
                    txts = {
                        key: val.to(self.device, non_blocking=True)#[:2, :]
                        for key, val in txt.items()
                    }

                    img_shape = imgs['pixel_values'].shape[2:]

                    images = imgs['pixel_values'].to(self.device, non_blocking=True).reshape((-1, *img_shape))
                    gold_examples = gold_example.to(self.device, non_blocking=True)\
                        .repeat_interleave(repeats=IMG_SAMPLES, dim=0)

                    out = self.model(text_data=txts, image_data=images, img_samples=IMG_SAMPLES)

                    # TODO: - Compute LiT/CLIP loss.

                    # Optim step.
                    self.optim.step()
                    # TODO: - Develop Metrics - MRR and Hit Rate @ 1.

                    bar.update()
                    bar.set_description(f'Training {epoch}/{self.epochs} - Loss {running_loss / idx:.3f} '
                                        f'MRR {running_mrr / idx:.3f} HR {running_hit_rate / idx :.3f}')

            # LR Scheduler Chaining.
            self.optim_lr_scheduler.step()
            self.cosine_annealing_lr.step()

            extra = {
                'loss': running_loss / len(self.train_dataloader),
                'mrr': running_mrr / len(self.train_dataloader),
                'hit_rate': running_hit_rate / len(self.train_dataloader)
            }

            self._save_model_state(current_epoch=epoch, **extra)

            running_loss, running_mrr, running_hit_rate = 0.0, 0.0, 0.0

            self.model.eval()

            with torch.no_grad():
                with tqdm(total=len(self.val_dataloader), colour='red', leave=False) as bar:
                    for idx, (txt, imgs, gold_example) in enumerate(self.val_dataloader, start=1):
                        txt = {key: val.to(self.device, non_blocking=True) for key, val in txt.items()}
                        imgs = imgs['pixel_values'].to(self.device, non_blocking=True)
                        gold_example = gold_example.to(self.device, non_blocking=True)

                        out = self.model(text_data=txt, image_data=imgs)

                        # TODO - Compute Validation Metrics for validation. Record on the three running metrics variables.

                        bar.update()
                        bar.set_description(f'Validation {epoch}/{self.epochs} - Loss {running_loss / idx:.3f} '
                                            f'MRR {running_mrr / idx:.3f} HR {running_hit_rate / idx :.3f}')

            running_loss, running_mrr, running_hit_rate = 0.0, 0.0, 0.0
