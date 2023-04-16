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
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import BertTokenizerFast, ConvNextImageProcessor
from transformers.optimization import Adafactor, AdafactorSchedule, get_cosine_schedule_with_warmup

from dataset import VWSDDataset, DatasetConfig

from models.lit import LiT
from models.encoders.bert_encoder import BERTEncoder
from models.encoders.convnextv2_encoder import ConvNextV2Encoder
from loss import LiTLoss
from metrics import RuntimeMetrics


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
        self.model_log_path = kwargs['model_log_path']
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.model_log_path, exist_ok=True)

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
        self.is_train = kwargs.get("execute")

        text_enc, text_model_path, tokenizer = self._select_text_encoder(text_config)
        vision_enc, vision_model_path, img_processor = self._select_img_encoder(vision_config)

        train_dataset = VWSDDataset(img_data_path=img_path, wsd_data_path=wsd_path, gold_data_path=gold_path,
                                    config=DatasetConfig.TRAIN, image_processor=img_processor, tokenizer=tokenizer,
                                    split=train_split_ratio, seed=splitting_seed, device=self.device)

        val_dataset = VWSDDataset(img_data_path=img_path, wsd_data_path=wsd_path, gold_data_path=gold_path,
                                  config=DatasetConfig.VAL, image_processor=img_processor, tokenizer=tokenizer,
                                  split=train_split_ratio, seed=splitting_seed, device=self.device)

        num_workers = min(os.cpu_count(), 0)

        self.writer = SummaryWriter(log_dir=self.model_log_path)

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size,
                                           shuffle=True, collate_fn=train_dataset.collate_dataset,
                                           num_workers=num_workers)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.val_batch_size,
                                         shuffle=False, collate_fn=val_dataset.collate_dataset,
                                         num_workers=num_workers)

        self.loss_criterion = LiTLoss()
        #  LiT uses modified AdaFactor.
        #  https://github.com/google-research/big_vision/blob/47ac2fd075fcb66cadc0e39bd959c78a6080070d/big_vision/optax.py#L157
        #  CLIP uses Adam.

        self.model = LiT(embedding_dim=embedding_dim, vision_model_path=vision_model_path,
                         text_model_path=text_model_path,
                         tokenizer_len=len(tokenizer), vision_encoder=vision_enc, text_encoder=text_enc,
                         logit_scale_init_value=scale_factor).to(device=self.device, non_blocking=True)

        self.optim = Adafactor(self.model.parameters(), warmup_init=True)

        self.optim_lr_scheduler = AdafactorSchedule(optimizer=self.optim, initial_lr=self.lr)

        # Cosine Annealing - As per the authors of LiT - Rescaled warmup_steps.
        warm_up_steps = int(10000 / 55000 * len(self.train_dataloader))
        total_train_steps = len(self.train_dataloader)

        self.cosine_annealing_lr = get_cosine_schedule_with_warmup(self.optim, num_warmup_steps=warm_up_steps,
                                                                   num_training_steps=total_train_steps)

        evaluation_epoch_index = kwargs.get("evaluation_epoch_index")
        self._load_model_state(index=evaluation_epoch_index)

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
        param index: Loads the checkpoint based on created date index. Defaults to -1 to load the latest checkpoint.
        """
        items = [os.path.join(self.model_save_path, item) for item in os.listdir(self.model_save_path)]
        if items:
            checkpoint_path = sorted(items, key=os.path.getctime)[index]

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
        running_loss, running_rr, running_hit_rate = 0.0, 0.0, 0.0
        IMG_SAMPLES = 10

        for epoch in range(self.current_epoch, self.epochs + 1):
            self.model.train()

            global_step = (epoch - 1) * len(self.train_dataloader)

            with tqdm(total=len(self.train_dataloader), colour='cyan', leave=True) as bar:
                for idx, (txt, imgs, gold_example) in enumerate(self.train_dataloader, start=1):
                    self.optim.zero_grad()

                    out = self.model(text_data=txt, image_data=imgs, img_samples=IMG_SAMPLES)

                    loss = self.loss_criterion(out, text_to_img_mapping=gold_example)
                    running_loss += loss.item()

                    loss.backward()

                    # Optim step.
                    self.optim.step()

                    batch_wise_rr = RuntimeMetrics.reciprocal_rank_per_batch(logit_scores=out, gold_indices=gold_example
                                                                             , top_k=IMG_SAMPLES).item()
                    running_rr += batch_wise_rr
                    batch_wise_hit_rate = RuntimeMetrics.hit_rate_at1(logit_scores=out, gold_indices=gold_example)\
                        .item()

                    running_hit_rate += batch_wise_hit_rate

                    avg_loss = running_loss / idx
                    mrr = running_rr / (idx * self.train_batch_size)
                    avg_hit_rate = running_hit_rate / (idx * self.train_batch_size)

                    self.writer.add_scalar('Loss/train', avg_loss, global_step=global_step)
                    self.writer.add_scalar('MRR/train', mrr, global_step=global_step)
                    self.writer.add_scalar('HR/train', avg_hit_rate, global_step=global_step)

                    global_step += idx

                    bar.update()
                    bar.set_description(f'Training {epoch}/{self.epochs} - Loss {avg_loss:.3f} MRR {mrr:.3f} '
                                        f'HR {avg_hit_rate :.3f}')

            # LR Scheduler Chaining.
            self.optim_lr_scheduler.step()
            self.cosine_annealing_lr.step()

            extra = {
                'loss': running_loss / len(self.train_dataloader),
                'mrr': running_rr / (self.train_batch_size * len(self.train_dataloader)),
                'hit_rate': running_hit_rate / (self.train_batch_size * len(self.train_dataloader))
            }

            self._save_model_state(current_epoch=epoch, **extra)

            running_loss, running_rr, running_hit_rate = 0.0, 0.0, 0.0

            self.model.eval()

            global_step = (epoch - 1) * len(self.val_dataloader)

            with torch.no_grad():
                with tqdm(total=len(self.val_dataloader), colour='red', leave=False) as bar:
                    for idx, (txt, imgs, gold_example) in enumerate(self.val_dataloader, start=1):
                        out = self.model(text_data=txt, image_data=imgs, img_samples=IMG_SAMPLES)

                        loss = self.loss_criterion(out, text_to_img_mapping=gold_example)
                        running_loss += loss.item()

                        batch_wise_rr = RuntimeMetrics.reciprocal_rank_per_batch(logit_scores=out,
                                                                                 gold_indices=gold_example,
                                                                                 top_k=IMG_SAMPLES).item()
                        running_rr += batch_wise_rr
                        batch_wise_hit_rate = RuntimeMetrics.hit_rate_at1(logit_scores=out,
                                                                          gold_indices=gold_example).item()
                        running_hit_rate += batch_wise_hit_rate

                        avg_loss = running_loss / idx
                        mrr = running_rr / (idx * self.val_batch_size)
                        avg_hit_rate = running_hit_rate / (idx * self.val_batch_size)

                        self.writer.add_scalar('Loss/val', avg_loss, global_step=global_step)
                        self.writer.add_scalar('MRR/val', mrr, global_step=global_step)
                        self.writer.add_scalar('HR/val', avg_hit_rate, global_step=global_step)

                        global_step += idx

                        bar.update()
                        bar.set_description(f'Validation {epoch}/{self.epochs} - Loss {avg_loss:.3f} '
                                            f'MRR {mrr:.3f} HR {avg_hit_rate :.3f}')

            running_loss, running_rr, running_hit_rate = 0.0, 0.0, 0.0

    def evaluate(self):
        self.model.eval()
        running_loss, running_rr, running_hit_rate = 0.0, 0.0, 0.0
        IMG_SAMPLES = 10
        with torch.no_grad():
            with tqdm(total=len(self.val_dataloader), colour='red', leave=False) as bar:
                for idx, (txt, imgs, gold_example) in enumerate(self.val_dataloader, start=1):
                    out = self.model(text_data=txt, image_data=imgs, img_samples=IMG_SAMPLES)

                    loss = self.loss_criterion(out, text_to_img_mapping=gold_example)
                    running_loss += loss.item()

                    batch_wise_rr = RuntimeMetrics.reciprocal_rank_per_batch(logit_scores=out,
                                                                             gold_indices=gold_example,
                                                                             top_k=IMG_SAMPLES).item()
                    running_rr += batch_wise_rr
                    batch_wise_hit_rate = RuntimeMetrics.hit_rate_at1(logit_scores=out,
                                                                      gold_indices=gold_example).item()
                    running_hit_rate += batch_wise_hit_rate

                    avg_loss = running_loss / idx
                    mrr = running_rr / (idx * self.val_batch_size)
                    avg_hit_rate = running_hit_rate / (idx * self.val_batch_size)

                    self.writer.add_scalar('Loss/val', avg_loss, global_step=idx)
                    self.writer.add_scalar('MRR/val', mrr, global_step=idx)
                    self.writer.add_scalar('HR/val', avg_hit_rate, global_step=idx)

                    bar.update()
                    bar.set_description(f'Validation Loss {avg_loss:.3f} '
                                        f'MRR {mrr:.3f} HR {avg_hit_rate :.3f}')

