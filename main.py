# main.py
#
# Created by Vaibhav Bhargava on 01-04-2023
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


from argparse import ArgumentParser, Namespace
from trainer import Trainer, TextConfig, VisionConfig


parser = ArgumentParser()

parser.add_argument('--execute', type=int, default=0, choices=[0, 1], help='0: train\n1: evaluate')

parser.add_argument('--train_split_ratio', type=float, default=0.8)
parser.add_argument('--splitting_seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--val_batch_size', type=int, default=16)

parser.add_argument('--base_path', type=str,
                    help='Directory where training/evaluation dataset is present.', required=True)
parser.add_argument('--model_save_path', type=str, help='Directory where model is to be saved.', required=True)
parser.add_argument('--model_log_path', type=str,
                    help='Directory where tensorboard logs are to be saved.', required=True)

parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--scale_factor', type=float, default=2.6592)

parser.add_argument('--text_encoder', type=int, default=1, help='Choose the text encoder.')
parser.add_argument('--vision_encoder', type=int, default=1, help='Choose the vision encoder.')
parser.add_argument('--evaluation_epoch_index', type=int, default=-1, help='Choose the evaluation epoch checkpoint.')


def create_trainer(arguments: Namespace) -> Trainer:
    text_enc_val, vision_enc_val = arguments.text_encoder, arguments.vision_encoder
    text_enc_config = [option for option in TextConfig if option.value == text_enc_val]

    if not text_enc_config:
        raise ValueError(f'Unexpected value for text_encoder {text_enc_val}')

    text_enc_config = text_enc_config[0]

    vision_enc_config = [option for option in VisionConfig if option.value == vision_enc_val]

    if not vision_enc_config:
        raise ValueError(f'Unexpected value for vision_encoder {vision_enc_val}')

    vision_enc_config = vision_enc_config[0]

    kwargs = arguments.__dict__

    return Trainer(text_config=text_enc_config, vision_config=vision_enc_config, **kwargs)


def train(arguments: Namespace):
    trainer = create_trainer(arguments)
    trainer.train()


def evaluate(arguments: Namespace):
    trainer = create_trainer(arguments)
    trainer.evaluate()


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    execution = args.execute

    if execution == 0:
        train(args)
    elif execution == 1:
        evaluate(args)
    else:
        raise ValueError(f'execute flag does not takes {execution} as an input.')
