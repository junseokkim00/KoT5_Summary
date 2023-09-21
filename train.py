import os
import argparse
import pandas as pd
import numpy as np

from loguru import logger



# Check where model and the metadata exists

if os.path.exists('./base_hf/config.json') and os.path.exists('./base_hf/pytorch_model.bin'):
    print("READY TO TRAIN")
    pass

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from model import KoT5ConditionalGeneration
from dataset import KoT5SummaryModule

from transformers import T5Tokenizer


parser=argparse.ArgumentParser(description="KoT5 Summarization")

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--train_file", type=str, default='./train_data_text_summarization/data/train/train.json',help='train file')
        parser.add_argument('--test_file', type=str, default='./train_data_text_summarization/data/test/test.json', help='test file')
        parser.add_argument('--batch_size', type=int, default=28, help='batch size')
        parser.add_argument('--checkpoint', type=str, default='checkpoint', help='checkpoint')
        parser.add_argument('--max_len', type=int, default=512, help='max length')
        parser.add_argument('--max_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--lr', type=float, default=3e-5, help='The initial learning rate')
        parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu'], help='choice acclerator')
        parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
        parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='gradient_clipping')

        return parser
    
if __name__ == '__main__':
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KoT5SummaryModule.add_model_specific_args(parser)
    tokenizer = T5Tokenizer.from_pretrained('./vocab/sentencepiece.model')
    args = parser.parse_args()
    logger.info(args)


    summaryModule = KoT5SummaryModule(train_file=args.train_file, test_file=args.test_file,tok=tokenizer, max_len=args.max_len, 
                                      batch_size=args.batch_size, num_workers=args.num_workers)
    summaryModule.setup('fit')

    model = KoT5ConditionalGeneration(args)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath=args.checkpoint, filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                          verbose=True,
                                          save_last=True,
                                          mode='min',
                                          save_top_k=3)

    wandb_logger = WandbLogger(project='KoT5_Summarize')

    trainer = L.Trainer(max_epochs=args.max_epochs,
                        accelerator=args.accelerator,
                        devices=args.num_gpus,
                        gradient_clip_val=args.gradient_clip_val,
                        callbacks=[checkpoint_callback],
                        logger = wandb_logger)
    trainer.fit(model, summaryModule)
    
    