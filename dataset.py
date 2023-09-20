import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

class KoT5SummaryDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_json(file)
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index
    
    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __getitem__(self, idx):
        df = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(df['passage'])
        input_ids = self.add_padding_data(input_ids)
        label_ids = self.tokenizer.encode(df['summary'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.bos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)
                }
    def __len__(self):
        return self.len