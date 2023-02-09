import numpy as np
import torch


class MusicDataset(torch.utils.data.Dataset):

    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    @torch.no_grad()
    def __getitem__(self, idx):

        input_ids = self.text_datasets['train'][idx]['input_ids']
        hidden_state = self.model_emb(torch.tensor(input_ids))

        # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
        arr = np.array(hidden_state, dtype=np.float32)

        out_kwargs = {
            'input_ids': np.array(self.text_datasets['train'][idx]['input_ids']),
            'input_mask': np.array(self.text_datasets['train'][idx]['input_mask'])
        }

        return arr, out_kwargs


__all__ = ('MusicDataset', )
