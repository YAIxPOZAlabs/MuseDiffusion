import numpy as np
import torch


class EmbeddingWrappedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, model_emb):
        super().__init__()
        self.dataset = dataset
        self.length = len(dataset)
        self.model_emb = model_emb.eval().requires_grad_(False)

    def __len__(self):
        return self.length

    @torch.no_grad()
    def __getitem__(self, idx):

        item = self.dataset[idx]
        input_ids, input_mask = item['input_ids'], item['input_mask']

        # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
        hidden_state = self.model_emb(torch.tensor(input_ids))
        arr = np.array(hidden_state, dtype=np.float32)

        out_kwargs = {'input_ids': np.array(input_ids), 'input_mask': np.array(input_mask)}

        return arr, out_kwargs


__all__ = ('EmbeddingWrappedDataset',)
