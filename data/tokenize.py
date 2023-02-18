from datasets import Dataset as ArrowDataset
import torch

def helper_tokenize(sentence_lst, num_proc=4):
    raw_datasets = ArrowDataset.from_dict(sentence_lst)

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        attn_mask = []
        length = []
        for i in range(len(group_lst['src'])):
            end_token = 1
            src = group_lst['src'][i]
            trg = group_lst['trg'][i]
            new_length = group_lst['length'][i] + 1 # eos +1

            lst.append(src + [end_token] + trg)
            mask.append([0] * (len(src) + 1))
            attn_mask.append([1] * (len(src) + 1 + len(trg)))
            length.append(new_length)
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        group_lst['attention_mask'] = attn_mask
        group_lst['length'] = length
        return group_lst

    merged_datasets = raw_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=num_proc,
        desc=f"merge and mask",
    )

    return merged_datasets


__all__ = ('helper_tokenize',)