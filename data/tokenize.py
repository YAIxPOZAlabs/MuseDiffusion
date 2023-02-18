from datasets import Dataset as ArrowDataset
import torch


def _collate_batch_helper(examples, pad_token_id, max_length):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
    return result


def helper_tokenize(sentence_lst, seq_len, num_proc=4):
    raw_datasets = ArrowDataset.from_dict(sentence_lst)

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        attn_mask = []
        for i in range(len(group_lst['src'])):
            end_token = 1
            src = group_lst['src'][i]
            trg = group_lst['trg'][i]

            lst.append(src + [end_token] + trg)
            mask.append([0] * (len(src) + 1))
            attn_mask.append([1] * (len(src) + 1 + len(trg)))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        group_lst['attention_mask'] = attn_mask
        return group_lst

    merged_datasets = raw_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=num_proc,
        desc=f"merge and mask",
    )

    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], 0, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        group_lst['attention_mask'] = _collate_batch_helper(group_lst['attention_mask'], 0, max_length)
        return group_lst

    padded_datasets = merged_datasets.map(
        pad_function,
        batched=True,
        num_proc=num_proc,
        desc=f"padding",
    )

    return padded_datasets


__all__ = ('helper_tokenize',)
