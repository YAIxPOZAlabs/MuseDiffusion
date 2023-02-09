from datasets import Dataset as ArrowDataset, DatasetDict
import torch


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    # mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = 0
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        # mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def helper_tokenize(sentence_lst, seq_len):
    raw_datasets = ArrowDataset.from_dict(sentence_lst)

    def tokenize_function(examples):
        result_dict = {'input_id_x': examples['src'], 'input_id_y': examples['trg']}
        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = 1
            src = group_lst['input_id_x'][i]
            trg = group_lst['input_id_y'][i]
            # while len(src) + len(trg) > seq_len - 2:
            #     if len(src)>len(trg):
            #         src.pop()
            #     elif len(src)<len(trg):
            #         trg.pop()
            #     else:
            #         src.pop()
            #         trg.pop()
            # src.append(end_token)
            # # trg.append(end_token)

            lst.append(src + [end_token] + trg)
            mask.append([0] * (len(src) + 1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )

    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], 0, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    raw_datasets = DatasetDict()
    raw_datasets['train'] = lm_datasets
    return raw_datasets


__all__ = ('helper_tokenize',)
