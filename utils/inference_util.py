from typing import List
from pathlib import Path
import numpy as np
from models.commu.preprocessor.encoder import EventSequenceEncoder
from models.commu.preprocessor.utils.container import MidiInfo
import torch

from miditoolkit import MidiFile

def post_process(
    generation_result,
    num_meta: int,
    input_x = None,
):
    '''
    이 함수는 추가로 작업 필요
    '''
    npy = np.array(generation_result)
    eos_idxs = np.where(npy == 1) # eos token == 1
    if len(eos_idxs) > 1:
        eos_idx = eos_idxs[-1]
        return generation_result[:eos_idx+1]
    else:
        print('error in note sequence, no eos token')


def decode_event_sequence(
        generation_result: List[int], # 형식: meta (11) + eos + midi
        num_meta: int
) -> MidiFile:
    encoded_meta = generation_result[: num_meta + 1]
    event_sequence = generation_result[num_meta + 2:]
    decoder = EventSequenceEncoder()
    decoded_midi = decoder.decode(
        midi_info=MidiInfo(*encoded_meta, event_seq=event_sequence),
    )
    return decoded_midi

def set_output_file_path(
    idx, output_dir
):
    return output_dir + "/"+ idx + ".mid"

def execute(model, sequences, meta_info_len: int, output_dir) -> Path:
    for idx, seq in enumerate(sequences):
        logits = model.get_logits(seq.unsqueeze(0))  # bsz, seqlen, vocab
        seq = torch.topk(logits, k=1, dim=-1)

        seq = post_process(seq.squeeze(0), meta_info_len)
        decoded_midi = decode_event_sequence(
            generation_result=seq,
            num_meta=meta_info_len,
        )
        output_file_path = set_output_file_path(idx, output_dir)
        decoded_midi.dump(output_file_path)
