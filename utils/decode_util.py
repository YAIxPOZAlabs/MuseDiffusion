import os
from typing import List, Union
import numpy as np
from models.commu.preprocessor.encoder import EventSequenceEncoder, TOKEN_OFFSET
from models.commu.preprocessor.utils.container import MidiInfo

from miditoolkit import MidiFile


class SeqeunceToMidi:
    def __init__(self) -> None:
        self.decoder = EventSequenceEncoder()

    @staticmethod
    def set_output_file_path(idx: int, output_dir: Union[str, os.PathLike]) -> str:
        return "{output_dir}/{idx}.mid".format(idx=idx, output_dir=output_dir)

    def post_process(self, generation_result):  # TODO
        '''
        TODO
        Future Work
        '''
        npy = np.array(generation_result)
        eos_idxs = np.where(npy == 1)[0] # eos token == 1
        if len(eos_idxs) >= 1:
            eos_idx = eos_idxs[0].item()
            return generation_result[:eos_idx + 1]
        else:
            raise Exception('Error in note sequence, no eos token')

    @staticmethod
    def validate_generated_sequence(seq: List[int]) -> bool:
        num_note = 0
        for idx, token in enumerate(seq):
            if idx + 2 > len(seq) - 1:
                break
            if token in range(TOKEN_OFFSET.NOTE_VELOCITY.value, TOKEN_OFFSET.CHORD_START.value):
                if (
                    seq[idx - 1] in range(TOKEN_OFFSET.POSITION.value, TOKEN_OFFSET.BPM.value)
                    and seq[idx + 1]
                    in range(TOKEN_OFFSET.PITCH.value, TOKEN_OFFSET.NOTE_VELOCITY.value)
                    and seq[idx + 2]
                    in range(TOKEN_OFFSET.NOTE_DURATION.value, TOKEN_OFFSET.POSITION.value)
                ):
                    num_note += 1
        return num_note > 0

    def decode_event_sequence(
            self,
            generation_result: List[int],  # 형식: meta (11) + eos + midi
            num_meta: int
    ) -> MidiFile:
        encoded_meta = generation_result[: num_meta + 1]
        event_sequence = generation_result[num_meta + 2:]
        decoded_midi = self.decoder.decode(
            midi_info=MidiInfo(*encoded_meta, event_seq=event_sequence),
        )
        return decoded_midi

    def __call__(self, sequences, output_dir, input_ids_mask_ori, seq_len) -> None:
        for idx, (seq, input_mask) in enumerate(zip(sequences, input_ids_mask_ori)):
            len_meta = seq_len - int((input_mask.sum()))
            assert len_meta == 12  # meta와 midi사이에 들어가는 meta eos까지 12(11+1)
            note_seq = seq[len_meta:]
            note_seq = self.post_process(note_seq)
            if self.validate_generated_sequence(note_seq):
                decoded_midi = self.decode_event_sequence(
                    generation_result=seq,
                    num_meta=len_meta,
                )
                output_file_path = self.set_output_file_path(idx=idx, output_dir=output_dir)
                decoded_midi.dump(output_file_path)
            else:
                raise ValueError("Validation of generated sequence failed:\n{!r}".format(note_seq))
