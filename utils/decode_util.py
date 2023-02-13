import os
from typing import List, Union
import numpy as np
from models.commu.preprocessor.encoder import EventSequenceEncoder, TOKEN_OFFSET
from models.commu.preprocessor.utils.container import MidiInfo

from miditoolkit import MidiFile

class SequenceToMidi:
    def __init__(self) -> None:
        self.decoder = EventSequenceEncoder()

    @staticmethod
    def set_output_file_path(idx: int, output_dir: Union[str, os.PathLike]) -> str:
        return "{output_dir}/{idx}.mid".format(idx=idx, output_dir=output_dir)

    def remove_padding(self, generation_result):
        '''
        TODO
        Future Work
        '''
        npy = np.array(generation_result)
        #assert npy.ndim == 1

        eos_idx = np.where(npy == 1)[0] # eos token == 1
        if len(eos_idx) > 0:
            eos_idx = eos_idx[0].item() # note seq의 첫 eos이후에 나온 토큰들은 모두 패딩이 잘못 생성된거로 간주
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
            encoded_meta,
            note_seq
    ) -> MidiFile:
        decoded_midi = self.decoder.decode(
            midi_info=MidiInfo(*encoded_meta, event_seq=note_seq),
        )
        return decoded_midi
        

    def __call__(self, sequences, output_dir, input_ids_mask_ori, seq_len) -> None:
        num_valid_seq = 0
        for idx, (seq, input_mask) in enumerate(zip(sequences, input_ids_mask_ori)):
            len_meta = seq_len - int((input_mask.sum()))
            assert len_meta == 12  # meta와 midi사이에 들어가는 meta eos까지 12(11+1)

            encoded_meta = seq[:len_meta-1] #meta의 eos 토큰 제외 11개만 가져오기
            note_seq = seq[len_meta:]
            note_seq = self.remove_padding(note_seq)

            if self.validate_generated_sequence(note_seq):
                decoded_midi = self.decode_event_sequence(
                    encoded_meta,
                    note_seq
                )
                output_file_path = self.set_output_file_path(idx=idx, output_dir=output_dir)
                decoded_midi.dump(output_file_path)
                num_valid_seq +=1
            else:
                print(f"{idx+1}th sequence is invalid")

        if num_valid_seq == 0 :
            raise ValueError("Validation of generated sequence failed:\n{!r}".format(note_seq))

    def save_tokens(self, input_tokens, output_tokens, output_dir, index):
        out_list = []
        for idx, (in_seq, out_seq) in enumerate(zip(input_tokens, output_tokens)):
            len_meta = 12 #seq_len - int((input_mask.sum()))
            #assert len_meta == 12  # meta와 midi사이에 들어가는 meta eos까지 12(11+1)

            encoded_meta = in_seq[:len_meta-1] #meta의 eos 토큰 제외 11개만 가져오기
            in_note_seq = in_seq[len_meta:]
            in_note_seq = self.remove_padding(in_note_seq)
            out_note_seq = out_seq[len_meta:]
            out_note_seq = self.remove_padding(out_note_seq)
            #output_file_path = self.set_output_file_path(idx=idx, output_dir=output_dir)
            out_list.append(np.concatenate((encoded_meta, in_note_seq, [0], out_note_seq)))
        path = os.path.join(output_dir, str(index), '.npy')
        np.save(path, out_list)