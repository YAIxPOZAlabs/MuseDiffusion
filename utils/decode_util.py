import os
import numpy as np
from miditoolkit import MidiFile

from models.commu.preprocessor.utils.container import MidiInfo
from models.commu.preprocessor.encoder import EventSequenceEncoder, TOKEN_OFFSET
# from models.commu.midi_generator.midi_inferrer import InferenceTask  # TODO


class SequenceToMidi:

    decoder: "EventSequenceEncoder"

    def __init__(self):
        self.decoder = EventSequenceEncoder()

    # Shorten set_output_file_path method
    @staticmethod
    def set_output_file_path(*, output_dir, original_index, batch_index, index):
        return "{output_dir}/{original_index}_batch{batch_index}_{index}.midi".format(
            output_dir=output_dir, original_index=original_index,
            batch_index=batch_index, index=index
        )

    @staticmethod
    def remove_padding(generation_result):
        """
        TODO
        Future Work
        """
        npy = np.array(generation_result)
        assert npy.ndim == 1

        eos_idx = np.where(npy == 1)[0]  # eos token == 1
        if len(eos_idx) > 0:
            eos_idx = eos_idx[0].item()  # note seq 의 첫 eos 이후에 나온 토큰은 모두 패딩이 잘못 생성된 거로 간주
            return generation_result[:eos_idx + 1]
        else:
            raise ValueError('Error in note sequence, no eos token')

    # # Get method from pre-declared class
    # validate_generated_sequence = InferenceTask.validate_generated_sequence  # TODO

    @staticmethod
    def validate_generated_sequence(seq):
        num_note = 0  # NOQA (코드 중복)
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
    ) -> "MidiFile":
        decoded_midi = self.decoder.decode(
            midi_info=MidiInfo(*encoded_meta, event_seq=note_seq),
        )
        return decoded_midi

    def __call__(
            self, sequences, output_dir, input_ids_mask_ori, seq_len, batch_index, batch_size
    ) -> "None":

        invalid_idxes = set()
        num_files_before_batch = batch_index * batch_size
        assert len(sequences) == batch_size, "Length of sequence differs from batch size"

        for idx, (seq, input_mask) in enumerate(zip(sequences, input_ids_mask_ori)):
            try:
                len_meta = seq_len - int((input_mask.sum()))
                assert len_meta == 12  # meta와 midi사이에 들어가는 meta eos까지 12(11+1)

                encoded_meta = seq[:len_meta-1]  # meta의 eos 토큰 제외 11개만 가져오기
                note_seq = seq[len_meta:]
                note_seq = self.remove_padding(note_seq)

                if self.validate_generated_sequence(note_seq):
                    decoded_midi = self.decode_event_sequence(
                        encoded_meta,
                        note_seq
                    )
                    output_file_path = self.set_output_file_path(
                        original_index=num_files_before_batch + idx,
                        batch_index=batch_index,
                        index=idx,
                        output_dir=output_dir
                    )
                    decoded_midi.dump(output_file_path)
                else:
                    print(f"Invalid sequence: Batch {batch_index} Index {idx} "
                          f"(Original Index: {num_files_before_batch + idx}).")
                    invalid_idxes.add(idx)
            except Exception as exc:
                print(f"Error: {type(exc)} occurred while generating midi of Batch {batch_index} Index {idx}\n"
                      f" (Original Index: {num_files_before_batch + idx}).")
                raise

        valid_count = len(sequences) - len(invalid_idxes)

        self.print_summary(
            batch_index=batch_index,
            batch_size=batch_size,
            valid_count=valid_count,
            invalid_idxes=invalid_idxes,
            output_dir=output_dir
        )

    @staticmethod
    def print_summary(batch_index, batch_size, valid_count, invalid_idxes, output_dir):
        invalid_idxes = sorted(invalid_idxes)
        print()
        print(f"{f' Summary of Batch {batch_index} ':=^40}")
        print(f"* Original index: from {batch_index * batch_size} to {(batch_index + 1) * batch_size - 1}")
        print(f"* {valid_count} valid sequences are converted to midi into path:"
              f"    {os.path.abspath(output_dir)}")
        print(f"* {len(invalid_idxes)} sequences are invalid.")
        if invalid_idxes:
            print(f"* Index (in batch {batch_index}) of invalid sequence:")
            print(f"    {invalid_idxes}")
        print("=" * 40)
        print()

    @classmethod
    def save_tokens(cls, input_tokens, output_tokens, output_dir, batch_index):
        out_list = []
        for idx, (in_seq, out_seq) in enumerate(zip(input_tokens, output_tokens)):
            len_meta = 12  # seq_len - int((input_mask.sum()))
            # assert len_meta == 12  # meta와 midi사이에 들어가는 meta eos까지 12(11+1)

            encoded_meta = in_seq[:len_meta-1]  # meta의 eos 토큰 제외 11개만 가져오기
            in_note_seq = in_seq[len_meta:]
            in_note_seq = cls.remove_padding(in_note_seq)
            out_note_seq = out_seq[len_meta:]
            out_note_seq = cls.remove_padding(out_note_seq)
            # output_file_path = self.set_output_file_path(idx=idx, output_dir=output_dir)
            out_list.append(np.concatenate((encoded_meta, in_note_seq, [0], out_note_seq)))
        path = os.path.join(output_dir, str(batch_index), '.npy')
        np.save(path, out_list)
