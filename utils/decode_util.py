import os
import numpy as np
from miditoolkit import MidiFile

from models.commu.preprocessor.utils.container import MidiInfo
from models.commu.preprocessor.encoder import EventSequenceEncoder
from models.commu.midi_generator.midi_inferrer import InferenceTask

import contextlib
import io


class SequenceToMidi:

    output_file_format = "{output_dir}/{original_index}_batch{batch_index}_{index}.midi"

    def __init__(self):
        self.decoder = EventSequenceEncoder()

    decoder: "EventSequenceEncoder"

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
            return None
            # raise ValueError('Error in note sequence, no eos token')

    # Get method from pre-declared class
    validate_generated_sequence = InferenceTask.validate_generated_sequence

    def decode_event_sequence(
            self,
            encoded_meta,
            note_seq
    ) -> "MidiFile":
        decoded_midi = self.decoder.decode(
            midi_info=MidiInfo(*encoded_meta, event_seq=note_seq),
        )
        return decoded_midi

    def _decode_single(self, seq, input_mask):  # Output: Midi, Errcode
        len_meta = len(seq) - int((input_mask.sum()))
        assert len_meta == 12

        encoded_meta = seq[:len_meta - 1]  # meta의 eos 토큰 제외 11개만 가져오기
        note_seq = seq[len_meta:]
        note_seq = self.remove_padding(note_seq)

        if note_seq is not None:
            if self.validate_generated_sequence(note_seq):
                decoded_midi = self.decode_event_sequence(encoded_meta, note_seq)
                return decoded_midi, 0
            else:
                return None, 2
        else:
            return None, 1

    _errmsg = {
        1: "NO EOS TOKEN",
        2: "VALIDATION OF SEQUENCE FAILED"
    }

    def decode_single(self, seq, input_mask, output_file_path):
        decoded_midi, errcode = self._decode_single(seq, input_mask)
        if errcode == 0:
            decoded_midi.dump(output_file_path)
        return errcode

    def decode_multi_verbose(
            self, sequences, output_dir, input_ids_mask_ori, batch_index, batch_size
    ) -> "None":

        invalid_idxes = set()
        num_files_before_batch = batch_index * batch_size
        assert len(sequences) == batch_size, "Length of sequence differs from batch size"

        for idx, (seq, input_mask) in enumerate(zip(sequences, input_ids_mask_ori)):
            logger = io.StringIO()
            try:
                with contextlib.redirect_stdout(logger):
                    decoded_midi, errcode = self._decode_single(seq, input_mask)
            except Exception as exc:
                print(logger.getvalue())
                print(f"<Error> {exc.__class__.__qualname__}: {exc} \n"
                      f"  occurred while generating midi of Batch {batch_index} Index {idx} "
                      f"(Original: {num_files_before_batch + idx}).")
                raise
            except BaseException:
                print(logger.getvalue())
                raise
            else:
                log = logger.getvalue()
                if errcode == 0:  # Validation Succeeded
                    if log:
                        print(f"<Warning> Batch {batch_index} Index {idx} "
                              f"(Original: {num_files_before_batch + idx})"
                              f" - {' '.join(log.splitlines())}")
                    output_file_path = self.output_file_format.format(
                        original_index=num_files_before_batch + idx,
                        batch_index=batch_index,
                        index=idx,
                        output_dir=output_dir
                    )
                    decoded_midi.dump(output_file_path)
                else:
                    print(f"<Warning> Batch {batch_index} Index {idx} "
                          f"(Original: {num_files_before_batch + idx}) "
                          f"- {self._errmsg[errcode]}")
                    if log:
                        print(log)
                    invalid_idxes.add(idx)

        valid_count = len(sequences) - len(invalid_idxes)

        self.print_summary(
            batch_index=batch_index,
            batch_size=batch_size,
            valid_count=valid_count,
            invalid_idxes=invalid_idxes,
            output_dir=output_dir
        )

    __call__ = decode_multi_verbose

    @staticmethod
    def print_summary(batch_index, batch_size, valid_count, invalid_idxes, output_dir):
        invalid_idxes = sorted(invalid_idxes)
        log = (
            "\n"
            f"{f' Summary of Batch {batch_index} ':=^40}\n"
            f" * Original index: from {batch_index * batch_size} to {(batch_index + 1) * batch_size - 1}\n"
            f" * {valid_count} valid sequences are converted to midi into path:\n"
            f"     {os.path.abspath(output_dir)}\n"
            f" * {len(invalid_idxes)} sequences are invalid.\n"
        )
        if invalid_idxes:
            log += (
                f" * Index (in batch {batch_index}) of invalid sequence:\n"
                f"    {invalid_idxes}\n"
            )
        log += ("=" * 40) + "\n"
        print(log)

    @classmethod
    def save_tokens(cls, input_tokens, output_tokens, output_dir, batch_index):
        out_list = []
        for (in_seq, out_seq) in zip(input_tokens, output_tokens):
            len_meta = 12
            # assert len_meta == 12  # meta와 midi사이에 들어가는 meta eos까지 12(11+1)

            encoded_meta = in_seq[:len_meta-1]  # meta의 eos 토큰 제외 11개만 가져오기
            in_note_seq = in_seq[len_meta:]
            in_note_seq = cls.remove_padding(in_note_seq)
            out_note_seq = out_seq[len_meta:]
            out_note_seq = cls.remove_padding(out_note_seq)
            # output_file_path = self.set_output_file_path(idx=idx, output_dir=output_dir)
            out_list.append(np.concatenate((encoded_meta, in_note_seq, [0], out_note_seq)))
        path = os.path.join(output_dir, f'token_batch{batch_index}.npy')
        np.save(path, out_list)
