import os
import io
import contextlib
import numpy as np
from miditoolkit import MidiFile

try:
    np.int
except AttributeError:
    np.int = int  # for backward compatibility

try:
    from commu.preprocessor.encoder import EventSequenceEncoder, MetaEncoder
    from commu.preprocessor.encoder.event_tokens import base_event, TOKEN_OFFSET
    from commu.preprocessor.utils.container import MidiInfo, MidiMeta
except ModuleNotFoundError as e:
    raise ImportError("In Sampling task, ComMU module is required.") from e


class MetaToSequence:

    def __init__(self):
        self.meta_encoder = MetaEncoder()
        self.chord_map = {
            i[0].upper() + i[1:]: j for j, i in
            enumerate([k[6:] for k in base_event if k.startswith("Chord_")], start=TOKEN_OFFSET.CHORD_START.value)
        }

    def encode_chord(self, chord_progression):
        encoded_chord = []
        chord_progression = chord_progression
        assert len(chord_progression) % 8 == 0

        for idx in range(0, len(chord_progression), 8):
            encoded_chord.append(432)
            chord_token = self.chord_map[chord_progression[idx]]
            encoded_chord.append(chord_token)
            recent_chord = chord_progression[idx]
            for i in range(1, 8):
                if recent_chord != chord_progression[idx + i]:
                    encoded_chord.append(432 + i * 16)
                    recent_chord = chord_progression[idx + i]
        return encoded_chord

    def encode_meta(self, midi_meta):
        return self.meta_encoder.encode(midi_meta)

    def execute(self, input_data: dict) -> "list[int]":
        midi_meta = MidiMeta(**input_data)
        chord_progression = input_data["chord_progression"].split("-")
        return self.encode_meta(midi_meta) + self.encode_chord(chord_progression)

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)


class SequenceToMidiError(Exception):
    pass


class SequenceToMidi:

    _decoder: EventSequenceEncoder

    @property
    def decoder(self) -> EventSequenceEncoder:  # Lazy getter (since decoding is repeated process)
        try:
            decoder = SequenceToMidi._decoder
        except AttributeError:
            decoder = SequenceToMidi._decoder = EventSequenceEncoder()
        return decoder

    @staticmethod
    def remove_padding(generation_result):
        npy = np.array(generation_result)
        assert npy.ndim == 1

        eos_idx = np.where(npy == 1)[0]  # eos token == 1
        if len(eos_idx) > 0:
            eos_idx = eos_idx[0].item()  # note seq 의 첫 eos 이후에 나온 토큰은 모두 패딩이 잘못 생성된 거로 간주
            return generation_result[:eos_idx + 1]
        else:
            raise SequenceToMidiError("NO EOS TOKEN")

    @staticmethod
    def restore_chord(seq, meta):
        """
        decode 시 remove padding 후에
        encoded meta 및 zero padding 없어진 seq input 으로 사용
        """
        new_meta = meta[:11]
        chord_info = meta[11:]
        bar_idx = np.where(seq == 2)[0]
        if len(bar_idx) == len(np.where(chord_info == 432)[0]):
            new_seq = np.concatenate((seq[:bar_idx[0] + 1], chord_info[:2]), axis=0)
            bar_count = 0
            last_idx = bar_idx[0]
        elif len(bar_idx) == len(np.where(chord_info == 432)[0]) + 1:
            new_seq = np.concatenate((seq[:bar_idx[1] + 1], chord_info[:2]), axis=0)
            bar_count = 1
            last_idx = bar_idx[1]
        elif len(bar_idx) < len(np.where(chord_info == 432)[0]):
            diff = len(np.where(chord_info == 432)[0]) - len(bar_idx)
            for _ in range(diff):
                seq = np.insert(seq, -1, 2)
            bar_idx = np.where(seq == 2)[0]
            new_seq = np.concatenate((seq[:bar_idx[0] + 1], chord_info[:2]), axis=0)
            bar_count = 0
            last_idx = bar_idx[0]
        else:
            raise SequenceToMidiError("RESTORE_CHORD FROM META FAILED")

        for i in range(2, len(chord_info), 2):
            if chord_info[i] == 432:
                new_seq = np.concatenate(
                    (new_seq, seq[last_idx + 1:bar_idx[bar_count + 1] + 1], chord_info[i:i + 2]),
                    axis=0
                )
                bar_count += 1
                last_idx = bar_idx[bar_count]

            else:
                candidate = np.where(np.logical_and(432 <= seq, seq < chord_info[i]))[0]
                if bar_count != len(bar_idx) - 1:
                    can_idx = np.where(
                        np.logical_and(bar_idx[bar_count] < candidate, candidate < bar_idx[bar_count + 1])
                    )[0]
                else:
                    can_idx = np.where(bar_idx[bar_count] < candidate)[0]

                if len(can_idx) == 0:
                    new_seq = np.concatenate(
                        (new_seq, chord_info[i:i + 2]), axis=0
                    )
                else:
                    new_seq = np.concatenate(
                        (new_seq, seq[last_idx + 1:candidate[can_idx[-1]] + 4], chord_info[i:i + 2]),
                        axis=0
                    )
                    last_idx = candidate[can_idx[-1]] + 3  # 무조건 note 일 것으로 예상됨

        new_seq = np.concatenate((new_seq, seq[last_idx + 1:]), axis=0)
        return new_seq, new_meta

    @staticmethod
    def validate_generated_sequence(seq):
        for idx, token in enumerate(seq):
            if idx + 2 > len(seq) - 1:
                break
            elif (
                    token in range(TOKEN_OFFSET.NOTE_VELOCITY.value, TOKEN_OFFSET.CHORD_START.value)
                    and seq[idx - 1] in range(TOKEN_OFFSET.POSITION.value, TOKEN_OFFSET.BPM.value)
                    and seq[idx + 1] in range(TOKEN_OFFSET.PITCH.value, TOKEN_OFFSET.NOTE_VELOCITY.value)
                    and seq[idx + 2] in range(TOKEN_OFFSET.NOTE_DURATION.value, TOKEN_OFFSET.POSITION.value)
            ):
                return
        raise SequenceToMidiError("VALIDATION OF SEQUENCE FAILED")

    def decode_event_sequence(
            self,
            encoded_meta,
            note_seq
    ) -> "MidiFile":
        decoded_midi = self.decoder.decode(
            midi_info=MidiInfo(*encoded_meta, event_seq=note_seq),
        )
        return decoded_midi

    def decode(self, seq, input_mask, output_file_path=None):
        len_meta = len(seq) - int((input_mask.sum()))
        encoded_meta = seq[:len_meta - 1]  # meta 에서 eos 토큰 제외 11개만 사용
        note_seq = seq[len_meta:]
        note_seq = self.remove_padding(note_seq)
        note_seq, encoded_meta = self.restore_chord(note_seq, encoded_meta)
        self.validate_generated_sequence(note_seq)
        decoded_midi = self.decode_event_sequence(encoded_meta, note_seq)
        if output_file_path:
            decoded_midi.dump(output_file_path)
        return decoded_midi
    
    def split_meta_midi(self, seq, input_mask):
        len_meta = len(seq) - int((input_mask.sum()))
        encoded_meta = seq[:len_meta - 1]  # meta 에서 eos 토큰 제외 11개만 사용
        note_seq = seq[len_meta:]
        note_seq = self.remove_padding(note_seq)
        note_seq, encoded_meta = self.restore_chord(note_seq, encoded_meta)
        return encoded_meta, note_seq

    def __call__(self, *args, **kwargs):
        return self.decode(*args, **kwargs)


def meta_to_batch(midi_meta_dict, batch_size, seq_len):
    import torch
    encoded_meta = MetaToSequence().execute(midi_meta_dict)
    encoded_meta = torch.tensor(encoded_meta)
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.int)
    input_ids[:, :len(encoded_meta)] = encoded_meta
    input_mask = torch.ones(batch_size, seq_len, dtype=torch.int)
    input_mask[:, :len(encoded_meta) + 1] = 0
    batch = {'input_ids': input_ids, 'input_mask': input_mask}
    return batch


def batch_decode_seq2seq(
        sequences,
        input_ids_mask_ori,
        batch_index,
        previous_count,
        output_dir,
        output_file_format="{original_index}_batch{batch_index}_{index}.midi"
):
    decoder = SequenceToMidi()
    invalid_idxes = set()

    for index, (seq, input_mask) in enumerate(zip(sequences, input_ids_mask_ori)):
        original_index = previous_count + index
        logger = io.StringIO()
        try:
            # we can't handle OOV error - since OOV is only logged into stdout
            # so we need stdout redirection
            with contextlib.redirect_stdout(logger):
                decoded_midi = decoder(seq, input_mask)
        except SequenceToMidiError as exc:  # exception that indicates decoding failure
            log = logger.getvalue()
            print(f"<Warning> Batch {batch_index} Index {index} "
                  f"(Original: {original_index}) "
                  f"- Generation Failure: {exc}")
            if log:
                print(log)
            invalid_idxes.add(index)
        except Exception as exc:  # normal exceptions - assure redirected stdout logs and print extra information
            print(logger.getvalue())
            print(f"<Error> {exc.__class__.__qualname__}: {exc} \n"
                  f"  occurred while generating midi of Batch {batch_index} Index {index} "
                  f"(Original: {original_index}).")
            raise
        except BaseException:  # special exceptions (KeyboardInterrupt, SystemExit...) - assure redirected stdout logs
            print(logger.getvalue())
            raise
        else:
            log = logger.getvalue()  # to check OOV
            if log:
                print(f"<Warning> Batch {batch_index} Index {index} "
                      f"(Original: {original_index}) "
                      f"- {' '.join(log.splitlines())}")
            output_file_path = output_file_format.format(
                original_index=original_index,
                batch_index=batch_index,
                index=index
            )
            decoded_midi.dump(os.path.join(output_dir, output_file_path))

    valid_count = len(sequences) - len(invalid_idxes)
    invalid_idxes = sorted(invalid_idxes)
    log = (
        "\n"
        f"{f' Summary of Batch {batch_index} ':=^60}\n"
        f" * Original index: from {previous_count} to {previous_count + len(sequences)}\n"
        f" * {valid_count} valid sequences are converted to midi into path:\n"
        f"     {os.path.abspath(output_dir)}\n"
        f" * {len(invalid_idxes)} sequences are invalid.\n"
    )
    if invalid_idxes:
        log += (
            f" * Index (in batch {batch_index}) of invalid sequence:\n"
            f"    {invalid_idxes}\n"
        )
    log += ("=" * 60) + "\n"
    print(log)

    return valid_count


def batch_decode_generate(
        sequences,
        input_ids_mask_ori,
        batch_index,
        previous_count,
        output_dir,
        output_file_format="generated_{valid_index}.midi"
):
    decoder = SequenceToMidi()
    valid_index = previous_count

    for seq, input_mask in zip(sequences, input_ids_mask_ori):
        logger = io.StringIO()
        try:
            # we can't handle OOV error - since OOV is only logged into stdout
            # so we need stdout redirection
            with contextlib.redirect_stdout(logger):
                decoded_midi = decoder(seq, input_mask)
        except SequenceToMidiError:
            continue  # in generation, we can skip decoding failures
        log = logger.getvalue()  # to check OOV
        if log:
            print(f"<Warning> Index {valid_index} "
                  f"- {' '.join(log.splitlines())}")
        output_file_path = output_file_format.format(valid_index=valid_index)
        decoded_midi.dump(os.path.join(output_dir, output_file_path))
        valid_index += 1

    log = (
        "\n"
        f"{f' Summary of Trial {batch_index} ':=^60}\n"
        f" * {valid_index - previous_count} valid sequences are converted to midi into path:\n"
        f"     {os.path.abspath(output_dir)}\n"
        f" * Totally {valid_index} sequences are converted."
    ) + ("=" * 60) + "\n"
    print(log)

    return valid_index
