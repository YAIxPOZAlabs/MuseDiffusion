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
    from commu.preprocessor.encoder import EventSequenceEncoder, MetaEncoder, TOKEN_OFFSET
    from commu.preprocessor.utils.container import MidiInfo, MidiMeta
except ModuleNotFoundError as e:
    raise ImportError(
        "In Sampling task, ComMU module is required. Try to install by command"
        " 'pip install git+https://github.com/YAIxPOZAlabs/ComMU-package'."
    ) from e


# Constants

CHORD_MAP = {
  'A': 195, 'A7': 196, 'A+': 197, 'Adim': 198, 'Am': 199, 'Am7': 200, 'Am7b5': 201, 'Amaj7': 202, 'Asus4': 203,
  'A#': 204, 'A#7': 205, 'A#+': 206, 'A#dim': 207, 'A#m': 208, 'A#m7': 209, 'A#m7b5': 210, 'A#maj7': 211, 'A#sus4': 212,
  'B': 213, 'B7': 214, 'B+': 215, 'Bdim': 216, 'Bm': 217, 'Bm7': 218, 'Bm7b5': 219, 'Bmaj7': 220, 'Bsus4': 221,
  'C': 222, 'C7': 223, 'C+': 224, 'Cdim': 225, 'Cm': 226, 'Cm7': 227, 'Cm7b5': 228, 'Cmaj7': 229, 'Csus4': 230,
  'C#': 231, 'C#7': 232, 'C#+': 233, 'C#dim': 234, 'C#m': 235, 'C#m7': 236, 'C#m7b5': 237, 'C#maj7': 238, 'C#sus4': 239,
  'D': 240, 'D7': 241, 'D+': 242, 'Ddim': 243, 'Dm': 244, 'Dm7': 245, 'Dm7b5': 246, 'Dmaj7': 247, 'Dsus4': 248,
  'D#': 249, 'D#7': 250, 'D#+': 251, 'D#dim': 252, 'D#m': 253, 'D#m7': 254, 'D#m7b5': 255, 'D#maj7': 256, 'D#sus4': 257,
  'E': 258, 'E7': 259, 'E+': 260, 'Edim': 261, 'Em': 262, 'Em7': 263, 'Em7b5': 264, 'Emaj7': 265, 'Esus4': 266,
  'F': 267, 'F7': 268, 'F+': 269, 'Fdim': 270, 'Fm': 271, 'Fm7': 272, 'Fm7b5': 273, 'Fmaj7': 274, 'Fsus4': 275,
  'F#': 276, 'F#7': 277, 'F#+': 278, 'F#dim': 279, 'F#m': 280, 'F#m7': 281, 'F#m7b5': 282, 'F#maj7': 283, 'F#sus4': 284,
  'G': 285, 'G7': 286, 'G+': 287, 'Gdim': 288, 'Gm': 289, 'Gm7': 290, 'Gm7b5': 291, 'Gmaj7': 292, 'Gsus4': 293,
  'G#': 294, 'G#7': 295, 'G#+': 296, 'G#dim': 297, 'G#m': 298, 'G#m7': 299, 'G#m7b5': 300, 'G#maj7': 301, 'G#sus4': 302,
  'NN': 303
}


# Pipelines

class MetaToSequence:

    @staticmethod
    def encode_chord(chord_progression):
        encoded_chord = []
        chord_progression = chord_progression
        assert len(chord_progression) % 8 == 0

        for idx in range(0, len(chord_progression), 8):
            encoded_chord.append(432)
            chord_token = CHORD_MAP[chord_progression[idx]]
            encoded_chord.append(chord_token)
            recent_chord = chord_progression[idx]
            for i in range(1, 8):
                if recent_chord != chord_progression[idx + i]:
                    encoded_chord.append(432 + i * 16)
                    recent_chord = chord_progression[idx + i]
        return encoded_chord

    @staticmethod
    def encode_meta(midi_meta):
        return MetaEncoder().encode(midi_meta)

    @classmethod
    def execute(cls, input_data: dict) -> "list[int]":
        midi_meta = MidiMeta(**input_data)
        chord_progression = input_data["chord_progression"].split("-")
        return cls.encode_meta(midi_meta) + cls.encode_chord(chord_progression)

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)


class MetaToBatch(MetaToSequence):

    @classmethod
    def execute(cls, input_data, batch_size, seq_len):  # NOQA
        import torch
        encoded_meta = super(MetaToBatch, cls).execute(input_data)
        encoded_meta = torch.tensor(encoded_meta)
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.int)
        input_ids[:, :len(encoded_meta)] = encoded_meta
        input_mask = torch.ones(batch_size, seq_len, dtype=torch.int)
        input_mask[:, :len(encoded_meta) + 1] = 0
        batch = {'input_ids': input_ids, 'input_mask': input_mask}
        return batch


class SequenceToMidiError(Exception):
    pass


class SequenceToMidi:

    decoder = EventSequenceEncoder()

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

    @classmethod
    def decode_event_sequence(
            cls,
            encoded_meta,
            note_seq
    ) -> "MidiFile":
        decoded_midi = cls.decoder.decode(
            midi_info=MidiInfo(*encoded_meta, event_seq=note_seq),
        )
        return decoded_midi

    @classmethod
    def extract_note_seq(cls, seq, input_mask):
        len_meta = len(seq) - int((input_mask.sum()))
        note_seq = seq[len_meta:]
        try:
            note_seq = cls.remove_padding(note_seq)
            return note_seq[np.where(note_seq < 559)]
        except SequenceToMidiError:
            return

    @classmethod
    def _decode(cls, seq, input_mask):  # Output: Midi, Errcode
        len_meta = len(seq) - int((input_mask.sum()))
        encoded_meta = seq[:len_meta - 1]  # meta 에서 eos 토큰 제외 11개만 사용
        note_seq = seq[len_meta:]
        note_seq = cls.remove_padding(note_seq)
        note_seq, encoded_meta = cls.restore_chord(note_seq, encoded_meta)
        cls.validate_generated_sequence(note_seq)
        decoded_midi = cls.decode_event_sequence(encoded_meta, note_seq)
        return decoded_midi

    @classmethod
    def decode(cls, seq, input_mask, output_file_path=None):
        decoded_midi = cls._decode(seq, input_mask)
        if output_file_path:
            decoded_midi.dump(output_file_path)
        return decoded_midi


def save_tokens(input_tokens, output_tokens, output_dir, batch_index):
    out_list = []
    len_meta = 12
    for (in_seq, out_seq) in zip(input_tokens, output_tokens):
        encoded_meta = in_seq[:len_meta - 1]
        in_note_seq = in_seq[len_meta:]
        in_note_seq = SequenceToMidi.remove_padding(in_note_seq)
        out_note_seq = out_seq[len_meta:]
        out_note_seq = SequenceToMidi.remove_padding(out_note_seq)
        # output_file_path = self.set_output_file_path(idx=idx, output_dir=output_dir)
        out_list.append(np.concatenate((encoded_meta, in_note_seq, [0], out_note_seq)))
    path = os.path.join(output_dir, f'token_batch{batch_index}.npy')
    np.save(path, out_list)


def batch_decode_seq2seq(
        sequences,
        input_ids_mask_ori,
        batch_index,
        previous_count,
        output_dir,
        output_file_format="{original_index}_batch{batch_index}_{index}.midi"
):

    invalid_idxes = set()

    for index, (seq, input_mask) in enumerate(zip(sequences, input_ids_mask_ori)):
        original_index = previous_count + index
        logger = io.StringIO()
        try:
            with contextlib.redirect_stdout(logger):
                decoded_midi = SequenceToMidi.decode(seq, input_mask)
        except SequenceToMidiError as exc:
            log = logger.getvalue()
            print(f"<Warning> Batch {batch_index} Index {index} "
                  f"(Original: {original_index}) "
                  f"- Generation Failure: {exc}")
            if log:
                print(log)
            invalid_idxes.add(index)
        except Exception as exc:
            print(logger.getvalue())
            print(f"<Error> {exc.__class__.__qualname__}: {exc} \n"
                  f"  occurred while generating midi of Batch {batch_index} Index {index} "
                  f"(Original: {original_index}).")
            raise
        except BaseException:
            print(logger.getvalue())
            raise
        else:
            log = logger.getvalue()
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

    valid_index = previous_count

    for seq, input_mask in zip(sequences, input_ids_mask_ori):
        logger = io.StringIO()
        try:
            with contextlib.redirect_stdout(logger):
                decoded_midi = SequenceToMidi.decode(seq, input_mask)
        except SequenceToMidiError:
            continue
        log = logger.getvalue()
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
