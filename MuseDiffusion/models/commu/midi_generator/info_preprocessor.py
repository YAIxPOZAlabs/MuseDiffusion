from typing import List, Any

from .container import TransXlInputData
from ..preprocessor.encoder import MetaEncoder
from ..preprocessor.utils.container import MidiMeta

CHORD_MAP = {'A': 195, 'A7': 196, 'A+': 197, 'Adim': 198, 'Am': 199, 'Am7': 200, 'Am7b5': 201, 'Amaj7': 202, 'Asus4': 203, 'A#': 204, 'A#7': 205, 'A#+': 206, 'A#dim': 207, 'A#m': 208, 'A#m7': 209, 'A#m7b5': 210, 'A#maj7': 211, 'A#sus4': 212, 
             'B': 213, 'B7': 214, 'B+': 215, 'Bdim': 216, 'Bm': 217, 'Bm7': 218, 'Bm7b5': 219, 'Bmaj7': 220, 'Bsus4': 221, 
             'C': 222, 'C7': 223, 'C+': 224, 'Cdim': 225, 'Cm': 226, 'Cm7': 227, 'Cm7b5': 228, 'Cmaj7': 229, 'Csus4': 230, 'C#': 231, 'C#7': 232, 'C#+': 233, 'C#dim': 234, 'C#m': 235, 'C#m7': 236, 'C#m7b5': 237, 'C#maj7': 238, 'C#sus4': 239, 
             'D': 240, 'D7': 241, 'D+': 242, 'Ddim': 243, 'Dm': 244, 'Dm7': 245, 'Dm7b5': 246, 'Dmaj7': 247, 'Dsus4': 248, 'D#': 249, 'D#7': 250, 'D#+': 251, 'D#dim': 252, 'D#m': 253, 'D#m7': 254, 'D#m7b5': 255, 'D#maj7': 256, 'D#sus4': 257, 
             'E': 258, 'E7': 259, 'E+': 260, 'Edim': 261, 'Em': 262, 'Em7': 263, 'Em7b5': 264, 'Emaj7': 265, 'Esus4': 266, 
             'F': 267, 'F7': 268, 'F+': 269, 'Fdim': 270, 'Fm': 271, 'Fm7': 272, 'Fm7b5': 273, 'Fmaj7': 274, 'Fsus4': 275, 'F#': 276, 'F#7': 277, 'F#+': 278, 'F#dim': 279, 'F#m': 280, 'F#m7': 281, 'F#m7b5': 282, 'F#maj7': 283, 'F#sus4': 284, 
             'G': 285, 'G7': 286, 'G+': 287, 'Gdim': 288, 'Gm': 289, 'Gm7': 290, 'Gm7b5': 291, 'Gmaj7': 292, 'Gsus4': 293, 'G#': 294, 'G#7': 295, 'G#+': 296, 'G#dim': 297, 'G#m': 298, 'G#m7': 299, 'G#m7b5': 300, 'G#maj7': 301, 'G#sus4': 302, 
             'NN': 303}

def parse_meta(**kwargs: Any) -> MidiMeta:
    return MidiMeta(**kwargs)


def encode_meta(meta_encoder: MetaEncoder, midi_meta: MidiMeta) -> List[int]:
    return meta_encoder.encode(midi_meta)


def normalize_chord_progression(chord_progression: str) -> List[str]:
    return chord_progression.split("-")


class PreprocessTask:
    def __init__(self):
        self.input_data = None
        self.midi_meta = None

    def get_meta_info_length(self):
        return len(self.midi_meta.__fields__)

    def normalize_input_data(self, input_data: dict):
        input_data["chord_progression"] = normalize_chord_progression(input_data["chord_progression"])
        self.input_data = TransXlInputData(**input_data)

    def preprocess(self) -> List[int]:
        self.midi_meta = parse_meta(**self.input_data.dict())
        meta_encoder = MetaEncoder()
        encoded_meta = encode_meta(
            meta_encoder=meta_encoder, midi_meta=self.midi_meta
        )
        return encoded_meta
    
    def encode_chord(self):
        # chord to token dict 필요
        encoded_chord = []
        chord_progression = self.input_data.chord_progression
        assert len(chord_progression) % 8 == 0

        for idx in range(0, len(chord_progression), 8):
            encoded_chord.append(432)
            chord_token = CHORD_MAP[chord_progression[idx]]
            encoded_chord.append(chord_token)
            recent_chord = chord_progression[idx]
            for i in range(1, 8):
                if recent_chord != chord_progression[idx+i]:
                    encoded_chord.append(432+i*16)
                    recent_chord = chord_progression[idx+i]
        return encoded_chord

    def excecute(self, input_data: dict) -> List[int]:
        if self.input_data is None:
            self.normalize_input_data(input_data)

        encoded_meta = self.preprocess()

        encoded_chord = self.encode_chord()

        encoded_meta = encoded_meta+encoded_chord

        return encoded_meta