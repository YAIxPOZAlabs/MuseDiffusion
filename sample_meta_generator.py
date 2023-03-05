from MuseDiffusion.utils.decode_util import META_CONSTANTS


def prompt(target, *, caster: "type|type(lambda: None)" = str, choice: dict = None):
    if choice is not None:
        print("Choose", target, "from:", *choice.keys())
    result = caster(input('%s : ' % target))
    if choice is not None:
        while result not in choice:
            result = caster(input('%s : ' % target))
    return result


def chord_caster(t_chord):
    mapping = {',': '-', '[': '', ']': '', "'": '', ' ': ''}
    return ''.join(mapping.get(c, c) for c in t_chord)


def get_meta():
    return {
        'bpm': prompt('bpm', caster=int), 'audio_key': prompt('audio_key', choice=META_CONSTANTS.audio_key),
        'time_signature': prompt('time_signature', choice=META_CONSTANTS.time_signature),
        'pitch_range': prompt('pitch_range', choice=META_CONSTANTS.pitch_range),
        'num_measures': prompt('num_measures', caster=int),
        'inst': prompt('instrument', choice=META_CONSTANTS.instrument),
        'genre': prompt('genre', choice=META_CONSTANTS.genre),
        'min_velocity': prompt('min_velocity', caster=int),
        'max_velocity': prompt('max_velocity', caster=int),
        'track_role': prompt('track_role', choice=META_CONSTANTS.track_role),
        'rhythm': prompt('rhythm', choice=META_CONSTANTS.rhythm),
        'chord_progression': prompt('chord_progression', caster=chord_caster)
    }


if __name__ == '__main__':
    META = get_meta()
    print(META)
    with open('meta_dict.py', "w") as f:
        import pprint
        print("META = ", end="", file=f)
        pprint.pprint(META, stream=f)
