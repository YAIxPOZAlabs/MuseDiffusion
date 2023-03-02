META = {}

with open("MuseDiffusion/models/commu/preprocessor/utils/constants.py") as f:
        namespace = {}
        exec(f.read(), namespace)
        key_map = namespace['KEY_MAP']
        time_map = namespace['TIME_SIG_MAP']
        pitch_map = namespace['PITCH_RANGE_MAP']
        inst_map = namespace['INST_MAP']
        genre_map = namespace['GENRE_MAP']
        track_map = namespace['TRACK_ROLE_MAP']
        rhythm_map = namespace['RHYTHM_MAP']

print(key_map.keys())
print(time_map.keys())
print(pitch_map.keys())
print(inst_map.keys())
print(genre_map.keys())
print(track_map.keys())
print(rhythm_map.keys())

bpm = int(input('bpm : '))
META['bpm'] = bpm

audio_key = ''
while audio_key not in key_map:
        audio_key = input('audio_key : ')
META['audio_key'] = audio_key

time = ''
while time not in time_map:
        time = input('time_signature : ')
META['time_signature'] = time

pitch = ''
while pitch not in pitch_map:
        pitch = input('pitch_range : ')
META['pitch_range'] = pitch

num = int(input('num_measures : '))
META['num_measures'] = num

inst = ''
while inst not in inst_map:
        inst = input('instrument : ')
META['instrument'] = inst

genre = ''
while genre not in genre_map:
        genre = input('genre : ')
META['genre'] = genre

min_vel = int(input('min_velosity : '))
META['min_velosity'] = min_vel
max_vel = int(input('max_velosity : '))
META['max_velosity'] = max_vel

track = ''
while track not in track_map:
        track = input('track_role : ')
META['track_role'] = track

rhythm = ''
while rhythm not in rhythm_map:
        rhythm = input('rhythm : ')
META['rhythm'] = rhythm

t_chord = input('chord_progression : ')
chord = ''

for ch in t_chord:
        if ch in ['[', ']', "'", ' ']:
                continue
        if ch == ',':
                chord += '-'
        else:
                chord += ch

META['chord_progression'] = chord

print(META)
with open('meta_dict.py', "w") as f:
        import pprint
        print("META = ", end="", file=f)
        pprint.pprint(META, stream=f)
