import torch


def get_vectors(midi, note_len=128):
    """
    get rhythm, melody, and harmony vector for MSIM
    """
    i = 0
    while midi[i] != 2:
        i += 1
    i += 1
    rhythm_vec = torch.tensor([1e-8] * 32, dtype=torch.float32)
    tmp_rhythm_vec = torch.tensor([1e-8] * 32, dtype=torch.float32)
    melody_vec = torch.tensor([1e-8] * 12, dtype=torch.float32)
    harmony_vec = torch.tensor([0] * 12, dtype=torch.float32)
    
    cur_highest_pitch = -1
    prev_highest_pitch = -1
    prev_startp = -1

    while True:
        # calc tmp_groove_vec and add tmp_groove_vec to groove_vec
        if midi[i] <= 2:
            tmp_rhythm_vec /= torch.norm(tmp_rhythm_vec)
            rhythm_vec += tmp_rhythm_vec
            tmp_rhythm_vec = torch.tensor([1e-32] * 32, dtype=torch.float32)
            i += 1
            if midi[i-1] == 2:
                prev_startp = -1
                continue
            if prev_startp != startp and prev_highest_pitch >= 0:
                melody_vec[(cur_highest_pitch - prev_highest_pitch) % 12] += 1
            break
        has_pos = (432 <= midi[i] <= 559)
        startp = (midi[i] - 432) if has_pos else prev_startp
        if not has_pos:
            i -= 1
        if 195 <= midi[i+1] <= 303:
            i += 2
            continue
        try:
            assert 131 <= midi[i+1] <= 194
            assert 3 <= midi[i+2] <= 130
            assert 304 <= midi[i+3] <= 431
        except:
            raise ValueError("wrong format midi file")
        pitch = midi[i+2]
        endp = startp + midi[i+3] - 303
        harmony_vec[pitch % 12] += 1
        for t in range(0, min(128, endp), 4):
            if t < startp:
                continue
            max_amplitude = (0.00542676376 * (midi[i+1] - 130) * 2 + 0.310801) ** 2
            tmp_rhythm_vec[t // 4] = max(tmp_rhythm_vec[t // 4],
                max_amplitude * max(0, 1 - (t-startp) / note_len))
        if cur_highest_pitch >= 0:
            if prev_startp != startp:
                if prev_highest_pitch >= 0:
                    melody_vec[(cur_highest_pitch - prev_highest_pitch) % 12] += 1
                prev_startp = startp
                prev_highest_pitch = cur_highest_pitch
                cur_highest_pitch = pitch

        cur_highest_pitch = max(pitch, cur_highest_pitch)
        prev_startp = startp
        i += 4
    rhythm_vec /= torch.norm(rhythm_vec)
    melody_vec /= torch.norm(melody_vec)
    harmony_vec /= torch.norm(harmony_vec)
    return rhythm_vec, melody_vec, harmony_vec


def MSIM(midi1, midi2, return_vectors=False):
    """
    compare two midi file and return similarity
    """
    r1, m1, h1 = get_vectors(midi1)
    r2, m2, h2 = get_vectors(midi2)
    msim = torch.dot(r1, r2) * torch.dot(m1, m2) * torch.dot(h1, h2)
    if return_vectors:
        return msim, [r1, m1, h1], [r2, m2, h2]
    return msim


def ONNC(midilist, return_vectors=False, return_MSIM=False, return_mostsim=False):
    """
    Calculate 1NNC based on MSIM
    1NNC < 0.5 means overfitting, 1NNC > 0.5 means underfitting, 1NNC == 0.5 is best.
    the first half midis of midilist are considered GT, and last haf midis are considered generated.
    """
    rhythm_vectors = []
    melody_vectors = []
    harmony_vectors = []
    for midi in midilist:
        r, m, h = get_vectors(midi)
        rhythm_vectors.append(r)
        melody_vectors.append(m)
        harmony_vectors.append(h)
    rhythm_vectors = torch.stack(rhythm_vectors)
    melody_vectors = torch.stack(melody_vectors)
    harmony_vectors = torch.stack(harmony_vectors)
    rhythm_sim = rhythm_vectors @ rhythm_vectors.T
    melody_sim = melody_vectors @ melody_vectors.T
    harmony_sim = harmony_vectors @ harmony_vectors.T
    msim = rhythm_sim * melody_sim * harmony_sim
    msim.fill_diagonal_(0)
    most_sim = torch.argmax(msim, dim=1)
    halflength = len(midilist) // 2
    onnc = ((most_sim[:halflength] < halflength).sum() + (most_sim[halflength:] >= halflength).sum()) / len(midilist)
    if return_vectors == return_MSIM == return_mostsim == False:
        return onnc
    toreturn = [onnc]
    if return_vectors:
        toreturn.append([[rhythm_vectors, melody_vectors, harmony_vectors]])
    if return_MSIM:
        toreturn.append(msim)
    if return_mostsim:
        toreturn.append(most_sim)
    return toreturn
