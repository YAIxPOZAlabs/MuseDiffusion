import torch


def get_vectors(midi, note_len=128):
    i = 0
    while midi[i] != 2:
        i += 1
    i += 1
    groove_vec = torch.tensor([1e-8] * 32, dtype=torch.float32)
    tmp_groove_vec = torch.tensor([1e-8] * 32, dtype=torch.float32)
    progression_vec = torch.tensor([1e-8] * 12, dtype=torch.float32)
    chroma_vec = torch.tensor([0] * 12, dtype=torch.float32)
    
    cur_highest_pitch = -1
    prev_highest_pitch = -1
    prev_startp = -1

    while True:
        # calc tmp_groove_vec and add tmp_groove_vec to groove_vec
        if midi[i] <= 2:
            tmp_groove_vec /= torch.norm(tmp_groove_vec)
            groove_vec += tmp_groove_vec
            tmp_groove_vec = torch.tensor([1e-32] * 32, dtype=torch.float32)
            i += 1
            if midi[i-1] == 2:
                continue
            if prev_startp != startp and prev_highest_pitch >= 0:
                progression_vec[(cur_highest_pitch - prev_highest_pitch) % 12] += 1
            break
        has_pos = (432 <= midi[i] <= 559)
        if not has_pos:
            i -= 1
        if 195 <= midi[i+1] <= 303:
            i += 2
            continue
        pitch = midi[i+2]
        startp = (midi[i] - 432) if has_pos else prev_startp
        endp = startp + midi[i+3] - 303
        chroma_vec[pitch % 12] += 1
        for t in range(0, min(128, endp), 4):
            if t < startp:
                continue
            max_amplitude = (0.00542676376 * (midi[i+1] - 130) * 2 + 0.310801) ** 2
            tmp_groove_vec[t // 4] = max(tmp_groove_vec[t // 4],
                max_amplitude * max(0, 1 - (t-startp) / note_len))
        if cur_highest_pitch >= 0:
            if prev_startp != startp:
                if prev_highest_pitch >= 0:
                    progression_vec[(cur_highest_pitch - prev_highest_pitch) % 12] += 1
                prev_startp = startp
                prev_highest_pitch = cur_highest_pitch
                cur_highest_pitch = pitch

        cur_highest_pitch = max(pitch, cur_highest_pitch)
        prev_startp = startp
        i += 4
    groove_vec /= torch.norm(groove_vec)
    progression_vec /= torch.norm(progression_vec)
    chroma_vec /= torch.norm(chroma_vec)
    return groove_vec, progression_vec, chroma_vec


def MSIM(midi1, midi2, return_vectors=False):
    g1, p1, c1 = get_vectors(midi1)
    g2, p2, c2 = get_vectors(midi2)
    msim = torch.dot(g1, g2) * torch.dot(p1, p2) * torch.dot(c1, c2)
    if return_vectors:
        return msim, [g1, p1, c1], [g2, p2, c2]
    return msim


def ONNC(midilist, return_vectors=False, return_MSIM=False, return_mostsim=False):
    """
    Calculate 1NNC based on MSIM
    1NNC < 0.5 means overfitting, 1NNC > 0.5 means underfitting, 1NNC == 0.5 is best.
    the first half midis of midilist are considered GT, and last haf midis are considered generated.
    """
    groove_vectors = []
    progression_vectors = []
    chroma_vectors = []
    for midi in midilist:
        g, p, c = get_vectors(midi)
        groove_vectors.append(g)
        progression_vectors.append(p)
        chroma_vectors.append(c)
    groove_vectors = torch.stack(groove_vectors)
    progression_vectors = torch.stack(progression_vectors)
    chroma_vectors = torch.stack(chroma_vectors)
    groove_sim = groove_vectors @ groove_vectors.T
    progression_sim = progression_vectors @ progression_vectors.T
    chroma_sim = chroma_vectors @ chroma_vectors.T
    msim = groove_sim * progression_sim * chroma_sim
    msim.fill_diagonal_(0)
    most_sim = torch.argmax(msim, dim=1)
    halflength = len(midilist) // 2
    onnc = ((most_sim[:halflength] < halflength).sum() + (most_sim[halflength:] >= halflength).sum()) / len(midilist)
    if return_vectors == return_MSIM == return_mostsim == False:
        return onnc
    toreturn = [onnc]
    if return_vectors:
        toreturn.append([[groove_vectors, progression_vectors, chroma_vectors]])
    if return_MSIM:
        toreturn.append(msim)
    if return_mostsim:
        toreturn.append(most_sim)
    return toreturn
