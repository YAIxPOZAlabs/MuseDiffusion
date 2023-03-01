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
    nbars = 0

    while True:
        # calc tmp_groove_vec and add tmp_groove_vec to groove_vec
        if midi[i] <= 2:
            nbars += 1
            tmp_groove_vec /= torch.norm(tmp_groove_vec)
            groove_vec += tmp_groove_vec
            tmp_groove_vec = torch.tensor([1e-32] * 32, dtype=torch.float32)
            i += 1
            if midi[i-1] == 2:
                continue
            break
        if 195 <= midi[i+1] <= 303:
            i += 2
            continue
        pitch = midi[i+2]
        startp = midi[i] - 432
        endp = startp + midi[i+3] - 303
        chroma_vec[pitch % 12] += 1
        for t in range(0, min(128, endp), 4):
            if t < midi[i] - 432:
                continue
            max_amplitude = (0.00542676376 * (midi[i+1] - 130) * 2 + 0.310801) ** 2
            tmp_groove_vec[t // 4] = max(tmp_groove_vec[t // 4],
                (max_amplitude * max(0, 1 - (t-startp) / note_len)) ** 2)
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
    groove_vec /= nbars
    progression_vec /= torch.norm(progression_vec)
    chroma_vec /= torch.norm(chroma_vec)
    return groove_vec, progression_vec, chroma_vec


def MSIM(midi1, midi2, return_vectors=False):
    g1, p1, c1 = get_vectors(midi1)
    g2, p2, c2 = get_vectors(midi2)
    msim = (torch.dot(g1, g2) + 1) * (torch.dot(p1, p2) + 1) * (torch.dot(c1, c2) + 1) / 8
    if return_vectors:
        return msim, [g1, p1, c1], [g2, p2, c2]
    return msim
