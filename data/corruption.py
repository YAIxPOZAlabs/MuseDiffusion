import random
import torch


def masking_token(seq: torch.Tensor, p: float, inplace: bool = False):
    """
    masking_token으로 변경하는 함수
    token 단위로 진행
    이거 쓰려면 masking token값으로 730 지정해주고 embedding layer의 dim 1 추가해야됨
    """
    corrupted = seq if inplace else torch.clone(seq)

    for i in range(len(seq[12:])):
        if seq[i+12] == 1:
            break
        rnd = torch.rand(1)[0]
        if rnd < p:
            corrupted[i+12] = 730
    return corrupted

def masking_note(seq: torch.Tensor, p: float, inplace: bool = False):
    """
    masking_token으로 변경하는 함수
    note (position - velocity - pitch - duration) 단위로 진행
    이거 쓰려면 masking token값으로 730 지정해주고 embedding layer의 dim 1 추가해야됨
    """
    corrupted = seq if inplace else torch.clone(seq)

    vel_idx = (131 >= seq).nonzero(as_tuple=True)
    for idx in vel_idx:
        if seq[idx] >194:
            continue
        rnd = torch.rand(1)[0]
        if rnd < p:
            corrupted[idx-1:idx+3] = [730]*4
    return corrupted


def randomize_note(seq: torch.Tensor, p: float, inplace: bool = False):
    """
    주어진 확률(혹은 비율) p 만큼의 note의 pitch, duration, velocity, position? 을 랜덤하게 변경하는 함수
    seq: meta+0+note_seq+zero_padding (torch type)
    p: 0.0~1.0 사이값
    note token 순서: position - velocity - pitch - duration 순으로 존재
    pitch token range: 3 ~ 130
    velocity: 131 ~ 194
    duration: 304~431
    position: 432~559
    """
    corrupted = seq if inplace else torch.clone(seq)

    # velocity token index 찾기 (velocity token 범위: 131~194)
    vel_idx = (131 >= seq).nonzero(as_tuple=True)
    for idx in vel_idx:
        if seq[idx] > 194:
            continue
        rnd = torch.rand(1)[0]
        if rnd < p:
            new_pitch = torch.randint(3, 131, (1,))[0]
            new_duration = torch.randint(304, 432, (1,))[0]
            new_velocity = torch.randint(131, 195, (1,))[0]
            corrupted[idx] = new_velocity
            corrupted[idx+1] = new_pitch
            corrupted[idx+2] = new_duration
            # seq[idx-1] = new_position --> 이건 쓸까말까
    return corrupted


def adding_token(seq: torch.Tensor, mask: torch.Tensor, p: float, inplace: bool = False):
    """
    단일토큰 추가하는 함수...인데 추가해버리면 input_mask에도 약간의 변화가 필요함
    """
    corrupted = seq if inplace else torch.clone(seq)
    ...


def random_rotating(seq: torch.Tensor, count: int, inplace: bool = False):
    """
    마디단위로 묶어서 마디의 순서를 랜덤하게 섞는다
    count: 몇번 마디 바꾸는것을 수행할것인가
    """
    rotated = seq if inplace else torch.clone(seq)
    bar_idx = (seq == 2).nonzero(as_tuple=True)
    eos_idx = (seq == 1).nonzero(as_tuple=True)[-1]

    for _ in range(count):
        first_idx = random.randint(0, len(bar_idx))
        second_idx = random.randint(0, len(bar_idx))
        while second_idx == first_idx:
            second_idx = random.randint(0, len(bar_idx))

        first_idx, second_idx = sorted([first_idx, second_idx])

        # find start and end of bars
        bar1_start = bar_idx[first_idx]
        bar2_start = bar_idx[second_idx]

        bar1_end = bar_idx[first_idx+1]
        bar2_end = bar_idx[second_idx+1] if second_idx < len(bar_idx)-1 else eos_idx

        # sequence에서 bar 2개를 잘라내고 순서를 바꿔서 concat
        start_to_bar1 = rotated[:bar1_start]
        bar1_array = rotated[bar1_start:bar1_end]
        bar1_to_bar2 = rotated[bar1_end:bar2_start]
        bar2_array = rotated[bar2_start:bar2_end]
        bar2_to_EOS = rotated[bar2_end:]
        
        rotated = torch.cat([start_to_bar1, bar2_array, bar1_to_bar2, bar1_array, bar2_to_EOS])
    return rotated


def get_corruption_from_configs():
    # corruption 수행
    # 랜덤하게 한 함수를 선택해서 수행하는 방식? 아니면 여러개 동시에 적용하는 방식?
    from functools import partial
    func = partial(randomize_note, p=0.5)
    return func
