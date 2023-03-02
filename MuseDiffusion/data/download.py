__all__ = ('guarantee_data', 'validate_checksum', 'download_data', 'get_data_dir')


DATASET_INFO = {
    'target_train.npy': (
        'https://github.com/YAIxPOZAlabs/ComMU-processed/blob/master/target_train.npy?raw=true',
        '3721233979443a35b66de1f55baf0c89'),
    'target_val.npy': (
        'https://github.com/YAIxPOZAlabs/ComMU-processed/blob/master/target_val.npy?raw=true',
        'cc1c15d9198bf7ba3316bf3676b4a997'),
    'input_val.npy': (
        'https://github.com/YAIxPOZAlabs/ComMU-processed/blob/master/input_val.npy?raw=true',
        'dd3b8d6f9aba6ffc40fd04c1195ea899'),
    'input_train.npy': (
        'https://github.com/YAIxPOZAlabs/ComMU-processed/blob/master/input_train.npy?raw=true',
        'bf4e8f53c5bd0b5b45fa430db95d8b54')
}  # type: dict[str, (str, str)]


def get_data_dir(data_dir=None):
    import os
    from pathlib import Path
    if data_dir is not None:
        return Path(data_dir).absolute()
    else:
        return Path(os.getcwd()) / 'datasets' / 'ComMU-processed'


def check_hash(data_dir):
    import hashlib
    download_path = get_data_dir(data_dir)
    for filename, (_, checksum) in DATASET_INFO.items():
        filepath = download_path.joinpath(filename)
        with open(filepath, 'rb') as f:
            assert checksum == hashlib.md5(f.read()).hexdigest(), \
                "Checksum of {} does not match!".format(filepath)
    return True


def download_data(data_dir, *, mode=0o755):
    from urllib.request import urlretrieve
    download_path = get_data_dir(data_dir)
    print('#' * 30 + '\nDownloading Datasets...\n' + '#' * 30)
    download_path.parent.mkdir(mode=mode, exist_ok=True)
    download_path.mkdir(mode=mode, exist_ok=True)
    for idx, (filename, (url, _)) in enumerate(DATASET_INFO.items(), start=1):
        filepath = download_path.joinpath(filename)
        print(filename, "({idx}/{total})".format(idx=idx, total=len(DATASET_INFO)))
        urlretrieve(url, str(filepath))
        filepath.chmod(mode=mode)
    check_hash(data_dir)


def validate_checksum(data_dir):
    download_path = get_data_dir(data_dir)
    return download_path.parent.is_dir() and \
        download_path.is_dir() and \
        all((download_path.joinpath(filename).is_file() for filename in DATASET_INFO.keys())) and \
        check_hash(data_dir)


def guarantee_data(data_dir, *, mode=0o755):
    if not validate_checksum(data_dir):
        download_data(data_dir, mode=mode)
    return True


if __name__ == '__main__':
    print("Dataset is {0} to {1}.".format(
        "downloaded and validated" if validate_checksum(None) else "not downloaded",
        get_data_dir(None)))
