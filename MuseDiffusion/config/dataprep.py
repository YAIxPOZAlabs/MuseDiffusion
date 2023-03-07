from .base import S, Item as _


class DataPrepSettings(S):
    data_dir: str = _('', 'path for dataset to be saved')
    num_proc: int = _(4, 'number of subprocesses to preparing dataset')
