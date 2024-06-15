from torch.utils.data import Dataset
from embodiedpose.data_loaders.scene_pose_dataset import ScenePoseDataset

class MultiHumPoseDataset(Dataset):
    """
    Dataloader to load two people pose data
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        cfg = args[0]
        swap_order = cfg.swap_order

        if swap_order:
            # SWAPPED OPERATION
            self.human2_dataset = ScenePoseDataset(*args, **kwargs)
            datafiles = args[1]
            p2_files = [datafiles[0].replace("_p1_", "_p2_")]
            mod_args = (args[0], p2_files)
            self.human1_dataset = ScenePoseDataset(*mod_args, **kwargs)
        else:
            # NORMAL OPERATION
            self.human1_dataset = ScenePoseDataset(*args, **kwargs)
            datafiles = args[1]
            p2_files = [datafiles[0].replace("_p1_", "_p2_")]
            mod_args = (args[0], p2_files)
            self.human2_dataset = ScenePoseDataset(*mod_args, **kwargs)

        self.data_keys = self.human1_dataset.data_keys
        self.get_sample_len_from_key = self.human1_dataset.get_sample_len_from_key
        self.data_raw = []
        self.data_raw.append(self.human1_dataset.data_raw)
        self.data_raw.append(self.human2_dataset.data_raw)

    def __len__(self):
        return len(self.human1_dataset)

    def __getitem__(self, idx):
        # get data from PROX
        human1_data = self.human1_dataset.__getitem__(idx)
        human2_data = self.human2_dataset.__getitem__(idx)

        return human1_data, human2_data

    def sample_seq(self, fr_num, fr_start):
        data1 = self.human1_dataset.sample_seq(fr_num=fr_num, fr_start=fr_start)
        data2 = self.human2_dataset.sample_seq(fr_num=fr_num, fr_start=fr_start)
        return data1, data2

    def get_sample_from_key(self, take_key, full_sample=True, return_batch=True, dyncam=False):
        # inside both, cam is obtained as: sample['cam'] = self.data_raw[take_key]['cam']
        data1 = self.human1_dataset.get_sample_from_key(take_key, full_sample=full_sample, return_batch=return_batch, dyncam=dyncam)
        data2 = self.human2_dataset.get_sample_from_key(take_key, full_sample=full_sample, return_batch=return_batch, dyncam=dyncam)
        return data1, data2


