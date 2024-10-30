'''
Modified from EmbodiedPose repo: https://github.com/ZhengyiLuo/EmbodiedPose
This is the env used for the eval_scene, it seems that here is where everything happens: the simulation and the forward step

'''

from torch.utils.data import Dataset
from embodiedpose.data_loaders.scene_pose_dataset import ScenePoseDataset

class MultiHumPoseDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__()
        cfg = args[0]
        swap_order = cfg.swap_order
        self.human_dataset_all = []
        self.n_agents = cfg.num_agents

        if swap_order:
            print("WARNING: DATALOADER: swapping option is not implemented for the n_agents version! only for 2 agents")
        # NORMAL OPERATION
        for n in range(cfg.num_agents):
            if n==0:
                self.human_dataset = ScenePoseDataset(*args, **kwargs)
            else:
                datafiles = args[1]
                p2_files = [datafiles[0].replace("_p1_", f"_p{n+1}_")]
                mod_args = (args[0], p2_files)
                self.human_dataset = ScenePoseDataset(*mod_args, **kwargs)
            self.human_dataset_all.append(self.human_dataset)

        self.data_keys = self.human_dataset_all[0].data_keys
        self.get_sample_len_from_key = self.human_dataset_all[0].get_sample_len_from_key

        self.data_raw = []
        for n in range(cfg.num_agents):
            self.data_raw.append(self.human_dataset_all[n].data_raw)
        print(f"DATALOADER: loaded {len(self.data_raw)} agents data")

    def __len__(self):
        return len(self.human_dataset_all[0])

    def __getitem__(self, idx):
        return_data = [data.__getitem__(idx) for data in self.human_dataset_all]
        return return_data

    def sample_seq(self, fr_num, fr_start):
        ret_data = [data.sample_seq(fr_num=fr_num, fr_start=fr_start) for data in self.human_dataset_all]
        return ret_data

    def get_sample_from_key(self, take_key, full_sample=True, return_batch=True, dyncam=False):
        ret_data = [data.get_sample_from_key(take_key, full_sample=full_sample, return_batch=return_batch, dyncam=dyncam)
                    for data in self.human_dataset_all]
        return ret_data

