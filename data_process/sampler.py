from training_utils import BaseSampler
import copy
import random
class Sampler(BaseSampler):
    def __init__(self):
        super().__init__()
        self.uid_text = {}

    def load_dataset(self, trn_dataset, val_dataset):
        if trn_dataset is not None:
            self.train_dataset = []
            for fid in range(len(trn_dataset["input_ids"])):
                data_row_dict = trn_dataset[fid]
                self.train_dataset.append(data_row_dict)
            self.train_indices = list(range(len(self.train_dataset)))

        self.val_dataset = []
        for fid in range(len(val_dataset["input_ids"])):
            data_row_dict = val_dataset[fid]
            self.val_dataset.append(data_row_dict)
        return

    def example_to_feature(self, example, *args, down_sampling=True, only_first=False, filter_spoiler_type=None, **kwargs):
        new_mapping = []
        for itm in example["offset_mapping"]:
            if itm is None:
                new_mapping.append([-1, -1])
            else:
                assert len(itm)==2
                new_mapping.append(itm)
        example_copied = copy.deepcopy(example)
        example_copied["offset_mapping"] = new_mapping

        # if down_sampling:
        #     if "start_target" in example and "end_target" in example:
        #         st_lab, ed_lab = example["start_target"], example["end_target"]
        #         if st_lab[0] == 1 and ed_lab[0]==1:
        #             zero_proportion = len([x for x in example["input_ids"] if x==0]) / len(example["input_ids"])
        #             drop_scale = 1.2 if zero_proportion > 0.9 else 1.0
        #             drop_prob = 0.6
        #             if random.random() < (drop_prob*drop_scale):
        #                 return None
        if only_first:
            if "is_first_feature" in example:
                if example["is_first_feature"] == 0:
                    return None
        if filter_spoiler_type is not None:
            if example["tags"] != filter_spoiler_type:
                return None
        return example_copied


