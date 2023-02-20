import datetime
import time
from tqdm import tqdm
import numpy as np
import random
import logging
import torch
from contextlib import contextmanager

class Logger():
    def __init__(self, log_path, log_name="default_log", stream_print=True):
        self.logger = logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        if stream_print:
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            logger.addHandler(sh)
        return

    def info(self, log_str):
        self.logger.info(log_str)

    def warning(self, warn_str):
        self.logger.warning(warn_str)

"""
记录运行时间
@timer
def fun1(s_time, logger=None):
    time.sleep(s_time)
    return s_time
fun1(1.0, logger=logger)
"""
def timer(func):
    def wrapper(*args, **kwargs):
        ts = datetime.datetime.now()
        res = func(*args, **kwargs)
        dur = (datetime.datetime.now()-ts).total_seconds()
        if "logger" in kwargs and kwargs["logger"] is not None:
            kwargs["logger"].info(str(func)+" run time cost={} sec".format(dur))
        else:
            logging.info(str(func)+" run time cost={} sec".format(dur))
        return res
    return wrapper


def clip_grads(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class BaseSampler():
    def __init__(self, logger=None):
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.current_epoch = 0
        self.train_indices = []
        self.is_large_file_roller = False
        self.logger = logger if logger is not None else logging

    def set_training_dataset_roller(self, train_file_path, line_function, *args, buffer_size=50000, batch_size_estimate=20, **kwargs):
        self.logger.warning("rolling train dataset has buffer size % bsz not equal to 0, try use integer copies of batch size in a dataset buffer")
        self.is_large_file_roller = True
        self.train_file_path = train_file_path
        self.train_fr = open(train_file_path, encoding="utf8", buffering=4096)
        self.deserialize_func = line_function
        self.deserialize_args = args
        self.deserialize_kwargs = kwargs
        self.rolling_buffer_size = buffer_size
        self.load_rolling_dataset(*self.deserialize_args, **self.deserialize_kwargs)

    def load_rolling_dataset(self, *args, **kwargs):
        self.train_dataset = []
        line = self.train_fr.readline()
        if not hasattr(self, "train_data_count"):
            self.train_data_count = 0
            self.train_rolling_pt = 0
        while line is not None and line != "":
            self.train_dataset.append(self.deserialize_func(line, *args, **kwargs))
            self.train_rolling_pt += 1
            if len(self.train_dataset) >= self.rolling_buffer_size:
                break
            else:
                if not self.train_data_count > 0:
                    self.train_data_count -= 1
            line = self.train_fr.readline()

        if len(self.train_dataset) < self.rolling_buffer_size:
            self.train_fr.close()
            self.train_data_count = abs(self.train_data_count)
            self.train_fr = open(self.train_file_path, encoding="utf8", buffering=4096)
            line = self.train_fr.readline()
            while line is not None and line != "":
                self.train_dataset.append(self.deserialize_func(line, *args, **kwargs))
                if len(self.train_dataset) >= self.rolling_buffer_size:
                    break
                line = self.train_fr.readline()
        self.train_indices = list(range(len(self.train_dataset)))
        random.shuffle(self.train_indices)
        return

    def train_batch_sampling(self, *args, batch_size=20, max_epoch=5, **kwargs):
        if self.train_indices is None or len(self.train_indices) != len(self.train_dataset):
            self.train_indices = list(range(len(self.train_dataset)))
        random.shuffle(self.train_indices)

        features = {}
        feature_types = {}
        default_key = None
        prev_epoch = kwargs["continue_epoch"] if ("continue_epoch" in kwargs and kwargs["continue_epoch"] is not None) else 0
        if prev_epoch > 0:
            self.current_epoch = prev_epoch
        if "dynamic_pad_id" in kwargs:
            dynamic_pad_id = int(kwargs["dynamic_pad_id"])
        else:
            dynamic_pad_id = None

        while self.current_epoch < max_epoch:
            for iid, pi in enumerate(tqdm(self.train_indices)):
                try:
                    processed = self.example_to_feature(self.train_dataset[pi], *args, **kwargs)

                except Exception as ex:
                    print("sampler feature processing error:")
                    print(ex)
                    continue
                if processed is None:
                    continue
                for _k, _v in processed.items():
                    if not isinstance(_v, list):
                        _v = [_v]
                    if default_key is None:
                        default_key = _k
                    if _k not in features:
                        features[_k] = []
                        feature_types[_k] = np.float
                        if isinstance(_v[0], int):
                            feature_types[_k] = np.long
                        elif isinstance(_v[0], float):
                            feature_types[_k] = np.float
                        elif isinstance(_v[0], str):
                            feature_types[_k] = "str"
                        elif isinstance(_v[0], list) and (isinstance(_v[0][0], int) or isinstance(_v[0][0], float)):
                            feature_types[_k] = np.float if isinstance(_v[0][0], float) else np.int
                        else:
                            print(_v[0])
                            raise Exception("data type not supported, should be among int, float, string")
                    features[_k].append(_v)

                if len(features[default_key]) >= batch_size:
                    out_features = {}
                    for _k, _vs in features.items():
                        if feature_types[_k] == "str":
                            out_features[_k] = [x[0] for x in _vs]
                        else:
                            if dynamic_pad_id is not None:
                                max_steps = max([len(x) for x in _vs])
                                paddings = [[dynamic_pad_id]*(max_steps-len(x)) for x in _vs]
                                _vs = [x1+pad for x1, pad in zip(_vs, paddings)]
                            out_features[_k] = np.array(_vs, dtype=feature_types[_k])
                    out_features["is_new_epoch"] = (prev_epoch != self.current_epoch)
                    # if not self.is_large_file_roller and self.current_epoch == max_epoch -1 and iid + batch_size >= len(self.train_indices):
                    #     out_features["is_new_epoch"] = True
                    yield out_features
                    features = {_k:[] for _k in features.keys()}
                    prev_epoch = self.current_epoch

            if self.is_large_file_roller:
                if self.train_data_count > 0 and self.train_rolling_pt >= self.train_data_count:
                    print("\nepoch {0} done, starting next epoch...".format(self.current_epoch))
                    self.logger.info("\nepoch {0} done, starting next epoch...".format(self.current_epoch))
                    self.current_epoch += 1
                    self.train_rolling_pt = 0
                else:
                    print("\nrolling to next section of datafile, progress={0}/{1}".format(self.train_rolling_pt, self.train_data_count))
                    self.logger.info(("\nrolling to next section of datafile, progress={0}/{1}".format(self.train_rolling_pt, self.train_data_count)))
            else:
                print("\nepoch {0} done, starting next epoch...".format(self.current_epoch))
                self.logger.info("\nepoch {0} done, starting next epoch...".format(self.current_epoch))
                random.shuffle(self.train_indices)
                self.current_epoch += 1

    def val_batch_sampling(self, *args, batch_size=20, dataset=None, **kwargs):
        if dataset is None:
            dataset = self.val_dataset
        if "dynamic_pad_id" in kwargs:
            dynamic_pad_id = int(kwargs["dynamic_pad_id"])
        else:
            dynamic_pad_id = None

        features = {}
        feature_types = {}
        default_key = None
        L = len(dataset)
        print("start sampling on dataset with size {0}".format(L))

        for di, data_example in enumerate(tqdm(dataset)):
            processed = self.example_to_val_feature(data_example, *args, down_sampling=False, **kwargs)
            if processed is None:
                continue
            for _k, _v in processed.items():
                if not isinstance(_v, list):
                    _v = [_v]
                if default_key is None:
                    default_key = _k
                if _k not in features:
                    features[_k] = []
                    feature_types[_k] = np.float
                    if isinstance(_v[0], int):
                        feature_types[_k] = np.long
                    elif isinstance(_v[0], float):
                        feature_types[_k] = np.float
                    elif isinstance(_v[0], str):
                        feature_types[_k] = "str"
                    elif isinstance(_v[0], list) and (isinstance(_v[0][0], int) or isinstance(_v[0][0], float)):
                        feature_types[_k] = np.float if isinstance(_v[0][0], float) else np.int
                    else:
                        print(_v[0])
                        raise Exception("data type not supported, should be among int, float, string")
                features[_k].append(_v)

            if len(features[default_key]) >= batch_size or di==L-1:
                out_features = {}
                for _k, _vs in features.items():
                    if feature_types[_k] == "str":
                        out_features[_k] = [x[0] for x in _vs]
                    else:
                        if dynamic_pad_id is not None:
                            max_steps = max([len(x) for x in _vs])
                            paddings = [[dynamic_pad_id] * (max_steps - len(x)) for x in _vs]
                            _vs = [x1 + pad for x1, pad in zip(_vs, paddings)]
                        out_features[_k] = np.array(_vs, dtype=feature_types[_k])
                yield out_features
                features = {_k:[] for _k in features.keys()}

    def example_to_val_feature(self, example, *args, **kwargs):
        return self.example_to_feature(example, *args, **kwargs)

    def example_to_feature(self, example, *args, **kwargs):
        return {"label":example[0], "ids":example[1], "text":example[2]}

    def load_dataset(self, *args, **kwargs):
        self.train_dataset = [(random.randint(0, 1), [i]*(4+i%2), str("inp={0}".format(i))) for i in range(102)]
        self.test_dataset = [(random.randint(0, 1), [i] * (4 + i % 2), str("inp={0}".format(i))) for i in range(101)]

class MeasureTracker():
    measures = {}
    keep_rescent = 50
    @classmethod
    def print_measures(cls):
        res = []
        for _k in list(MeasureTracker.measures.keys()):
            if len(MeasureTracker.measures[_k]) > MeasureTracker.keep_rescent:
                MeasureTracker.measures[_k] = MeasureTracker.measures[_k][-MeasureTracker.keep_rescent:]
            avg_measure = sum(MeasureTracker.measures[_k])/len(MeasureTracker.measures[_k])
            res.append(_k+": "+str(avg_measure))
        return "\n"+(", ".join(res))

    @classmethod
    def track_record(cls, *measures, track_names=None):
        if track_names is None:
            measure_num = len(measures)
            track_names = ["measure_{0}".format(i) for i in range(measure_num)]
        else:
            assert len(track_names)==len(measures)

        results = []
        for ms in measures:
            if isinstance(ms, torch.Tensor):
                if ms.shape==():
                    res_scalar = float(ms.clone().detach().to("cpu"))
                else:
                    res_scalar = float(ms.mean().clone().detach().to("cpu"))
            elif isinstance(ms, float):
                res_scalar = ms
            else:
                try:
                    res_scalar = float(ms)
                except Exception as ex:
                    print(ex)
                    raise Exception("only support track of scalar tensor of float")
            results.append(res_scalar)

        for nid, res in enumerate(results):
            nm = track_names[nid]
            if nm not in MeasureTracker.measures:
                MeasureTracker.measures[nm] = []
            MeasureTracker.measures[nm].append(res)

"""
def cosine_decay_lr(_i, _imax):
    max_lr = 5e-5
    min_lr = 1e-6
    num_oscillations = 6
    if _i < 10:
        lr = min_lr + (max_lr-min_lr) / 10.0 * _i
    else:
        progress = _i / _imax
        decayed_cosine = (0.5 * (1+np.cos(num_oscillations*progress*np.pi))*(max_lr-min_lr)+min_lr) * (1-progress)
        lr = max(decayed_cosine, min_lr)
    return lr
# with adjust_lr_optimizer(optimizer, cosine_decay_lr, iter_t, iter_max) as scheduled_optimizer:
#   ...
"""
@contextmanager
def adjust_lr_optimizer(optimizer, udf_func, *args, **kwargs):
    new_lr = udf_func(*args, **kwargs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    yield optimizer