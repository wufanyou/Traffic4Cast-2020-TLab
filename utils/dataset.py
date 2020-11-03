# Created by fw at 8/14/20

import torch
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import Dataset as _Dataset

# from typing import Union,List
import lmdb
import io
import os


def get_dataset(cfg, city, dataset_type):
    cfg = cfg.DATASET
    assert city.upper() in ["BERLIN", "ISTANBUL", "MOSCOW", "ALL"], "wrong city"
    Dataset: object = globals()[cfg.NAME]

    if city.upper() == "ALL":
        d = []
        for c in ["BERLIN", "ISTANBUL", "MOSCOW"]:
            d.append(Dataset(cfg, c, dataset_type))
        dataset = torch.utils.data.ConcatDataset(d)
    else:
        dataset = Dataset(cfg, city, dataset_type)

    return dataset


# 2019-01-01 TUESDAY
def _get_weekday_feats(index):
    dayofyear = index // 288 + 1
    weekday = np.zeros([7, 495, 436], dtype=np.float32)
    weekday[(dayofyear + 1) % 7] = 1
    return weekday


def _get_time_feats(index):
    index = index % 288
    theta = index / 287 * 2 * np.pi
    time = np.zeros([2, 495, 436], dtype=np.float32)
    time[0] = np.cos(theta)
    time[1] = np.sin(theta)
    return time


# map to [0,255]
def _get_weekday_feats_v2(index) -> np.array:
    dayofyear = index // 288 + 1
    weekday = np.zeros([7, 495, 436], dtype=np.float32)
    weekday[(dayofyear + 1) % 7] = 255
    return weekday


# map to [0,255]
def _get_time_feats_v2(index) -> np.array:
    index = index % 288
    theta = index / 287 * 2 * np.pi
    time = np.zeros([2, 495, 436], dtype=np.float32)
    time[0] = (np.cos(theta) + 1) / 2 * 255
    time[1] = (np.sin(theta) + 1) / 2 * 255
    return time


class PretrainDataset(_Dataset):
    def __init__(self, cfg, city="berlin", dataset_type="train"):
        self.city = city.upper()
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.sample = self._sample(dataset_type)
        self.env = None
        self.transform_env = None

    # TODO
    def __len__(self):
        return len(self.sample)

    def _sample(self, dataset_type):
        assert dataset_type in ["train", "valid"], "wrong dataset type"
        if dataset_type == "train":
            return range(105120)
        if dataset_type == "valid":
            return np.random.choice(range(105120), 1024)

    # TODO
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        # print(idx)
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x = np.concatenate(x)
        y = [self._get_item(start_idx + i) for i in [12, 13, 14, 17, 20, 23]]
        y = np.concatenate(y)
        extra = np.concatenate(
            [_get_time_feats_v2(start_idx), _get_weekday_feats_v2(start_idx)]
        )

        return {"x": x, "y": y, "extra": extra}

    def _get_item(self, idx):
        idx = str(idx).encode("ascii")
        try:
            with self.env.begin() as txn:
                data = txn.get(idx)
            data = np.load(io.BytesIO(data))
            x = np.zeros(495 * 436 * 3, dtype=np.uint8)
            x[data["x"]] = data["y"]
            x = x.reshape([495, 436, 3])
            x = np.moveaxis(x, -1, 0)
        except:
            x = np.zeros([3, 495, 436], dtype=np.uint8)
        return x


class BaseDataset(_Dataset):
    def __init__(self, cfg, city="berlin", dataset_type="train"):
        self.city = city.upper()
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.sample = self._sample(dataset_type)
        self.env = None
        self.transform_env = None

    # TODO
    def __len__(self):
        return len(self.sample)

    def _sample(self, dataset_type):
        assert dataset_type in ["train", "valid", "test"], "wrong dataset type"
        self.valid_index = np.load(self.cfg.VALID_INDEX)["index"]
        self.test_index = np.load(self.cfg.TEST_INDEX)["index"]
        self.valid_and_text_index = np.append(self.test_index, self.valid_index)
        self.valid_and_text_index.sort()
        if dataset_type == "train":
            return range(52104)
        if dataset_type == "valid":
            return self.valid_index
        if dataset_type == "test":
            return self.test_index

    # TODO
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        # print(idx)
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x = np.concatenate(x)
        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            return {"x": x, "y": y}
        else:
            return {"x": x}

    def _get_item(self, idx):
        idx = str(idx).encode("ascii")
        try:
            with self.env.begin() as txn:
                data = txn.get(idx)
            data = np.load(io.BytesIO(data))
            x = np.zeros(495 * 436 * 9, dtype=np.uint8)
            x[data["x"]] = data["y"]
            x = x.reshape([495, 436, 9])
            x = np.moveaxis(x, -1, 0)
        except:
            x = np.zeros([9, 495, 436], dtype=np.uint8)
        return x

    def sample_by_month(self, month):
        if type(month) is int:
            month = [month]
        sample = []
        one_day = pd.to_datetime("2019-01-02") - pd.to_datetime("2019-01-01")
        start_date = pd.to_datetime("2019-01-01")
        for i in self.sample:
            j = start_date + (i // 288) * one_day
            j = j.month
            if j in month:
                sample.append(i)
        sample = np.array(sample)
        self.sample = sample


# BaseDataset -> DatasetV2 : add extra feats
class DatasetV2(BaseDataset):
    def __init__(self, cfg, city="berlin", dataset_type="train"):
        super().__init__(cfg, city, dataset_type)
        static_feature = np.load(os.path.join(cfg.DATA_PATH, f"{self.city}_static.npz"))
        self.static_feature = static_feature["array"]

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x = np.concatenate(x)
        extra = np.concatenate(
            [_get_time_feats(start_idx), _get_weekday_feats(start_idx)]
        )
        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            return {"x": x, "y": y, "extra": extra}
        else:
            return {"x": x, "extra": extra}


def _augment_flip(x, y, prob=0.5):
    if np.random.rand() < prob:
        x = np.flip(x, 1)
        y = np.flip(y, 1)
    if np.random.rand() < prob:
        x = np.flip(x, 2)
        y = np.flip(y, 2)
    x = x.copy()
    y = y.copy()
    return x, y


# DatasetV2 -> DatasetV3 : add flip augmentation
class DatasetV3(DatasetV2):
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x = np.concatenate(x)

        extra = np.concatenate(
            [_get_time_feats(start_idx), _get_weekday_feats(start_idx)]
        )
        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            if self.dataset_type == "train":
                x, y = _augment_flip(x, y, prob=0.25)
            return {"x": x, "y": y, "extra": extra}
        else:
            return {"x": x, "extra": extra}


# DatasetV4 -> DatasetV2 : update extra feats to [0,255]
class DatasetV4(DatasetV2):
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x = np.concatenate(x)
        extra = np.concatenate(
            [_get_time_feats_v2(start_idx), _get_weekday_feats_v2(start_idx)]
        )
        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            return {"x": x, "y": y, "extra": extra}
        else:
            return {"x": x, "extra": extra}


# DatasetV2 -> DatasetV5 : update extra feats to [0,255]
class DatasetV5(DatasetV2):
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x = np.concatenate(x)
        extra = np.concatenate(
            [_get_time_feats_v2(start_idx), _get_weekday_feats_v2(start_idx)]
        )
        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            return {"x": x, "y": y, "extra": extra}
        else:
            return {"x": x, "extra": extra}


__MIN_RANGE__ = 48


# DatasetV5 -> DatasetV6 : update more aggregated feats
# DatasetV6 is too slow
class DatasetV6(DatasetV5):
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x = np.concatenate(x)
        z = self._random_n_feats(idx)
        extra = np.concatenate(
            [_get_time_feats_v2(start_idx), _get_weekday_feats_v2(start_idx)]
        )
        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            return {"x": x, "y": y, "z": z, "extra": extra}
        else:
            return {"x": x, "z": z, "extra": extra}

    def _random_n_samples(self, dayofyear):
        if 1 <= dayofyear <= 181:
            idx_num = np.random.choice(range(1, 6), p=[0.02, 0.23, 0.5, 0.23, 0.02])
            idx = np.random.randint(0, 60, [5]) + np.arange(0, 288, 60)
            idx = np.clip(idx, 0, 276)
            idx += (dayofyear - 1) * 288
            idx = np.random.choice(idx, idx_num)
            idx.sort()
            return idx
        else:
            idx = self.valid_and_text_index[
                self.valid_and_text_index >= (dayofyear - 1) * 288
            ]
            idx = idx[idx < dayofyear * 288]
            if len(idx) == 0:
                if dayofyear == 0:
                    return self._random_n_samples(2)
                elif dayofyear == 366:
                    return self._random_n_samples(365)
                else:
                    return self._random_n_samples(dayofyear - 1)
            elif 1 <= len(idx) <= 5:
                return idx
            else:
                idx_num = np.random.choice(range(1, 6), p=[0.02, 0.23, 0.5, 0.23, 0.02])
                idx = np.random.randint(0, 60, [5]) + np.arange(0, 288, 60)
                idx = np.clip(idx, 0, 276)
                idx += (dayofyear - 1) * 288
                idx = np.random.choice(idx, idx_num)
                idx.sort()
                return idx

    def _random_n_feats(self, idx, days=[-1, 1]):
        dayofyear = idx // 288 + 1
        all = []
        mask = []
        for d in days:
            idx = self._random_n_samples(dayofyear - d)
            value = [self._get_item(j + i)[:-1] for i in idx for j in range(12)]
            value = np.stack(value)  # [12*[1,5],8,495,436]
            value = value.reshape(-1, 12, 8, 495, 436)
            mask.append(len(value))
            zeros = np.zeros(5, 12, 8, 495, 436, dtype=np.uint8)
            zeros[: len(value)] = value
            all.append(zeros)
        mask = np.array(mask)
        all = np.stack(all)
        return {"value": all, "mask": mask}


# DatasetV6 -> DatasetV7 : pre-compute aggregated feats
class DatasetV7(DatasetV6):
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x.append(self._random_n_feats(start_idx, self.cfg.DAYS))
        x = np.concatenate(x)
        extra = np.concatenate(
            [_get_time_feats_v2(start_idx), _get_weekday_feats_v2(start_idx)]
        )
        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            return {"x": x, "y": y, "extra": extra}
        else:
            return {"x": x, "extra": extra}

    def _random_n_feats(self, idx, days=[-7, -3, -2, -1, 1, 2, 3, 7]):
        dayofyear = idx // 288 + 1
        array = np.concatenate([self._get_day_item(dayofyear + d)[:-1] for d in days])
        return array

    def _get_day_item(self, dayofyear):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        dayofyear = max(1, dayofyear)
        dayofyear = min(dayofyear, 365)
        # print(dayofyear)
        idx = np.random.randint(50) if dayofyear <= 181 else 0
        idx = f"dayofyear/{dayofyear}/{idx}".encode("ascii")
        # print(idx)
        with self.env.begin() as txn:
            data = txn.get(idx)
        data = np.load(io.BytesIO(data))
        x = np.zeros(9 * 495 * 436, dtype=np.uint8)
        x[data["x"]] = data["y"]
        x = x.reshape([9, 495, 436])
        return x


# DatasetV7 -> DatasetV8 : add holiday feats
class DatasetV8(DatasetV7):
    def __init__(self, cfg, city="berlin", dataset_type="train"):
        super().__init__(cfg, city, dataset_type)
        self.holidays = joblib.load(cfg.HOLIDAYS)

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x.append(self._random_n_feats(start_idx, self.cfg.DAYS))
        x = np.concatenate(x)
        extra = np.concatenate(
            [
                _get_time_feats_v2(start_idx),
                _get_weekday_feats_v2(start_idx),
                self._get_holiday(start_idx),
            ]
        )

        result = {"x": x, "extra": extra}

        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            result["y"] = y

        if self.cfg.USE_POS_WEIGHT:
            pos_weight = np.array(self.cfg.POS_WEIGHT)
            result["pos_weight"] = pos_weight

        return result

    def _get_holiday(self, idx):
        value = self.holidays[self.city.upper()][idx // 288] * 255
        array = np.zeros([1, 495, 436], dtype=np.float32)
        if value != 0:
            array[:] = value
        return array


# DatasetV8 -> DatasetV9 : map input by gauss rank
# too slow
class DatasetV9(DatasetV8):
    def __init__(self, cfg, city="berlin", dataset_type="train"):
        super().__init__(cfg, city, dataset_type)
        self.transform_map = np.load(
            os.path.join(cfg.DATA_PATH, f"{city.upper()}_value_map.npz")
        )["array"]

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self._random_n_feats(start_idx, self.cfg.DAYS))
        x = np.concatenate(x)
        extra = np.concatenate(
            [
                self.static_feature,
                _get_time_feats_v2(start_idx),
                _get_weekday_feats_v2(start_idx),
                self._get_holiday(start_idx),
            ]
        )

        result = {"x": x, "extra": extra}

        if self.dataset_type != "test":
            y = [
                self._get_item(start_idx + i, transform=False)[:-1]
                for i in [12, 13, 14, 17, 20, 23]
            ]
            y = np.concatenate(y)
            result["y"] = y

        if self.cfg.USE_POS_WEIGHT:
            pos_weight = np.array(self.cfg.POS_WEIGHT)
            result["pos_weight"] = pos_weight

        return result

    def _get_item(self, idx, transform=True):
        if transform:
            idx = str(idx).encode("ascii")
            try:
                with self.env.begin() as txn:
                    data = txn.get(idx)
                data = np.load(io.BytesIO(data))
                x = np.zeros(495 * 436 * 9, dtype=np.uint8)
                x[data["x"]] = data["y"]
                x = x.reshape([495, 436, 9])
                x = np.moveaxis(x, -1, 0)
                new_x = np.zeros([9, 495, 436], dtype=np.float32)
                for i in range(9):
                    new_x[i] = self.transform_map[i][x[i]]
            except:
                new_x = np.zeros([9, 495, 436], dtype=np.float32)
            return new_x
        else:
            idx = str(idx).encode("ascii")
            try:
                with self.env.begin() as txn:
                    data = txn.get(idx)
                data = np.load(io.BytesIO(data))
                x = np.zeros(495 * 436 * 9, dtype=np.uint8)
                x[data["x"]] = data["y"]
                x = x.reshape([495, 436, 9])
                x = np.moveaxis(x, -1, 0)
            except:
                x = np.zeros([9, 495, 436], dtype=np.uint8)
            return x

    def _get_day_item(self, dayofyear):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )
        dayofyear = max(1, dayofyear)
        dayofyear = min(dayofyear, 365)
        # print(dayofyear)
        idx = np.random.randint(50) if dayofyear <= 181 else 0
        idx = f"dayofyear/{dayofyear}/{idx}".encode("ascii")
        # print(idx)
        with self.env.begin() as txn:
            data = txn.get(idx)
        data = np.load(io.BytesIO(data))
        x = np.zeros(9 * 495 * 436, dtype=np.uint8)
        x[data["x"]] = data["y"]
        x = x.reshape([9, 495, 436])
        new_x = np.zeros([9, 495, 436], dtype=np.float32)
        for i in range(9):
            new_x[i] = self.transform_map[i][x[i]]
        return new_x


# DatasetV8 -> DatasetV10 : map input by gauss rank, the transformation is pre-computed.
class DatasetV10(DatasetV8):
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )

        if self.transform_env is None:
            self.transform_env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, f"{self.city}_GK"), readonly=True
            )

        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i, True) for i in range(12)]
        x.append(self._random_n_feats(start_idx, self.cfg.DAYS))
        x = np.concatenate(x)
        extra = np.concatenate(
            [
                self.static_feature,
                _get_time_feats_v2(start_idx),
                _get_weekday_feats_v2(start_idx),
                self._get_holiday(start_idx),
            ]
        )

        result = {"x": x, "extra": extra}

        if self.dataset_type != "test":
            y = [
                self._get_item(start_idx + i, transform=False)[:-1]
                for i in [12, 13, 14, 17, 20, 23]
            ]
            y = np.concatenate(y)
            result["y"] = y

        if self.cfg.USE_POS_WEIGHT:
            pos_weight = np.array(self.cfg.POS_WEIGHT)
            result["pos_weight"] = pos_weight

        return result

    def _get_item(self, idx, transform=True):
        if transform:
            env = self.transform_env
        else:
            env = self.env
        idx = str(idx).encode("ascii")
        try:
            with env.begin() as txn:
                data = txn.get(idx)
            data = np.load(io.BytesIO(data))
            x = np.zeros(495 * 436 * 9, dtype=np.float32 if transform else np.uint8)
            x[data["x"]] = data["y"]
            x = x.reshape([495, 436, 9])
            x = np.moveaxis(x, -1, 0)
        except:
            x = np.zeros([9, 495, 436], dtype=np.float32 if transform else np.uint8)
        return x


# DatasetV8 -> DatasetV11 : include valid in training
class DatasetV11(DatasetV8):
    def _sample(self, dataset_type):
        assert dataset_type in ["train", "valid", "test"], "wrong dataset type"
        self.valid_index = np.load(self.cfg.VALID_INDEX)["index"]
        self.test_index = np.load(self.cfg.TEST_INDEX)["index"]
        self.valid_and_text_index = np.append(self.test_index, self.valid_index)
        self.valid_and_text_index.sort()

        if dataset_type == "train":
            sample = np.append(np.arange(52104), self.valid_index)
            sample.sort()
            return sample

        if dataset_type == "valid":
            return self.valid_index
        if dataset_type == "test":
            return self.test_index


# DatasetV8 -> DatasetV12 : include geo code embed
class DatasetV12(DatasetV8):
    def __init__(self, cfg, city="berlin", dataset_type="train"):
        super().__init__(cfg, city, dataset_type)
        self._create_geo_embed()

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )

        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x.append(self._random_n_feats(start_idx, self.cfg.DAYS))
        x = np.concatenate(x)
        extra = np.concatenate(
            [
                _get_time_feats_v2(start_idx),
                _get_weekday_feats_v2(start_idx),
                self._get_holiday(start_idx),
            ]
        )

        result = {"x": x, "extra": extra, "geo": self.geo_array}

        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            result["y"] = y

        if self.cfg.USE_POS_WEIGHT:
            pos_weight = np.array(self.cfg.POS_WEIGHT)
            result["pos_weight"] = pos_weight

        return result

    def _create_geo_embed(self):
        # scale = self.cfg.EMBED_SCALE
        self.geo_array = np.arange(495 * 436).reshape(495, 436).astype(np.long)


# DatasetV12 -> DatasetV13 : include valid in training
class DatasetV13(DatasetV12):
    def _sample(self, dataset_type):
        assert dataset_type in ["train", "valid", "test"], "wrong dataset type"
        self.valid_index = np.load(self.cfg.VALID_INDEX)["index"]
        self.test_index = np.load(self.cfg.TEST_INDEX)["index"]
        self.valid_and_text_index = np.append(self.test_index, self.valid_index)
        self.valid_and_text_index.sort()

        if dataset_type == "train":
            sample = np.append(np.arange(52104), self.valid_index)
            sample.sort()
            return sample

        if dataset_type == "valid":
            return self.valid_index
        if dataset_type == "test":
            return self.test_index


# DatasetV12 -> DatasetV15: add weather feats
class DatasetV15(DatasetV12):
    def __init__(self, cfg, city="berlin", dataset_type="train"):
        super().__init__(cfg, city, dataset_type)
        self.weather = np.load(
            os.path.join(cfg.WEATHER, f"{city.upper()}_weather.npz")
        )["array"]

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )

        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x.append(self._random_n_feats(start_idx, self.cfg.DAYS))
        x = np.concatenate(x)
        extra = np.concatenate(
            [
                _get_time_feats_v2(start_idx),
                _get_weekday_feats_v2(start_idx),
                self._get_holiday(start_idx),
                self._get_weather(start_idx),
            ]
        )

        result = {"x": x, "extra": extra, "geo": self.geo_array}

        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            result["y"] = y

        if self.cfg.USE_POS_WEIGHT:
            pos_weight = np.array(self.cfg.POS_WEIGHT)
            result["pos_weight"] = pos_weight

        return result

    def _get_weather(self, start_idx):
        x = self.weather[start_idx].reshape(-1).astype(np.float32)
        x = x.repeat(495 * 436).reshape(-1, 495, 436).astype(np.float32)
        return x


# DatasetV16 -> DatasetV15: add sun feats
class DatasetV16(DatasetV15):
    def __init__(self, cfg, city="berlin", dataset_type="train"):
        super().__init__(cfg, city, dataset_type)
        self.sun = np.load(os.path.join(cfg.SUN, f"{city.upper()}_sun.npz"))["array"]

    def _get_sun(self, start_idx):
        index = start_idx // 288
        x = self.sun[index].reshape(-1).astype(np.float32)
        x = x.repeat(495 * 436).reshape(-1, 495, 436).astype(np.float32)
        return x

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )

        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x.append(self._random_n_feats(start_idx, self.cfg.DAYS))
        x = np.concatenate(x)
        extra = np.concatenate(
            [
                _get_time_feats_v2(start_idx),
                _get_weekday_feats_v2(start_idx),
                self._get_holiday(start_idx),
                self._get_weather(start_idx),
                self._get_sun(start_idx),
            ]
        )

        result = {"x": x, "extra": extra, "geo": self.geo_array}

        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            result["y"] = y

        if self.cfg.USE_POS_WEIGHT:
            pos_weight = np.array(self.cfg.POS_WEIGHT)
            result["pos_weight"] = pos_weight

        return result


# DatasetV17 -> DatasetV15: add sun feats remove weather
class DatasetV17(DatasetV15):
    def __init__(self, cfg, city="berlin", dataset_type="train"):
        super().__init__(cfg, city, dataset_type)
        self.sun = np.load(os.path.join(cfg.SUN, f"{city.upper()}_sun.npz"))["array"]

    def _get_sun(self, start_idx):
        index = start_idx // 288
        x = self.sun[index].reshape(-1).astype(np.float32)
        x = x.repeat(495 * 436).reshape(-1, 495, 436).astype(np.float32)
        return x

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )

        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x.append(self._random_n_feats(start_idx, self.cfg.DAYS))
        x = np.concatenate(x)
        extra = np.concatenate(
            [
                _get_time_feats_v2(start_idx),
                _get_weekday_feats_v2(start_idx),
                self._get_holiday(start_idx),
                # self._get_weather(start_idx),
                self._get_sun(start_idx),
            ]
        )

        result = {"x": x, "extra": extra, "geo": self.geo_array}

        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            result["y"] = y

        if self.cfg.USE_POS_WEIGHT:
            pos_weight = np.array(self.cfg.POS_WEIGHT)
            result["pos_weight"] = pos_weight
        return result


# fix DST in Berlin
def _get_time_feats_v3(index, city) -> np.array:
    if city == "BERLIN" and index in (25656, 86136):
        index += 12
    index = index % 288
    theta = index / 287 * 2 * np.pi
    time = np.zeros([2, 495, 436], dtype=np.float32)
    time[0] = (np.cos(theta) + 1) / 2 * 255
    time[1] = (np.sin(theta) + 1) / 2 * 255
    return time


# fix DST in Berlin
def _get_weekday_feats_v3(index, city) -> np.array:
    if city == "BERLIN" and index in (25656, 86136):
        index += 12
    dayofyear = index // 288 + 1
    weekday = np.zeros([7, 495, 436], dtype=np.float32)
    weekday[(dayofyear + 1) % 7] = 255
    return weekday


# DatasetV18 -> DatasetV15: fix DST in Berlin
class DatasetV18(DatasetV16):
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )

        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x.append(self.static_feature)
        x.append(self._random_n_feats(start_idx, self.cfg.DAYS))
        x = np.concatenate(x)
        extra = np.concatenate(
            [
                _get_time_feats_v3(start_idx, self.city),
                _get_weekday_feats_v3(start_idx, self.city),
                self._get_holiday(start_idx),
                self._get_weather(start_idx),
                self._get_sun(start_idx),
            ]
        )

        result = {"x": x, "extra": extra, "geo": self.geo_array}

        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            result["y"] = y

        if self.cfg.USE_POS_WEIGHT:
            pos_weight = np.array(self.cfg.POS_WEIGHT)
            result["pos_weight"] = pos_weight
        return result


# DatasetV19 -> DatasetV18: include valid in train
class DatasetV19(DatasetV18):
    def _sample(self, dataset_type):
        assert dataset_type in ["train", "valid", "test"], "wrong dataset type"
        self.valid_index = np.load(self.cfg.VALID_INDEX)["index"]
        self.test_index = np.load(self.cfg.TEST_INDEX)["index"]
        self.valid_and_text_index = np.append(self.test_index, self.valid_index)
        self.valid_and_text_index.sort()

        if dataset_type == "train":
            sample = np.append(np.arange(52104), self.valid_index)
            sample.sort()
            return sample

        if dataset_type == "valid":
            return self.valid_index
        if dataset_type == "test":
            return self.test_index


# DatasetV18 -> DatasetV20: 3D input, include valid, no sun, no weather
class DatasetV20(DatasetV19):
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                os.path.join(self.cfg.DATA_PATH, self.city), readonly=True
            )

        start_idx = self.sample[idx]
        x = [self._get_item(start_idx + i) for i in range(12)]
        x = np.stack(x)
        x = x.swapaxes(0, 1)
        extra = np.concatenate(
            [
                self.static_feature,
                self._random_n_feats(start_idx, self.cfg.DAYS),
                _get_time_feats_v3(start_idx, self.city),
                _get_weekday_feats_v3(start_idx, self.city),
                self._get_holiday(start_idx),
                self._get_weather(start_idx),
                self._get_sun(start_idx),
            ]
        )

        result = {"x": x, "extra": extra, "geo": self.geo_array}

        if self.dataset_type != "test":
            y = [self._get_item(start_idx + i)[:-1] for i in [12, 13, 14, 17, 20, 23]]
            y = np.concatenate(y)
            result["y"] = y

        if self.cfg.USE_POS_WEIGHT:
            pos_weight = np.array(self.cfg.POS_WEIGHT)
            result["pos_weight"] = pos_weight
        return result
