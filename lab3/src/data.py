import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src import config
from src.tables import read_csv, write_csv

PART_DIRS = {
    "train": "seg_train/seg_train",
    "validation": "seg_test/seg_test",
    "test": "seg_test/seg_test",
}


def sorted_files(folder):
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() == ".jpg")


def build_split(data_dir):
    train_dir = data_dir / PART_DIRS["train"]
    test_dir = data_dir / PART_DIRS["validation"]
    rows = []

    rng = np.random.default_rng(0)
    for c in config.CLASSES:
        files = sorted_files(train_dir / c)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        for i in sorted(idx[:config.TRAIN_PER_CLASS]):
            rows.append((files[i], c, "train"))

    rng = np.random.default_rng(0)
    for c in config.CLASSES:
        files = sorted_files(test_dir / c)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        for i in sorted(idx[:cut]):
            rows.append((files[i], c, "validation"))
        for i in sorted(idx[cut:]):
            rows.append((files[i], c, "test"))

    return rows


def save_split(rows, path):
    write_csv(path, ["file", "label", "part"], rows)


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(config.MEAN, config.STD),
    ])


def eval_transform():
    return T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(config.MEAN, config.STD),
    ])


class SplitDataset(Dataset):
    def __init__(self, data_dir, split_path, part, transform):
        rows = [r for r in read_csv(split_path) if r["part"] == part]
        root = data_dir / PART_DIRS[part]
        self.files = [r["file"] for r in rows]
        self.labels = [config.CLASSES.index(r["label"]) for r in rows]
        self.paths = [root / r["label"] / r["file"] for r in rows]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def train_loader(data_dir, split_path):
    ds = SplitDataset(data_dir, split_path, "train", train_transform())
    return DataLoader(ds, batch_size=config.TRAIN_BATCH, shuffle=True,
                      drop_last=False, num_workers=0)


def eval_loader(data_dir, split_path, part):
    ds = SplitDataset(data_dir, split_path, part, eval_transform())
    return DataLoader(ds, batch_size=config.EVAL_BATCH, shuffle=False, num_workers=0)
