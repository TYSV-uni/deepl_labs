CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 160

EPOCHS = 8
TRAIN_BATCH = 32
EVAL_BATCH = 64
TRAIN_PER_CLASS = 200
WEIGHT_DECAY = 1e-4

REGIMES = {
    "scratch": {
        "pretrained": False,
        "trainable": None,
        "eval": [],
        "groups": [(None, 3e-3)],
    },
    "frozen": {
        "pretrained": True,
        "trainable": ["fc"],
        "eval": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
        "groups": [(["fc"], 3e-3)],
    },
    "unfreeze": {
        "pretrained": True,
        "trainable": ["layer4", "fc"],
        "eval": ["conv1", "bn1", "layer1", "layer2", "layer3"],
        "groups": [(["fc"], 3e-3), (["layer4"], 3e-4)],
    },
}

RUNS = [
    ("scratch", 1),
    ("frozen", 1), ("frozen", 2), ("frozen", 3),
    ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3),
]
