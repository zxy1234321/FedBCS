config = {
    "data": {
        "dataset": "tnbc",
        "dataset_path": "./data",
        "num_classes": 2,
    },
    "eval_checkpoint": "model_best.pth",
    "gpu": 0,
    "logging": {
        "group": "FedBCS",
        "level": "Debug",
        "log_comment": "FedBCS",
        "log_dir": "./logs",
        "mode": "online",
        "notes": "FedBCS",
        "project": "Cell-Segmentation",
    },
    "loss": {
        "dice": {"loss_fn": "dice_loss", "weight": 1},
        "ce": {"loss_fn": "CrossEntropyLoss", "weight": 1}
    },
    "model": {
        "backbone": "fsr"
    },
    "random_seed": 42,
    "training": {
        "batch_size": 6,
        "epochs": 400,
        "local_epoch": 1,
        "optimizer": "SGD",
        "optimizer_hyperparameter": {
            "lr": 0.01,
            "weight_decay": 0.0001
        },
    },
}
