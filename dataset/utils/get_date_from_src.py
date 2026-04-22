from dataset.utils.mydataset import FedDataset

DATA_SOURCES = {
    "example": {
        "tnbc": {
            "base_dir": "./data",
            "paths": {
                "tnbc": {
                    "train": "TNBC/train.txt",
                    "val": "TNBC/val.txt"
                },
                "kirc": {
                    "train": "KIRC/train.txt",
                    "val": "KIRC/val.txt"
                },
                "tcia": {
                    "train": "TCIA/train.txt",
                    "val": "TCIA/val.txt"
                },
                "crc": {
                    "train": "CRC/train.txt",
                    "val": "CRC/val.txt"
                },
            }
        },
    },
}


def get_datasets(using_list, source_key, name, train_transform=None, val_transform=None):
    print(f"Using source: {source_key}, dataset: {name}")
    source = DATA_SOURCES[source_key][name]
    base_dir = source["base_dir"]

    paths = source["paths"]

    datasets = {}
    for domain in using_list:
        if domain not in paths:
            raise ValueError(f"Unknown domain: {domain}")

        train_file = f"{base_dir}/{paths[domain]['train']}"
        val_file = f"{base_dir}/{paths[domain]['val']}"

        train_dataset = FedDataset(base_dir=base_dir, labeled_file=train_file, transform=train_transform)
        val_dataset = FedDataset(base_dir=base_dir, labeled_file=val_file, transform=val_transform)

        datasets[domain] = {"train": train_dataset, "val": val_dataset}

    return datasets


FOLD_PATHS = {
    "tnbc": {
        "base_dir": "./data",
        "domains": {
            "tnbc": "TNBC/5fold/TNBC",
            "kirc": "KIRC/5fold/KIRC",
            "tcia": "TCIA/5fold/TCIA",
            "crc": "CRC/5fold/CRC",
        }
    },
    "mri": {
        "base_dir": "./data/MRI",
        "domains": {
            "BIDMC": "BIDMC/5fold/BIDMC",
            "HK": "HK/5fold/HK",
            "I2CVB": "I2CVB/5fold/I2CVB",
            "ISBI": "ISBI/5fold/ISBI",
            "ISBI_1.5": "ISBI_1.5/5fold/ISBI_1.5",
            "UCL": "UCL/5fold/UCL",
        }
    }
}


def get_datasets_5fold(using_list, name, fold_num, train_transform=None, val_transform=None):
    if fold_num < 1 or fold_num > 5:
        raise ValueError(f"fold_num must be 1-5, got {fold_num}")

    if name not in FOLD_PATHS:
        raise ValueError(f"5-fold not supported for dataset: {name}")

    config = FOLD_PATHS[name]
    base_dir = config["base_dir"]
    domains = config["domains"]

    print(f"Loading 5-fold dataset: {name}, fold {fold_num}")
    print(f"Domains: {using_list}")

    datasets = {}
    for domain in using_list:
        if domain not in domains:
            raise ValueError(f"Unknown domain: {domain}")

        domain_prefix = domains[domain]
        train_file = f"{base_dir}/{domain_prefix}_fold{fold_num}_train.txt"
        val_file = f"{base_dir}/{domain_prefix}_fold{fold_num}_val.txt"

        print(f"  {domain}: train={train_file}, val={val_file}")

        train_dataset = FedDataset(base_dir=base_dir, labeled_file=train_file, transform=train_transform)
        val_dataset = FedDataset(base_dir=base_dir, labeled_file=val_file, transform=val_transform)

        datasets[domain] = {"train": train_dataset, "val": val_dataset}

    return datasets


def get_mri_datasets_5fold(using_list, fold_num):
    from dataset.utils.mri_dataset import Prostate

    if fold_num < 1 or fold_num > 5:
        raise ValueError(f"fold_num must be 1-5, got {fold_num}")

    config = FOLD_PATHS["mri"]
    base_dir = config["base_dir"]
    domains = config["domains"]

    print(f"Loading MRI 5-fold dataset: fold {fold_num}")
    print(f"Sites: {using_list}")

    train_datasets = []
    val_datasets = []

    for site in using_list:
        if site not in domains:
            raise ValueError(f"Unknown site: {site}")

        site_prefix = domains[site]
        train_file = f"{base_dir}/{site_prefix}_fold{fold_num}_train.txt"
        val_file = f"{base_dir}/{site_prefix}_fold{fold_num}_val.txt"

        print(f"  {site}: train={train_file}, val={val_file}")

        train_dataset = Prostate(site=site, base_path=base_dir, split='train',
                                  fold=fold_num, fold_file=train_file)
        val_dataset = Prostate(site=site, base_path=base_dir, split='test',
                                fold=fold_num, fold_file=val_file)

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    return train_datasets, val_datasets
