from argparse import ArgumentParser
from dataset import Priv_NAMES as DATASET_NAMES
from utils.best_args import best_args
from utils.conf import set_random_seed


def add_experiment_args():
    parser = ArgumentParser(description='Federated Learning')
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--device_id', type=int, default=1)
    parser.add_argument('--communication_epoch', type=int, default=400)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='fedbcs')
    parser.add_argument('--arch', type=str, default='fsr')
    parser.add_argument('--structure', type=str, default='homogeneity')
    parser.add_argument('--dataset', type=str, default='tnbc',
                        choices=['tnbc', 'mri'])
    parser.add_argument('--source_key', type=str, default='zhy')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--pri_aug', type=str, default='weak')
    parser.add_argument('--online_ratio', type=float, default=1)
    parser.add_argument('--learning_decay', type=bool, default=False)
    parser.add_argument('--averaing', type=str, default='weight', choices=['weight', 'equal'])

    parser.add_argument('--infoNCET', type=float, default=0.005)
    parser.add_argument('--T', type=float, default=0.05)
    parser.add_argument('--weight', type=int, default=1)

    parser.add_argument('--reserv_ratio', type=float, default=0.1)
    parser.add_argument('--save_best_model', type=bool, default=True)
    parser.add_argument('--channel_ratio', type=float, default=0)
    parser.add_argument('--mode', type=str, default='ori')
    parser.add_argument('--txt_log', action='store_true', default=True)
    parser.add_argument('--backbone_type', type=str, choices=["fastvit_sa36", 'fastvit_t8', "fastvit_sa24", "fastvit_sa12", "fastvit_s12",
                                 "fastvit_t12", "fastvit_ma36"], default="fastvit_t12")
    parser.add_argument('--wHEAL', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--use_mask', type=int, default=1)
    parser.add_argument('--use_dynamic_weight', type=int, default=1)
    parser.add_argument('--alp', type=float, default=1)
    parser.add_argument('--layer_config', type=str, default='0,1,2,3')

    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--log_root', type=str, default='logs')

    args = parser.parse_args()
    args.num_nuclei_classes = 6
    args.num_tissue_classes = 19
    args.drop_rate = 0

    if args.model in best_args.get(args.dataset, {}):
        best = best_args[args.dataset][args.model]
        print(f" Using the best args for {args.model}")
    elif 'fedbcs' in best_args.get(args.dataset, {}):
        best = best_args[args.dataset]['fedbcs']
        print(f" Using the best args for fedbcs")
    else:
        best = {}

    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)
    return args
