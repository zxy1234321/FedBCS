import datetime
import os
import setproctitle
import torch.multiprocessing
from args import add_experiment_args
os.environ['PYTHONHASHSEED'] = str(42)
args = add_experiment_args()
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
from fdmodels import get_model
warnings.filterwarnings("ignore")

from dataset import get_prive_dataset
from fd_trainer.training import train


def generate_log_filename(args):
    fold_str = f"_fold{args.fold}" if args.fold > 0 else ""
    log_filename = '{}_{}_{}_{}_{}_{}_{}_{}{}'.format(args.model, args.parti_num,
        args.dataset, args.communication_epoch, args.local_epoch, args.averaing,
             datetime.datetime.now().strftime('%m%d-%H%M%S'), args.description, fold_str)
    return log_filename


def main(args=args):

    priv_dataset = get_prive_dataset(args)
    args.parti_num = len(priv_dataset.DOMAINS_LIST)
    print('parti_num:', args.parti_num)

    if args.fold > 0:
        print(f'===== 5-Fold Cross Validation: Fold {args.fold} =====')

    backbones_list = priv_dataset.get_backbone(args.parti_num, [args.arch] * args.parti_num, args, args.channel_ratio, args.mode)

    log_filename = generate_log_filename(args)
    logs_parent_dir = os.path.join(args.log_root, args.dataset)
    log_filepath = os.path.join(logs_parent_dir, log_filename)
    os.makedirs(log_filepath, exist_ok=True)
    args.log_dir = log_filepath
    model = get_model(backbones_list, args, priv_dataset.get_transform())

    print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))
    setproctitle.setproctitle('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

    train(model, priv_dataset, args, log_filepath)


if __name__ == '__main__':
    main()
