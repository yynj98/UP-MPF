import argparse
import torch
import numpy as np
from os.path import join
from pprint import pprint
import random

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=34, help="Random seed for initialization")
parser.add_argument("--cuda", type=str, default='0')
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

# directory settings
parser.add_argument("--data_dir", type=str, default='datasets')
parser.add_argument("--img_dir", type=str, default=None)
parser.add_argument("--out_dir", type=str, default='out')

# dataset settings
parser.add_argument("--dataset", type=str, default='t2015')
parser.add_argument("--train_file", type=str, default='train-few1.tsv')
parser.add_argument("--dev_file", type=str, default='dev-few1.tsv')

# model settings
parser.add_argument("--model_name", type=str, default='bert-base-uncased')
VISUAL_MODELS = ['nf_resnet50', 'resnet50', 'resnetv2_50x1_bitm']
parser.add_argument("--visual_model_name", type=str, choices=VISUAL_MODELS, default='nf_resnet50')

# template settings
parser.add_argument("--template", type=int, default=0)
parser.add_argument("--prompt_token", type=str, default='[unused1]')
parser.add_argument("--prompt_shape", type=str)
parser.add_argument("--lstm_dropout", type=float, default=0.0)
parser.add_argument("--img_token", type=str, default='[unused2]')
parser.add_argument("--pooling_scale", type=str, default='77')
parser.add_argument("--img_template_mode", type=str, default='default')

# training settings
parser.add_argument("--do_pretrain", action="store_true", default=False)
parser.add_argument("--pretrain_epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr_lm_model", type=float, default=0)
parser.add_argument("--lr_resnet", type=float, default=0)
parser.add_argument("--lr_visual_mlp", type=float, default=0)
parser.add_argument("--decay_rate", type=float, default=0.99)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--early_stop", type=int, default=20)

# loading settings
parser.add_argument("--load_visual_encoder", action="store_true", default=False)
parser.add_argument("--up_model_path", type=str, default=None)


args = parser.parse_args()
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()  # only used to set random seeds

args.data_dir = join(args.data_dir, args.dataset)
args.img_token_len = int(args.pooling_scale[0]) * int(args.pooling_scale[1])

if args.img_dir:
    args.no_img = False
else:
    args.no_img = True

if args.train_file == 'train.tsv':
    train_abbr = '[Full]'
elif '1' in args.train_file:
    train_abbr = '[s1]'
elif '2' in args.train_file:
    train_abbr = '[s2]'
else:
    train_abbr = '[s0]'
args.train_abbr = train_abbr

if args.dev_file == 'dev.tsv':
    dev_abbr = '[Full]'
elif '1' in args.dev_file:
    dev_abbr = '[d1]'
elif '2' in args.dev_file:
    dev_abbr = '[d2]'
else:
    dev_abbr = '[d0]'
args.dev_abbr = dev_abbr


def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


set_seed(args)
pprint(f'[#] args: \n{args}')
