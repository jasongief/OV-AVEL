import argparse

parser = argparse.ArgumentParser(description="A project implemented in pyTorch")

# =========================== Learning Configs ============================
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--n_epoch', type=int, default=50)
# parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('--train_batch_size', type=int)
parser.add_argument('--val_batch_size', type=int)
parser.add_argument('--test_batch_size', type=int)

parser.add_argument('--seed', type=int)
parser.add_argument('--lr', type=float, default=0.0001)
# parser.add_argument('--gpu', type=str)
parser.add_argument('--snapshot_pref', type=str)
# parser.add_argument('--split', type=str)
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--clip_gradient', type=float, default=0.8)

parser.add_argument('--val_data_type', type=str, default="total")
parser.add_argument('--test_data_type', type=str, default="total")
parser.add_argument('--test_strategy_type', type=str, default="v0")

# parser.add_argument('--spatial_attn_av_both_firstK', action='store_true')
# parser.add_argument('--spatial_attn_av_both_lastK', action='store_true')
# parser.add_argument('--spatial_attn_K', type=int)
# parser.add_argument('--spatial_attn_av_evenly_firstK', action='store_true')
# parser.add_argument('--spatial_attn_av_evenly_lastK', action='store_true')

# parser.add_argument('--spatial_attn_av_fixed_blkids', action='store_true')
# parser.add_argument("--sattn_av_fixed_audio_lids", default=[], nargs='+', type=int, help='add spatial attention in which audio bocks: [0, 1, 2, 3]')
# parser.add_argument("--sattn_av_fixed_visual_lids", default=[], nargs='+', type=int, help='add spatial attention in which audio bocks: [0, 1, 2, 3]')




parser.add_argument('--start_epoch', type=int)
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
parser.add_argument('--weight_decay', '--wd', type=float,
                    metavar='W', help='weight decay (default: 5e-4)')


# =========================== Model Configs ==============================

# =========================== Display Configs ============================
parser.add_argument('--print_iter_freq', type=int, default=5)
parser.add_argument('--val_print_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--eval_freq', type=int, default=1)


