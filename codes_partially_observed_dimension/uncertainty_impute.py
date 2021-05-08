import os
import sys
import pdb
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import torch
import numpy as np
from pprint import pformat, pprint

from datasets import get_dataset
from utils.hparams import HParams
from utils.test_utils import run_imputation
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from transformers import Encoder
from exnode import ExnodeEncoder

import torch.nn as nn
import torch.optim as optim

ckpt_path_dict = dict()
if True:
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/gas_all_rand_miss/ckpt/'
        ckpt_dir = os.path.join(ckpt_root_dir, 'epoch_70.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/air_quality_all_rand_miss/ckpt/'
        ckpt_dir = os.path.join(ckpt_root_dir, 'epoch_470.pt')
    if True:
        main_ckpt_dir = '/media/leo/work/projects/asyn_time_transformer/log/air_quality_0.8_miss/ckpt/epoch_160.pt'
        uncertrainty_ckpt_dir = '/media/leo/work/projects/asyn_time_transformer/log/air_quality_all_rand_miss_confidence/ckpt/epoch_777.pt'
if False:
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/nfl_per_gap_8_layer_12_heads_no_dropout_unnormalize_max_level_4_mse/ckpt/'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_690.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1398.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1685.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2653.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_3079.pt')
    if True:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/nfl_per_gap_8_layer_12_heads_no_dropout_unnormalize_max_level_4_confidence/ckpt/'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_692.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1398.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1676.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2658.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_3098.pt')
if False:
    if True:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/mujoco_per_gap_8_layer_12_heads_no_dropout_unnormalize_max_level_4_mse_0.975_miss_att_d_128_model_d_1024/ckpt/'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_693.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1363.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1681.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2666.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_3037.pt')

if False:
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/per_gap_8_layer_12_heads_no_dropout_unnormalize_max_level_4_mse_0.975_miss_correct_test/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_592.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1188.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1790.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2580.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_6489.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/per_gap_8_layer_8_heads_no_dropout_unnormalize_max_level_4_mse_0.975_miss/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_592.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1173.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1771.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2621.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_6653.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/per_gap_8_layer_8_heads_no_dropout_unnormalize_max_level_4_confidence_0.975_miss/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_592.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1188.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1793.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2637.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_6651.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/per_gap_8_layer_12_heads_no_dropout_unnormalize_max_level_4_mse_0.975_miss_gp/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_592.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1188.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1790.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2580.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_6656.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/per_gap_8_layer_12_heads_no_dropout_unnormalize_max_level_4_mse_0.975_miss_att_d_128_model_d_1024/ckpt/'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_248.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_435.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_744.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_1345.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_6645.pt')
    if True:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/per_gap_8_layer_12_heads_no_dropout_unnormalize_max_level_4_mse_0.975_miss_att_d_128_model_d_1024_gp/ckpt/'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_248.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_435.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_744.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_1345.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_6650.pt')
if False:
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/traffic_per_gap_8_layer_8_heads_dropout_0.0_unnormalize_max_level_4_mse/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_580.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1041.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1222.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2483.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_4199.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/traffic_per_gap_4_layer_4_heads_dropout_0.0_unnormalize_max_level_4_mse/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_584.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1185.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1670.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2575.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_4090.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/traffic_per_gap_4_layer_4_heads_dropout_0.4_unnormalize_max_level_4_mse/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_587.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1103.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1572.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2651.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_4334.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/traffic_per_gap_8_layer_12_heads_dropout_0.2_unnormalize_max_level_4_mse/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_722.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1552.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1922.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2461.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_4566.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/traffic_per_gap_8_layer_12_heads_dropout_0.0_unnormalize_max_level_4_mse/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_714.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1477.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_1933.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2490.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_4042.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/traffic_per_gap_8_layer_12_heads_128_att_dims_1024_d_model_dropout_0.0_unnormalize_max_level_4_mse/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_972.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1715.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_2163.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2666.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_3298.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/traffic_per_gap_8_layer_12_heads_128_att_dims_1024_d_model_dropout_0.0_unnormalize_max_level_4_mse/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_972.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1715.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_2163.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2666.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_3298.pt')
    if False:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/traffic_per_gap_8_layer_12_heads_128_att_dims_1024_d_model_dropout_0.0_unnormalize_max_level_4_mse_independent/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_966.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1721.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_2116.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2634.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_5067.pt')
    if True:
        ckpt_root_dir = '/playpen1/scribble/ssy/log/traffic_per_gap_8_layer_12_heads_128_att_dims_1024_d_model_dropout_0.0_unnormalize_max_level_4_mse_knn/ckpt'
        ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'epoch_921.pt')
        ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'epoch_1287.pt')
        ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'epoch_2110.pt')
        ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'epoch_2307.pt')
        ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'epoch_4126.pt')
main_ckpt = torch.load(main_ckpt_dir)
certrainty_ckpt = torch.load(uncertrainty_ckpt_dir)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--num_missing', type=int)
parser.add_argument('--save_fig', type=int)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


# creat exp dir
if not os.path.exists(params.exp_dir):
    os.mkdir(params.exp_dir)
if not os.path.exists(os.path.join(params.exp_dir, 'gen')):
    os.mkdir(os.path.join(params.exp_dir, 'gen'))
if not os.path.exists(os.path.join(params.exp_dir, 'ckpt')):
    os.mkdir(os.path.join(params.exp_dir, 'ckpt'))
if not os.path.exists(os.path.join(params.exp_dir, 'impute')):
    os.mkdir(os.path.join(params.exp_dir, 'impute'))


train_data, val_data, test_data = get_dataset(params.data_root, params.dataset, False)

train_mean = torch.mean(train_data, 0)
test_mean = torch.mean(test_data, 0)
main_model = eval(params.model_name)(
    max_time_scale=params.max_time_scale,
    time_enc_dim=params.time_enc_dim,
    time_dim=params.time_dim,
    expand_dim=params.expand_dim,
    mercer=params.mercer,
    n_layers=params.n_layers,
    n_head=params.n_heads,
    d_k=params.att_dims,
    d_v=params.att_dims,
    d_model=params.model_dims,
    d_inner=params.inner_dims,
    d_data=train_data.shape[-1],
    dropout=params.dropout,
    use_layer_norm=params.layer_norm,
    use_gap_encoding=params.use_gap_encoding,
    adapter=params.adapter,
    use_mask=params.att_mask,
    confidence=0
)
certainty_model = eval(params.model_name)(
    max_time_scale=params.max_time_scale,
    time_enc_dim=params.time_enc_dim,
    time_dim=params.time_dim,
    expand_dim=params.expand_dim,
    mercer=params.mercer,
    n_layers=params.n_layers,
    n_head=params.n_heads,
    d_k=params.att_dims,
    d_v=params.att_dims,
    d_model=params.model_dims,
    d_inner=params.inner_dims,
    d_data=train_data.shape[-1],
    dropout=params.dropout,
    use_layer_norm=params.layer_norm,
    use_gap_encoding=params.use_gap_encoding,
    adapter=params.adapter,
    use_mask=params.att_mask,
    confidence=1
)

main_model = nn.DataParallel(main_model).to(device)
certainty_model = nn.DataParallel(certainty_model).to(device)
print(main_model)
print("Start Imputation")
main_model.load_state_dict(main_ckpt)
certainty_model.load_state_dict(certrainty_ckpt)
loss = run_imputation(main_model, certainty_model, params.mode, val_data.repeat(10,1,1), args.num_missing, confidence=params.confidence , max_level=params.max_level, fig_path = os.path.join(params.exp_dir, 'impute'), 
                    save_all_imgs=args.save_fig, dataset=params.dataset, train_mean=train_mean, test_mean=test_mean, gp=params.gp)

output_str = 'Testing_Loss: %4f' % (loss)
print(output_str)