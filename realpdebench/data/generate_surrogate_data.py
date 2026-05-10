import torch
import numpy as np
import yaml
import h5py
import re
import os

from fno import FNO3d

from data.flame_surrogate_dataset import SurrogateDataset
from data.data_normalizer import IdentityNormalizer, GaussianNormalizer, RangeNormalizer

device = 'cuda:0'
os.makedirs('/wutailin/real_benchmark/combustion/surrogate/', exist_ok=True)

config_path = 'real_benchmark/configs/single_injector/surrogate_model/fno.yaml'
with open(config_path, 'r') as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)

step = 10
batch_size = 50
sub_s = 1

normalizer_dataset = SurrogateDataset(dataset_name=kwargs['dataset_name'], dataset_root=kwargs['dataset_root'], mode='train', step=step)
data_normalizer = GaussianNormalizer(normalizer_dataset, device=device, is_save=False)

model = FNO3d(
    modes1=4, 
    modes2=16, 
    modes3=16, 
    n_layers=4,
    width=64, 
    shape_in=(step, 128, 128, 17), 
    shape_out=(step, 128, 128, 1),
    ).to(device)
cp_path = 'real_benchmark/results/fno/surrogate_fno_single_injector/2025-09-13_15-08-31/model_0700.pth' 

cp = torch.load(cp_path, map_location=device)
model.load_state_dict(cp['model_state_dict'])
model.eval()

for file_name in os.listdir('/wutailin/real_benchmark/combustion/numerical'):
    print(file_name)

    data_path = '/wutailin/real_benchmark/combustion/numerical/' + file_name
    with h5py.File(data_path, 'r') as hf:
        field_names = hf['channel_names'][:]
        traj_numerical = hf['measured_data'][:]

        time = hf['time'][:] 
        x = hf['x'][:]
        y = hf['y'][:]

    data_path = '/wutailin/real_benchmark/combustion/real/' + file_name
    with h5py.File(data_path, 'r') as hf:
        trajectory = hf['trajectory'][:]

    pred_list = []
    match = re.match(r'(\d+)NH3_(\d+\.?\d*).h5', file_name)
    gas_ratio = int(match.group(1)) 
    equivalence_ratio = float(match.group(2))
    print(gas_ratio, equivalence_ratio)
    for i in range(0, traj_numerical.shape[0]-1, batch_size*step):
        with torch.no_grad():
            traj_numerical_batch = torch.tensor(traj_numerical[i:i+batch_size*step, ::sub_s, ::sub_s], dtype=torch.float).reshape(-1,step,128,128,15).to(device)
            gas_ratio_batch = torch.ones_like(traj_numerical_batch[..., [0]]) * gas_ratio
            equivalence_ratio_batch = torch.ones_like(traj_numerical_batch[..., [0]]) * equivalence_ratio
            traj_numerical_batch = torch.cat([traj_numerical_batch, gas_ratio_batch, equivalence_ratio_batch], dim=-1)

            traj_numerical_batch, _ = data_normalizer.preprocess(traj_numerical_batch, traj_numerical_batch)
            pred_traj_batch = model(traj_numerical_batch)
            _, pred_traj_batch = data_normalizer.postprocess(pred_traj_batch, pred_traj_batch)
            # print(pred_traj_batch.shape)
            pred_list.append(pred_traj_batch.reshape(-1, 128, 128).cpu().numpy())

    with torch.no_grad():
        traj_numerical_batch = torch.tensor(traj_numerical[-step:, ::sub_s, ::sub_s], dtype=torch.float).reshape(1,step,128,128,15).to(device)
        gas_ratio_batch = torch.ones_like(traj_numerical_batch[..., [0]]) * gas_ratio
        equivalence_ratio_batch = torch.ones_like(traj_numerical_batch[..., [0]]) * equivalence_ratio
        traj_numerical_batch = torch.cat([traj_numerical_batch, gas_ratio_batch, equivalence_ratio_batch], dim=-1)

        traj_numerical_batch, _ = data_normalizer.preprocess(traj_numerical_batch, traj_numerical_batch)
        pred_traj_batch = model(traj_numerical_batch)
        _, pred_traj_batch = data_normalizer.postprocess(pred_traj_batch, pred_traj_batch)
        # print(pred_traj_batch.shape)
        pred_list.append(pred_traj_batch.reshape(-1, 128, 128)[[-1]].cpu().numpy())

    pred_traj = np.concatenate(pred_list, axis=0)
    print(pred_traj.shape)

    save_path = '/wutailin/real_benchmark/combustion/surrogate/' + file_name
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('measured_data', data=pred_traj)
        f.create_dataset('time', data=time)
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)

    print(f"Saved {save_path}")
