import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

exp_folder = Path('exp')
exp_name = str(max(int(name.name) for name in exp_folder.iterdir()) + 1)
writer = SummaryWriter(f'exp/{exp_name}')

pose_filename = 'traj.txt'
pose_format = 'kitti'
valset_interval = 10
logging.basicConfig(level=logging.DEBUG #设置日志输出格式
                    # ,filename="demo.log" #log日志输出的文件位置和文件名
                    # ,filemode="w" #文件的写入格式，w为重新写入文件，默认是追加
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )
logger = logging.getLogger()


def read_pose_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    if pose_format == 'kitti':
        pose = [([eval(num) for num in line.split()]) for line in lines]
        pose = np.array(pose, dtype=np.float32).reshape((-1, 12))
    return pose


class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.net1 = nn.Sequential(
                nn.Linear(1, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
        self.net2 = nn.Sequential(
                nn.Linear(257, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 12),
                )

    def forward(self, t):
        mid = self.net1(t)
        mid = torch.cat([mid, t], -1)
        return self.net2(mid)


device = 'cuda:0'
pose = read_pose_file(pose_filename)
n_frames = len(pose)
ts = np.linspace(0, n_frames - 1, n_frames)
pose, ts = torch.tensor(pose, dtype=torch.float32).to(device), torch.tensor(ts, dtype=torch.float32).to(device)
ts = ts.view(-1, 1)
train_idx = [i for i in range(n_frames) if i % valset_interval != 0]
val_idx = [i for i in range(n_frames) if i % valset_interval == 0]
n_train = len(train_idx)
n_val = len(val_idx)

logger.info(f"frame num: {n_frames}, train set: {n_train}, val set: {n_val}")
train_ts, val_ts = ts[train_idx], ts[val_idx]
train_pose, val_pose = pose[train_idx], pose[val_idx]
network = PoseNet().to(device)
optimizer = torch.optim.Adam(list(network.parameters()), lr=1e-3)
train_epoch = 60000
val_interval = 2000

bar = tqdm(range(train_epoch))
for epoch in bar:
    output = network(train_ts)
    optimizer.zero_grad()
    loss = F.mse_loss(output, train_pose, reduction='mean')
    loss.backward()
    optimizer.step()
    writer.add_scalar('train/loss', loss, global_step=epoch)

    bar.set_postfix_str(f'loss={loss}')
    if epoch % val_interval == 0 and epoch != 0:
        with torch.inference_mode(True):
            output = network(val_ts)
            loss = F.mse_loss(output, val_pose, reduction='mean')
            writer.add_scalar('val/loss', loss, global_step=epoch)
            print(f'validation loss = {loss}')

torch.save(network, "checkpoint.pt")
test_ts = ts
output = network(test_ts).cpu().numpy()

output_filename = 'out_traj.txt'
np.savetxt(output_filename, output)