import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct
import seaborn as sns
import argparse
from SpatialGAN import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--region_i", type=int, default=10, help="i of region index.")
parser.add_argument("--region_j", type=int, default=24, help="j of region index.")
#parser.add_argument("--test_demand", type=float, default=65., help="Desired demand of test region")
parser.add_argument("--adj_bar", type=float, default=0.47, help="adj bar")
parser.add_argument("--init_dim", type=int, default=100, help="Dimensionality of the latent code")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--Ghid', type=int, default=64, help='Hidden feature number of G.')

opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

# hyper parameters
region_width = 10
region_length = 10
city_size = 50

# get min max of x and y
x = np.loadtxt(open('./inflow_region.csv', "rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open('./label_region.csv', "rb"), delimiter=",", skiprows=0)
max_x = x.max()
max_y0 = y[:, 0].max()
max_y1 = y[:, 1].max()
max_y2 = y[:, 2].max()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

if not os.path.exists('./spatial_gan_test'):
    os.mkdir('./spatial_gan_test')


################################ True Value in Reality #################################
########################################################################################
#path1 = './speed_city.csv'
#speed = np.loadtxt(open(path1, "rb"), delimiter=",", skiprows=0).reshape(-1, city_size, city_size)

path2 = './inflow_city.csv'
inflow = np.loadtxt(open(path2, "rb"), delimiter=",", skiprows=0).reshape(-1, city_size, city_size)

path3 = './demand_city.csv'
demand = np.loadtxt(open(path3, "rb"), delimiter=",", skiprows=0).reshape(-1, city_size, city_size)

# select the test region
#spd = speed[:, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]
flow = inflow[:, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]
dmd = demand[:, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]

# real demand sum
dmd = np.sum(dmd, axis=1)
dmd = np.sum(dmd, axis=1)

# show the statistics of the region
'''
#################### speed #################################
# 1. real average demand
mean_dmd = np.mean(dmd)
print("Average of real demand: ", mean_dmd)

# 2. mean speed distribution
mean_spd = np.mean(spd, axis = 0)
vmax = mean_spd.max()    # max number of inflow in the region
print("max num in the real mean speed distribution: ", vmax)

# 3. heatmap of speed
sns.heatmap(mean_spd, annot=True, vmin=0, vmax=vmax)
label = 'Region latitude {}, longitude {}, travel demand {}'.format(region_i, region_j, int(mean_dmd))
plt.title(label)
plt.show()
'''
#################### inflow ###################
# 1. real average demand
mean_dmd = np.mean(dmd)
print('Average of real demand of adj bar ' + str(opt.adj_bar) + ' and region ' + str(opt.region_i) + '_' + str(opt.region_j) + ': ', mean_dmd)


selected_idx = np.where(((mean_dmd - 5) <= dmd) & (dmd <= (mean_dmd + 5)))[0]
dmd = dmd[selected_idx]
flow = flow[selected_idx]
print("Number of real maps: ", dmd.shape[0])

# 2. mean inflow distribution
mean_flow = np.mean(flow, axis=0)
vmax = mean_flow.max()    # max number of inflow in the region
print("max num in the real mean inflow distribution: ", vmax)
np.savetxt('./spatial_gan_test/real_region_mean_' + str(opt.region_i) + '_' + str(opt.region_j) + '.csv', mean_flow, delimiter=',')

# 3. heatmap of inflow
sns.heatmap(mean_flow, annot=True, vmin=0, vmax=vmax)
label = 'Region latitude {}, longitude {}, travel demand {}'.format(opt.region_i, opt.region_j, int(mean_dmd))
plt.title(label)
plt.savefig('./spatial_gan_test/real_region_mean_' + str(opt.region_i) + '_' + str(opt.region_j) + '.png')
plt.close()

################################ Load Model ##########################################
########################################################################################
G = Generator(103, opt.Ghid, 1).to(device)
G.load_state_dict(torch.load('./spatial_gan_train/G_params_' + str(opt.adj_bar) + '.pkl', map_location='cpu'))


########################## Prepare test data (input of G) ##########################
######################################################################################
# prepare noise
batch_x = torch.randn(flow.shape[0], opt.init_dim * opt.init_dim)
batch_x = batch_x.view(-1, region_width * region_length, opt.init_dim)
batch_x = Variable(batch_x.to(device))

# prepare adjacency matrix (need to be normalized)
all_adj = np.loadtxt(open('./all_region_inflow_adjacency.csv', "rb"), delimiter=",", skiprows=0)
all_adj = all_adj.reshape(-1, region_width * region_length, region_width * region_length)
current_region_idx = int(opt.region_i * (50 - region_length + 1) + opt.region_j)
current_region_adj = all_adj[current_region_idx]
# normalize current_region_adj
current_region_adj[current_region_adj < opt.adj_bar] = 0   # remove the negative corr
for i in range(current_region_adj.shape[0]):
    if sum(current_region_adj[i]) == 0:
        current_region_adj[i, i - int(i / 100) * 100] = 1
    if sum(current_region_adj[i]) != 0:
        row_sum = sum(current_region_adj[i])
        current_region_adj[i] = current_region_adj[i] / row_sum
# repeat current_region_adj for opt.out_num times
batch_adj = np.tile(current_region_adj, (flow.shape[0], 1))
batch_adj = batch_adj.reshape(-1, region_width * region_length, region_width * region_length)
batch_adj = torch.tensor(batch_adj)
print(batch_adj.size())
batch_adj = Variable(batch_adj.to(device)).float()

# prepare conditions
region_i = opt.region_i / max_y0
region_j = opt.region_j / max_y1
test_demand = int(mean_dmd) / max_y2

y1 = torch.zeros(flow.shape[0] * region_width * region_length, 1) + region_i
y2 = torch.zeros(flow.shape[0] * region_width * region_length, 1) + region_j
batch_y = torch.cat((y1, y2), 1)
y3 = torch.zeros(flow.shape[0] * region_width * region_length, 1) + test_demand
batch_y = torch.cat((batch_y, y3), 1).view(flow.shape[0], region_width * region_length, -1)
# print(batch_y.data.numpy()[0:5])
batch_y = Variable(batch_y.to(device)).float()


################################ Test Result ##########################################
########################################################################################
# Set model's test mode
G.eval()
test_reg = G(batch_x, batch_adj, batch_y)

##################### get generated images ####################
regs = test_reg.cpu().data.numpy().reshape(-1, region_width, region_length)
regs = (regs * 0.5 + 0.5) * max_x

#total_taxi_per_reg = np.sum(regs, axis=1)
#total_taxi_per_reg = np.sum(total_taxi_per_reg, axis=1)
# print("total_taxi_per_region: ", total_taxi_per_reg)
#np.savetxt(path8, total_taxi_per_reg, delimiter=',')

##################### plot the heatmap of generated mean img ###################
# compute the mean region of generated regions
mean_reg = np.mean(regs, axis=0)
#vmax = mean_reg.max()
np.savetxt('./spatial_gan_test/fake_region_mean_' + str(opt.region_i) + '_' + str(opt.region_j) + '_adj_bar_' + str(opt.adj_bar) + '.csv', mean_reg, delimiter=',')

# print("generated mean region: ", mean_reg)
# draw the heatmap of mean_reg
sns.heatmap(mean_reg, annot=True, vmin=0, vmax=vmax)
label = 'Region latitude {}, longitude {}, travel demand {}'.format(opt.region_i, opt.region_j, int(mean_dmd))
plt.title(label)
plt.savefig('./spatial_gan_test/fake_region_mean_' + str(opt.region_i) + '_' + str(opt.region_j) + '_adj_bar_' + str(opt.adj_bar) + '.png')
plt.close()

###################### Pixel Distance ###############################
# sort flow based on dmd
real_dist = flow.reshape(-1, 100)
sorted_idx = np.argsort(dmd, axis=0, kind='quicksort')
sorted_real_dist = real_dist[sorted_idx]

# prepare conditions (with the same dmd in order)
region_i = opt.region_i / max_y0
region_j = opt.region_j / max_y1
sorted_test_demand = dmd[sorted_idx] / max_y2
sorted_test_demand = torch.tensor(sorted_test_demand.reshape(-1, 1)).float()

y1 = torch.zeros(flow.shape[0] * region_width * region_length, 1) + region_i
y2 = torch.zeros(flow.shape[0] * region_width * region_length, 1) + region_j
batch_y = torch.cat((y1, y2), 1)
sorted_y3 = torch.zeros(flow.shape[0], 1) + sorted_test_demand
sorted_y3 = sorted_y3.numpy()
sorted_y3 = sorted_y3.repeat(region_width * region_length, axis=0)
sorted_y3 = torch.tensor(sorted_y3)
batch_y = torch.cat((batch_y, sorted_y3), 1).view(flow.shape[0], region_width * region_length, -1)
sorted_batch_y = Variable(batch_y.to(device)).float()

G.eval()
sorted_SGAN_dist = G(batch_x, batch_adj, sorted_batch_y)
sorted_SGAN_dist = sorted_SGAN_dist.cpu().data.numpy().reshape(-1, region_width, region_length)
sorted_SGAN_dist = (sorted_SGAN_dist * 0.5 + 0.5) * max_x
sorted_SGAN_dist = sorted_SGAN_dist.reshape(-1, 100)

dst = sorted_real_dist - sorted_SGAN_dist

dst_sum = []
for i in range(sorted_real_dist.shape[1]):
    tmp = np.linalg.norm(dst[:, i])
    dst_sum.append(tmp)
dst_sum = np.array(dst_sum)
dst_mean = np.mean(dst_sum)
dst_std = np.std(dst_sum)

np.savetxt('./spatial_gan_test/pixel_L2_SpacialGAN_' + str(opt.region_i) + '_' + str(opt.region_j) + '_adj_bar_' + str(opt.adj_bar) + '.csv', dst_sum, delimiter=',')
print("L2 mean pixel distance between real ones and SpacialGAN: ", dst_mean)
print("L2 pixel distance std between real ones and SpacialGAN: ", dst_std)

########################## Whole distance ##################################
whole_real_dist = sorted_real_dist.reshape(-1)
whole_SGAN_dist = sorted_SGAN_dist.reshape(-1)
whole_dst = np.linalg.norm(whole_real_dist - whole_SGAN_dist)
print("whole distance (L2): ", whole_dst)

###################### Mean-to-mean distance ###############################
# Euclidian Distance (L2)
mean_to_mean_dist = mean_flow - mean_reg
mean_to_mean_dist = np.linalg.norm(mean_to_mean_dist)
print("Mean-to-mean distance (L2 distance): ", mean_to_mean_dist)
