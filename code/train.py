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
import argparse
from SpatialGAN import Generator
from SpatialGAN import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--adj_bar", type=float, default=0.4, help="adj bar")
parser.add_argument("--init_dim", type=int, default=100, help="dimensionality of the latent code")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--Ghid', type=int, default=64, help='Hidden feature number of G.')
parser.add_argument('--Dhid', type=int, default=128, help='Hidden feature number of D.')

opt = parser.parse_args()
print(opt)
print(opt.adj_bar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

region_width = 10
region_length = 10

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

if not os.path.exists('./spatial_gan_train'):
    os.mkdir('./spatial_gan_train')


################################ Data cleaning ##########################################
########################################################################################
# if use speed, set the speeds larger than 200 to be 200, then normalize
x = np.loadtxt(open('./inflow_region.csv', "rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open('./label_region.csv', "rb"), delimiter=",", skiprows=0)
adj = np.loadtxt(open('./inflow_region_adjacency_' + str(opt.adj_bar) + '.csv', "rb"), delimiter=",", skiprows=0)

x = x.reshape(-1, region_width * region_length, 1)
adj = adj.reshape(-1, region_width * region_length, region_width * region_length)


print(x.shape)
print(adj.shape)
print(y.shape)
print(y[103000:103032])

# remove the last 10 rows and related regions
for i in [35, 40]:
    for j in range(0, 41, 5):
        arr1 = np.where(y[:, 0] == i)[0]
        arr2 = np.where(y[:, 1] == j)[0]
        idx = np.intersect1d(arr1, arr2)

        x = np.delete(x, idx, axis=0)
        y = np.delete(y, idx, axis=0)
        adj = np.delete(adj, idx, axis=0)

print(x.shape)
print(adj.shape)
print(y.shape)

################################ Data loading ##########################################
########################################################################################
# min-max normalization: x -> [0,1]
# x is a numpy array


def min_max_normal(x):
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    return x


'''
# input normalization: x -> [-1,1]
x = min_max_normal(x)
'''
x = x / x.max()
x = (x - 0.5) / 0.5
x = torch.tensor(x)

adj = torch.tensor(adj)

y = y[:, 0:3]     # remove feature: period
y = y.repeat(region_width * region_length, axis=0)
# label normalization: each label -> [0,1]
for i in [0, 1, 2]:
    y[:, i] = min_max_normal(y[:, i])
    #max_y = y[:, i].max()
    #print("max number in label: ", max_y)
    #y[:, i] = y[:, i] / max_y
    #y[:, i] = (y[:, i] - 0.5) / 0.5

print('normalized label: ', y[0:20, :])
y = y.reshape(-1, region_width * region_length, 3)
y = torch.tensor(y)

dataset = Data.TensorDataset(x, y, adj)
train_loader = Data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)


################################ Spatial GAN ##########################################
########################################################################################
D = Discriminator(4, opt.Dhid, 1).to(device)
G = Generator(103, opt.Ghid, 1).to(device)

opt_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
opt_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

################################ Loss histgram ##########################################
########################################################################################


def show_train_hist(hist, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)


################################ Training ##########################################
########################################################################################
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

# training
for epoch in range(opt.epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if epoch == 10:  # or epoch == 15:
        opt_G.param_groups[0]['lr'] /= 10
        opt_D.param_groups[0]['lr'] /= 10

    for step, (b_x, b_y, b_adj) in enumerate(train_loader):
        ######################### Train Discriminator #######################
        D.zero_grad()
        num_img = b_x.size(0)              # batch size

        real_img = Variable(b_x.to(device)).float()     # put tensor in Variable
        img_label = Variable(b_y.to(device)).float()
        img_adj = Variable(b_adj.to(device)).float()
        prob_real_img_right_pair = D(real_img, img_adj, img_label)

        noise = torch.randn(num_img, opt.init_dim * region_length * region_width).view(num_img, region_length * region_width, opt.init_dim)
        noise = Variable(noise.to(device))  # randomly generate noise

        fake_img = G(noise, img_adj, img_label)
        prob_fake_img_pair = D(fake_img, img_adj, img_label)

        # sample real imgs from database(just shuffle this batch imgs)
        shuffled_row_idx = torch.randperm(num_img)
        real_shuffled_img = b_x[shuffled_row_idx]
        real_shuffled_img = Variable(real_shuffled_img.to(device)).float()
        shuffled_adj = b_adj[shuffled_row_idx]
        shuffled_adj = Variable(shuffled_adj.to(device)).float()

        prob_real_img_wrong_pair = D(real_shuffled_img, shuffled_adj, img_label)

        D_loss = - torch.mean(torch.log(prob_real_img_right_pair) +
                              torch.log(1. - prob_fake_img_pair) + torch.log(1. - prob_real_img_wrong_pair))

        D_loss.backward()
        opt_D.step()

        D_losses.append(D_loss.item())

        ########################### Train Generator #############################
        G.zero_grad()
        # compute loss of fake_img
        noise2 = torch.randn(num_img, opt.init_dim * region_length * region_width).view(num_img, region_length * region_width, opt.init_dim)
        noise2 = Variable(noise2.to(device))  # randomly generate noise

        # create random label
        y_real = Variable(torch.ones(num_img).to(device))
        G_result = G(noise2, img_adj, img_label)
        D_result = D(G_result, img_adj, img_label).squeeze()
        # print("D score: ", D_result.cpu().data.numpy())

        G_loss = BCE_loss(D_result, y_real)

        G_loss.backward()
        opt_G.step()

        G_losses.append(G_loss.item())

        #print('Epoch [{}/{}], D_loss: {:.6f}, G_loss: {:.6f} '.format(epoch + 1, opt.epoch, D_loss.item(), G_loss.item()))

    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

show_train_hist(train_hist, path='./spatial_gan_train/train_loss_hist_' + str(opt.adj_bar) + '.png')
torch.save(G.state_dict(), './spatial_gan_train/G_params_' + str(opt.adj_bar) + '.pkl')   # save parameters
torch.save(D.state_dict(), './spatial_gan_train/D_params_' + str(opt.adj_bar) + '.pkl')
