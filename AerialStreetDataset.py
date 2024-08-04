import os
import glob
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from equilib import equi2pers
from torch.utils import data
from torch.utils.data import DataLoader


class AerialDataset(data.Dataset):
    def __init__(self, root, train=True, test=False):
        self.root = root
        self.train = train
        self.test = test

        # load image file name
        target_file = 'train' if train else ('test' if test else 'val')
        with open(os.path.join(root, target_file+'.txt'), 'r') as f:
            self.im_ids = [i.strip() for i in f.readlines()]

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4485537493150323, 0.47377605990113797, 0.4520951120606348), 
                                 std=(0.1799316263889618, 0.1567772635735416, 0.14737897874657677)),
        ])

    def __getitem__(self, item):
        aerial = cv2.imread(os.path.join(self.root, 'aerial', self.im_ids[item]+'_aerial.tif')).astype(np.float32)
        aerial = cv2.cvtColor(aerial, cv2.COLOR_BGR2RGB) / 255.0
        ndsm = cv2.imread(os.path.join(self.root, 'ndsm', self.im_ids[item]+'_ndsm.tif'), cv2.IMREAD_UNCHANGED)
        
        transformed_aerial = self.transforms(aerial)
        transformed_ndsm = transforms.ToTensor()(ndsm)
        return transformed_aerial, transformed_ndsm

    def __len__(self):
        return len(self.im_ids)


class AerialStreetDataset(data.Dataset):
    def __init__(self, root, train=True, test=False, sample_num=2048, sample_avoid_sky=False, require_st_cons=False):
        self.root = root
        self.train = train
        self.test = test

        self.sample_num = sample_num
        self.sample_avoid_sky = sample_avoid_sky
        self.require_st_cons = require_st_cons

        # load image file name
        target_file = 'train' if train else ('test' if test else 'val')
        with open(os.path.join(root, target_file+'.txt'), 'r') as f:
            self.im_ids = [i.strip() for i in f.readlines()]

        self.aerial_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4485537493150323, 0.47377605990113797, 0.4520951120606348), 
                                 std=(0.1799316263889618, 0.1567772635735416, 0.14737897874657677)),
        ])
        self.pano_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5530911549660469, 0.5883250463786659, 0.6191158691144336), 
                                 std=(0.2013322671333761, 0.20358427316458785, 0.259951481179659)),
        ])
    
    def __getitem__(self, item):
        aerial = cv2.imread(os.path.join(self.root, 'aerial', self.im_ids[item]+'_aerial.tif')).astype(np.float32)
        aerial = cv2.cvtColor(aerial, cv2.COLOR_BGR2RGB) / 255.0
        pano = cv2.imread(os.path.join(self.root, 'pano', self.im_ids[item]+'.png')).astype(np.float32)
        pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB) / 255.0
        ndsm = cv2.imread(os.path.join(self.root, 'ndsm', self.im_ids[item]+'_ndsm.tif'), cv2.IMREAD_UNCHANGED)

        transformed_aerial = self.aerial_transforms(aerial)
        transformed_pano = self.pano_transforms(pano)
        transformed_ndsm = transforms.ToTensor()(ndsm)
        
        if not self.require_st_cons or not self.train:
            return transformed_aerial, transformed_ndsm, transformed_pano
        
        sky_mask = cv2.imread(os.path.join(self.root, 'sky', self.im_ids[item]+'_sky.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        sky_mask = sky_mask / 255.0
        transformed_sky_mask = transforms.ToTensor()(sky_mask)

        depth = torch.zeros((4, 1, 256, 256), dtype=torch.float32)
        ordinal = torch.zeros((4, self.sample_num, 5)).long()

        for h in range(4):
            d = cv2.imread(os.path.join(self.root, 'pers_depth', self.im_ids[item]+'_'+str(h*90).zfill(3)+'.tif'), cv2.IMREAD_UNCHANGED)

            b = d != 0
            d[b] = 1 - d[b]
            d[b] = (d[b] - d[b].min()) / (d[b].max() - d[b].min())

            transformed_d = transforms.ToTensor()(d)
            depth[h, ...] = transformed_d

            sample_fn = os.path.join(self.root, 'sample_points', f'{self.sample_num}_10_30_NOSKY{int(self.sample_avoid_sky)}',
                                      self.im_ids[item]+'_'+str(h*90).zfill(3)+'.npy')
            ordinal[h, ...] = torch.from_numpy(np.load(sample_fn)).long()

        return transformed_aerial, transformed_ndsm, transformed_pano, transformed_sky_mask, depth, ordinal

    def __len__(self):
        return len(self.im_ids)


def test_aerial():
    train_dataset = AerialDataset(root='./dataset', train=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, drop_last=True)

    for aerial, ndsm in train_loader:
        print(aerial.shape, ndsm.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(aerial[0, ...].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(ndsm[0, ...].squeeze().cpu().numpy())
        plt.show()


def test_aerialstreet():
    train_dataset = AerialStreetDataset(root='./dataset', train=True, require_st_cons=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, drop_last=True)
    for aerial, ndsm, pano, sky, depth, ordinal in train_loader:
        print(aerial.shape, ndsm.shape, pano.shape, sky.shape, depth.shape, ordinal.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(aerial[0, ...].permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(ndsm[0, ...].squeeze())
        plt.show()

        plt.subplot(1, 2, 1)
        plt.imshow(pano[0, ...].permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(sky[0, ...].squeeze())
        plt.show()

        for h in range(4):
            rots = [{'roll': 0., 'pitch': 0, 'yaw': h * 90 / 180 * np.pi}] * 4
            t = equi2pers(pano, rots, 256, 256, 90, z_down=True)
            
            plt.subplot(2, 4, h+1)
            plt.imshow(depth[0, h, ...].squeeze())
            plt.subplot(2, 4, h+5)
            plt.imshow(t[0, ...].permute(1,2,0))
        plt.show()
        
        plt.imshow(sky[0, ...].squeeze())
        plt.show()

        for h in range(4):
            xa = ordinal[:, h, :, 0]
            ya = ordinal[:, h, :, 1]
            xb = ordinal[:, h, :, 2]
            yb = ordinal[:, h, :, 3]
            r = ordinal[:, h, :, 4]

            batch_indices = torch.arange(4)[:, None]

            a = depth[batch_indices, h, 0, xa, ya]
            b = depth[batch_indices, h, 0, xb, yb]
            rr = torch.where(torch.abs(a-b)<1e-3, 0, torch.where(a < b, 1, 2)).squeeze()
            print(torch.equal(r, rr))
            plt.plot([ya[0, :], yb[0, :]], [xa[0, :], xb[0, :]],  markersize=1, linewidth=0.5, c='w')
            
            # c = depth[0, h, 0, ...].numpy()
            # c[xa, ya] = 1.0
            # c[xb, yb] = 1.0
            plt.imshow(depth[0, h, 0, ...])
            plt.show()


if __name__ == "__main__":
    test_aerialstreet()
    