import os
import glob
import argparse
import random

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from model.DualResUNet import DualResUNet
from model.Renderer import HeightRenderer, PanoRenderer

from mayavi import mlab


def show_density(args):
    net_type = args.net_type
    path_to_model = args.path_to_model
    
    model = None
    if net_type == "resunet":
        model = DualResUNet(3, 3, 64)

    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['net'])
    model = model.cuda()
    
    pano_renderer = PanoRenderer(1)
    im_ids = []
    with open('dataset/test.txt', 'r') as f:
        for i in f:
            im_ids.append(i.strip())
    random.shuffle(im_ids)
    
    aerial_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4485537493150323, 0.47377605990113797, 0.4520951120606348), 
                                std=(0.1799316263889618, 0.1567772635735416, 0.14737897874657677)),
    ])
    pano_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5530911549660469, 0.5883250463786659, 0.6191158691144336), 
                                std=(0.2013322671333761, 0.20358427316458785, 0.259951481179659)),
    ])
    
    with torch.no_grad():
        for im_id in im_ids:
            aerial = cv2.imread(os.path.join('dataset', 'aerial', im_id+'_aerial.tif')).astype(np.float32)
            aerial = cv2.cvtColor(aerial, cv2.COLOR_BGR2RGB) / 255.0
            transformed_aerial = aerial_transforms(aerial).unsqueeze(0).cuda()
            
            pano = cv2.imread(os.path.join('dataset', 'pano', im_id+'.png')).astype(np.float32)
            pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB) / 255.0
            transformed_pano = pano_transforms(pano).unsqueeze(0).cuda()
            
            density = model(transformed_aerial, transformed_pano)
            pano, _ = pano_renderer(density)
            
            plt.imshow(pano.squeeze().cpu())
            plt.show()
            density = torch.nn.functional.sigmoid(density)
            mlab.pipeline.volume(mlab.pipeline.scalar_field(density.squeeze().permute(1, 2, 0).cpu().numpy()))
            mlab.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize density field")
    parser.add_argument("-n", "--net_type", choices=['resunet'], default='resunet', help="type of the network")
    parser.add_argument("-p", "--path_to_model", required=True, help="path to the model. e.g. DENSITY_XXX/model_best.pth")
    args = parser.parse_args()

    show_density(args)
    