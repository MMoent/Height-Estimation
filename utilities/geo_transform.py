import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F


def get_pano_direction(H=256, W=512):
    h = torch.linspace(0, 1.0, H)
    w = torch.linspace(0, 1.0, W)
    _x, _y = torch.meshgrid(h, w, indexing='ij')
    lat = (_x - 0.5) * torch.pi
    lon = 2 * _y * torch.pi
    axis1 = torch.cos(lat) * torch.cos(lon)
    axis2 = torch.cos(lat) * torch.sin(lon)
    axis3 = torch.sin(lat)
    pano_direction = torch.concatenate([axis1[..., None], axis2[...,None], axis3[..., None]], dim=-1)
    return pano_direction


def unproject_pano(pano, origin, X=256, Y=256, Z=256):
    _x, _y, _z = torch.arange(0, X), torch.arange(0, Y), torch.arange(0, Z)
    zt, xt, yt = torch.meshgrid(_z, _x, _y, indexing='ij')
    xt, yt, zt = xt - origin[0], yt - origin[1], zt - origin[2]
    denominator = torch.sqrt(xt**2 + yt**2 + zt**2) + 1e-8
    xt, yt, zt = xt / denominator, yt / denominator, zt / denominator

    theta = torch.arcsin(-zt) / (0.5 * torch.pi)
    phi = (torch.atan2(yt, -xt)) / torch.pi
    bs, c, h, w = pano.shape
    grid = torch.cat([phi.view(X*Y, Z, 1).expand(bs, -1, -1, -1), theta.view(X*Y, Z, 1).expand(bs, -1, -1, -1)], dim=-1)
    voxel_sampled = F.grid_sample(pano, grid, mode='bilinear', align_corners=True).view(bs, c, Z, X, Y)
    return voxel_sampled


def test(pano, voxel_sampled):
    # pano_direction = get_pano_direction(H=256, W=512)
    pano = cv2.imread('./demo.png')
    pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB) / 255.0
    # pano[0:10, :, :] = (1, 0, 0)
    # pano[120:128, :, :] = (0, 1, 0)
    # pano[245:255, :, :] = (0, 0, 1)
    pano[:, 0:10, :] = (1, 0, 0)
    pano[:, 117:127, :] = (1, 1, 0)
    pano[:, 245:255, :] = (0, 1, 0)
    pano[:, 373:383, :] = (0, 1, 1)
    pano[:, 501:511, :] = (0, 0, 1)
    # pano[117:127, :, :] = (1, 1, 1)
    pano = torch.from_numpy(pano).float().permute(2, 0, 1).expand(4, -1, -1, -1)
    print(pano.shape)
    voxel_sampled = unproject_pano(pano=pano, origin=(128, 128, 128), X=256, Y=256, Z=256)
    test(pano, voxel_sampled)
    plt.subplot(6, 2, 1)
    plt.imshow(pano[0, ...].permute(1, 2, 0))
    plt.axis('off')
    plt.subplot(6, 2, 2)
    plt.imshow(voxel_sampled[0, :, 0, ...].permute(1, 2, 0))
    plt.axis('off')
    for i in range(7):
        plt.subplot(6, 2, i+2)
        t = i*30
        plt.imshow(voxel_sampled[0, :, t, ...].permute(1, 2, 0))
        plt.axis('off')
        plt.title(t)
    plt.show()
    # plt.subplot(6, 2, 1)
    # plt.imshow(pano[0, ...].permute(1, 2, 0))
    # plt.axis('off')
    # plt.subplot(6, 2, 2)
    # plt.imshow(voxel_sampled[0, :, :, 0, :].permute(1, 2, 0))
    # plt.axis('off')
    # for i in range(1, 7):
    #     plt.subplot(6, 2, i+2)
    #     t = i*20+60
    #     plt.imshow(voxel_sampled[0, :, :, t, :].permute(1, 2, 0))
    #     plt.axis('off')
    #     plt.title(t)
    # plt.show()


if __name__ == "__main__":
    origin_z = (-1 + (2 * 2 / 128)) * torch.ones(4, 1)
    origin_x, origin_y = torch.zeros(4, 1), torch.zeros(4, 1)
    origin = torch.cat([origin_x, origin_y, origin_z], dim=-1)[:, None, None, None, :]
    print(origin.shape)
    pano_direction = get_pano_direction()
    print(pano_direction.shape)
    pano_direction = pano_direction[None, ..., None, :].expand(4, -1, -1, -1, -1)

    sample_len = (torch.arange(300)+1) / 300
    sample_len = sample_len.view(1, 1, 1, -1, 1)
    print(sample_len.shape, pano_direction.shape)
    sample_point = origin + sample_len * pano_direction
    print(sample_point.shape)

    voxel_min = -1
    voxel_max = -1 + 64/(128/2) 
    grid = sample_point.permute(0, 3, 1, 2, 4)
    grid[...,2]   = ((grid[...,2]-voxel_min)/(voxel_max-voxel_min)) * 2 - 1
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # points = sample_point[0, ..., 299, :]
    # ax.scatter(points[..., 0], points[..., 1], points[..., 2], c='r', s=0.001)
    
    points = grid[0, 299, ...]
    ax.scatter(points[..., 0], points[..., 1], points[..., 2], c='g', s=0.001)
    ax.set_zlim(-1, 1)
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    plt.show()