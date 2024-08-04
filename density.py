import os
import argparse
import random
import math
from datetime import datetime

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import *

from model import ResUNet, HeightRenderer

from loss import ScaleInvariantLoss
from skimage.metrics import structural_similarity as compare_ssim
from AerialStreetDataset import AerialDataset


def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    mae = np.mean(np.abs(gt - pred))
    ssim = compare_ssim(gt, pred, data_range=gt.max()-gt.min())

    return mae, rmse, ssim

    
def train(args):
    restore_from = args.restore_from
    seed = args.seed
    eps = args.epochs
    bs = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    net_type = args.net_type.lower()
    lr_decay = args.lr_decay
    num_workers = args.num_workers
        
    checkpoint = None
    if restore_from is not None:
        checkpoint = torch.load(restore_from)
        eps = checkpoint['epochs'] if eps is None else eps
        seed, bs, lr, weight_decay, lr_decay, net_type = checkpoint['seed'], checkpoint['batch_size'], checkpoint['lr'], checkpoint['weight_decay'], checkpoint['lr_decay'], checkpoint['net_type']
        save_dir_path = os.path.split(restore_from)[0]
        print(f'Training restored. Model will be saved in {save_dir_path}')
        best_mae, best_rmse, best_ep, current_ep = checkpoint['best_mae'], checkpoint['best_rmse'], checkpoint['best_ep'], checkpoint['current_ep']
        
        log_path = os.path.join(save_dir_path, 'log.npy')
        log = np.load(log_path).tolist() if os.path.exists(log_path) else []
    else:
        time_now = datetime.now().strftime('%y%m%d%H%M%S')
        save_dir_path = os.path.join('experiments', 'density', 
                                        f'DENSITY_{time_now}_{net_type.upper()}_EP{str(eps).zfill(3)}_BS{str(bs).zfill(2)}_LR{lr:.6f}'+ \
                                        ('' if lr_decay is None else f'_LD{str(int(lr_decay[0])).zfill(3)}+{lr_decay[1]:.2f}')
                                    )
        os.makedirs(save_dir_path, exist_ok=True)
        print(f'Model will be saved in {save_dir_path}')
        best_mae, best_rmse, best_ep, current_ep = 1e7, 1e7, 0, 0
        log = []
        
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

    # dataset
    train_dataset = AerialDataset(root='./dataset', train=True)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataset = AerialDataset(root='./dataset', train=False, test=False)
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=num_workers, shuffle=True, drop_last=True)
    
    # define network
    model = ResUNet(in_channels=3, out_channels=64)
    model = model.cuda()
    torch.compile(model)
    
    height_renderer = HeightRenderer(BS=bs).cuda()
    
    # define optimizer & lr scheduler
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    step_size, gamma = (100, 0.1) if lr_decay is None else (int(lr_decay[0]), lr_decay[1])
    lr_scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    
    if restore_from is not None:
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_decay:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
    # parallelize
    parallel = False
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu_ids is not None:
        gpu_ids = list(map(int, gpu_ids.split(',')))
    if gpu_ids is not None and len(gpu_ids) > 1:
        parallel = True 
        model = torch.nn.DataParallel(model)
        
    # define loss
    scale_invariant_loss = ScaleInvariantLoss().cuda()
    mae = L1Loss().cuda()
    mse = MSELoss().cuda()

    for ep in range(current_ep+1, eps+1):
        """training"""
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
        train_loss = []
        for step, (aerial, ndsm) in loop:
            aerial, ndsm = aerial.cuda(), ndsm.cuda()
            # clear grad
            optimizer.zero_grad()
            density = model(aerial)
            
            height = height_renderer(density)
            total_loss = scale_invariant_loss(height, ndsm)
            train_loss.append(total_loss)

            total_loss.backward()
            optimizer.step()
                
            loop.set_description(f"D -- train on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}: epoch [{ep}/{eps}]")
            loop.set_postfix(train_loss=total_loss.item())

        """validation"""
        model.eval()
        with torch.no_grad():
            loop = tqdm(enumerate(val_loader), total=len(val_loader))
            val_loss, val_mae, val_rmse = [], [], []
            for step, (aerial, ndsm) in loop:
                aerial, ndsm = aerial.cuda(), ndsm.cuda()
                density = model(aerial)
                output = height_renderer(density)
                val_loss.append(scale_invariant_loss(output, ndsm))
                val_mae.append(mae(output, ndsm))
                val_rmse.append(torch.sqrt(mse(output, ndsm)))

                loop.set_description(f"D -- val on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}: epoch [{ep}/{eps}]")
                loop.set_postfix(val_loss=val_loss[-1].item())

        ave_train_loss = sum(train_loss) / len(train_loss)
        ave_train_loss = ave_train_loss.item()
        ave_val_loss = sum(val_loss) / len(val_loss)
        ave_val_loss = ave_val_loss.item()
        ave_val_mae = sum(val_mae) / len(val_mae)
        ave_val_mae = ave_val_mae.item()
        ave_val_rmse = sum(val_rmse) / len(val_rmse)
        ave_val_rmse = ave_val_rmse.item()
        log.append([ave_train_loss, ave_val_loss, ave_val_mae, ave_val_rmse])
        
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        if lr_decay is not None:
            lr_scheduler.step() # lr decay

        """save log"""
        np.save(os.path.join(save_dir_path, 'log.npy'), np.array(log))
        """save best model"""
        save_best = False
        if ave_val_mae < best_mae and ave_val_rmse < best_rmse:
            best_mae = ave_val_mae
            best_rmse = ave_val_rmse
            best_ep = ep
            save_best = True
        """save model for restore"""
        checkpoint = {
            'net': model.state_dict(),
            'seed': seed,
            'epochs': eps,
            'current_ep': ep,
            'batch_size': bs,
            'lr': lr,
            'current_lr': current_lr,
            'optimizer': optimizer.state_dict(),
            'weight_decay': weight_decay,
            'lr_decay': lr_decay,
            'lr_scheduler': None if lr_decay is None else lr_scheduler.state_dict(),
            'net_type': net_type,
            'best_mae': best_mae,
            'best_rmse': best_rmse,
            'best_ep': best_ep
        }
        checkpoint_path = os.path.join(save_dir_path, 'model.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'CURRENT EPOCH {ep}: model saved to {checkpoint_path}.')
        if save_best:
            best_checkpoint_path = os.path.join(save_dir_path, 'model_best.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f'BEST EPOCH {best_ep}: model saved to {best_checkpoint_path}.')

        """log training info"""
        log_time = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        log_verbose = f'{log_time} -- epoch {str(ep).zfill(3)} batch size {str(bs).zfill(2)} lr {current_lr:.6f} | '\
                    f'train loss:{ave_train_loss:10.6f} val loss:{ave_val_loss:10.6f} '\
                    f'val MAE:{ave_val_mae:10.6f} val RMSE:{ave_val_rmse:10.6f} | '\
                    f'best epoch {str(best_ep).zfill(3)} best MAE:{best_mae:10.6f} best RMSE:{best_rmse:10.6f}\n'
        print(log_verbose)
        with open(os.path.join(save_dir_path, 'log_verbose.txt'), 'a') as f:
            f.write(log_verbose)

    """test"""
    print('training completed')
    print('testing...')
    test(save_dir_path, net_type, parallel)
    print('test over')
    print(100*'-')


def test(save_dir_path, net_type, parallel):
    model = ResUNet(3, 64).cuda()
    torch.compile(model)
    height_renderer = HeightRenderer(BS=1).cuda()
    
    checkpoint = torch.load(os.path.join(save_dir_path, 'model_best.pth'))
    if parallel:
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    else:
        model.load_state_dict(checkpoint['net'])
        
    pred_save_path = os.path.join(save_dir_path, 'prediction')
    os.makedirs(pred_save_path, exist_ok=True)

    fns = []
    with open('./dataset/test.txt', 'r') as f:
        for i in f:
            fns.append(i.strip())
    
    aerial_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4485537493150323, 0.47377605990113797, 0.4520951120606348), 
                                std=(0.1799316263889618, 0.1567772635735416, 0.14737897874657677)),
    ])

    metrics = []
    model.eval()
    with torch.no_grad():
        loop = tqdm(enumerate(fns), total=len(fns))
        for step, fn in loop:
            aerial = cv2.imread(os.path.join('./dataset/aerial', fn+'_aerial.tif')).astype(np.float32)
            aerial = cv2.cvtColor(aerial, cv2.COLOR_BGR2RGB) / 255.0
            transformed_aerial = aerial_transforms(aerial).unsqueeze(0).cuda()

            density = model(transformed_aerial)
            prediction = height_renderer(density).squeeze().cpu().numpy()
            prediction[prediction < 0.1] = 0.
            
            cv2.imwrite(os.path.join(pred_save_path, fn+'_pred.tif'), prediction)
            mask = cv2.imread(os.path.join('./dataset/ndsm', fn+'_ndsm.tif'), cv2.IMREAD_UNCHANGED)
            metrics.append(compute_errors(mask, prediction))
    
    test_results = np.mean(np.array(metrics), axis=0)
    print(test_results)
    with open(os.path.join(save_dir_path, 'test_results.txt'), 'w') as f:
        f.write(f'MAE: {test_results[0]:10.6f}\n')
        f.write(f'RMSE:{test_results[1]:10.6f}\n')
        f.write(f'SSIM:{test_results[2]:10.6f}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train baseline")

    parser.add_argument('-n', '--net_type', choices=['resunet'], default='resunet')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-e', '--epochs', type=int, help='epochs', required=True)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--lr', type=float, help='learning rate', required=True)
    parser.add_argument('--lr_decay', nargs=2, type=float, help='learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--restore_from', help='path to checkpoint. e.g. DENSITY_XXX/model_best.pth')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--test_path')
    args = parser.parse_args()
    
    test_path = args.test_path
    train(args) if test_path is None else test(save_dir_path=test_path, parallel=False)
    