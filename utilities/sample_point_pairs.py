import os
import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


def random_sample(sample_num, min_dist, max_dist, avoid_sky):
    save_folder = os.path.join('./dataset/sample_points', f'{sample_num}_{min_dist}_{max_dist}_NOSKY{int(avoid_sky)}')
    os.makedirs(save_folder, exist_ok=True)
    im_ids = []
    with open('./dataset/all.txt', 'r') as f:
        for i in f:
            im_ids.append(i.strip())
    
    for step, im_id in enumerate(im_ids):
        for h in range(4):
            im = cv2.imread(os.path.join('./dataset/pers', im_id+'_'+str(h*90).zfill(3)+'.png'))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(os.path.join('./dataset/pers_depth', im_id+'_'+str(h*90).zfill(3)+'.tif'), cv2.IMREAD_UNCHANGED)

            b = depth != 0
            depth[b] = 1 - depth[b]
            depth[b] = (depth[b] - depth[b].min()) / (depth[b].max() - depth[b].min())

            all_coords = [(x, y) for x in range(256) for y in range(256)]
            sampled_pairs = set()

            # Randomly sample N non-repeated point pairs
            while len(sampled_pairs) < sample_num:
                # Randomly select two points from the list
                point_a, point_b = random.sample(all_coords, 2)

                # Calculate the distance between the two points
                dx = point_b[0] - point_a[0]
                dy = point_b[1] - point_a[1]
                distance = np.sqrt(dx**2 + dy**2)

                # Check the distance constraint and that the reversed pair hasn't been sampled
                if min_dist <= distance <= max_dist and (point_b, point_a) not in sampled_pairs:
                    if not avoid_sky or (depth[point_a[1], point_a[0]] != 0 or depth[point_b[1], point_b[0]] != 0):
                        sampled_pairs.add((point_a, point_b))
            #             im[point_a[1], point_a[0], :], im[point_b[1], point_b[0], :] = 255, 255 
            # for pair in sampled_pairs:
            #     plt.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], c='w', marker='o', markersize=1, linewidth=0.5)
            # plt.imshow(im)
            # plt.show()
            ordinal_relation = np.zeros((sample_num, 5), dtype=np.uint8)
            for idx, (point_a, point_b) in enumerate(sampled_pairs):
                xa, ya, xb, yb = point_a[1], point_a[0], point_b[1], point_b[0]
                depth_a, depth_b = depth[xa, ya], depth[xb, yb]

                r = 0 if abs(depth_a-depth_b) < 1e-3 else (1 if depth_a < depth_b else 2)
                ordinal_relation[idx, :] = [xa, ya, xb, yb, r]
            #     if r == 0:
            #         plt.plot([ya, yb], [xa, xb], markersize=1, linewidth=0.5)
            # plt.imshow(depth)
            # plt.show()
            np.save(os.path.join(save_folder, im_id+'_'+str(h*90).zfill(3)+'.npy'), ordinal_relation)
        print(step, '/', len(im_ids), end='\r')
    print(save_folder, 'completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sample random pixel pairs")

    parser.add_argument('-n', '--number', default=1024, type=int, help='number of pairs to sample')
    parser.add_argument('--min_dist', default=10, type=int, help='min distance between two points in each pair')
    parser.add_argument('--max_dist', default=30, type=int, help='max distance between two points in each pair')
    parser.add_argument('--avoid_sky', action='store_true', help='whether to avoid sampling in sky area')
    
    args = parser.parse_args()

    random_sample(args.number, args.min_dist, args.max_dist, args.avoid_sky)
