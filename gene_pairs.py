import sys
import sys, os
import argparse
import time
import copy
import numpy as np


def gene_local_points(grid_size, pair_num, local_size=1):
    neighbor = []
    for j in range(-local_size, local_size + 1):
        for h in range(-local_size, local_size + 1):
            if j == 0 and h == 0:
                pass
            else:
                item = [j, h]
                neighbor.append(item)

    neighbor = np.array(neighbor)

    tot_pairs = []
    for k in range(pair_num):
        while True:
            x1 = np.random.randint(0, grid_size)
            y1 = np.random.randint(0, grid_size)
            point1 = x1 * grid_size + y1

            np.random.shuffle(neighbor)
            x2 = x1 + neighbor[0][0]
            y2 = y1 + neighbor[0][1]
            if x2 < 0:
                x2 = 0
            elif x2 >= grid_size:
                x2 = grid_size - 1
            if y2 < 0:
                y2 = 0
            elif y2 >= grid_size:
                y2 = grid_size - 1
            point2 = x2 * grid_size + y2

            if point1 == point2:  # bugFix
                continue

            if [point1, point2] in tot_pairs or [point2, point1] in tot_pairs:
                continue
            else:
                break
        tot_pairs.append(list([point1, point2]))

    tot_pairs = np.array(tot_pairs)
    return tot_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--ratios", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1], type=list,
                        help='ratios of context')
    parser.add_argument("--sample_num", default=100, type=int, help='sample num of S')
    parser.add_argument("--grid_size", default=16, type=int, help='number of grids of each img')
    parser.add_argument("--pair_num", default=200, type=int, help='number of point pair of each test img')
    parser.add_argument("--targeted", action="store_true", dest="targeted", help="whether use the targeted attack (True for targeted attack, False for untargeted attack)")
    # prefix of save path
    parser.add_argument('--distance', default='l_inf', type=str,help="type of adversarial attacks, currently only support 'l_inf'")
    parser.add_argument('--arch', default="resnet18", type=str, help="model name")
    parser.add_argument("--adv_model", action="store_true", dest="adv_model", help="the type of model (True for adversarially learned DNN, False for standardly learned DNN)")

    args = parser.parse_args()
    np.random.seed(args.seed)

    if not args.targeted:
        args.img_adv = "advImgs_untarget"
        prefix = "{}/{}/untarget/".format(args.distance, args.arch)
    else:
        args.img_adv = "advImgs_target"
        prefix = "{}/{}/target/".format(args.distance, args.arch)

    if args.adv_model:
        args.selected_imgs = os.listdir(os.path.join(prefix + args.img_adv, "adv_model"))
        args.point_path = os.path.join(args.img_adv, "adv_model", "points")
    else:
        args.selected_imgs = os.listdir(os.path.join(prefix + args.img_adv, "ori_model"))
        args.point_path = os.path.join(args.img_adv, "ori_model", "points")

    args.selected_imgs.sort()

    point_dir = os.path.join(prefix, args.point_path)
    if not os.path.exists(point_dir):
        os.makedirs(point_dir)

    # gene pairs for each image
    for im in args.selected_imgs:
        if not im.startswith('ILSVRC'):
            continue
        img_name = im.replace('.npy', '')
        print('Image ', img_name)
        save_path = os.path.join(point_dir, "img{}".format(img_name))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

            tot_pairs = gene_local_points(args.grid_size, args.pair_num, local_size=1)
            np.save(save_path + '/points.npy', tot_pairs)

            # tot_pairs = np.load(save_path + '/points.npy')
            for r in args.ratios:
                print('Ratio:', r)
                players = []
                for p, pt in enumerate(tot_pairs):
                    point1, point2 = pt[0], pt[1]
                    # m-order interactions
                    context = list(range(args.grid_size ** 2))
                    context.remove(point1)
                    context.remove(point2)

                    players_thispair = []
                    m = int((args.grid_size ** 2 - 2) * r)  # m-order
                    for k in range(args.sample_num):
                        players_thispair.append(np.random.choice(context, m, replace=False))

                    players.append(players_thispair)

                player_save_path = os.path.join(point_dir, "img{}/ratio{}_S.npy".format(img_name, int(r * 100)))
                players = np.array(players)
                np.save(player_save_path, players)


if __name__ == "__main__":
    main()
