import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Mast to bbox converter')
    # parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='mask_data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='npy_and_mask')
    # parser.add_argument("--mode", 
    #                     help="true: matching pair, false: continuous id list", 
    #                     type=bool, default=True)
    # parser.add_argument("--min_hits", 
    #                     help="Minimum number of associated detections before track is initialised.", 
    #                     type=int, default=3)
    # parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    ins_names = []
    img_names = []
    colours = np.random.rand(32, 3)
    
    if not os.path.exists('mask_data'):
        print('\n\tERROR: mask_data link not found!\n\n    Create a symbolic link to the mask data\n   $ ln -s ../dataset stuttgart_00 mask_data\n\n')
        exit()

    if not os.path.exists('bbox'):
        os.makedirs('bbox')

    data_path = os.path.join(args.seq_path, args.phase)
    npyListing = os.listdir(data_path)
    total_seq = len(npyListing) // 2

    for i in range(total_seq):
        if (i+1) // 10 < 1:
            ins_names.append('stuttgart_00_000000_00000{}_leftImg8bit.npy'.format(i+1))
            img_names.append('stuttgart_00_000000_00000{}_leftImg8bit.png'.format(i+1))
        elif (i+1) // 10 < 10:
            ins_names.append('stuttgart_00_000000_0000{}_leftImg8bit.npy'.format(i+1))
            img_names.append('stuttgart_00_000000_0000{}_leftImg8bit.png'.format(i+1))
        else:
            ins_names.append('stuttgart_00_000000_000{}_leftImg8bit.npy'.format(i+1))
            img_names.append('stuttgart_00_000000_000{}_leftImg8bit.png'.format(i+1))

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')


    for frame, f in enumerate(ins_names):
        frame_bbox = []
        instance = np.load(data_path + '/' + f)
        out_file = os.path.join('bbox', 'bbox_%s'%(frame))

        fn = os.path.join(args.seq_path, 'original', img_names[frame]) 
        im = io.imread(fn)
        ax1.imshow(im)
        plt.title(' Tracked Targets')
        
        for c in instance:
            one_pair = np.stack(np.where(c==1), axis=0)
            y1, x1 = min(one_pair[0]), min(one_pair[1])
            y2, x2 = max(one_pair[0]), max(one_pair[1])
            b = [x1, y1, x2, y2, 1]
            w, h = x2-x1, y2-y1
            frame_bbox.append(b)
            # print(np.array(frame_bbox))
            color = np.random.randint(100)
            ax1.add_patch(patches.Rectangle((b[0],b[1]),w,h,fill=False,lw=3, ec=colours[color%32,:]))
        fig.canvas.flush_events()
        plt.draw()
        ax1.cla()
        np.save(out_file, np.array(frame_bbox))

            