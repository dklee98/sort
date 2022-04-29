from nis import match
from operator import and_
import os
import numpy as np
import argparse
import time
from sort import *
# from sort2 import *

def iou_dk(past_map, now_map):    
    tmp_now = np.expand_dims(now_map, 1)
    tmp_past = np.expand_dims(past_map, 0)

    and_mask = np.logical_and(tmp_now, tmp_past)
    or_mask = np.logical_or(tmp_now, tmp_past)

    ## too much computation time
    overlap = and_mask.sum(axis=(2,3)) / or_mask.sum(axis=(2,3))
    print(overlap.shape)
    return overlap

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT mask2 dongkyu')
    # parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='mask_data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='npy_and_mask')
    parser.add_argument("--bbox_path", 
                        help="Path to bbox.", type=str, default='bbox')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    bbox_path = args.bbox_path
    display = args.display
    seq_path = args.seq_path
    phase = args.phase
    total_time = 0
    ins_names = []
    img_names = []
    colours = np.random.rand(32, 3) #used only for display


    if(display):
        if not os.path.exists('mask_data'):
            print('\n\tERROR: mask_data link not found!\n\n    Create a symbolic link to the mask data\n   $ ln -s ../dataset stuttgart_00 mask_data\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')
    
    if not os.path.exists('bbox'):
        print('\n\tERROR: bbox link not found!\n')
        exit()

    if not os.path.exists('output'):
        os.makedirs('output')
    
    npyListing = os.listdir(bbox_path)
    total_frames = len(npyListing)

    for i in range(total_frames):
        if (i+1) // 10 < 1:
            ins_names.append('stuttgart_00_000000_00000{}_leftImg8bit.npy'.format(i+1))
            img_names.append('stuttgart_00_000000_00000{}_leftImg8bit.png'.format(i+1))
        elif (i+1) // 10 < 10:
            ins_names.append('stuttgart_00_000000_0000{}_leftImg8bit.npy'.format(i+1))
            img_names.append('stuttgart_00_000000_0000{}_leftImg8bit.png'.format(i+1))
        else:
            ins_names.append('stuttgart_00_000000_000{}_leftImg8bit.npy'.format(i+1))
            img_names.append('stuttgart_00_000000_000{}_leftImg8bit.png'.format(i+1))

    mot_tracker = Sort()
    ins_list = []
    data_path = os.path.join(seq_path, phase)

    with open(os.path.join('output', 'aaaaa.txt'),'w') as out_file:
        for frame, f in enumerate(ins_names):
            detection = np.load(bbox_path + '/' + 'bbox_%s.npy'%(frame))
            ins_list.append(np.load(data_path + '/' + f))
            if len(ins_list) < 2:
                continue

            if(display):
                fn = os.path.join(seq_path, 'original', img_names[frame]) 
                im = io.imread(fn)
                ax1.imshow(im)
                plt.title('%d Tracked Targets'%(frame))

            start_time = time.time()
            ## import sort
            track_bbs_ids = mot_tracker.update(detection)
            ## import sort2
            # track_bbs_ids = mot_tracker.update(ins_list, detection)
            cycle_time = time.time() - start_time
            print(f"{cycle_time:.5f} sec")
            total_time += cycle_time

            for d in track_bbs_ids:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
                if(display):
                    d = d.astype(np.int32)
                    ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

            if(display):
                fig.canvas.flush_events()
                fig_n = os.path.join('output', 'img_bbox')
                plt.savefig(fig_n + '/' + 'img_%s.png'%(frame), dpi=300)
                plt.draw()
                ax1.cla()
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

