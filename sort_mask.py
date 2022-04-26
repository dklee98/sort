from nis import match
from operator import and_
import os
import numpy as np
import argparse
import time

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_batch(past_map, now_map):
    # now_bool = now_map[:,] > 0
    # past_bool = past_map[:,] > 0
    
    tmp_now = np.expand_dims(now_map, 1)
    tmp_past = np.expand_dims(past_map, 0)

    and_mask = np.logical_and(tmp_now, tmp_past)
    or_mask = np.logical_or(tmp_now, tmp_past)

    ##### solved #####
    # start = time.time()

    # sum_and = []
    # sum_or = []
    # for and_p in and_mask:
    #     m = 0
    #     idx = 0
    #     for j, and_n in enumerate(and_p):
    #         if not np.any(and_n):
    #             continue
    #         else:
    #             if m < np.count_nonzero(and_n):
    #                 # m = np.maximum(m, np.count_nonzero(and_n))
    #                 m = np.count_nonzero(and_n)
    #                 idx = j
    #     sum_and.append([idx,m])

    # for i, or_p in enumerate(or_mask):
    #     sum_or.append(np.count_nonzero(or_p[sum_and[i][0]]))
    #     sum_and[i] = sum_and[i][1]
    
    # # print(sum_and, sum_or)

    # overlap = np.array(sum_and) / np.array(sum_or)
    # # print(overlap)

    # end = time.time()
    # print(f"{end - start:.5f} sec")

    ## too much computation time
    overlap = and_mask.sum(axis=(2,3)) / or_mask.sum(axis=(2,3))

    return overlap
    

def cal_association(past_map, now_map, iou_threshold):
    if(len(now_map) == 0):
        return np.empty((0,2), dtype=int), np.arange(len(past_map))

    iou_matrix = iou_batch(past_map, now_map)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis = 1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
        # print(matched_indices)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_past_maps = []
    for p, p_map in enumerate(past_map):
        if p not in matched_indices[:, 0]:
            unmatched_past_maps.append(p)
    unmatched_now_maps = []
    for n, n_map in enumerate(now_map):
        if n not in matched_indices[:,1]:
            unmatched_now_maps.append(n)
            
    matched_pair = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_past_maps.append(m[0])
            unmatched_now_maps.append(m[1])
        else:
            matched_pair.append(m.reshape(1,2))

    if len(matched_pair) == 0:
        matched_pair = np.empty((0,2), dtype=int)
    else:
        matched_pair = np.concatenate(matched_pair, axis = 0)

    return matched_pair, np.array(unmatched_past_maps)

class Pair(object):
    id = 0
    def __init__(self):
        self.id = Pair.id
        Pair.id += 1
        self.matching_pair = []
        self.hits = 0

    def add_pair(self, p):
        self.matching_pair.append(p)
        self.hits += 1


class Tracker(object):
    def __init__(self, min_hits, iou_threshold=0.3):
        self.trackers = []
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.frame_count = 0

    def matching(self, map, mode):
        self.frame_count += 1
        ret = []
        if len(map) < 2:
            matched, unmatched_past = np.empty((0,2), dtype=int), np.arange(len(map[-1]))
        else:
            past, now = map[-2], map[-1]
            matched, unmatched_past = cal_association(past, now, self.iou_threshold)

        if mode:
            return matched

        print(self.frame_count, matched)
        # if len(self.trackers) == 0:
        #     for i in range(len(past)):
        #         self.trackers.append(Pair(i))
        #         # print(self.trackers[i].id)
        if self.frame_count < 3:
            for m in matched:
                self.trackers[m[0]].add_pair(m[1])
        else:
            for m in matched:
                for t in self.trackers:
                    if len(t.matching_pair) == 0:
                        continue
                    else: 
                        if t.matching_pair[-1] == m[0]:
                            t.add_pair(m[1])

            
        for i in unmatched_past:
            self.trackers.append(Pair())

        num = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                ret.append([trk.id] + trk.matching_pair)
            num -= 1

        if len(ret) > 0:
            return ret        
        # for i in self.trackers:
        #     print(i.id, i.matching_pair)
        # print('============')
        return np.empty((0,3))



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT mask dongkyu')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='mask_data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='npy_and_mask')
    parser.add_argument("--mode", 
                        help="true: matching pair, false: continuous id list", 
                        type=bool, default=True)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    display = args.display
    phase = args.phase
    mode = args.mode
    file_names = []
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3) #used only for display
    
    if not os.path.exists('mask_data'):
        print('\n\tERROR: mask_data link not found!\n\n    Create a symbolic link to the mask data\n   $ ln -s ../dataset stuttgart_00 mask_data\n\n')
        exit()

    if not os.path.exists('output'):
        os.makedirs('output')
    
    data_path = os.path.join(args.seq_path, phase)
    npyListing = os.listdir(data_path)
    total_seq = len(npyListing) // 2
    for i in range(total_seq):
        if (i+1) // 10 < 1:
            file_names.append('stuttgart_00_000000_00000{}_leftImg8bit.npy'.format(i+1))
        elif (i+1) // 10 < 10:
            file_names.append('stuttgart_00_000000_0000{}_leftImg8bit.npy'.format(i+1))
        else:
            file_names.append('stuttgart_00_000000_000{}_leftImg8bit.npy'.format(i+1))
    
    map_list = []
    mot_tracker = Tracker(min_hits = args.min_hits, iou_threshold = args.iou_threshold)

    ## just matching pair
    with open(os.path.join('output', 'matching_pair.txt'),'w') as out_file:
        
        for frame, f in enumerate(file_names):
            start = time.time()
            map_list.append(np.load(data_path + '/' + f))

            track_results = mot_tracker.matching(map_list, mode)

            for d in track_results:
                print('%d'%(frame) + ', ' + ', '.join(map(str, d)), file=out_file)
            end = time.time()
            print(f"{end - start:.5f} sec")
            print(frame)
