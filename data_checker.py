import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as patcol
from skimage import io
import time
import argparse
import cv2

import pycocotools.mask as maskUtils



def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(np.bool)
    return bitmap_mask

def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='data checker dongkyu')
    # parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--threshold", help="folder name", type=str, default='0.7')
    # parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='npy_and_mask')
    # parser.add_argument("--bbox_path", 
    #                     help="Path to bbox.", type=str, default='bbox')
    # parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    folder = args.threshold
    if not os.path.exists('dk_data'):
        print('\n\tERROR: dk_data link not found!\n\n    Create a symbolic link to the dk_data\n    (https://motchallenge.net/data/2D_MOT_2015/#download).')
        exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    colours = np.random.rand(8, 3)


    bbox_path = os.path.join('dk_data', folder, 'bbox')
    mask_path = os.path.join('dk_data', folder, 'mask')
    label_path = os.path.join('dk_data', folder, 'label')
    image_path = os.path.join('dk_data', folder, 'mask_rcnn', 'stuttgart')
    origin_path = os.path.join('dk_data', 'original_images')

    for i in range(104):
        b_p = os.path.join(bbox_path, 'bbox_%06d.npy' % (i+1))
        m_p = os.path.join(mask_path, 'mask_%06d.npy' % (i+1))
        l_p = os.path.join(label_path, 'label_%06d.npy' % (i+1))
        i_p = os.path.join(image_path, 'stuttgart_00_000000_%06d_leftImg8bit.png' % (i+1))
        o_r = os.path.join(origin_path, 'stuttgart_00_000000_%06d_leftImg8bit.png' % (i+1))
        
        im = io.imread(o_r)

        bbox = np.load(b_p, allow_pickle=True)
        mask = np.load(m_p, allow_pickle=True)
        label = np.load(l_p, allow_pickle=True)

        

        for i, b in enumerate(bbox):
            b = b.astype(np.int32)
            ax1.add_patch(patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, lw=2, ec=colours[label[i]%8,:]))
            # print(b[-1])
        

        alpha = 0.5 # Transparency
        for i, m in enumerate(mask):
            # start = time.time()
            color = colours[label[i]%8, :]
            contours, _ = bitmap_to_polygon(m)
            for i, c in enumerate(contours):
                epsilon = 0.1 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                approx = np.concatenate(approx)
                contours[i] = approx
            polygon = [patches.Polygon(c) for c in contours]
            # end = time.time()
            # print(f"{end - start:.5f} sec")
            m = m.astype(bool)
            im[m] = im[m] * (1 - alpha) + color * alpha
            p = patcol.PatchCollection(polygon, facecolor=color, edgecolors='w', linewidths=1, alpha=0.8)
            ax1.add_collection(p)


        ax1.imshow(im)
        plt.title('1')

        fig.canvas.flush_events()
        plt.draw()
        ax1.cla()
        # time.sleep(1)
        

