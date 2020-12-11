import cv2
import math
import argparse
import skimage.transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pathlib import Path
from tqdm import tqdm
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects

import sknw


def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    """
    Calaculate the prependicuar distance of given point from the line having
    start and end points.
    """
    if start == end:
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1])
            - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d


def rdp(points, epsilon):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.

    @param points: Series of points for a line geometry represnted in graph.
    @param epsilon: Tolerance required for RDP algorithm to aproximate the
                    line geometry.

    @return: Aproximate series of points for approximate line geometry
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[: index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results


def simplify_graph(graph, max_distance=2):
    """
    :type graph: MultiGraph
    """
    all_segments = []
    for (s, e) in graph.edges():
        for _, val in graph[s][e].items():
            ps = val['pts']
            full_segments = np.row_stack([
                graph.nodes[s]['o'],
                ps,
                graph.nodes[e]['o']
            ])

            # segments = simplify_edge(full_segments, max_distance=max_distance)
            segments = rdp(full_segments.tolist(), max_distance)
            all_segments.append(segments)

    return all_segments


def simplify_edge(ps: np.ndarray, max_distance=2):
    """
    Combine multiple points of graph edges to line segments
    so distance from points to segments <= max_distance
    :param ps: array of points in the edge, including node coordinates
    :param max_distance: maximum distance, if exceeded new segment started
    :return: ndarray of new nodes coordinates
    """
    res_points = []
    cur_idx = 0
    # combine points to the single line while distance from the line to any point < max_distance
    for i in range(1, len(ps) - 1):
        segment = ps[cur_idx:i + 1, :] - ps[cur_idx, :]
        angle = -math.atan2(segment[-1, 1], segment[-1, 0])
        ca = math.cos(angle)
        sa = math.sin(angle)
        # rotate all the points so line is alongside first column coordinate
        # and the second col coordinate means the distance to the line
        segment_rotated = np.array([[ca, -sa], [sa, ca]]).dot(segment.T)
        distance = np.max(np.abs(segment_rotated[1, :]))
        if distance > max_distance:
            res_points.append(ps[cur_idx, :])
            cur_idx = i
    if len(res_points) == 0:
        res_points.append(ps[0, :])
    res_points.append(ps[-1, :])

    return np.array(res_points)


def segmets_to_linestrings(segments):
    linestrings = []
    for segment in segments:
        linestring = segment_to_linestring(segment)
        if len(linestring) > 0:
            linestrings.append(linestring)
    if len(linestrings) == 0:
        linestrings = ['LINESTRING EMPTY']
    return linestrings


def segment_to_linestring(segment):
    if len(segment) < 2:
        return []

    linestring = 'LINESTRING ({})'
    sublinestring = ''

    for i, node in enumerate(segment):
        if i == 0:
            sublinestring = sublinestring + '{:.1f} {:.1f}'.format(node[1], node[0])
        else:
            if node[0] == segment[i - 1][0] and node[1] == segment[i - 1][1]:
                if len(segment) == 2:
                    return []
                continue
            if i > 1 and node[0] == segment[i - 2][0] and node[1] == segment[i - 2][1]:
                continue
            sublinestring = sublinestring + ', {:.1f} {:.1f}'.format(node[1], node[0])
    linestring = linestring.format(sublinestring)
    return linestring


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def flatten(l):
    return [item for sublist in l for item in sublist]


def remove_small_terminal(graph):
    deg = graph.degree()
    terminal_points = [i for i, d in dict(deg).items() if d == 1]
    edges = list(graph.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in graph[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                graph.remove_edge(s, e)
                continue
        vals = flatten([[v] for v in graph[s][e].values()])
        for ix, val in enumerate(vals):
            if s in terminal_points and val.get('weight', 0) < 30:
                graph.remove_node(s)
            if e in terminal_points and val.get('weight', 0) < 30:
                graph.remove_node(e)


def process_masks(mask_paths, thre=0.9):
    replicate = 5
    clip = 2
    rec = replicate + clip
    lnstr_df = pd.DataFrame()
    for msk_pth in tqdm(mask_paths):
        img_id = Path(msk_pth).stem.replace('_mask', '')
        msk = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
        msk = cv2.copyMakeBorder(msk, replicate, replicate, replicate, replicate, cv2.BORDER_REPLICATE)
        b_msk = msk > 255 * thre
        remove_small_holes(b_msk, 300, in_place=True)
        remove_small_objects(b_msk, 300, in_place=True)
        ske = skeletonize(b_msk).astype(np.uint16)
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)

        # build graph from skeleton
        graph = sknw.build_sknw(ske, multi=True)
        num_nodes = len(graph.nodes)
        remove_small_terminal(graph)
        while len(graph.nodes) != num_nodes:
            num_nodes = len(graph.nodes)
            remove_small_terminal(graph)
        segments = simplify_graph(graph)

        # plt segments
        # plt.imshow(b_msk)
        # for segment in segments:
        #     plt.plot(np.array(segment)[:, 1], np.array(segment)[:, 0], 'green')
        #     ps = np.array([segment[0], segment[-1]])
        #     plt.plot(ps[:, 1], ps[:, 0], 'r.')
        # plt.show()

        # plt graph
        # plt.imshow(msk)
        # for (s, e) in graph.edges():
        #     ps = graph[s][e][0]['pts']
        #     plt.plot(ps[:, 1], ps[:, 0], 'green')
        # nodes = graph.nodes()
        # ps = np.array([nodes[i]['o'] for i in nodes])
        # plt.plot(ps[:, 1], ps[:, 0], 'r.')

        linestrings_ = segmets_to_linestrings(segments)
        linestrings = unique(linestrings_)

        local = pd.DataFrame()
        local['WKT_Pix'] = linestrings
        local['ImageId'] = img_id
        lnstr_df = pd.concat([lnstr_df, local], ignore_index=True)

    return lnstr_df


def masks_to_csv(root):
    fname = [str(f) for f in root.glob('*.png')]

    print('Processing masks into linestrings...')
    lstrs = process_masks(fname)
    lstrs = lstrs[['ImageId', 'WKT_Pix']].copy()
    lstrs = lstrs.drop_duplicates()

    out_root = Path('./solutions')
    if 'test' in root.name:
        out_file = out_root / root.parts[7] /'lstrs_prop_sn.csv'
        Path.mkdir(out_file.parent, exist_ok=True)
    elif 'masks' in root.name:
        out_file = out_root / 'lstrs_gt_sn.csv'
        Path.mkdir(out_root, exist_ok=True)
    else:
        out_file = out_root / 'test.csv'
        Path.mkdir(out_root, exist_ok=True)

    lstrs.to_csv(str(out_file), index = False)
    print('Generate csv successfully!')


def main():
    parser = argparse.ArgumentParser(description='Masks into linestrings')
    parser.add_argument('-r', '--root', type=str, required=True,
                        help='Root directory to masks.')

    args = parser.parse_args()

    # mask to csv
    masks_to_csv(Path(args.root))


if __name__ == '__main__':
    main()