import copy
import argparse
import scipy
import logging
import shapely.wkt
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from logging.handlers import RotatingFileHandler
from scipy.stats import hmean
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point, LineString


# --------------- wkt-to-G ------------------ #
def wkt_to_G(wkt_list, node_iter=0, edge_iter=0):
    #
    node_loc_dic, edge_dic = wkt_list_to_nodes_edges(wkt_list,
                                                     node_iter=node_iter,
                                                     edge_iter=edge_iter)
    G0 = nodes_edges_to_G(node_loc_dic, edge_dic)

    return G0


def wkt_list_to_nodes_edges(wkt_list, node_iter=0, edge_iter=0):
    """
    Convert wkt list to nodes and edges
    Make an edge between each node in linestring. Since one linestring
    may contain multiple edges, this is the safest approach
    """
    node_loc_set = set()  # set of edge locations
    node_loc_dic = {}  # key = node idx, val = location
    node_loc_dic_rev = {}  # key = location, val = node idx
    edge_loc_set = set()  # set of edge locations
    edge_dic = {}  # edge properties

    for i, lstring in enumerate(wkt_list):
        # get lstring properties
        shape = shapely.wkt.loads(lstring)
        xs, ys = shape.coords.xy
        length = shape.length

        # iterate through coords in line to create edges between every point
        for j, (x, y) in enumerate(zip(xs, ys)):
            loc = (x, y)
            # for first item just make node, not edge
            if j == 0:
                # if not yet seen, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1

            # if not first node in edge, retrieve previous node and build edge
            else:
                prev_loc = (xs[j - 1], ys[j - 1])
                prev_node = node_loc_dic_rev[prev_loc]

                # if new, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                # if seen before, retrieve node properties
                else:
                    node = node_loc_dic_rev[loc]

                # add edge, which is start_node to end_node
                edge_loc = (loc, prev_loc)
                edge_loc_rev = (prev_loc, loc)
                # shouldn't be duplicate edges, so break if we see one
                if (edge_loc in edge_loc_set) or (edge_loc_rev in edge_loc_set):
                    # print("Oops, edge already seen, returning:", edge_loc)
                    # return
                    print("Oops, edge already seen, skipping:", edge_loc)
                    continue

                # get distance to prev_loc and current loc
                proj_prev = shape.project(Point(prev_loc))
                proj = shape.project(Point(loc))
                # edge length is the diffence of the two projected lengths
                #   along the linestring
                edge_length = abs(proj - proj_prev)
                # make linestring
                line_out = LineString([prev_loc, loc])
                line_out_wkt = line_out.wkt

                edge_props = {'start': prev_node,
                              'start_loc_pix': prev_loc,
                              'end': node,
                              'end_loc_pix': loc,
                              'length': edge_length,
                              'wkt_pix': line_out_wkt,
                              'geometry': line_out,
                              'osmid': i}

                edge_loc_set.add(edge_loc)
                edge_dic[edge_iter] = edge_props
                edge_iter += 1

    return node_loc_dic, edge_dic


def nodes_edges_to_G(node_loc_dic, edge_dic):
    '''Take output of wkt_list_to_nodes_edges(wkt_list) and create networkx
    graph'''

    G = nx.MultiDiGraph()

    # add nodes
    # for key,val in node_loc_dic.iteritems():
    for key in node_loc_dic.keys():
        val = node_loc_dic[key]
        attr_dict = {'osmid': key,
                     'x': val[0],
                     'y': val[1]}
        G.add_node(key, **attr_dict)

    # add edges
    # for key,val in edge_dic.iteritems():
    for key in edge_dic.keys():
        val = edge_dic[key]
        attr_dict = val
        u = attr_dict['start']
        v = attr_dict['end']

        G.add_edge(u, v, **attr_dict)

    G1 = G.to_undirected()

    return G1


# ----------- execute utils ----------- #
def make_graphs(G_gt_, G_p_, linestring_delta=20, is_curved_eps=0.03):
    # ---------------- ground truth ----------------- #
    for i, (u, v, data) in enumerate(G_gt_.edges(keys=False, data=True)):
        try:
            line = data['geometry']
        except KeyError:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    # create graph with midpoints
    G_gt0 = G_gt_.to_undirected()
    G_gt_cp, xms, yms = create_graph_midpoints(
        G_gt0.copy(),
        linestring_delta=linestring_delta,
        is_curved_eps=is_curved_eps)

    # get ground truth control points
    control_points_gt = []
    for n in G_gt_cp.nodes():
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])

    # get ground truth paths
    all_pairs_lengths_gt_native = dict(
        nx.shortest_path_length(G_gt_cp, weight='length'))

    # ---------------- proposal ----------------- #
    for i, (u, v, data) in enumerate(G_p_.edges(keys=False, data=True)):
        try:
            line = data['geometry']
        except KeyError:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    # create graph with midpoints
    G_p = G_p_.to_undirected()
    G_p_cp, xms_p, yms_p = create_graph_midpoints(
        G_p.copy(),
        linestring_delta=linestring_delta,
        is_curved_eps=is_curved_eps)

    # get proposal control points
    control_points_prop = []
    for n in G_p_cp.nodes():
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])

    # get paths
    all_pairs_lengths_prop_native = dict(
        nx.shortest_path_length(G_p_cp, weight='length'))

    # insert gt control points into proposal
    G_p_cp_prime, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt)

    # now insert control points into ground truth
    G_gt_cp_prime, xn_gt, yn_gt = insert_control_points(
        G_gt_, control_points_prop)

    # get paths
    all_pairs_lengths_gt_prime = dict(
        nx.shortest_path_length(G_gt_cp_prime, weight='length'))
    all_pairs_lengths_prop_prime = dict(
        nx.shortest_path_length(G_p_cp_prime, weight='length'))

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
           control_points_gt, control_points_prop, \
           all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
           all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime


def create_graph_midpoints(G_, linestring_delta=20, is_curved_eps=0.03):
    """
    Insert midpoint nodes into long edges on the graph.

    :param G_:
    :param linestring_delta:
    :param is_curved_eps:

    :return:
    """
    if len(G_.nodes()) == 0:
        return G_, [], []

    # midpoints
    xms, yms = [], []
    Gout = G_.copy()
    midpoint_name_val, midpoint_name_inc = np.max(G_.nodes()) + 1, 1
    for u, v, key, data in G_.edges(keys=True, data=True):
        # curved line
        if 'geometry' in data:

            # first edge props and  get utm zone and letter
            edge_props_init = G_.edges([u, v])

            linelen = data['length']
            line = data['geometry']

            xs, ys = line.xy  # for plotting

            # check if curved or not
            minx, miny, maxx, maxy = line.bounds
            # get euclidean distance
            dst = scipy.spatial.distance.euclidean([minx, miny], [maxx, maxy])

            # ignore if almost straight
            # if np.abs(dst - linelen) / linelen < is_curved_eps:
            #     continue

            # also ignore super short lines
            if linelen < 0.75 * linestring_delta:
                continue

            # interpolate midpoints
            # if edge is short, use midpoint, else get evenly spaced points
            if linelen <= linestring_delta:
                interp_dists = [0.5 * line.length]
            else:
                # get evenly spaced points
                npoints = len(np.arange(0, linelen, linestring_delta)) + 1
                interp_dists = np.linspace(0, linelen, npoints)[1:-1]

            # create nodes
            node_id_new_list = []
            xms_tmp, yms_tmp = [], []
            for j, d in enumerate(interp_dists):
                midPoint = line.interpolate(d)
                xm0, ym0 = midPoint.xy
                xm = xm0[-1]
                ym = ym0[-1]
                point = Point(xm, ym)
                xms.append(xm)
                yms.append(ym)
                xms_tmp.append(xm)
                yms_tmp.append(ym)

                # add node to graph, with properties of u
                node_id = midpoint_name_val
                # node_id = np.round(u + midpoint_name_val,2)
                midpoint_name_val += midpoint_name_inc
                node_id_new_list.append(node_id)

                # add to graph
                Gout, node_props, xn, yn = insert_point_into_G(Gout, point, node_id=node_id)

    return Gout, xms, yms


def insert_point_into_G(G_, point, node_id=100000, max_distance_meters=10,
                        nearby_nodes_set=set([])):
    """
    Insert a new node in the graph closest to the given point.

    :param G_:
    :param point:
    :param node_id:
    :return:
    """
    best_edge, min_dist, best_geom = get_closest_edge_from_G(G_, point, nearby_nodes_set=nearby_nodes_set)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    if min_dist > max_distance_meters:
        return G_, {}, -1, -1
    else:
        if node_id in G_node_set:
            return G_, {}, -1, -1
        line_geom = best_geom
        # Length along line that is closest to the point
        line_proj = line_geom.project(point)

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_proj)
        x, y = new_point.x, new_point.y

        # create new node
        node_props = {'osmid': node_id, 'x': x, 'y': y}
        G_.add_node(node_id, **node_props)

        # assign, then update edge props for new edge
        _, _, edge_props_new = copy.deepcopy(
            list(G_.edges([u, v], data=True))[0])

        split_line = cut_linestring(line_geom, line_proj)

        # if cp.is_empty:
        if len(split_line) == 1:
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']
            buff = 0.05  # meters
            if (abs(x_p - x_u) <= buff) and (abs(y_p - y_u) <= buff):
                outnode = u
                outnode_x, outnode_y = x_u, y_u
            elif (abs(x_p - x_v) <= buff) and (abs(y_p - y_v) <= buff):
                outnode = v
                outnode_x, outnode_y = x_v, y_v
            else:
                # return
                return G_, {}, 0, 0

            # if the line cannot be split, that means that the new node
            # is coincident with an existing node.  Relabel, if desired
            node_props = G_.nodes[outnode]
            # A dictionary with the old labels as keys and new labels
            #  as values. A partial mapping is allowed.
            mapping = {outnode: node_id}
            Gout = nx.relabel_nodes(G_, mapping)

            return Gout, node_props, x_p, y_p
        else:
            # else, create new edges
            line1, line2 = split_line

            # get distances
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            if dist_to_v < dist_to_u:
                line2, line1 = split_line

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1

            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2

            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)

            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, **edge_props_line1)
                G_.add_edge(node_id, v, **edge_props_line2)
            else:
                G_.add_edge(node_id, u, **edge_props_line1)
                G_.add_edge(v, node_id, **edge_props_line2)

            G_.remove_edge(u, v, key)

            return G_, node_props, x, y


def get_closest_edge_from_G(G_, point, nearby_nodes_set=set([])):
    """Return closest edge to point, and distance to said edge."""
    dist_list = []
    edge_list = []
    geom_list = []
    p = point  # Point(point_coords)
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        if len(nearby_nodes_set) > 0:
            if (u not in nearby_nodes_set) and (v not in nearby_nodes_set):
                continue
        line = data['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u, v, key])
    # get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom


def cut_linestring(line, distance):
    """Cuts a shapely linestring at a specified distance from its starting point."""
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    # iterate through coorda and check if interpolated point has been passed
    # already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if pdl == distance:
            return [LineString(coords[:i + 1]),
                    LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:])]

    # if we've reached here then that means we've encountered a self-loop and
    # the interpolated point is between the final midpoint and the the original
    # node
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [LineString(coords[:i] + [(cp.x, cp.y)]),
            LineString([(cp.x, cp.y)] + coords[i:])]


def insert_control_points(G_, control_points, max_distance_meters=10):
    """
    Wrapper around insert_point_into_G() for all control_points.

    :param G_:
    :param control_points:
    :param max_distance_meters:
    :return:
    """
    # insertion can be super slow so construct kdtree if a large graph
    if len(G_.nodes()) > 1000:
        # construct kdtree of ground truth
        kd_idx_dic, kdtree, pos_arr = G_to_kdtree(G_)

    Gout = G_.copy()
    new_xs, new_ys = [], []
    if len(G_.nodes()) == 0:
        return Gout, new_xs, new_ys

    for i, [node_id, x, y] in enumerate(control_points):
        point = Point(x, y)
        # if large graph, determine nearby nodes
        if len(G_.nodes()) > 1000:
            node_names, dists_m_refine = nodes_near_point(
                n_neighbors=20)
            nearby_nodes_set = set(node_names)
        else:
            nearby_nodes_set = set([])

        # insert point
        Gout, node_props, xnew, ynew = insert_point_into_G(
            Gout, point, node_id=node_id,
            max_distance_meters=max_distance_meters,
            nearby_nodes_set=nearby_nodes_set)

        if (xnew != -1) and (ynew != -1):
            new_xs.append(xnew)
            new_ys.append(ynew)

    return Gout, new_xs, new_ys


def compute_apls_metric(all_pairs_lengths_gt_native,
                        all_pairs_lengths_prop_native,
                        all_pairs_lengths_gt_prime,
                        all_pairs_lengths_prop_prime,
                        control_points_gt, control_points_prop,
                        res_dir=None, min_path_length=10):
    """
    Compute APLS metric and plot results (optional)
    :param all_pairs_lengths_gt_native:
    :param all_pairs_lengths_prop_native:
    :param all_pairs_lengths_gt_prime:
    :param all_pairs_lengths_prop_prime:
    :param control_points_gt:
    :param control_points_prop:
    :param res_dir:
    :param min_path_length:
    :param verbose:
    :param super_verbose:
    :return:
    """

    # compute metric (gt to prop)
    control_nodes = [z[0] for z in control_points_gt]
    C_gt_onto_prop, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True)

    if res_dir:
        scatter_png = str(res_dir.parent / ('scatter_g2p_' + res_dir.name + '.png'))
        hist_png = str(res_dir.parent / ('hist_g2p_' + res_dir.name + '.png'))
        # can't plot route names if there are too many...
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]
        plot_metric(
            C_gt_onto_prop, diffs, routes_str=routes_str,
            figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
            scatter_png=scatter_png,
            hist_png=hist_png)

    # compute metric (prop to gt)
    control_nodes = [z[0] for z in control_points_prop]
    C_prop_onto_gt, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True)

    if res_dir:
        scatter_png = str(res_dir.parent / ('scatter_p2g_'+res_dir.name+'.png'))
        hist_png = str(res_dir.parent / ('hist_p2g_'+res_dir.name+'.png'))
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]
        plot_metric(
            C_prop_onto_gt, diffs, routes_str=routes_str,
            figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
            scatter_png=scatter_png,
            hist_png=hist_png)

    # Total
    if (C_gt_onto_prop <= 0) or (C_prop_onto_gt <= 0) \
            or (np.isnan(C_gt_onto_prop)) or (np.isnan(C_prop_onto_gt)):
        C_tot = 0
    else:
        C_tot = hmean([C_gt_onto_prop, C_prop_onto_gt])
        if np.isnan(C_tot):
            C_tot = 0

    return C_tot, C_gt_onto_prop, C_prop_onto_gt


def path_sim_metric(all_pairs_lengths_gt, all_pairs_lengths_prop,
                    control_nodes=[], min_path_length=10,
                    diff_max=1, missing_path_len=-1, normalize=True):
    """Compute metric for multiple paths."""
    diffs = []
    routes = []
    diff_dic = {}
    gt_start_nodes_set = set(all_pairs_lengths_gt.keys())
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())

    if len(gt_start_nodes_set) == 0:
        return 0, [], [], {}

    # set nodes to inspect
    if len(control_nodes) == 0:
        good_nodes = list(all_pairs_lengths_gt.keys())
    else:
        good_nodes = control_nodes

    # iterate overall start nodes
    # for start_node, paths in all_pairs_lengths.iteritems():
    for start_node in good_nodes:
        node_dic_tmp = {}

        # if we are not careful with control nodes, it's possible that the
        # start node will not be in all_pairs_lengths_gt, in this case use max
        # diff for all routes to that node
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to that node
        if start_node not in gt_start_nodes_set:
            print("for ss, node", start_node, "not in set")
            print("   skipping N paths:", len(
                list(all_pairs_lengths_prop[start_node].keys())))
            for end_node, len_prop in all_pairs_lengths_prop[start_node].items():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            return

        paths = all_pairs_lengths_gt[start_node]

        # CASE 1
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to the start node
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.items():
                if (end_node != start_node) and (end_node in good_nodes):
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
                    node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            # print ("start_node missing:", start_node)
            continue

        # else get proposed paths
        else:
            paths_prop = all_pairs_lengths_prop[start_node]

            # get set of all nodes in paths_prop, and missing_nodes
            end_nodes_gt_set = set(paths.keys()).intersection(good_nodes)
            # end_nodes_gt_set = set(paths.keys()) # old version with all nodes

            end_nodes_prop_set = set(paths_prop.keys())
            missing_nodes = end_nodes_gt_set - end_nodes_prop_set

            # iterate over all paths from node
            for end_node in end_nodes_gt_set:
                # for end_node, len_gt in paths.iteritems():

                len_gt = paths[end_node]
                # skip if too short
                if len_gt < min_path_length:
                    continue

                # get proposed path
                if end_node in end_nodes_prop_set:
                    # CASE 2, end_node in both paths and paths_prop, so
                    # valid path exists
                    len_prop = paths_prop[end_node]
                else:
                    # CASE 3: end_node in paths but not paths_prop, so assign
                    # length as diff_max
                    len_prop = missing_path_len

                # compute path difference metric
                diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                diffs.append(diff)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff

            diff_dic[start_node] = node_dic_tmp

    if len(diffs) == 0:
        return 0, [], [], {}

    # compute Cost
    diff_tot = np.sum(diffs)
    if normalize:
        norm = len(diffs)
        diff_norm = diff_tot / norm
        C = 1. - diff_norm
    else:
        C = diff_tot

    return C, diffs, routes, diff_dic


def single_path_metric(len_gt, len_prop, diff_max=1):
    """Compute APLS metric for single path."""
    if len_gt <= 0:
        return 0
    elif len_prop < 0 and len_gt > 0:
        return diff_max
    else:
        diff_raw = np.abs(len_gt - len_prop) / len_gt
        return np.min([diff_max, diff_raw])


# ------------------ apls_utils ------------------- #
def G_to_kdtree(G_, x_coord='x', y_coord='y'):
    """
    Create kd tree from node positions.
    :param G_:
    :param x_coord:
    :param y_coord:
    :param verbose:
    :return:
    """
    nrows = len(G_.nodes())
    ncols = 2
    kd_idx_dic = {}
    arr = np.zeros((nrows, ncols))
    # populate node array
    for i, n in enumerate(G_.nodes()):
        n_props = G_.node[n]
        if x_coord == 'lon':
            lat, lon = n_props['lat'], n_props['lon']
            x, y = lon, lat
        else:
            x, y = n_props[x_coord], n_props[y_coord]

        arr[i] = [x, y]
        kd_idx_dic[i] = n

    # now create kdtree from numpy array
    kdtree = scipy.spatial.KDTree(arr)
    return kd_idx_dic, kdtree, arr


def nodes_near_point(x, y, kdtree, kd_idx_dic, n_neighbors=-1, radius_m=60):
    """Get nodes near the given point."""

    point = [x, y]

    # query kd tree for nodes of interest
    if n_neighbors > 0:
        node_names, idxs_refine, dists_m_refine = _query_kd_nearest(
            kdtree, kd_idx_dic, point, n_neighbors=n_neighbors)
    else:
        node_names, idxs_refine, dists_m_refine = _query_kd_ball(
            kdtree, kd_idx_dic, point, radius_m)

    return node_names, dists_m_refine  # G_sub


def _query_kd_nearest(kdtree, kd_idx_dic, point, n_neighbors=10,
                      distance_upper_bound=1000):
    '''
    Query the kd-tree for neighbors
    Return nearest node names, distances, nearest node indexes
    If not keep_point, remove the origin point from the list
    '''

    dists_m, idxs = kdtree.query(point, k=n_neighbors,
                                 distance_upper_bound=distance_upper_bound)

    idxs_refine = list(np.asarray(idxs))
    dists_m_refine = list(dists_m)
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


def _query_kd_ball(kdtree, kd_idx_dic, point, r_meters, keep_point=True):
    '''
    Query the kd-tree for neighbors within a distance r of the point
    Return nearest node names, distances, nearest node indexes
    if not keep_point, remove the origin point from the list
    '''

    dists_m, idxs = kdtree.query(point, k=500, distance_upper_bound=r_meters)
    # keep only points within distance and greaater than 0?
    if not keep_point:
        f0 = np.where((dists_m <= r_meters) & (dists_m > 0))
    else:
        f0 = np.where((dists_m <= r_meters))
    idxs_refine = list(np.asarray(idxs)[f0])
    dists_m_refine = list(dists_m[f0])
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


# --------------------- apls_plots ------------------ #
def plot_metric(C, diffs, routes_str=[],
                figsize=(10, 5), scatter_png='', hist_png='',
                scatter_alpha=0.3, scatter_size=2, scatter_cmap='jet',
                dpi=300):
    ''' Plot output of cost metric in both scatterplot and histogram format'''

    # plot diffs
    title = 'Path Length Similarity: ' + str(np.round(C, 2))
    fig, (ax0) = plt.subplots(1, 1, figsize=(1 * figsize[0], figsize[1]))
    # ax0.plot(diffs)
    ax0.scatter(list(range(len(diffs))), diffs, s=scatter_size, c=diffs,
                alpha=scatter_alpha,
                cmap=scatter_cmap)
    if len(routes_str) > 0:
        xticklabel_pad = 0.1
        ax0.set_xticks(list(range(len(diffs))))
        ax0.set_xticklabels(routes_str, rotation=50, fontsize=4)
        ax0.tick_params(axis='x', which='major', pad=xticklabel_pad)

    ax0.set_ylabel('Length Diff (Normalized)')
    ax0.set_xlabel('Path ID')
    ax0.set_title(title)
    # plt.tight_layout()
    if scatter_png:
        plt.savefig(scatter_png, dpi=dpi)
    plt.close('all')

    # plot and plot diffs histo
    bins = np.linspace(0, 1, 30)
    bin_centers = np.mean(list(zip(bins, bins[1:])), axis=1)
    # digitized = np.digitize(diffs, bins)
    # bin_means = [np.array(diffs)[digitized == i].mean() for i in range(1, len(bins))]
    hist, bin_edges = np.histogram(diffs, bins=bins)
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    # ax1.plot(bins[1:],hist, type='bar')
    # ax1.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0] )
    ax1.bar(bin_centers, 1. * hist / len(diffs),
            width=bin_centers[1] - bin_centers[0])
    ax1.set_xlim([0, 1])
    # ax1.set_ylabel('Num Routes')
    ax1.set_ylabel('Frac Num Routes')
    ax1.set_xlabel('Length Diff (Normalized)')
    ax1.set_title('Length Diff Histogram - Score: ' + str(np.round(C, 2)))
    ax1.grid(True)
    # plt.tight_layout()
    if hist_png:
        plt.savefig(hist_png, dpi=dpi)
    plt.close('all')


def gather_files(gt_wkt_file, prop_wkt_file):
    """
     Build lists of ground truth and proposal graphs
    :param gt_wkt_file:
    :param prop_wkt_file:

    :return:
    -------
    gt_list, gt_raw_list, gp_list, root_list, im_loc_list : tuple
        gt_list is a list of ground truth graphs.
        gp_list is a list of proposal graphs
        root_list is a list of names
    """
    gt_list, gp_list, root_list = [], [], []
    df_wkt_gt = pd.read_csv(gt_wkt_file)
    df_wkt = pd.read_csv(prop_wkt_file)

    image_ids = set(df_wkt_gt['ImageId'])
    print('Build lists of ground truth and proposal graphs... \n')
    for image_id in tqdm(image_ids):
        # ground truth
        df_filt = df_wkt_gt['WKT_Pix'][df_wkt_gt['ImageId'] == image_id]
        wkt_list = df_filt.values

        if (len(wkt_list) == 0) or (wkt_list[0] == 'LINESTRING EMPTY'):
            continue

        node_iter, edge_iter = 10000, 10000  # TODO ?
        G_gt_init = wkt_to_G(wkt_list, node_iter=node_iter, edge_iter=edge_iter)

        # propsal
        df_filt = df_wkt['WKT_Pix'][df_wkt['ImageId'] == image_id]
        wkt_list = df_filt.values

        if (len(wkt_list) == 0) or (wkt_list[0] == 'LINESTRING EMPTY'):
            continue

        node_iter, edge_iter = 1000, 1000
        G_p_init = wkt_to_G(wkt_list, node_iter=node_iter, edge_iter=edge_iter)

        gt_list.append(G_gt_init)
        gp_list.append(G_p_init)
        root_list.append(image_id)

    return gt_list, gp_list, root_list


def execute(gt_list, gp_list, root_list, linestring_delta=15,
            min_path_length=3, n_plots=0, id='0'):
    """
    Compute APLS for the input data in gt_list, gp_list
    """
    logger = logging.getLogger('APLS')
    # make dirs
    outdir = Path('output')
    outdir.mkdir(exist_ok=True, parents=True)

    total_C = 0
    print("Compute APLS... \n")
    for i in tqdm(range(len(root_list))):

        outroot = root_list[i]
        G_gt_init = gt_list[i]
        G_p_init = gp_list[i]

        # get graphs with midpoints and geometry
        G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime \
            = make_graphs(G_gt_init, G_p_init, linestring_delta)

        # Metric
        if i < n_plots:
            res_dir = outdir/outroot
        else:
            res_dir = None

        C, C_gt_onto_prop, C_prop_onto_gt = compute_apls_metric(
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
            control_points_gt, control_points_prop,
            min_path_length=min_path_length, res_dir=res_dir)

        total_C += C

    print('avg_apls = {}'.format(total_C / (i + 1)))
    logger.info('{}: avg_apls = {}'.format(id, total_C / (i + 1)))


def main():
    parser = argparse.ArgumentParser(description='Masks into linestrings')
    parser.add_argument('-g', '--gt-wkt-file', type=str, required=True,
                        help='Path to ground truth wkt file.')
    parser.add_argument('-p', '--proposal-wkt-file', type=str, required=True,
                        help='Path to proposal wkt file.')
    parser.add_argument('--linestring_delta', default=20, type=int,
                        help='Distance between midpoints on edges')

    args = parser.parse_args()

    Path('output').mkdir(parents=True, exist_ok=True
                         )
    logger = logging.getLogger('APLS')
    logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler('output/info.log', maxBytes=10485760, backupCount=20, encoding="utf8")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # Gather files
    gt_list, gp_list, root_list = gather_files(args.gt_wkt_file, args.proposal_wkt_file)
    # Compute
    execute(gt_list, gp_list, root_list, linestring_delta=args.linestring_delta, id=args.proposal_wkt_file.split('/')[1])


if __name__ == '__main__':
    main()
