# coding = utf-8
# usr/bin/env python

'''
Author: Jantory
Email: zhang_ze_yu@outlook.com

Time: 2021/9/8 12:24 上午
Software: PyCharm
File: noise.py
desc:
'''
import heapq
import copy
import numpy as np
import random
import math
from networkx.convert_matrix import from_numpy_matrix


import os
import scipy.sparse as sp
from collections import defaultdict

np.random.seed(1)
def no_operation(p, mx):
    "Do not modify the original graph"
    return mx


'''
We manipulate the graph using according to the local information
'''
def del_edge_degree_avg(p, mx):
    """
      For nodes have k-highest degree (k is determined by a probability): If their
      degrees are higher than the average, we delete existing edges so that the
      degrees after modification is at the average degree of the original graph;
      If their degrees are smaller than the average, we add new edges so that the
      degrees after modification is also at the average. In case of digital decimal,
      the whole procedure chooses to round floor.
      Input: p - Probability
             mx - Non-sparse adjacency matrix for the graph (require to be symmetric).
    """
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    ## Degree of each node
    degree = dict(G.degree)

    avg = sum(degree.values()) / len(adj)
    k = int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    for i in top_k_vertex:
        idx = list(adj[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx) > avg:
            idx_del = random.sample(idx, int(len(idx) - avg))
            for j in idx_del:
                adj[i][j] = 0
                adj[j][i] = 0
        else:
            continue

    return adj


def del_edge_degree_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    ## Degree of each node
    degree = dict(G.degree)

    deg_values = np.array(list(degree.values()))
    counts = np.bincount(deg_values)
    mode = np.argmax(counts)
    k = int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    for i in top_k_vertex:
        idx = list(adj[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx) > mode:
            idx_del = random.sample(idx, int(len(idx) - mode))
            for j in idx_del:
                adj[i][j] = 0
                adj[j][i] = 0
        else:
            continue

    return adj


def flip_edge_degree_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    ## Degree of each node
    degree = dict(G.degree)

    avg = sum(degree.values()) / len(adj)
    k = int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    for i in top_k_vertex:
        idx_ones = list(mx[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx_ones) > avg:
            idx_zeros = list(np.where(mx[i] == 0)[0])
            idx_del = random.sample(idx_ones, int(len(idx_ones) - avg - 1))
            idx_add = random.sample(idx_zeros, int(len(idx_ones) - avg - 1))
            for j in idx_del:
                adj[i][j] = 0
                adj[j][i] = 0
            for j in idx_add:
                adj[i][j] = 1
                adj[j][i] = 1

    return adj


def flip_edge_degree_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    ## Degree of each node
    degree = dict(G.degree)

    deg_values = np.array(list(degree.values()))
    counts = np.bincount(deg_values)
    mode = np.argmax(counts)

    k = int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    for i in top_k_vertex:
        idx_ones = list(mx[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx_ones) > mode:
            idx_zeros = list(np.where(mx[i] == 0)[0])
            idx_del = random.sample(idx_ones, int(len(idx_ones) - mode - 1))
            idx_add = random.sample(idx_zeros, int(len(idx_ones) - mode - 1))
            for j in idx_del:
                adj[i][j] = 0
                adj[j][i] = 0
            for j in idx_add:
                adj[i][j] = 1
                adj[j][i] = 1

    return adj


def add_edge_degree_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    ## Degree of each node
    degree = dict(G.degree)

    avg = sum(degree.values()) / len(adj)
    k = int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    s = 0
    for i in top_k_vertex:
        idx_ones = list(mx[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx_ones) > avg:
            idx_zeros = list(np.where(mx[i] == 0)[0])
            idx_add = random.sample(idx_zeros, int((len((idx_ones)) - avg) / 1.11))
            s += len(idx_add)

            for j in idx_add:
                adj[i][j] = 1
                adj[j][i] = 1

    return adj


def add_edge_degree_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    ## Degree of each node
    degree = dict(G.degree)

    deg_values = np.array(list(degree.values()))
    counts = np.bincount(deg_values)
    mode = np.argmax(counts)

    k = int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    s = 0
    for i in top_k_vertex:
        idx_ones = list(mx[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx_ones) > mode:
            idx_zeros = list(np.where(mx[i] == 0)[0])
            idx_add = random.sample(idx_zeros, int((len(idx_ones) - mode) / 1.2))

            for j in idx_add:
                adj[i][j] = 1
                adj[j][i] = 1

    return adj

'''
We manipulate the graph using according to the community information
'''
def graph_partition(G):
    '''
    This function is used for get the best graph partition. We use cached file to avoid
    unnecessary overhead. The dataset here is Cora.
    Input G - The networkx type graph we want to know the partiton
    '''
    import community as community_louvain
    import json

    file_path = 'community.txt'

    if not os.path.isfile(file_path):
        partition = community_louvain.best_partition(G)
        json.dump(partition, open(file_path, 'w'))
        return partition
    else:
        partition = json.load(open(file_path))
        new_partition = dict()
        for key in partition.keys():
            new_partition[eval(key)] = partition[key]
        return new_partition


def graph_edge_partition(G):
    '''
    This function is used for getting the edge partitions. Edges whose endpoints are in
    the same partition are grouped, and edges connect two partitions are grouped
    together.
    Input: partition - the Leuvan partition of graph G
           G - The networkx type graph we want to know the partiton
    '''
    edge_partition = defaultdict(set)
    partition = graph_partition(G)

    for edge in G.edges():
        if partition[edge[0]] == partition[edge[1]]:
            edge_partition[partition[edge[0]]].add((edge[0], edge[1]))
        else:
            edge_partition[-1].add((edge[0], edge[1]))
    return edge_partition, partition


def del_edge_community_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition = graph_edge_partition(G)[0]
    for key in edge_partition.keys():
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def del_edge_community_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition = graph_edge_partition(G)[0]
    for key in edge_partition.keys():
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def flip_edge_community_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, partition = graph_edge_partition(G)
    partition_trans = defaultdict(set)

    for key in partition.keys():
        partition_trans[partition[key]].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        if key == -1:
            while num_operate > 0:
                sample_population = list(edge_partition.keys())
                sample_population.remove(-1)
                c1, c2 = random.sample(sample_population, 2)
                source = random.sample(partition_trans[c1], 1)[0]
                target = random.sample(partition_trans[c2], 1)[0]
                (source_, target_) = (source, target) if source < target else (target, source)
                if (source_, target_) not in edge_partition[key] \
                        and (source_, target_) not in add_edges:
                    add_edges.add((source_, target_))
                    num_operate = num_operate - 1
        else:
            while num_operate > 0:
                if len(partition_trans[key]) > 6:  ## to avoid dead loop, because some partitions are too small
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_ and (source_, target_) not in edge_partition[key] and \
                            (source_, target_) not in add_edges:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
                else:
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def flip_edge_community_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, partition = graph_edge_partition(G)
    partition_trans = defaultdict(set)

    for key in partition.keys():
        partition_trans[partition[key]].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        if key == -1:
            while num_operate > 0:
                sample_population = list(edge_partition.keys())
                sample_population.remove(-1)
                c1, c2 = random.sample(sample_population, 2)
                source = random.sample(partition_trans[c1], 1)[0]
                target = random.sample(partition_trans[c2], 1)[0]
                (source_, target_) = (source, target) if source < target else (target, source)
                if (source_, target_) not in edge_partition[key] \
                        and (source_, target_) not in add_edges:
                    add_edges.add((source_, target_))
                    num_operate = num_operate - 1
        else:
            while num_operate > 0:
                if len(partition_trans[key]) > 0.01 * len(adj):  ## to avoid dead loop, because some partitions are too small
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_ and (source_, target_) not in edge_partition[key] and \
                            (source_, target_) not in add_edges:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
                else:
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def add_edge_community_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, partition = graph_edge_partition(G)
    partition_trans = defaultdict(set)

    for key in partition.keys():
        partition_trans[partition[key]].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        if key == -1:
            while num_operate > 0:
                sample_population = list(edge_partition.keys())
                sample_population.remove(-1)
                c1, c2 = random.sample(sample_population, 2)
                source = random.sample(partition_trans[c1], 1)[0]
                target = random.sample(partition_trans[c2], 1)[0]
                (source_, target_) = (source, target) if source < target else (target, source)
                if (source_, target_) not in edge_partition[key] \
                        and (source_, target_) not in add_edges:
                    add_edges.add((source_, target_))
                    num_operate = num_operate - 1
        else:
            while num_operate > 0:
                if len(partition_trans[key]) > 6:  ## to avoid dead loop
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_ and (source_, target_) not in edge_partition[key] and \
                            (source_, target_) not in add_edges:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
                else:
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def add_edge_community_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, partition = graph_edge_partition(G)
    partition_trans = defaultdict(set)

    for key in partition.keys():
        partition_trans[partition[key]].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        if key == -1:
            while num_operate > 0:
                sample_population = list(edge_partition.keys())
                sample_population.remove(-1)
                c1, c2 = random.sample(sample_population, 2)
                source = random.sample(partition_trans[c1], 1)[0]
                target = random.sample(partition_trans[c2], 1)[0]
                (source_, target_) = (source, target) if source < target else (target, source)
                if (source_, target_) not in edge_partition[key] \
                        and (source_, target_) not in add_edges:
                    add_edges.add((source_, target_))
                    num_operate = num_operate - 1
        else:
            while num_operate > 0:
                if len(partition_trans[key]) > 0.01 * len(adj):  ## to avoid dead loop
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_ and (source_, target_) not in edge_partition[key] and \
                            (source_, target_) not in add_edges:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
                else:
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


'''
We manipulate the graph using according to the global information
'''
def graph_roles(G):
    '''
    This function is used for get the roles in the graph. We set the number of roles
    6. The result is cached to avoid unnecessary overhead. The dataset here is Cora.
    Input G - The networkx type graph we want to know the partiton
    '''
    from graphrole import RecursiveFeatureExtractor, RoleExtractor
    import json

    file_path = 'global.txt'

    if not os.path.isfile(file_path):
        # extract features
        feature_extractor = RecursiveFeatureExtractor(G)
        features = feature_extractor.extract_features()

        role_extractor = RoleExtractor(n_roles=6)
        role_extractor.extract_role_factors(features)
        node_roles = role_extractor.roles
        json.dump(node_roles, open(file_path, 'w'))
        return node_roles
    else:
        node_roles = json.load(open(file_path))
        new_node_roles = dict()
        for key in node_roles.keys():
            new_node_roles[eval(key)] = node_roles[key]
        return new_node_roles


def role_edge_partition(G):
    '''
    This function is used for getting the edge partitions according to graph roles.
    Edges whose endpoints are the same role are grouped, and edges connect different
    roles are grouped
    together.
    Input: G - The networkx type graph we want to know the partiton
    '''
    edge_partition = defaultdict(set)
    node_roles = graph_roles(G)

    for edge in G.edges():
        source = eval(node_roles[edge[0]][-1])
        target = eval(node_roles[edge[1]][-1])
        if source == target:
            edge_partition[(source, target)].add((edge[0], edge[1]))
        else:
            edge_partition[(source, target) if source < target else (target, source)].add((edge[0], edge[1]))
    return edge_partition, node_roles


def del_edge_global_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition = role_edge_partition(G)[0]
    for key in edge_partition.keys():
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def del_edge_global_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition = role_edge_partition(G)[0]
    for key in edge_partition.keys():
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def flip_edge_global_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, node_roles = role_edge_partition(G)
    node_roles_trans = defaultdict(set)

    for key in node_roles.keys():
        node_roles_trans[eval(node_roles[key][-1])].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        while num_operate > 0:
            source = random.sample(node_roles_trans[key[0]], 1)[0]
            target = random.sample(node_roles_trans[key[1]], 1)[0]
            (source_, target_) = (source, target) if source < target else (target, source)
            if source_ != target_ and (source_, target_) not in edge_partition[key] and (
            source_, target_) not in add_edges:
                add_edges.add((source_, target_))
                num_operate = num_operate - 1
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def flip_edge_global_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, node_roles = role_edge_partition(G)
    node_roles_trans = defaultdict(set)

    for key in node_roles.keys():
        node_roles_trans[eval(node_roles[key][-1])].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        while num_operate > 0:
            source = random.sample(node_roles_trans[key[0]], 1)[0]
            target = random.sample(node_roles_trans[key[1]], 1)[0]
            (source_, target_) = (source, target) if source < target else (target, source)
            if source_ != target_ and (source_, target_) not in edge_partition[key] and (
            source_, target_) not in add_edges:
                add_edges.add((source_, target_))
                num_operate = num_operate - 1
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def add_edge_global_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, node_roles = role_edge_partition(G)
    node_roles_trans = defaultdict(set)

    for key in node_roles.keys():
        node_roles_trans[eval(node_roles[key][-1])].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        while num_operate > 0:
            source = random.sample(node_roles_trans[key[0]], 1)[0]
            target = random.sample(node_roles_trans[key[1]], 1)[0]
            (source_, target_) = (source, target) if source < target else (target, source)
            if source_ != target_ and (source_, target_) not in edge_partition[key] and (
            source_, target_) not in add_edges:
                add_edges.add((source_, target_))
                num_operate = num_operate - 1
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def add_edge_global_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_matrix(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, node_roles = role_edge_partition(G)
    node_roles_trans = defaultdict(set)

    for key in node_roles.keys():
        node_roles_trans[eval(node_roles[key][-1])].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        while num_operate > 0:
            source = random.sample(node_roles_trans[key[0]], 1)[0]
            target = random.sample(node_roles_trans[key[1]], 1)[0]
            (source_, target_) = (source, target) if source < target else (target, source)
            if source_ != target_ and (source_, target_) not in edge_partition[key] and (
            source_, target_) not in add_edges:
                add_edges.add((source_, target_))
                num_operate = num_operate - 1
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


func_dict = {0: no_operation,
             1: del_edge_degree_avg, 2: del_edge_degree_mode,
             3: del_edge_community_avg, 4:del_edge_community_mode,
             5: del_edge_global_avg, 6: del_edge_global_mode,
             7:flip_edge_degree_avg, 8: flip_edge_degree_mode,
             9: flip_edge_community_avg, 10: flip_edge_community_mode,
             11: flip_edge_global_avg, 12: flip_edge_global_mode,
             13: add_edge_degree_avg, 14: add_edge_degree_mode,
             15: add_edge_community_avg, 16: add_edge_community_mode,
             17: add_edge_global_avg, 18: add_edge_global_mode}
