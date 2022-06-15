import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
import networkx as nx
from collections import defaultdict

import torch.nn.functional as F
from noise import func_dict
import copy
import argparse
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import pickle as pkl


"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            # samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
            samp_neighs = [samp_neigh.union(set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]
            # print(samp_neighs)

        # for item in samp_neighs:
        #     if len(item) == 0:
        #         print("I have empty sample")
        #     break

        unique_nodes_list = list(set.union(*samp_neighs))
      #  print ("\n unl's size=",len(unique_nodes_list))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
                self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_cora(p=0.01, f=0):
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("./cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = [float(x) for x in info[1:-1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    adj = np.zeros((2708, 2708))
    with open("./cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            adj[paper1][paper2] = 1
            adj[paper2][paper1] = 1

    mod_adj = func_dict[f](p=p, mx=adj)
    new_adj_lists = defaultdict(set)
    if p > 0 and p <= 1:
      for key in range(len(mod_adj)):
        for value in range(len(mod_adj)):
          if mod_adj[key][value] == 1:
            new_adj_lists[key].add(value)
      return feat_data, labels, new_adj_lists

    return feat_data, labels, adj_lists


def run_cora(p=0, f=0):
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora(p=p, f=f)
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True, gcn=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False, gcn=True)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        # print (batch, loss.item())

    val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    # print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    # print("Test Accuracy:", accuracy_score(labels[test], test_output.data.numpy().argmax(axis=1)))
    return accuracy_score(labels[test], test_output.data.numpy().argmax(axis=1))
    # print ("Average batch time:", np.mean(times))


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_cite(p=0, f=0):
    dataset_str = 'citeseer'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./citeseer/ind.{}.{}".format(dataset_str, names[i]), 'rb') as file:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(file, encoding='latin1'))
            else:
                objects.append(pkl.load(file))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./citeseer/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))

    adj = adj.toarray()
    if p > 0 and p <= 1:
        adj = func_dict[f](p=p, mx=adj)

    adj_lists = defaultdict(set)
    for key in range(len(adj)):
        for value in range(len(adj)):
            if adj[key][value] == 1:
                adj_lists[key].add(value)

    labels = np.argmax(labels, axis=1).reshape(-1, 1)
    return features, adj_lists, labels

def run_cite(p=0, f=0):
    feat_data, adj_lists, labels = load_cite(p=p, f=f)

    num_nodes = 3327
    features = nn.Embedding(3327, 3703)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True, gcn=True)
    enc1 = Encoder(features, 3703, 256, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False, gcn=True)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 256, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(6, enc2)
    #    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print (batch, loss.item())

    val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Test Accuracy:", accuracy_score(labels[test], test_output.data.numpy().argmax(axis=1)))
    return accuracy_score(labels[test], test_output.data.numpy().argmax(axis=1))
    print ("Average batch time:", np.mean(times))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=float, default=0,
                        help='Proportion of nodes required to be modified.')
    parser.add_argument('--f', type=int, default=0,
                        help='The operation we want to have on the graph')
    parser.add_argument('--seed', type=int, default=6,
                        help='The random seed we want to fix')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    acc = run_cite(args.p, args.f)
    print(func_dict[args.f].__name__, '\tp:', args.p, '\taccuracy:', acc)

