import sys
import os

# take a root path of project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from EDA.DataReader import DataReader
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from tqdm import tqdm
import torch
import numpy as np
import dgl
from collections import Counter
from itertools import chain, combinations
from sklearn.preprocessing import LabelEncoder


def custom_tokenizer(text):
    return text.split()


class BuildGraph:
    def __init__(self, name: str):
        # Add variable
        self.test_mask = None
        self.val_mask = None
        self.training_mask = None
        self.n = None
        self.vocab_map = None
        self.vocab = None
        self.corpus_matrix = None
        dataReader = DataReader(name)
        self.df_total = dataReader.df_total
        self.df_train = dataReader.df_train
        self.df_dev = dataReader.df_dev
        self.df_test = dataReader.df_test
        self.nxg = nx.DiGraph()
        # add creat function
        self.pre_processing()
        self.word_doc()
        self.word_word()
        # create graph dgl from graph networkx
        self.g = dgl.from_networkx(self.nxg, edge_attrs=['weight'])
        del self.nxg
        # setup graph
        self.setup_graph()

    def pre_processing(self):
        print("step pre processing")
        tfidf = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words=[])
        self.corpus_matrix = tfidf.fit_transform(self.df_total["corpus"])
        self.vocab = tfidf.get_feature_names_out()
        self.vocab_map = {}
        for idx, v in enumerate(self.vocab):
            self.vocab_map[v] = idx

    def word_doc(self):
        print("step add word doc edge")
        self.n = len(self.df_total["corpus"])

        for doc_id, doc in tqdm(enumerate(self.df_total["corpus"]), desc="Processing documents"):
            words = doc.split()
            word_indices = [self.vocab_map[w] for w in words if w in self.vocab_map]

            word_node_ids = np.array([self.n + wid for wid in word_indices])
            word_idx = self.corpus_matrix[doc_id].toarray()[0]
            # Thêm cạnh vào đồ thị
            for i in range(len(word_indices)):
                self.nxg.add_edge(doc_id, word_node_ids[i], weight=word_idx[word_indices[i]])
                self.nxg.add_edge(word_node_ids[i], doc_id, weight=word_idx[word_indices[i]])

    def word_word(self):
        print("step add word word edge")
        window_size = 10

        # preprocessing list of corpus
        tokenized_corpus = [doc.split() for doc in self.df_total["corpus"]]

        # create list window
        windows = list(chain.from_iterable(
            (words[i:i + window_size] for i in range(len(words) - window_size + 1))
            if len(words) > window_size else [words]
            for words in tokenized_corpus
        ))

        # number of window
        num_window = len(windows)

        # calculate frequency
        word_window_freq = Counter()
        for window in windows:
            appeared = {self.n + self.vocab_map[word] for word in window if word in self.vocab_map}
            word_window_freq.update(appeared)
        word_pair_count = Counter()

        for window in tqdm(windows, desc='Constructing word pair count frequency'):
            word_ids = [self.n + self.vocab_map[word] for word in window if word in self.vocab_map]

            for word_i_id, word_j_id in combinations(word_ids, 2):
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
        # calculate pmi
        edges = []

        for (i, j), count in tqdm(word_pair_count.items(), desc='Adding word_word edges'):
            word_freq_i = word_window_freq[i]
            word_freq_j = word_window_freq[j]

            pmi = np.log((count / num_window) / ((word_freq_i * word_freq_j) / (num_window ** 2)))

            if pmi > 0:
                edges.append((i, j, pmi))

        # Add all edges in the same time
        self.nxg.add_edges_from([(i, j, {'weight': w}) for i, j, w in edges])

    def setup_graph(self):
        print("step setup graph")
        # feature
        feature_vectors = np.concatenate(
            (self.corpus_matrix.toarray(), np.eye(len(self.vocab))),
            axis=0,
            dtype=np.float32
        )
        self.g.ndata['x'] = torch.from_numpy(feature_vectors)
        # label
        le = LabelEncoder()
        y = le.fit_transform(self.df_total["label"])

        word_labels = np.full(
            shape=len(self.vocab),
            fill_value=0,
        )
        yy = np.concatenate((y, word_labels), axis=0)
        yy = torch.from_numpy(yy).to(torch.float32)
        self.g.ndata["label"] = torch.tensor(yy)
        # range
        graph_nodes = self.n + len(self.vocab)

        train_range = range(len(self.df_train))
        val_range = range(len(self.df_train), len(self.df_train) + len(self.df_dev))
        test_range = range(len(self.df_train) + len(self.df_dev), len(self.df_total))
        # mask
        training_mask = np.zeros(graph_nodes)
        training_mask[train_range] = 1
        val_mask = np.zeros(graph_nodes)
        val_mask[val_range] = 1
        test_mask = np.zeros(graph_nodes)
        test_mask[test_range] = 1
        training_mask = training_mask.astype(np.bool_)
        val_mask = val_mask.astype(np.bool_)
        test_mask = test_mask.astype(np.bool_)
        # build graph
        self.g.ndata['train_mask'] = torch.from_numpy(training_mask)
        self.g.ndata['test_mask'] = torch.from_numpy(test_mask)
        self.g.ndata['val_mask'] = torch.from_numpy(val_mask)
        self.g.edata['weight'] = self.g.edata['weight'].to(torch.float32)
