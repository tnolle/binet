# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""Process mining algorithms."""

import numpy as np


class Miner(object):
    """Abstract miner class."""

    def __init__(self):
        pass

    def mine(self, eventlog_name):
        """Mine a process model from the given event log.

        This method must be implemented by the subclasses.

        :param eventlog_name:
        :return:
        """
        raise NotImplementedError


class HeuristicsMiner(Miner):
    def __init__(self, dependency=0.9, relative_to_best=0.05, length_one_loops=0.9, length_two_loops=0.9,
                 all_tasks_connected=True, adj_mat=None):
        super(HeuristicsMiner, self).__init__()

        # Adjacency matrix
        self.adj_mat = adj_mat

        # Thresholds
        self.dependency = dependency
        self.relative_to_best = relative_to_best
        self.length_one_loops = length_one_loops
        self.length_two_loops = length_two_loops

        # Heuristics
        self.all_tasks_connected = all_tasks_connected

    def mine(self, features):
        traces = features.reshape(features.shape[:2])

        # Direct sequence relation and length two loops relation
        dsr = HeuristicsMiner.get_direct_sequence_relation(traces)
        l2lr = HeuristicsMiner.get_length_two_loops(traces)

        # Depedency threshold
        self.adj_mat = dsr >= self.dependency

        # Calculate relative to best
        self.adj_mat[dsr < np.max(dsr, axis=1).T - self.relative_to_best] = False

        # Length one loops
        self.adj_mat[np.diag_indices_from(self.adj_mat)] = np.diag(dsr) >= self.length_one_loops

        # Length two loops
        length_one_loops = np.logical_and(np.diag(self.adj_mat), np.diag(self.adj_mat).T)
        length_two_loops = np.logical_and(~length_one_loops, l2lr >= self.length_two_loops)
        self.adj_mat[length_two_loops] = True

        # All tasks connected heuristic
        if self.all_tasks_connected:
            idx = ~np.any(self.adj_mat, axis=1)
            idx[:2] = False  # Start and end activity do not count
            self.adj_mat[idx, :] = dsr[idx] == np.expand_dims(dsr[idx].max(axis=1), axis=1)

        return self.adj_mat

    @staticmethod
    def get_direct_sequence_relation(traces):
        num_act = int(traces.max())
        counts = np.zeros((num_act, num_act), dtype=int)
        ngrams = HeuristicsMiner.get_two_grams(traces, flatten=True)
        for ngram in ngrams:
            if not np.any(ngram == 0):
                counts[ngram[0] - 1, ngram[1] - 1] += 1
        np.fill_diagonal(counts.T, 0)
        return (counts - counts.T) / (counts + counts.T + 1)

    @staticmethod
    def get_length_two_loops(traces):
        num_act = int(traces.max())
        counts = np.zeros((num_act, num_act), dtype=int)
        ngrams = HeuristicsMiner.get_three_grams(traces, flatten=True).astype(int)
        ngrams = ngrams[np.logical_and(ngrams[:, 0] == ngrams[:, 2], ngrams[:, 0] != ngrams[:, 1])][:, :-1]
        for ngram in ngrams:
            counts[ngram[0] - 1, ngram[1] - 1] += 1
        return (counts - counts.T) / (counts + counts.T + 1)

    def conformance_check(self, features):
        traces = features.reshape(features.shape[:2])
        ngrams = HeuristicsMiner.get_two_grams(traces, flatten=False).astype(int)
        conformance = np.zeros(ngrams.shape[:-1]).astype(bool)
        for (i, j), _ in np.ndenumerate(conformance):
            a = ngrams[i, j, 0] - 1
            b = ngrams[i, j, 1] - 1
            if a == -1 or b == -1:
                conformance[i, j] = True
            else:
                conformance[i, j] = self.adj_mat[a, b]
        return np.pad(conformance, ((0, 0), (1, 0)), mode='constant', constant_values=1)

    @staticmethod
    def get_two_grams(traces, flatten=True):
        ngrams = np.dstack((traces[:, :-1], traces[:, 1:]))
        if flatten:
            ngrams = ngrams.reshape((np.product(ngrams.shape[:-1]), 2)).astype(int)
        return ngrams

    @staticmethod
    def get_three_grams(traces, flatten=True):
        ngrams = np.dstack((traces[:, :-2], traces[:, 1:-1], traces[:, 2:]))
        if flatten:
            ngrams = ngrams.reshape((np.product(ngrams.shape[:-1]), 3)).astype(int)
        return ngrams
