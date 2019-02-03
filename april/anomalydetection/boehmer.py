#  Copyright 2018 Timo Nolle
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  ==============================================================================

import networkx as nx
import numpy as np

from april.anomalydetection import AnomalyDetectionResult
from april.anomalydetection import AnomalyDetector
from april.enums import Heuristic
from april.enums import Strategy


class LikelihoodAnomalyDetector(AnomalyDetector):
    def __init__(self, model=None):
        super(LikelihoodAnomalyDetector, self).__init__(model=model)

    def load(self, file_name):
        super(LikelihoodAnomalyDetector, self).load(file_name)

    def fit(self, dataset):
        self._model = self.get_extended_likelihood_graph(dataset)

    def detect(self, dataset):
        raise NotImplementedError()

    def get_edges(self, dataset, flatten=True, positional=False):
        cases = dataset.flat_features.astype(int)
        cases = np.pad(cases, ((0, 0), (1, 0), (0, 0)), mode='constant')
        cases = np.dstack((cases[:, :-1], cases[:, 1:]))

        if flatten:
            edges = []
        else:
            edges = np.empty((*dataset.flat_features.shape, 2), dtype=object)
        for i, case in enumerate(cases):
            for j, event in enumerate(case):
                if flatten and event[0] == 0 and j != 0:
                    continue

                last_event = event[:dataset.num_attributes]
                event = event[dataset.num_attributes:]
                new_event = [f'{last_event[0]}-{dataset.num_attributes - 1}-{last_event[-1]}']
                for k, e in enumerate(event):
                    act = f'{j}-{e}' if positional else f'{e}'
                    new_event.append(f'{event[0]}-{k}-{e}' if k != 0 else act)
                for k, edge in enumerate(zip(new_event[:-1], new_event[1:])):
                    if flatten:
                        edges.append(edge)
                    else:
                        edges[i, j, k] = edge
        return edges

    def get_extended_likelihood_graph(self, dataset, positional=False):
        edges = self.get_edges(dataset, positional=positional)
        edges, counts = np.unique(np.array(edges), return_counts=True, axis=0)

        graph = nx.DiGraph()
        graph.add_weighted_edges_from([(s, t, w) for (s, t), w in zip(edges, counts)])

        for node in graph:
            successors = list(graph.successors(node))
            node_volume = sum([graph.edges[node, s]['weight'] for s in successors])
            for successor in successors:
                graph.edges[node, successor]['weight'] = graph.edges[node, successor]['weight'] / node_volume

        return graph


class LikelihoodPlusAnomalyDetector(LikelihoodAnomalyDetector):
    """Implements the likelihood graph inspired by (Böhmer 2016).

    First builds a likelihood graph of all activities. Then this graph is extended to include all attributes.
    The anomaly score is derived from the probabilities in the likelihood graph.
    """

    abbreviation = 'likelihood+'
    name = 'Likelihood+'

    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supports_attributes = True

    def __init__(self, model=None):
        super(LikelihoodPlusAnomalyDetector, self).__init__(model=model)

    def detect(self, dataset):
        edges = self.get_edges(dataset, flatten=False)
        scores = np.zeros_like(dataset.flat_features)
        for (i, j, k), _ in np.ndenumerate(scores):
            p = self.model.get_edge_data(*edges[i, j, k], default=dict(weight=0))['weight']
            scores[i, j, k] = 1 - p

        return AnomalyDetectionResult(scores=scores)


class BoehmerLikelihoodAnomalyDetector(LikelihoodAnomalyDetector):
    """Implements the likelihood graph method from (Böhmer 2016).

    First builds a likelihood graph of all activities. Then this graph is extended to include all attributes.
    The anomaly score is derived from the probabilities in the likelihood graph.
    """

    abbreviation = 'likelihood'
    name = 'Likelihood'

    supported_strategies = [Strategy.SINGLE]
    supported_heuristics = [Heuristic.DEFAULT]

    def __init__(self, model=None):
        super(BoehmerLikelihoodAnomalyDetector, self).__init__(model=model)

    def fit(self, dataset):
        self._model = self.get_extended_likelihood_graph(dataset, positional=True)

        # edges = self.get_edges(dataset, flatten=False, positional=True)
        # for i in range(dataset.num_cases):
        #     p = 0
        #     for j in range(dataset.max_len):
        #         if j != 0 and np.sum(dataset.flat_features[i, j]) == 0:
        #             continue
        #         for k in range(dataset.num_attributes):
        #             s, t = edges[i, j, k]
        #             p += -np.log(self.model.get_edge_data(s, t, default=dict(weight=0))['weight'])
        #             if 'threshold' not in self.model.nodes[t]:
        #                 self.model.nodes[t]['threshold'] = []
        #             self.model.nodes[t]['threshold'].append(p)
        #
        # for node in self.model:
        #     if 'threshold' in self.model.nodes[node]:
        #         self.model.nodes[node]['threshold'] = np.min(self.model.nodes[node]['threshold'])

    def detect(self, dataset):
        edges = self.get_edges(dataset, flatten=False, positional=True)
        scores = np.zeros_like(dataset.flat_features)
        # for i in range(dataset.num_cases):
        #     p = 0
        #     for j in range(dataset.max_len):
        #         if j != 0 and np.sum(dataset.flat_features[i, j]) == 0:
        #             continue
        #         for k in range(dataset.num_attributes):
        #             s, t = edges[i, j, k]
        #             p += -np.log(self.model.get_edge_data(s, t, default=dict(weight=0))['weight'])
        #             tau = self.model.nodes[t]['threshold']
        #             scores[i, j, k] = p > tau
        return AnomalyDetectionResult(scores=scores)
