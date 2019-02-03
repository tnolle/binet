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

import _pickle as pickle
import os

import networkx as nx
import numpy as np
import untangle
from matplotlib import pyplot as plt

from april.fs import PLOT_DIR
from april.fs import PROCESS_MODEL_DIR
from april.processmining import Case
from april.processmining import Event
from april.processmining.log import EventLog
from april.utils import microsoft_colors


class ProcessMap(object):
    def __init__(self, graph=None):
        self.graph = graph
        self.start_event = EventLog.start_symbol
        self.end_event = EventLog.end_symbol

        self._variants = None
        self._variant_probabilities = None

    def load(self, file):
        """
        Load from a pickle file

        :param file:
        :return:
        """
        with open(file, 'rb') as f:
            self.graph = pickle.load(f)

    def save(self, file):
        """
        Save to a pickle file

        :param file:
        :return:
        """
        with open(file, 'wb') as f:
            pickle.dump(self.graph, f)

    def _check_edge(self, edge):
        """
        Returns whether the edge is an anomaly or not.
        True = anomaly
        False = normal

        :param edge: edge
        :return: boolean
        """
        return edge in self.graph.edges()

    def _check_edges(self, edges):
        """
        Returns for a list of given edges whether an edge is an anomaly. Cf. check_edge()

        :param edges: list of edges
        :return: list of booleans
        """
        return np.array([self._check_edge(e) for e in edges])

    def _check_trace(self, trace):
        """
        Returns a list of booleans representing whether a transition within the trace is an anomaly or not.
        True = anomaly
        False = normal

        :param trace: Trace object
        :return: list of booleans
        """

        # zip(...) generates the edges from the traces
        return self._check_edges(zip(trace[:-1], trace[1:]))

    def check_traces(self, traces):
        """
        Returns a list of booleans for each trace. See check_trace().

        :param traces: list of traces
        :return: list of list of booleans
        """
        return np.array([self._check_trace(s) for s in traces])

    def _get_variants(self):
        # variants
        variants = sorted(nx.all_simple_paths(self.graph, source=self.start_event, target=self.end_event))
        traces = [Case(id=i + 1, events=[Event(name=e) for e in v[1:-1]]) for i, v in enumerate(variants)]

        # probabilities
        def get_num_successors(x):
            return len([edge[1] for edge in self.graph.edges() if edge[0] == x])

        probabilities = [np.product([1 / max(1, get_num_successors(node)) for node in path]) for path in variants]

        # set globally
        self._variants = EventLog(cases=traces)
        self._variant_probabilities = probabilities

        return self._variants, self._variant_probabilities

    @property
    def activities(self):
        return sorted(n for n in self.graph if n != EventLog.start_symbol and n != EventLog.end_symbol)

    @property
    def variants(self):
        if self._variants is None:
            self._get_variants()
        return self._variants

    @property
    def variant_probabilities(self):
        if self._variant_probabilities is None:
            self._get_variants()
        return self._variant_probabilities

    @staticmethod
    def from_plg(file_path):
        """Load a process model from a plg file (the format PLG2 uses).

        Gates will be ignored in the resulting process map.

        :param file_path: path to plg file
        :return: ProcessMap object
        """

        if not file_path.endswith('.plg'):
            file_path += '.plg'
        if not os.path.isabs(file_path):
            file_path = os.path.join(PROCESS_MODEL_DIR, file_path)

        with open(file_path) as f:
            file_content = untangle.parse(f.read())

        start_event = int(file_content.process.elements.startEvent['id'])
        end_event = int(file_content.process.elements.endEvent['id'])

        id_activity = dict((int(task['id']), str(task['name'])) for task in file_content.process.elements.task)
        id_activity[start_event] = EventLog.start_symbol
        id_activity[end_event] = EventLog.end_symbol

        activities = id_activity.keys()

        gateways = [int(g['id']) for g in file_content.process.elements.gateway]
        gateway_followers = dict((id_, []) for id_ in gateways)
        followers = dict((id_, []) for id_ in activities)

        for sf in file_content.process.elements.sequenceFlow:
            source = int(sf['sourceRef'])
            target = int(sf['targetRef'])
            if source in gateways:
                gateway_followers[source].append(target)

        for sf in file_content.process.elements.sequenceFlow:
            source = int(sf['sourceRef'])
            target = int(sf['targetRef'])
            if source in activities and target in activities:
                followers[source].append(target)
            elif source in activities and target in gateways:
                followers[source] = gateway_followers.get(target)

        graph = nx.DiGraph()
        graph.add_nodes_from([id_activity.get(activity) for activity in activities])
        for source, targets in followers.items():
            for target in targets:
                graph.add_edge(id_activity.get(source), id_activity.get(target))

        return ProcessMap(graph)

    def plot_process_map(self, name=None, figsize=None):
        g = self.graph

        # Draw
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')

        # Set figure size
        if figsize is None:
            figsize = (8, 8)
        fig = plt.figure(3, figsize=figsize)

        color_map = []
        for node in g:
            if node in [EventLog.start_symbol, EventLog.end_symbol]:
                color_map.append(microsoft_colors[0])
            else:
                color_map.append(microsoft_colors[2])

        nx.draw(g, pos, node_color=color_map, with_labels=True)

        if name is not None:
            # Save to disk
            plt.tight_layout()
            fig.savefig(str(PLOT_DIR / name))
            plt.close()
        else:
            plt.show()
