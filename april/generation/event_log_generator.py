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

import itertools
import uuid

import networkx as nx
import numpy as np

from april.fs import PLOT_DIR
from april.generation import AttributeGenerator
from april.generation import NoneAnomaly
from april.processmining import ProcessMap
from april.processmining.log import EventLog


class EventLogGenerator(object):
    def __init__(self, process_map=None, event_attributes=None):
        self.process_map = None
        self.likelihood_graph = None
        self.event_attributes = self._check_attributes(event_attributes)

        if process_map is not None:
            if isinstance(process_map, str):
                self.process_map = ProcessMap.from_plg(process_map)
            elif isinstance(process_map, ProcessMap):
                self.process_map = process_map
            else:
                raise TypeError('Only String and ProcessMap are supported.')

    @staticmethod
    def _check_attributes(attributes):
        if isinstance(attributes, list):
            if not all([isinstance(a, AttributeGenerator) for a in attributes]):
                raise TypeError('Not all attributes are of class Attribute.')
            else:
                return attributes
        else:
            return []

    def build_likelihood_graph(self,
                               activity_dependency_p=0.0,
                               attribute_dependency_p=0.0,
                               probability_variance_max=None,
                               seed=None):

        def add_attribute_dependency_between(source, target, p):
            attribute_values = []
            attribute_min_groups = []
            attribute_max_groups = []

            for attribute in self.event_attributes:
                attribute_min_groups.append(attribute.min_group)
                attribute_max_groups.append(attribute.max_group)

                num_values = np.random.randint(attribute.min_group, attribute.max_group + 1)

                values = np.random.choice(attribute.values, num_values, replace=False)

                attribute_values.append(values)

            combinations = np.array(list(itertools.product(*attribute_values)))
            if np.random.uniform(0, 1) >= p:
                random_indices = range(len(combinations))
            else:
                random_indices = np.random.choice(
                    range(len(combinations)),
                    np.random.randint(
                        np.max(attribute_min_groups),
                        np.max(attribute_max_groups)
                    )
                )

            nodes = {source: source, target: target}
            for attribute_values in combinations[random_indices]:
                path = [source, *attribute_values, target]
                names = [a.name for a in self.event_attributes]

                for i, (s, t) in enumerate(zip(path[:-1], path[1:])):
                    if s not in nodes:
                        nodes[s] = uuid.uuid1()
                        self.likelihood_graph.add_node(nodes[s], name=names[i - 1], value=s)

                    if t not in nodes:
                        nodes[t] = uuid.uuid1()
                        self.likelihood_graph.add_node(nodes[t], name=names[i], value=t)

                    self.likelihood_graph.add_edge(nodes[s], nodes[t])

        def add_activity_dependency_to(g, source):
            source_value = self.likelihood_graph.nodes[source]['value']

            if source_value == EventLog.end_symbol:
                return
            else:
                targets = []
                for target in g.successors(source_value):
                    if target not in nodes:
                        nodes[target] = []

                    split_activity = np.random.uniform(0, 1) <= activity_dependency_p
                    if (split_activity or not nodes[target]) and target != EventLog.end_symbol:
                        identifier = uuid.uuid1()
                        nodes[target].append(identifier)
                        self.likelihood_graph.add_node(identifier, value=target, name='name')
                        targets.append(identifier)
                    else:
                        targets.append(np.random.choice(nodes[target]))

                for target in targets:
                    if source_value != EventLog.start_symbol:
                        if source not in edges:
                            edges[source] = []

                        if target not in edges[source]:
                            if len(self.event_attributes) > 0:
                                add_attribute_dependency_between(source, target, attribute_dependency_p)
                            else:
                                self.likelihood_graph.add_edge(source, target)
                            edges[source].append(target)
                    else:
                        self.likelihood_graph.add_edge(source, target)

                    add_activity_dependency_to(g, target)

        # Set seed for consistency
        if seed is not None:
            np.random.seed(seed)

        # Init graph
        self.likelihood_graph = nx.DiGraph()

        # Init helper dictionaries
        nodes = {}
        edges = {}
        for node in self.process_map.graph:
            if node in [EventLog.start_symbol, EventLog.end_symbol]:
                self.likelihood_graph.add_node(node, value=node, name='name')
                nodes[node] = [node]

        # Add attribute and activity dependencies
        add_activity_dependency_to(self.process_map.graph, EventLog.start_symbol)

        # Annotate with probabilities
        for node in self.likelihood_graph:
            if node == EventLog.end_symbol:
                continue

            successors = list(self.likelihood_graph.successors(node))

            if probability_variance_max is not None:
                variance = np.random.random() * np.abs(probability_variance_max) + .0001
                probabilities = np.abs(np.random.normal(0, variance, len(successors)))
                probabilities /= np.sum(probabilities)
            else:
                probabilities = np.ones(len(successors)) / len(successors)

            for successor, probability in zip(successors, probabilities):
                self.likelihood_graph.node[successor]['probability'] = probability
                self.likelihood_graph.edges[node, successor]['probability'] = np.round(probability, 2)

        return self.likelihood_graph

    def generate(self,
                 size,
                 anomalies=None,
                 anomaly_p=None,
                 anomaly_type_p=None,
                 activity_dependency_p=.5,
                 attribute_dependency_p=.5,
                 probability_variance_max=None,
                 seed=None,
                 show_progress='tqdm',
                 likelihood_graph=None):

        def random_walk(g):
            node = EventLog.start_symbol

            # Random walk until we reach the end event
            path = []
            while node != EventLog.end_symbol:
                # Skip the start node
                if node != EventLog.start_symbol:
                    path.append(node)

                # Get successors for node
                successors = list(g.successors(node))

                # Retrieve probabilities from nodes
                p = [g.edges[node, s]['probability'] for s in successors]

                # Check for and fix rounding errors
                if np.sum(p) != 0:
                    p /= np.sum(p)

                # Chose random successor based on probabilities
                node = np.random.choice(successors, p=p)

            return path

        if seed is not None:
            np.random.seed(seed)

        # Build the likelihood graph
        # TODO: Persist the likelihood graph
        if likelihood_graph is not None:
            self.likelihood_graph = likelihood_graph
        else:
            self.build_likelihood_graph(
                activity_dependency_p=activity_dependency_p,
                attribute_dependency_p=attribute_dependency_p,
                probability_variance_max=probability_variance_max,
                seed=seed
            )

        # Add metadata to anomalies
        activities = sorted(list(set([self.likelihood_graph.nodes[node]['value'] for node in self.likelihood_graph
                                      if self.likelihood_graph.nodes[node]['name'] == 'name'
                                      and self.likelihood_graph.nodes[node]['value'] not in
                                      [EventLog.start_symbol, EventLog.end_symbol]])))
        none_anomaly = NoneAnomaly()
        none_anomaly.activities = activities
        none_anomaly.graph = self.likelihood_graph
        none_anomaly.attributes = self.event_attributes
        for anomaly in anomalies:
            anomaly.activities = activities
            anomaly.graph = self.likelihood_graph
            anomaly.attributes = self.event_attributes

        # Generate the event log
        if show_progress == 'tqdm':
            from tqdm import tqdm
            iter = tqdm(range(size), desc='Generate event log')
        elif show_progress == 'tqdm_notebook':
            from tqdm import tqdm_notebook
            iter = tqdm_notebook(range(size), desc='Generate event log')
        else:
            iter = range(size)

        # Apply anomalies and add case id
        cases = []
        for case_id, path in enumerate([random_walk(self.likelihood_graph) for _ in iter], start=1):
            if np.random.uniform(0, 1) <= anomaly_p:
                anomaly = np.random.choice(anomalies, p=anomaly_type_p)
            else:
                anomaly = none_anomaly
            case = anomaly.apply_to_path(path)
            case.id = case_id
            cases.append(case)

        event_log = EventLog(cases=cases)

        event_log.attributes['generation_parameters'] = dict(
            size=size,
            attributes=[a.json for a in self.event_attributes],
            anomalies=[a.json for a in anomalies],
            anomaly_p=anomaly_p,
            anomaly_type_p=anomaly_type_p,
            activity_dependency_p=activity_dependency_p,
            attribute_dependency_p=attribute_dependency_p,
            probability_variance_max=probability_variance_max,
            seed=int(seed)
        )

        return event_log

    def plot_likelihood_graph(self, file_name=None, figsize=None):
        from april.utils import microsoft_colors
        from matplotlib import pylab as plt

        l = self.likelihood_graph
        pos = nx.drawing.nx_agraph.graphviz_layout(l, prog='dot')

        if figsize is None:
            figsize = (10, 14)
        fig = plt.figure(1, figsize=figsize)

        attribute_names = [a.name for a in self.event_attributes]
        attribute_colors = microsoft_colors[3:]
        colors = dict(zip(attribute_names, attribute_colors))

        color_map = []
        for node in l:
            if node in [EventLog.start_symbol, EventLog.end_symbol]:
                color_map.append(microsoft_colors[0])
            elif l.nodes[node]['name'] == 'name':
                color_map.append(microsoft_colors[2])
            else:
                color_map.append(colors[l.nodes[node]['name']])
        nx.draw(l, pos, node_color=color_map)
        nx.draw_networkx_labels(l, pos, labels=nx.get_node_attributes(l, 'value'))
        nx.draw_networkx_edge_labels(l, pos, edge_labels=nx.get_edge_attributes(l, 'probability'))

        if file_name is not None:
            # Save to disk
            fig.savefig(str(PLOT_DIR / file_name))
            plt.close()
        else:
            plt.show()
