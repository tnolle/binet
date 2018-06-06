# BINet: A neural network architecture for anomaly detection in business process event logs.
#
# Copyright (C) 2018 Timo Nolle
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

import numpy as np

from binet.generator.attributes import Attribute, NormalAttribute, NumericAttribute
from binet.generator.attributes import CategoricalAttribute
from binet.generator.attributes import UniformAttribute
from binet.processmining import Case
from binet.processmining import Flowchart, EventLog


class EventLogGenerator(object):
    def __init__(self, flowchart, trace_attributes=None, event_attributes=None):
        if isinstance(flowchart, str):
            self.flowchart = Flowchart.from_plg(flowchart)
        elif isinstance(flowchart, Flowchart):
            self.flowchart = flowchart
        else:
            raise TypeError('Only String and Flowchart are supported.')

        self.start_symbol = EventLog.start_symbol
        self.end_symbol = EventLog.end_symbol

        self.trace_attributes = self._check_attributes(trace_attributes)
        self.event_attributes = self._check_attributes(event_attributes)

        self._possible_traces = None

        if len(self.event_attributes) > 0:
            self._annotate_graph()

    def _annotate_graph(self):
        g = self.flowchart.graph
        attributes = self.event_attributes
        for key in sorted(g.node.keys()):
            for attribute in attributes:
                node_attribute = None

                # Check for attribute type and make copy of attribute with altered parameters
                if isinstance(attribute, CategoricalAttribute):
                    selection = np.random.randint(attribute.min_group, attribute.max_group)
                    node_values = np.sort(np.random.choice(attribute.values, selection, replace=False)).tolist()
                    node_attribute = CategoricalAttribute(
                        name=attribute.name,
                        values=node_values,
                        domain=attribute.domain
                    )
                elif isinstance(attribute, UniformAttribute):
                    low, high = sorted(np.random.randint(attribute.low, attribute.high, 2))
                    node_attribute = UniformAttribute(
                        name=attribute.name,
                        low=low,
                        high=high
                    )
                elif isinstance(attribute, NormalAttribute):
                    mu = np.random.randint(0, 2 * attribute.mu)
                    sigma = np.random.randint(1, 2 * attribute.sigma)
                    node_attribute = NormalAttribute(
                        name=attribute.name,
                        mu=mu,
                        sigma=sigma
                    )

                if node_attribute is not None:
                    g.node[key][attribute.name] = node_attribute
                else:
                    raise TypeError('Attribute type could not be resolved. Should be Categorical, Uniform, or Normal')

    def generate_valid_cases(self, size=100, variant_probabilities=None):
        variants = self.flowchart.variants
        attributes = self.event_attributes

        if variant_probabilities is None:
            variant_probabilities = self.flowchart.variant_probabilities

        if attributes is None:
            return variants
        else:
            variants = np.random.choice(variants, size, replace=True, p=variant_probabilities)

        cases = []
        for case in variants:
            case = Case.clone(case)
            for event in case:
                node = self.flowchart.graph.node[event.name]
                event.attributes = dict((attr.name, attr.random_value()) for attr in node.values() if
                                        isinstance(attr, CategoricalAttribute))
            cases.append(case)

        return cases

    @property
    def valid_traces(self):
        if self._possible_traces is None:
            self._possible_traces = self.generate_valid_cases()
        return self._possible_traces

    @valid_traces.setter
    def valid_traces(self, traces):
        self._possible_traces = traces

    def _get_event_attributes_dict(self):
        attributes = {}

        for event in self.flowchart.graph.nodes():
            if event not in [EventLog.start_symbol, EventLog.end_symbol]:
                node = self.flowchart.graph.node[event]
                attributes[event] = [attribute.to_json() for attribute in node.values()]

        return attributes

    def _get_trace_attributes_dict(self):
        return [attr.to_json() for attr in self.trace_attributes]

    @staticmethod
    def _generate_attributes(size):
        attributes = np.random.choice(range(1, 4), size)
        attr = []
        for i, attribute in enumerate(attributes):
            if attribute == 1:
                attr.append(CategoricalAttribute(name='attr_{}'.format(i), values=np.random.randint(5, 25)))
            elif attribute == 2:
                attr.append(UniformAttribute(name='attr_{}'.format(i)))
            elif attribute == 3:
                attr.append(NormalAttribute(name='attr_{}'.format(i)))
        return attr

    def _check_attributes(self, attributes):
        if isinstance(attributes, int):
            return self._generate_attributes(size=attributes)
        elif isinstance(attributes, list):
            if not all([isinstance(a, Attribute) for a in attributes]):
                raise TypeError('Not all attributes are of class Attribute.')
            else:
                return attributes
        else:
            return []

    def generate(self, size, anomalies=None, p=None, variant_probabilities=None, complexity=.2):
        if self._possible_traces is None:
            self.valid_traces = self.generate_valid_cases(variant_probabilities=variant_probabilities,
                                                          size=int(np.floor(size * complexity)))

        poss = self.valid_traces

        idx = np.random.choice(np.arange(len(poss)), size, replace=size > len(poss))
        traces = [Case.clone(poss[i]) for i in idx]

        for i, trace in enumerate(traces):
            trace.id = i + 1
            trace.attributes['label'] = 'normal'

            for attribute in self.trace_attributes:
                trace.attributes[attribute.name] = attribute.random_value()

            for event in trace:
                node = self.flowchart.graph.node[event.name]
                attributes = dict((attr.name, attr.random_value()) for attr in node.values() if
                                  isinstance(attr, NumericAttribute))
                event.attributes = {**event.attributes, **attributes}

            if anomalies is not None and p is not None:
                if np.random.uniform(0, 1) <= p:
                    anomaly = np.random.choice(anomalies)
                    anomaly.graph = self.flowchart.graph
                    anomaly.apply(trace)
                    # if trace in poss:
                    #     trace.attributes['label'] = 'normal'

        return EventLog(
            cases=traces,
            event_attributes=self._get_event_attributes_dict(),
            trace_attributes=self._get_trace_attributes_dict(),
            p=p,
            valid_traces=[t.to_json() for t in traces]
        )
