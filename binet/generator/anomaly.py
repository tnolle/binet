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

from binet.processmining import Event


class Anomaly(object):
    def __init__(self):
        self.graph = None

    def __str__(self):
        return str(self.__class__.__name__)

    def apply(self, trace):
        """
        This method applies the anomaly to a given trace

        :param trace: the input trace
        :return: a new trace after the anomaly has been applied
        """
        pass


class DuplicateSequenceAnomaly(Anomaly):
    def __init__(self, max_sequence_size=1):
        self.max_sequence_size = max_sequence_size
        super().__init__()

    def apply(self, trace):
        if len(trace) <= 1:
            trace.attributes['label'] = 'normal'
            return

        sequence_size = np.random.randint(1, min(len(trace), self.max_sequence_size + 1))
        start = np.random.randint(0, len(trace) - sequence_size)

        t = trace.events
        dupe_event = Event.clone(t[start])

        # # set a new random user for the duplicated event
        # user = np.random.choice(dupe_event.attr['_possible_users'])
        # dupe_event.attr['user'] = user

        anomalous_trace = t[:start] + [dupe_event] + t[start:]
        trace.events = anomalous_trace

        trace.attributes["label"] = {
            "anomaly": 'DuplicateSequence',
            "attr": {
                "size": int(sequence_size),
                "start": int(start)
            }
        }


class SkipSequenceAnomaly(Anomaly):
    def __init__(self, max_sequence_size=1):
        self.max_sequence_size = max_sequence_size
        super().__init__()

    def apply(self, trace):
        if len(trace) <= 1:
            trace.attributes['label'] = 'normal'
            return

        sequence_size = np.random.randint(1, min(len(trace), self.max_sequence_size + 1))
        start = np.random.randint(0, len(trace) - sequence_size)
        end = start + sequence_size

        t = trace.events
        trace.events = t[:start] + t[end:]

        trace.attributes["label"] = {
            "anomaly": 'SkipSequence',
            "attr": {
                "size": int(sequence_size),
                "start": int(start),
                "skipped_event": trace[start].to_json()
            }
        }


class SwitchEventsAnomaly(Anomaly):
    def __init__(self, max_distance=1):
        self.max_distance = max_distance
        super().__init__()

    def apply(self, trace):
        if len(trace) <= 3:
            trace.attributes['label'] = 'normal'
            return

        distance = np.random.randint(1, min(len(trace) - 1, self.max_distance + 1))

        first = np.random.randint(0, len(trace) - 1 - distance)
        second = first + distance

        t = trace.events
        trace.events = t[:first] + [t[second]] + t[first + 1:second] + [t[first]] + t[second + 1:]

        trace.attributes["label"] = {
            "anomaly": 'SwitchEvents',
            "attr": {
                "first": int(first),
                "second": int(second)
            }
        }


class IncorrectAttributeAnomaly(Anomaly):
    def __init__(self):
        super().__init__()

    def apply(self, trace):
        index = np.random.randint(0, len(trace))
        random_event = trace[index]
        attributes = self.graph.node[random_event.name].values()
        random_attribute = np.random.choice(list(attributes))
        original_attribute = random_event.attributes[random_attribute.name]
        random_event.attributes[random_attribute.name] = random_attribute.incorrect_value()

        trace.attributes["label"] = {
            "anomaly": 'IncorrectAttribute',
            "attr": {
                "index": int(index),
                "affected": random_attribute.name,
                "original": original_attribute
            }
        }
