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

import inspect
import sys

import numpy as np

from april.enums import Class
from april.processmining.case import Case
from april.processmining.event import Event


class Anomaly(object):
    """Base class for anomaly implementations."""

    def __init__(self):
        self.graph = None
        self.activities = None
        self.attributes = None
        self.name = self.__class__.__name__[:-7]

    def __str__(self):
        return self.name

    @property
    def json(self):
        return dict(anomaly=str(self),
                    parameters=dict((k, v) for k, v in vars(self).items() if k not in ['graph', 'attributes']))

    @property
    def event_len(self):
        n = 1
        if self.attributes is not None:
            n += len(self.attributes)
        return n

    @staticmethod
    def targets(label, num_events, num_attributes):
        """Return targets for the anomaly."""
        return np.zeros((num_events, num_attributes), dtype=int) + Class.NORMAL

    @staticmethod
    def pretty_label(label):
        """Return a text version of the label."""
        return 'Normal'

    def apply_to_case(self, case):
        """
        This method applies the anomaly to a given case

        :param case: the input case
        :return: a new case after the anomaly has been applied
        """
        pass

    def apply_to_path(self, path):
        """
        This method applies the anomaly to a given path in the graph.

        Requires self.graph to be set.

        :param path: the path containing node identifiers for the graph
        :return: a new case after anomaly has been applied
        """
        return self.apply_to_case(self.path_to_case(path))

    def path_to_case(self, p, label=None):
        """
        Converts a given path to a case by traversing the graph and returning a case.

        :param p: path of node identifiers
        :param label: is used to label the case
        :return: a case
        """
        g = self.graph

        case = Case(label=label)
        for i in range(0, len(p), self.event_len):
            event = Event(name=g.node[p[i]]['value'])
            for j in range(1, self.event_len):
                att = g.node[p[i + j]]['name']
                value = g.node[p[i + j]]['value']
                event.attributes[att] = value
            case.add_event(event)

        return case

    def generate_random_event(self):
        if self.activities is None:
            raise RuntimeError('activities has not bee set.')

        event = Event(name=f'Random activity {np.random.randint(1, len(self.activities))}')
        if self.attributes is not None:
            event.attributes = dict(
                (a.name, f'Random {a.name} {np.random.randint(1, len(a.values))}') for a in self.attributes)
        return event


class NoneAnomaly(Anomaly):
    """Return the case unaltered, i.e., normal."""

    def __init__(self):
        super(NoneAnomaly, self).__init__()
        self.name = 'Normal'

    def apply_to_case(self, case):
        case.attributes['label'] = 'normal'
        return case


class ReworkAnomaly(Anomaly):
    """Insert 1 sequence of n events coming from the case later in the case."""

    def __init__(self, max_sequence_size=2, max_distance=0):
        self.max_sequence_size = max_sequence_size
        self.max_distance = max_distance
        super(ReworkAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 1:
            return NoneAnomaly().apply_to_case(case)

        size = np.random.randint(2, min(len(case), self.max_sequence_size) + 1)
        start = np.random.randint(0, len(case) - size + 1)
        distance = np.random.randint(0, min(len(case) - (start + size), self.max_distance) + 1)

        t = case.events
        dupe_sequence = [Event.clone(e) for e in t[start:start + size]]

        inserted = [e.json for e in dupe_sequence]

        anomalous_trace = t[:start + size + distance] + dupe_sequence + t[start + size + distance:]
        case.events = anomalous_trace

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                size=int(size),
                start=int(start + size + distance),
                inserted=inserted
            )
        )

        return case

    @staticmethod
    def targets(label, num_events, num_attributes):
        targets = Anomaly.targets(label, num_events, num_attributes)
        start = label['attr']['start'] + 1
        size = label['attr']['size']
        targets[start:start + size, 0] = Class.REWORK
        return targets

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        inserted = label['attr']['inserted']
        return f'{name} {", ".join([e["name"] for e in inserted])} at {start}'


class SkipSequenceAnomaly(Anomaly):
    """Skip 1 sequence of n events."""

    def __init__(self, max_sequence_size=1):
        self.max_sequence_size = max_sequence_size
        super(SkipSequenceAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 2:
            return NoneAnomaly().apply_to_case(case)

        size = np.random.randint(1, min(len(case) - 1, self.max_sequence_size) + 1)
        start = np.random.randint(0, len(case) - size)
        end = start + size

        t = case.events
        skipped = [s.json for s in t[start:end]]
        case.events = t[:start] + t[end:]

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                size=int(size),
                start=int(start),
                skipped=skipped
            )
        )

        return case

    @staticmethod
    def targets(label, num_events, num_attributes):
        targets = Anomaly.targets(label, num_events, num_attributes)
        start = label['attr']['start'] + 1
        targets[start, 0] = Class.SKIP
        return targets

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        skipped = label['attr']['skipped']
        return f'{name} {", ".join([e["name"] for e in skipped])} at {start}'


class LateAnomaly(Anomaly):
    """Shift 1 sequence of `n` events by a distance `d` to the right."""

    def __init__(self, max_distance=1, max_sequence_size=1):
        self.max_distance = max_distance
        self.max_sequence_size = max_sequence_size
        super(LateAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 2:
            return NoneAnomaly().apply_to_case(case)

        size = np.random.randint(1, min(len(case) - 1, self.max_sequence_size + 1))
        distance = np.random.randint(1, min(len(case) - size, self.max_distance + 1))
        s = np.random.randint(0, len(case) - size - distance)
        i = s + distance

        t = case.events

        case.events = t[:s] + t[s + size:i + size] + t[s:s + size] + t[i + size:]

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                shift_from=int(s),
                shift_to=int(i),
                size=int(size)
            )
        )

        return case

    @staticmethod
    def targets(label, num_events, num_attributes):
        targets = Anomaly.targets(label, num_events, num_attributes)
        s = label['attr']['shift_from'] + 1
        e = label['attr']['shift_to'] + 1
        size = label['attr']['size']
        targets[s, 0] = Class.SHIFT
        targets[e:e + size, 0] = Class.LATE
        return targets

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        a = label['attr']['shift_from'] + 1
        b = label['attr']['shift_to'] + 1
        size = label['attr']['size']
        return f'{name} {size} from {a} to {b}'


class EarlyAnomaly(Anomaly):
    """Shift 1 sequence of `n` events by a distance `d` to the right."""

    def __init__(self, max_distance=1, max_sequence_size=1):
        self.max_distance = max_distance
        self.max_sequence_size = max_sequence_size
        super(EarlyAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 2:
            return NoneAnomaly().apply_to_case(case)

        size = np.random.randint(1, min(len(case) - 1, self.max_sequence_size + 1))
        distance = np.random.randint(1, min(len(case) - size, self.max_distance + 1))
        s = np.random.randint(distance, len(case) - size)
        i = s - distance

        t = case.events

        case.events = t[:i] + t[s:s + size] + t[i:s] + t[s + size:]

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                shift_from=int(s + size),
                shift_to=int(i),
                size=int(size)
            )
        )

        return case

    @staticmethod
    def targets(label, num_events, num_attributes):
        targets = Anomaly.targets(label, num_events, num_attributes)
        s = label['attr']['shift_from'] + 1
        e = label['attr']['shift_to'] + 1
        size = label['attr']['size']
        targets[s, 0] = Class.SHIFT
        targets[e:e + size, 0] = Class.EARLY
        return targets

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        a = label['attr']['shift_from'] + 1
        b = label['attr']['shift_to'] + 1
        size = label['attr']['size']
        return f'{name} {size} from {a} to {b}'


class AttributeAnomaly(Anomaly):
    """Replace n attributes in m events with an incorrect value."""

    def __init__(self, max_events=1, max_attributes=1):
        super(AttributeAnomaly, self).__init__()
        self.max_events = max_events
        self.max_attributes = max_attributes

    def apply_to_case(self, case):
        n = np.random.randint(1, min(len(case), self.max_events) + 1)
        event_indices = sorted(np.random.choice(range(len(case)), n, replace=False))

        indices = []
        original_attribute_values = []
        affected_attribute_names = []
        for event_index in event_indices:
            m = np.random.randint(1, min(len(self.attributes), self.max_attributes) + 1)
            attribute_indices = sorted(np.random.choice(range(len(self.attributes)), m, replace=False))
            for attribute_index in attribute_indices:
                affected_attribute = self.attributes[attribute_index]
                original_attribute_value = case[event_index].attributes[affected_attribute.name]

                indices.append(int(event_index))
                original_attribute_values.append(original_attribute_value)
                affected_attribute_names.append(affected_attribute.name)

                # Set the new value
                case[event_index].attributes[affected_attribute.name] = affected_attribute.random_value()

        attribute_names = sorted([a.name for a in self.attributes])
        attribute_indices = [attribute_names.index(a) for a in affected_attribute_names]

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                index=indices,
                attribute_index=attribute_indices,
                attribute=affected_attribute_names,
                original=original_attribute_values
            )
        )

        return case

    def apply_to_path(self, path):
        case_len = int(len(path) / self.event_len)

        n = np.random.randint(1, min(case_len, self.max_events) + 1)
        idx = sorted(np.random.choice(range(case_len), n, replace=False))

        indices = []
        original_attribute_values = []
        affected_attribute_names = []
        attribute_domains = []
        for index in idx:
            m = np.random.randint(1, min(len(self.attributes), self.max_attributes) + 1)
            attribute_indices = sorted(np.random.choice(range(len(self.attributes)), m, replace=False))
            for attribute_index in attribute_indices:
                affected_attribute = self.attributes[attribute_index]

                predecessor = path[index * self.event_len + attribute_index]

                attribute_values = [self.graph.nodes[s]['value'] for s in self.graph.successors(predecessor)]
                attribute_domain = [x for x in affected_attribute.domain if x not in attribute_values]
                original_attribute_value = self.graph.nodes[path[index * self.event_len + attribute_index + 1]]['value']

                indices.append(int(index))
                original_attribute_values.append(original_attribute_value)
                affected_attribute_names.append(affected_attribute.name)
                attribute_domains.append(attribute_domain)

        attribute_names = sorted([a.name for a in self.attributes])
        attribute_indices = [attribute_names.index(a) for a in affected_attribute_names]

        label = dict(
            anomaly=str(self),
            attr=dict(
                index=indices,
                attribute_index=attribute_indices,
                attribute=affected_attribute_names,
                original=original_attribute_values
            )
        )

        case = self.path_to_case(path, label)
        for index, affected_attribute, attribute_domain in zip(indices, affected_attribute_names, attribute_domains):
            case[index].attributes[affected_attribute] = np.random.choice(attribute_domain)

        return case

    @staticmethod
    def targets(label, num_events, num_attributes):
        targets = Anomaly.targets(label, num_events, num_attributes)
        indices = label['attr']['index']
        attribute_indices = label['attr']['attribute_index']
        for i, j in zip(indices, attribute_indices):
            targets[i + 1, j + 1] = Class.ATTRIBUTE
        return targets

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        affected = label['attr']['attribute']
        index = [i + 1 for i in label['attr']['index']]
        original = label['attr']['original']
        return f'{name} {affected} at {index} was {original}'


class ReplaceAnomaly(Anomaly):
    """Replace n events by random events coming from the case."""

    def __init__(self, max_replacements=1):
        self.max_replacements = max_replacements
        super(ReplaceAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 2:
            return NoneAnomaly().apply_to_case(case)

        num_replacements = np.random.randint(1, min(int(len(case) / 3), self.max_replacements) + 1)
        places = sorted(np.random.choice(np.arange(len(case) - 1, step=2), num_replacements, replace=False))

        t = case.events

        replaced = [t[i].json for i in places]

        for place in places:
            t = t[:place] + [self.generate_random_event()] + t[place + 1:]
        case.events = t

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                indices=[int(i) for i in places],
                replaced=replaced
            )
        )

        return case

    @staticmethod
    def targets(label, num_events, num_attributes):
        targets = Anomaly.targets(label, num_events, num_attributes)
        for i in label['attr']['indices']:
            targets[i + 1, 0] = Class.REPLACE
            targets[i + 1, 1:] = Class.ATTRIBUTE
        return targets

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        replaced = ', '.join([e['name'] for e in label['attr']['replaced']])
        indices = ', '.join([str(i + 1) for i in label['attr']['indices']])
        return f'{name} {replaced} at {indices}'


class InsertAnomaly(Anomaly):
    """Add n random events."""

    def __init__(self, max_inserts=1):
        self.max_inserts = max_inserts
        super(InsertAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) < 2:
            return NoneAnomaly().apply_to_case(case)

        num_inserts = np.random.randint(1, min(int(len(case) / 2), self.max_inserts) + 1)
        insert_places = sorted(np.random.choice(range(len(case) - 1), num_inserts, replace=False))
        insert_places += np.arange(len(insert_places))

        t = case.events
        for place in insert_places:
            t = t[:place] + [self.generate_random_event()] + t[place:]
        case.events = t

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                indices=[int(i) for i in insert_places]
            )
        )

        return case

    @staticmethod
    def targets(label, num_events, num_attributes):
        targets = Anomaly.targets(label, num_events, num_attributes)
        for i in label['attr']['indices']:
            targets[i + 1, 0] = Class.INSERT
            targets[i + 1, 1:] = Class.ATTRIBUTE
        return targets

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        indices = label['attr']['indices']
        return f'{name} at {", ".join([str(i + 1) for i in indices])}'


class SkipAnomaly(Anomaly):
    """Skip n single events."""

    def __init__(self, max_skips=2):
        self.max_skips = max_skips
        super(SkipAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 4:
            return NoneAnomaly().apply_to_case(case)

        num_skips = np.random.randint(2, min(int(len(case) / 2), self.max_skips) + 1)
        skip_places = sorted(np.random.choice(range(len(case) - 1), num_skips, replace=False))

        skipped = [case.events[i].json for i in skip_places]
        t = [e for i, e in enumerate(case.events) if i not in skip_places]

        case.events = t

        skip_places -= np.arange(len(skip_places))
        skip_places = list(set(skip_places))

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                indices=[int(i) for i in skip_places],
                skipped=skipped
            )
        )

        return case

    @staticmethod
    def targets(label, num_events, num_attributes):
        targets = Anomaly.targets(label, num_events, num_attributes)
        for i in label['attr']['indices']:
            targets[i + 1, 0] = Class.SKIP
        return targets

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        indices = ', '.join([str(i + 1) for i in label['attr']['indices']])
        skipped = ', '.join([e['name'] for e in label['attr']['skipped']])
        return f'{name} {skipped} at {indices}'


class SkipAndInsertAnomaly(Anomaly):
    """Skip n single events and insert m single events."""

    def __init__(self, max_skips=1, max_inserts=1):
        self.max_skips = max_skips
        self.max_inserts = max_inserts
        super(SkipAndInsertAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 2:
            return NoneAnomaly().apply_to_case(case)

        num_skips = np.random.randint(1, min(int(len(case) / 2), self.max_skips) + 1)
        skip_places = sorted(np.random.choice(range(len(case) - 1), num_skips, replace=False))

        skipped = [case.events[i].json for i in skip_places]
        t = [case.events[i] for i in range(len(case)) if i not in skip_places]

        skip_places -= np.arange(len(skip_places))

        num_inserts = np.random.randint(1, min(int(len(t) / 2), self.max_inserts) + 1)
        insert_places = sorted(np.random.choice(range(len(t) - 1), num_inserts, replace=False))
        insert_places2 = insert_places + np.arange(len(insert_places))

        for place in insert_places2:
            t = t[:place] + [self.generate_random_event()] + t[place:]

        case.events = t

        skip_places = skip_places + np.array([np.sum(insert_places < s) for s in skip_places])
        skip_places = list(set(skip_places))

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                skips=[int(i) for i in skip_places],
                skipped=skipped,
                inserts=[int(i) for i in insert_places2]
            )
        )

        return case

    @staticmethod
    def targets(label, num_events, num_attributes):
        targets = Anomaly.targets(label, num_events, num_attributes)
        for i in label['attr']['inserts']:
            targets[i + 1, 0] = Class.INSERT
            targets[i + 1, 1:] = Class.ATTRIBUTE
        for i in label['attr']['skips']:
            targets[i + 1, 0] = Class.SKIP
        return targets

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        inserts = ', '.join([str(i + 1) for i in label['attr']['inserts']])
        skips = ', '.join([str(i + 1) for i in label['attr']['skips']])
        skipped = ', '.join([e['name'] for e in label['attr']['skipped']])
        return f'{name} skip {skipped} at {skips} and insert at {inserts}'


ANOMALIES = dict((s[:-7], anomaly) for s, anomaly in inspect.getmembers(sys.modules[__name__], inspect.isclass))


def label_to_targets(label, num_events, num_attributes):
    if label == 'normal':
        return NoneAnomaly.targets(label, num_events, num_attributes)
    else:
        return ANOMALIES.get(label['anomaly']).targets(label, num_events, num_attributes)


def prettify_label(label):
    if label == 'normal':
        return 'Normal'
    else:
        return ANOMALIES.get(label['anomaly']).pretty_label(label)
