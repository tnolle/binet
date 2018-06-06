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

"""Core classes for process mining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import json
import os

import numpy as np
import pandas as pd
from lxml import etree

from binet.folders import EVENTLOG_DIR
from binet.utils import AttributeType


class Event(object):
    def __init__(self, name, timestamp=None, **kwargs):
        self.name = name
        self.timestamp = timestamp
        self.attributes = dict(kwargs)

    def __str__(self):
        _s = f'Event: name = {self.name}, timestamp = {self.timestamp}'

        attributes = [f'{key} = {value}' for key, value in self.attributes.items()]
        if len(attributes) > 0:
            _s += ', {}'.format(', '.join(attributes))

        return _s

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.name == other.name and self.attributes == other.attributes

    def to_json(self):
        """Return the event object as a json compatible python dictionary."""
        return dict(name=self.name, timestamp=self.timestamp, attributes=self.attributes)

    @staticmethod
    def clone(event):
        return Event(name=event.name, timestamp=event.timestamp, **dict(event.attributes))


class Case(object):
    def __init__(self, id=None, events=None, **kwargs):
        self.id = id
        if events is None:
            self.events = []
        else:
            self.events = events
        self.attributes = dict(kwargs)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        return all([a == b for a, b in zip(self, other)]) and self.attributes == other.attributes

    def __iter__(self):
        return iter(self.events)

    def __str__(self):
        s = []
        s.append(f'Case {self.id}: #events = {self.get_num_events()}')

        attributes = [f'{key} = {value}' for key, value in self.attributes.items()]
        if len(attributes) > 0:
            s.append(', {}'.format(', '.join(attributes)))

        s.append('-' * len(s[0]))

        for i, event in enumerate(self.events):
            _s = 'Event {}: name = {}, timestamp = {}'.format(i + 1, event.name, event.timestamp)

            attributes = ['{} = {}'.format(key, value) for key, value in event.attributes.items()]
            if len(attributes) > 0:
                _s += ', {}'.format(', '.join(attributes))

            s.append(_s)

        s.append('')

        return '\n'.join(s)

    def __getitem__(self, index):
        return self.events[index]

    def __setitem__(self, index, value):
        self.events[index] = value

    def __len__(self):
        return len(self.events)

    def index(self, index):
        return self.events.index(index)

    def add_event(self, event):
        self.events.append(event)

    def get_num_events(self):
        return len(self.events)

    def get_trace(self):
        return [str(event.name) for event in self.events]

    def get_all_attributes(self):
        return set().union(*(event.attributes.keys() for event in self.events))

    def to_json(self):
        """Return the trace object as a json compatible python dictionary."""
        return dict(id=self.id, events=[event.to_json() for event in self.events], attributes=self.attributes)

    @staticmethod
    def clone(trace):
        events = [Event.clone(event) for event in trace.events]
        return Case(id=trace.id, events=events, **dict(trace.attributes))


class EventLog(object):
    start_symbol = 'BOS'
    end_symbol = 'EOS'

    def __init__(self, cases=None, **kwargs):
        if cases is None or len(cases) == 0:
            self.cases = []
            self.activities = []
            self.max_length = 0
        else:
            self.cases = cases
            self.activities = self.get_activities()
            self.max_length = self.get_case_lens().max()
        self._variants = None
        self._variant_probabilities = None
        self._variant_counts = None
        self._labels = None
        self._text_labels = None
        self.attributes = dict(kwargs)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, index, ):
        return self.cases[index]

    def __setitem__(self, index, value):
        self.cases[index] = value

    def __str__(self):
        return f'Event Log: #cases: {len(self.cases)}, #events: {self.get_num_events()}, ' \
               f'#activities: {len(self.activities)}, Max length: {self.max_length}'

    def __len__(self):
        return len(self.cases)

    def get_attribute_names(self):
        attributes = ['name']
        if 'global_attributes' in self.attributes.keys() and 'event' in self.attributes['global_attributes'].keys():
            ignored = ['concept:name', 'time:timestamp', 'lifecycle:transition', 'EventID', 'activityNameEN',
                       'activityNameNL', 'dateFinished', 'question', 'product', 'EventOrigin', 'Action',
                       'organization involved', 'impact']

            attributes += sorted(
                [key for key in self.attributes['global_attributes']['event'].keys() if key not in ignored])
        else:
            attributes += sorted(
                [key for key in list(self.cases[0].events[0].attributes.keys()) if not key.startswith('_')])
        return attributes

    def get_attribute_types(self, attributes=None):
        def get_type(a):
            from numbers import Number
            if isinstance(a, Number):
                return AttributeType.NUMERICAL
            else:
                return AttributeType.CATEGORICAL

        if attributes is None:
            attributes = self.get_attribute_names()
        attribute_types = []  # name is always categorical
        for a in attributes:
            if a == 'name':
                a = self.cases[0][0].name
            else:
                a = self.cases[0][0].attributes[a]
            attribute_types.append(get_type(a))
        return attribute_types

    def add_case(self, case):
        for event in case:
            if event.name not in self.activities:
                self.activities.append(event.name)
        self.max_length = max(self.max_length, len(case))
        self.cases.append(case)

    def get_activities(self):
        return list(set([event.name for case in self.cases for event in case]))

    def get_case_lens(self):
        return np.array([case.get_num_events() for case in self.cases])

    def get_num_events(self):
        return self.get_case_lens().sum()

    def _get_variants(self):
        variants = [case.get_trace() for case in self.cases]
        self._variants, self._variant_counts = np.unique(variants, return_counts=True)
        self._variants = self._variants.tolist()
        self._variant_probabilities = self.variant_counts / float(len(self.cases))

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

    @property
    def variant_counts(self):
        if self._variant_counts is None:
            self._get_variants()
        return self._variant_counts

    @property
    def labels(self):
        if self._labels is None:
            self._labels = np.asarray(
                [t.attributes['label'] for t in self.cases if t.attributes and 'label' in t.attributes.keys()])
        return self._labels

    @property
    def text_labels(self):
        if self._text_labels is None:
            self._text_labels = np.array(['Normal' if l == 'normal' else l['anomaly'] for l in self.labels])
        return self._text_labels

    def get_targets(self):
        # +1 for EOS and +1 for the activity name
        labels = np.zeros((len(self.cases), max(self.get_case_lens()) + 1, len(self.get_attribute_names())),
                          dtype=bool)

        # set padding and mask
        mask = np.zeros_like(labels)
        for i, j in enumerate(self.get_case_lens() + 1):
            labels[i, j:] = 0
            mask[i, j:] = 1

        for i, label in enumerate(self.labels):
            # set labels to true where the anomaly happens
            if isinstance(label, dict):
                anomaly_type = label['anomaly']
                if anomaly_type in ['SkipSequence']:
                    idx = self.get_attribute_names().index('name')
                    labels[i, label['attr']['start'], idx] = True
                elif anomaly_type in ['DuplicateSequence']:
                    idx = self.get_attribute_names().index('name')
                    start = label['attr']['start'] + 1
                    end = start + label['attr']['size']
                    labels[i, start:end, idx] = True
                elif anomaly_type in ['IncorrectAttribute']:
                    idx = self.get_attribute_names().index(label['attr']['affected'])
                    labels[i, label['attr']['index'], idx] = True
                elif anomaly_type == 'SwitchEvents':
                    idx = self.get_attribute_names().index('name')
                    labels[i, label['attr']['first'], idx] = True
                    labels[i, label['attr']['second'], idx] = True

        return np.ma.array(labels, mask=mask, dtype=int)

    def to_json(self, file_path):
        """
        Save the event log to a JSON file.

        :param file_path: absolute path for the JSON file
        :return:
        """
        event_log = {"traces": [case.to_json() for case in self.cases], "attributes": self.attributes}
        with gzip.open(file_path, 'wt') as outfile:
            json.dump(event_log, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    def to_csv(self, file_path):
        """
        Save the event log to a CSV file.

        :param file_path: absolute path for the CSV file
        :return:
        """
        if not file_path.endswith('.csv'):
            '.'.join((file_path, 'csv'))
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)

    def to_dataframe(self):
        """
        Return pandas DataFrame containing the event log in matrix format.

        :return: pandas.DataFrame
        """
        frames = []
        for case_id, case in enumerate(self.cases):
            if case.id is not None:
                case_id = case.id
            for event_pos, event in enumerate(case):
                frames.append({
                    'trace_id': case_id,
                    'event_pos': event_pos,
                    'event': event.name,
                    'timestamp': event.timestamp,
                    **dict([i for i in event.attributes.items() if not i[0].startswith('_')])
                })
        return pd.DataFrame(frames)

    def to_feature_columns(self, include_attributes=None):
        """
        Return current event log as feature columns.

        Attributes are integer encoded. Shape of feature columns is (#traces, max_len, #attributes).

        :param include_attributes:

        :return: feature_columns, case_lens
        """

        if include_attributes is None:
            include_attributes = self.get_attribute_names()

        attribute_types = self.get_attribute_types(include_attributes)

        feature_columns = {'name': []}
        case_lens = []

        def get_default(a, t):
            if t == AttributeType.NUMERICAL:
                return 0.0
            else:
                return a

        bos = dict((a, get_default(EventLog.start_symbol, t)) for a, t in zip(include_attributes, attribute_types))
        bos = Event(timestamp=None, **bos)
        eos = dict((a, get_default(EventLog.end_symbol, t)) for a, t in zip(include_attributes, attribute_types))
        eos = Event(timestamp=None, **eos)

        for i, case in enumerate(self.cases):
            case_lens.append(len(case) + 2)
            for event in [bos] + case.events + [eos]:
                for attribute in self.get_attribute_names():
                    if attribute == 'name':
                        attr = event.name
                    elif attribute in include_attributes:
                        attr = event.attributes[attribute]
                    else:
                        continue
                    if attribute not in feature_columns.keys():
                        feature_columns[attribute] = []
                    feature_columns[attribute].append(attr)

        for key, attribute_type in zip(feature_columns.keys(), attribute_types):
            if attribute_type == AttributeType.CATEGORICAL:
                from sklearn.preprocessing import LabelEncoder
                feature_columns[key] = LabelEncoder().fit_transform(feature_columns[key]) + 1
            else:
                f = np.asarray(feature_columns[key])
                feature_columns[key] = (f - f.mean()) / f.std()

        # transform back into sequences
        case_lens = np.array(case_lens)
        offsets = np.concatenate(([0], np.cumsum(case_lens)[:-1]))
        features = [np.zeros((case_lens.shape[0], case_lens.max())) for _ in range(len(feature_columns))]
        for i, (offset, case_len) in enumerate(zip(offsets, case_lens)):
            for k, key in enumerate(feature_columns):
                x = feature_columns[key]
                features[k][i, :case_len] = x[offset:offset + case_len]

        return features, case_lens, attribute_types

    @staticmethod
    def load(eventlog_name):
        if not os.path.isabs(eventlog_name):
            eventlog_name = EVENTLOG_DIR / eventlog_name
        if eventlog_name.name.endswith('.xes') or eventlog_name.name.endswith('.xes.gz'):
            return EventLog.from_xes(eventlog_name)
        elif eventlog_name.name.endswith('.json') or eventlog_name.name.endswith('.json.gz'):
            return EventLog.from_json(eventlog_name)
        else:
            return EventLog.from_json(str(eventlog_name) + '.json.gz')

    @staticmethod
    def from_json(file_path: str):
        """
        Parse event log from JSON.

        JSON can be gzipped

        :param file_path: path to json file
        :return:
        """
        if not isinstance(file_path, str):
            file_path = str(file_path)

        if file_path.endswith('gz'):
            import gzip
            open = gzip.open

        # read the file
        with open(file_path, 'rb') as f:
            log = json.loads(f.read().decode('utf-8'))

        event_log = EventLog(**log['attributes'])

        for case in log['traces']:
            _case = Case(id=case['id'], **case['attributes'])
            for e in case['events']:
                event = Event(name=e['name'], timestamp=e['timestamp'], **e['attributes'])
                _case.add_event(event)
            event_log.add_case(_case)

        return event_log

    @staticmethod
    def from_xes(file_path):
        """
        Load an event log from an XES file

        :param file_path: path to xes file
        :return: EventLog object
        """

        # parse the log with lxml
        log = etree.parse(file_path).getroot()

        def parse_case(case):
            events = []
            attr = {}
            for child in case:
                tag = etree.QName(child).localname
                if tag == 'event':
                    event = parse_event(child)
                    if event is not None:
                        events.append(event)
                else:
                    attr[child.attrib['key']] = child.attrib['value']

            case_id = None
            if 'concept:name' in attr:
                case_id = attr['concept:name']

            return Case(id=case_id, events=events, **attr)

        def parse_event(event):
            attr = dict((attr.attrib['key'], attr.attrib['value']) for attr in event)

            timestamp = None
            # if 'time:timestamp' in global_attr['event'].keys():
            if 'time:timestamp' in attr:
                timestamp = attr['time:timestamp']

            name = ''
            if len(classifiers) > 0:
                keys = classifiers[0]['keys']
                check_keys = [key for key in keys if key not in attr]
                if len(check_keys) > 0:
                    print('Classifier key(s) {} could not be found in event.'.format(', '.join(check_keys)))
                    return None
                values = [attr[key] for key in keys]
                name = '+'.join(values)

            return Event(name=name, timestamp=timestamp, **attr)

        def parse_attribute(attribute):
            nested = len(attribute)
            attr = {
                'type': etree.QName(attribute.tag).localname,
                'value': attribute.attrib['value']
            }
            if nested:
                nested_attr = [parse_attribute(a) for a in attribute]
                attr['attr'] = dict([attr for attr in nested_attr if attr[0] is not None])
            if 'key' not in attribute.attrib:
                print('Key field was not found in attribute.')
                return None, None
            else:
                return attribute.attrib['key'], attr

        ext = []
        global_attr = {}
        classifiers = []
        cases = []
        attr = {}

        for child in log:
            tag = etree.QName(child).localname
            if tag == 'extension':
                ext.append(dict(child.attrib))
            elif tag == 'global':
                scope = child.attrib['scope']
                global_attr[scope] = {}
                for attribute in child:
                    attr_dict = {
                        'type': etree.QName(attribute.tag).localname,
                        'value': attribute.attrib['value']
                    }
                    global_attr[scope][attribute.attrib['key']] = attr_dict
            elif tag == 'classifier':
                name = child.attrib['name']
                keys = child.attrib['keys']
                keys = keys.split(' ')
                classifiers.append({'name': name, 'keys': keys})
            elif tag == 'trace':
                cases.append(parse_case(child))
            elif tag in ['string', 'date', 'int', 'float', 'boolean', 'id', 'list', 'container']:
                if child.attrib['key']:
                    key, attribute = parse_attribute(child)
                    if key is not None:
                        attr[key] = attribute
                else:
                    continue

        return EventLog(cases=cases, extensions=ext, global_attributes=global_attr, classifiers=classifiers, **attr)

    @staticmethod
    def from_csv(file_path):
        """
        Load an event log from a CSV file

        :param file_path: path to CSV file
        :return: EventLog object
        """
        # parse file as pandas dataframe
        df = pd.read_csv(file_path)

        # create event log
        event_log = EventLog()

        # iterate by distinct trace_id
        for case_id in np.unique(df['trace_id']):
            _case = Case(id=case_id)
            # iterate over rows per trace_id
            for index, row in df[df.trace_id == case_id].iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                event_name = row['event']
                user = row['user']
                _event = Event(name=event_name, timestamp=start_time, end_time=end_time, user=user)
                _case.add_event(_event)
            event_log.add_case(_case)

        return event_log

    @staticmethod
    def from_sql(server, database, user, password, schema='pm'):
        import pyodbc
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 13 for SQL Server}};'
            f'SERVER={{{server}}};'
            f'DATABASE={{{database}}};'
            f'UID={{{user}}};'
            f'PWD={{{password}}}'
        )

        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM {schema}.EventLog ORDER BY CaseId, Timestamp, SortKey, EventName')

        event_log = EventLog()
        case = None
        current_case_id = None
        for row in cursor.fetchall():
            case_id = row[0]
            timestamp = '' if row[3] is None else row[3].strftime('"%Y-%m-%d %H:%M:%S"')
            name = row[1]
            user = row[2]

            if case_id != current_case_id:
                case = Case(id=case_id)
                event_log.add_case(case)
                current_case_id = case_id
            case.add_event(Event(name=name, timestamp=timestamp, user=user))

        return event_log
