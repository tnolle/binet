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

import gzip
import json
import os

import numpy as np
import pandas as pd
from lxml import etree

from april.enums import AttributeType
from april.fs import EVENTLOG_CACHE_DIR
from april.fs import EVENTLOG_DIR
from april.processmining.case import Case
from april.processmining.event import Event


class EventLog(object):
    start_symbol = '▶'
    end_symbol = '■'

    def __init__(self, cases=None, **kwargs):
        if cases is None or len(cases) == 0:
            self.cases = []
        else:
            self.cases = cases
        self.attributes = dict(kwargs)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, indices):
        return np.asarray(self.cases)[indices]

    def __setitem__(self, index, value):
        self.cases[index] = value

    def __str__(self):
        return f'Event Log: #cases: {self.num_cases}, #events: {self.num_events}, ' \
            f'#activities: {self.num_activities}, Max case length: {self.max_case_len}'

    def __len__(self):
        return len(self.cases)

    @property
    def event_attribute_keys(self):
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

    @property
    def num_event_attributes(self):
        return len(self.event_attribute_keys)

    def get_attribute_types(self, attributes=None):
        def get_type(a):
            from numbers import Number
            if isinstance(a, Number):
                return AttributeType.NUMERICAL
            else:
                return AttributeType.CATEGORICAL

        if attributes is None:
            attributes = self.event_attribute_keys
        attribute_types = []  # name is always categorical
        for a in attributes:
            if a == 'name':
                a = self.cases[0][0].name
            else:
                a = self.cases[0][0].attributes[a]
            attribute_types.append(get_type(a))
        return attribute_types

    def get_unique_attribute_values(self, attribute_key):
        if attribute_key == 'name':
            return sorted(self.unique_activities)
        else:
            return sorted(list(set([e.attributes[attribute_key] for case in self.cases for e in case])))

    def add_case(self, case):
        self.cases.append(case)

    @property
    def unique_activities(self):
        return list(set([event.name for case in self.cases for event in case]))

    @property
    def unique_attribute_values(self):
        return dict((k, self.get_unique_attribute_values(k)) for k in self.event_attribute_keys)

    @property
    def num_activities(self):
        return len(self.unique_activities)

    @property
    def num_cases(self):
        return len(self.cases)

    @property
    def case_lens(self):
        return np.array([case.num_events for case in self.cases])

    @property
    def max_case_len(self):
        return self.case_lens.max()

    @property
    def num_events(self):
        return self.case_lens.sum()

    def get_traces(self, return_counts=False):
        return np.unique([case.trace for case in self.cases], return_counts=return_counts, axis=0)

    @property
    def traces(self):
        return self.get_traces()

    @property
    def trace_probabilities(self):
        return self.trace_counts / float(self.num_cases)

    @property
    def trace_counts(self):
        traces, counts = self.get_traces(return_counts=True)
        return counts

    @staticmethod
    def load(eventlog_name):
        """
        Load event log from file system.

        Supports JSON and XES files. Files can be gzipped.

        :param eventlog_name:
        :return:
        """
        if not os.path.isabs(eventlog_name):
            eventlog_name = EVENTLOG_DIR / eventlog_name
        if eventlog_name.name.endswith('.xes') or eventlog_name.name.endswith('.xes.gz'):
            return EventLog.from_xes(eventlog_name)
        elif eventlog_name.name.endswith('.json') or eventlog_name.name.endswith('.json.gz'):
            return EventLog.from_json(eventlog_name)
        else:
            return EventLog.from_json(str(eventlog_name) + '.json.gz')

    @staticmethod
    def from_dict(log):
        event_log = EventLog(**log['attributes'])
        for case in log['cases']:
            _case = Case(id=case['id'], **case['attributes'])
            for e in case['events']:
                event = Event(name=e['name'], timestamp=e['timestamp'], **e['attributes'])
                _case.add_event(event)
            event_log.add_case(_case)
        return event_log

    @staticmethod
    def from_json(file_path):
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

        # Read the file
        with open(file_path, 'rb') as f:
            log = json.loads(f.read().decode('utf-8'))

        event_log = EventLog(**log['attributes'])

        # Compatibility: Check for traces in log
        if 'traces' in log:
            case_key = 'traces'
        else:
            case_key = 'cases'

        for case in log[case_key]:
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
                    print(f'Classifier key(s) {", ".join(check_keys)} could not be found in event.')
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

        # iterate by distinct case id
        for case_id in np.unique(df['case_id']):
            _case = Case(id=case_id)
            # iterate over rows per case
            for index, row in df[df.case_id == case_id].iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                event_name = row['event']
                user = row['user']
                _event = Event(name=event_name, timestamp=start_time, end_time=end_time, user=user)
                _case.add_event(_event)
            event_log.add_case(_case)

        return event_log

    @staticmethod
    def from_sql(server, database, resource, password, schema='pm'):
        import pyodbc
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={{{server}}};'
            f'DATABASE={{{database}}};'
            f'UID={{{resource}}};'
            f'PWD={{{password}}}'
        )

        cursor = conn.cursor()
        cursor.execute(f'SELECT CaseId, ActivityName, Resource, Timestamp FROM {schema}.EventLog ORDER BY CaseId, Timestamp, SortKey, ActivityName')

        event_log = EventLog()
        case = None
        current_case_id = None
        for row in cursor.fetchall():
            case_id = row[0]
            timestamp = '' if row[3] is None else row[3].strftime('"%Y-%m-%d %H:%M:%S"')
            name = row[1]
            resource = row[2]

            if case_id != current_case_id:
                case = Case(id=case_id)
                event_log.add_case(case)
                current_case_id = case_id
            case.add_event(Event(name=name, timestamp=timestamp, resource=resource))

        return event_log

    @property
    def json(self):
        """Return json dictionary."""
        return {"cases": [case.json for case in self.cases], "attributes": self.attributes}

    @property
    def dataframe(self):
        """
        Return pandas DataFrame containing the event log in matrix format.

        :return: pandas.DataFrame
        """

        start_event = Event(timestamp=None, **dict((a, EventLog.start_symbol) for a in self.event_attribute_keys))
        end_event = Event(timestamp=None, **dict((a, EventLog.end_symbol) for a in self.event_attribute_keys))

        frames = []
        for case_id, case in enumerate(self.cases):
            if case.id is not None:
                case_id = case.id
            for event_pos, event in enumerate([start_event] + case.events + [end_event]):
                frames.append({
                    'case_id': case_id,
                    'event_position': event_pos,
                    'name': event.name,
                    'timestamp': event.timestamp,
                    **dict([i for i in event.attributes.items() if not i[0].startswith('_')])
                })
        return pd.DataFrame(frames)

    def save(self, name, p=0.0, number=1):
        base_name = f'{name}-{p:.1f}-{number}'
        cache_file = EVENTLOG_CACHE_DIR / f'{base_name}.pkl.gz'

        if cache_file.exists():
            os.remove(cache_file)

        self.save_json(EVENTLOG_DIR / f'{base_name}.json.gz')

    def save_json(self, file_path):
        """
        Save the event log to a JSON file.

        :param file_path: absolute path for the JSON file
        :return:
        """
        event_log = self.json
        with gzip.open(file_path, 'wt') as outfile:
            json.dump(event_log, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    def save_csv(self, file_path):
        """
        Save the event log to a CSV file.

        :param file_path: absolute path for the CSV file
        :return:
        """
        if not file_path.endswith('.csv'):
            '.'.join((file_path, 'csv'))
        df = self.dataframe
        df.to_csv(file_path, index=False)
