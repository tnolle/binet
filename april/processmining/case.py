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

import numpy as np

from april.processmining.event import Event


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
        s = [f'Case {self.id}: #events = {self.num_events}']

        attributes = [f'{key} = {value}' for key, value in self.attributes.items()]
        if len(attributes) > 0:
            s.append(f', {", ".join(attributes)}')

        s.append('-' * len(s[0]))

        for i, event in enumerate(self.events):
            _s = f'Event {i + 1}: name = {event.name}, timestamp = {event.timestamp}'

            attributes = [f'{key} = {value}' for key, value in event.attributes.items()]
            if len(attributes) > 0:
                _s += f', {", ".join(attributes)}'

            s.append(_s)

        s.append('')

        return '\n'.join(s)

    def __getitem__(self, indices):
        return np.asarray(self.events)[indices]

    def __setitem__(self, index, value):
        self.events[index] = value

    def __len__(self):
        return len(self.events)

    def index(self, index):
        return self.events.index(index)

    def add_event(self, event):
        self.events.append(event)

    @property
    def num_events(self):
        return len(self.events)

    @property
    def trace(self):
        return [str(event.name) for event in self.events]

    @property
    def attribute_names(self):
        return set().union(*(event.attributes.keys() for event in self.events))

    @property
    def json(self):
        """Return the case object as a json compatible python dictionary."""
        return dict(id=self.id, events=[event.json for event in self.events], attributes=self.attributes)

    @staticmethod
    def clone(trace):
        events = [Event.clone(event) for event in trace.events]
        return Case(id=trace.id, events=events, **dict(trace.attributes))
