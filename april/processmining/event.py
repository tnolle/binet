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

class Event(object):
    def __init__(self, name, timestamp=None, **kwargs):
        self.name = name
        self.timestamp = timestamp
        self.attributes = dict(kwargs)

    def __str__(self):
        _s = f'Event: name = {self.name}, timestamp = {self.timestamp}'

        attributes = [f'{key} = {value}' for key, value in self.attributes.items()]
        if len(attributes) > 0:
            _s += f', {", ".join(attributes)}'

        return _s

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.name == other.name and self.attributes == other.attributes

    @property
    def json(self):
        """Return the event object as a json compatible python dictionary."""
        return dict(name=self.name, timestamp=self.timestamp, attributes=self.attributes)

    @staticmethod
    def clone(event):
        return Event(name=event.name, timestamp=event.timestamp, **dict(event.attributes))
