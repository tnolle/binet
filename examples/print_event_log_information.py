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

from pprint import pprint

from binet import Dataset
from binet.utils import get_event_logs

if __name__ == '__main__':
    logs = [e.name for e in get_event_logs() if 'bpic17-0.3-1' in e.name or 'bpic13-0.3-3' in e.name]

    for log in sorted(logs):
        d = Dataset(log)

        print(log)
        print(d.event_log)
        pprint(dict(zip(d.event_log.get_attribute_names(), d.attribute_dims.astype(int))))
        print('Max length', d.max_len)
        print()
