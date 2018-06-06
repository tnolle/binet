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

import os

from binet.folders import BPIC_DIR
from binet.folders import EVENTLOG_DIR
from binet.processmining import EventLog


def main():
    xes_files = [
        'BPIC12.xes.gz',
        'BPIC13_closed_problems.xes.gz',
        'BPIC13_incidents.xes.gz',
        'BPIC13_open_problems.xes.gz',
        'BPIC15_1.xes.gz',
        'BPIC15_2.xes.gz',
        'BPIC15_3.xes.gz',
        'BPIC15_4.xes.gz',
        'BPIC15_5.xes.gz',
        'BPIC17.xes.gz',
        'BPIC17_offer_log.xes.gz'
    ]
    json_files = [
        'bpic12-0.0-1.json.gz',
        'bpic13-0.0-1.json.gz',
        'bpic13-0.0-2.json.gz',
        'bpic13-0.0-3.json.gz',
        'bpic15-0.0-1.json.gz',
        'bpic15-0.0-2.json.gz',
        'bpic15-0.0-3.json.gz',
        'bpic15-0.0-4.json.gz',
        'bpic15-0.0-5.json.gz',
        'bpic17-0.0-1.json.gz',
        'bpic17-0.0-2.json.gz'
    ]

    for xes_file, json_file in zip(xes_files, json_files):
        event_log = EventLog.from_xes(os.path.join(BPIC_DIR, xes_file))
        event_log.to_json(os.path.join(EVENTLOG_DIR, json_file))


if __name__ == '__main__':
    main()
