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

"""Adds anomalies to an existing event log."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm

from binet.folders import EVENTLOG_DIR
from binet.generator.anomaly import DuplicateSequenceAnomaly, Anomaly
from binet.generator.anomaly import SkipSequenceAnomaly
from binet.generator.anomaly import SwitchEventsAnomaly
from binet.processmining import EventLog
from binet.utils import get_event_logs
from examples.utils import company_names


def generate(combinations):
    el_path, p = combinations

    # np.random.seed(seed)
    anomalies = [
        SkipSequenceAnomaly(max_sequence_size=1),
        DuplicateSequenceAnomaly(max_sequence_size=1),
        SwitchEventsAnomaly(max_distance=1),
    ]

    el = EventLog.from_json(el_path)

    if len(el.get_attribute_names()) > 1:
        anomalies.append('IncorrectAttribute')

    # add anomalies to real eventlog
    for trace in el:
        if np.random.uniform() <= p:
            anomaly = np.random.choice(anomalies)
            if isinstance(anomaly, Anomaly):
                anomaly.apply(trace)
            else:
                index = np.random.randint(0, len(trace))
                random_event = trace[index]
                random_attribute = np.random.choice(el.get_attribute_names()[1:])
                original_attribute = random_event.attributes[random_attribute]
                random_event.attributes[random_attribute] = np.random.choice(company_names)
                trace.attributes['label'] = {
                    "anomaly": 'IncorrectAttribute',
                    "attr": {
                        "index": int(index),
                        "affected": random_attribute,
                        "original": original_attribute
                    }
                }
        else:
            trace.attributes['label'] = 'normal'

    el_name = os.path.basename(el_path)
    el_name, noise, dataset_id = el_name.split('-')

    el.to_json(str(EVENTLOG_DIR / f'{el_name}-{p}-{dataset_id}'))


def main():
    p = 0.3
    el_paths = [e.path for e in get_event_logs(PAF_DIR)]
    combinations = list(itertools.product(el_paths, [p]))

    with Pool() as p:
        for _ in tqdm(p.imap(generate, combinations), total=len(combinations)):
            pass


if __name__ == '__main__':
    main()
