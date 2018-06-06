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

"""Generates random event logs from process models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from binet.folders import EVENTLOG_CACHE_DIR
from binet.folders import EVENTLOG_DIR
from binet.folders import PROCESS_MODEL_DIR
from binet.generator.anomaly import DuplicateSequenceAnomaly
from binet.generator.anomaly import IncorrectAttributeAnomaly
from binet.generator.anomaly import SkipSequenceAnomaly
from binet.generator.anomaly import SwitchEventsAnomaly
from binet.generator.attributes import CategoricalAttribute
from binet.generator.attributes import NormalAttribute
from binet.generator.attributes import UniformAttribute
from binet.generator.generator import EventLogGenerator
from binet.processmining import Flowchart
from examples.utils import user_names, company_names, countries_iso


def generate(seed, model, index, p, size, complexity, anomalies, trace_attr, event_attr):
    # for reproducibility
    np.random.seed(seed)

    generator = EventLogGenerator(
        Flowchart.from_plg(model),
        event_attributes=event_attr,
        trace_attributes=trace_attr
    )

    event_log = generator.generate(
        size=size,
        anomalies=anomalies,
        variant_probabilities=None,
        p=p,
        complexity=complexity
    )

    event_log.attributes['generation_params'] = {
        'seed': seed,
        'model': model,
        'index': index,
        'p': p,
        'size': size,
        'complexity': complexity,
    }

    # Remove old cached files
    cache_file = os.path.join(EVENTLOG_CACHE_DIR, '{}-{:.1f}-{}.pkl.gz'.format(model, p, index))
    if os.path.isfile(cache_file):
        os.remove(cache_file)

    # save the file to json
    event_log.to_json(os.path.join(EVENTLOG_DIR, '{}-{:.1f}-{}.json.gz'.format(model, p, index)))


def main():
    # available models in models dir
    models = [f.stem for f in PROCESS_MODEL_DIR.glob('*.plg')]

    # for reproducibility
    np.random.seed(42)

    # parameters
    attr_dim = 40
    max_group = 40

    users = np.random.choice(user_names, attr_dim, replace=False).tolist()
    countries = np.random.choice(countries_iso, attr_dim, replace=False).tolist()
    companies = np.random.choice(company_names, attr_dim, replace=False).tolist()

    def get_cat_attributes(n=0):
        cat_attributes = [
            CategoricalAttribute(name='user', values=users, min_group=1, max_group=max_group),
            CategoricalAttribute(name='supervisor', values=users, min_group=1, max_group=max_group),
            CategoricalAttribute(name='country', values=countries, min_group=1, max_group=max_group),
            CategoricalAttribute(name='vendor', values=companies, min_group=1, max_group=max_group),
            CategoricalAttribute(name='agency', values=companies, min_group=1, max_group=max_group),
        ]
        return cat_attributes[:n]

    def get_num_attributes(n=0):
        num_attributes = [
            NormalAttribute(name='duration', mu=10000, sigma=100),
            UniformAttribute(name='price', low=0, high=1000000)
        ]
        return num_attributes[:n]

    def get_attributes_and_anomalies(cat=0, num=0):
        anomalies = [
            SkipSequenceAnomaly(max_sequence_size=1),
            DuplicateSequenceAnomaly(max_sequence_size=1),
            SwitchEventsAnomaly(max_distance=1),
            IncorrectAttributeAnomaly()
        ]
        if num + cat == 0:
            anomalies = anomalies[:-1]
        return get_cat_attributes(cat) + get_num_attributes(num), anomalies

    trace_attributes = []
    event_attributes = []
    anomalies = []
    for n in [0, 1, 3, 5]:
        attr, anoms = get_attributes_and_anomalies(cat=n, num=0)
        event_attributes.append(attr)
        trace_attributes.append([])
        anomalies.append(anoms)

    ps = [.3]
    n = 1
    sizes = [12500]
    complexities = [.8]

    combinations = list(
        itertools.product(range(n), sizes, complexities, zip(anomalies, trace_attributes, event_attributes)))
    params = []
    for model in models:
        for p in ps:
            for i, c in enumerate(combinations, start=1):
                params.append((np.random.randint(0, max(sizes)), model, i, p, *c[1:-1], *c[-1]))

    Parallel(n_jobs=-1)(delayed(generate)(*c) for c in tqdm(params))


if __name__ == '__main__':
    main()
