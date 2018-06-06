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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm

from binet.anomalydetection import AD
from binet.anomalydetection import DAEAnomalyDetector
from binet.anomalydetection import NNAnomalyDetector
from binet.anomalydetection import RNNAnomalyDetector
from binet.datasets import Dataset
from binet.utils import get_event_logs


def fit_and_save(dataset_name, abbr, **kwargs):
    anomaly_detector = AD.get(abbr)(**kwargs)
    anomaly_detector.fit(Dataset(dataset_name))
    anomaly_detector.save(dataset_name)
    if isinstance(anomaly_detector, NNAnomalyDetector):
        from keras.backend import clear_session
        clear_session()


def main():
    # Possible datasets
    # Artificial 'p2p', 'small', 'medium', 'large', 'huge', 'wide'
    # Real 'bpic12', 'bpic13', 'bpic15', 'bpic17'
    datasets = [e.name for e in get_event_logs() if e.model == 'p2p' and e.id in [1]]

    # Anomaly Detectors
    ads = [
        {'abbr': RNNAnomalyDetector.abbreviation, 'embedding': True, 'epochs': 10, 'recurrent_attr': True,
         'batch_size': 1000},
        {'abbr': DAEAnomalyDetector.abbreviation, 'epochs': 50, 'batch_size': 1000, 'hidden_layers': 1,
         'hidden_size_factor': .05, 'noise': None}
    ]

    for ad in ads:
        [fit_and_save(d, **ad) for d in tqdm(datasets, desc=ad['abbr'])]


if __name__ == '__main__':
    main()
