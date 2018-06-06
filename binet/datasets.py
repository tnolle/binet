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

import gzip
import logging
import pickle as pickle
import time

import numpy as np

from binet.processmining import EventLog
from binet.utils import AttributeType, EventLogFile


class Dataset(object):
    def __init__(self, dataset_name=None, compress=False):
        # Public properties
        self.attribute_types = None
        self.dataset_name = None
        self.counts = None
        self.inverse_indices = None
        self.compress = compress

        # Private properties
        self._attribute_dims = None
        self._targets = None
        self._labels = None
        self._trace_lens = None
        self._features = None
        self._onehot_features = None
        self._flat_features = None
        self._flat_features_2d = None
        self._flat_onehot_features = None
        self._flat_onehot_features_2d = None
        self._train_targets = None
        self._event_log = None
        self._total_examples = None
        self._text_labels = None
        self._unique_text_labels = None
        self._unique_anomaly_text_labels = None

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(40)

        # Load dataset
        if dataset_name is not None:
            self.load(dataset_name)

    @staticmethod
    def get_2d(x):
        return x.reshape((x.shape[0], np.product(x.shape[1:])))

    @property
    def mask(self):
        if self.compress:
            return range(self._features[0].shape[0])
        else:
            return self.inverse_indices

    @property
    def features(self):
        return [f[self.mask] for f in self._features]

    @property
    def targets(self):
        return self._targets[self.mask]

    @property
    def labels(self):
        return self._labels[self.mask]

    @property
    def trace_lens(self):
        return self._trace_lens[self.mask]

    @property
    def attribute_dims(self):
        if self._attribute_dims is None:
            self._attribute_dims = np.asarray(
                [f.max() if t == AttributeType.CATEGORICAL else 1 for f, t in
                 zip(self._features, self.attribute_types)])
        return self._attribute_dims

    @property
    def num_attributes(self):
        return len(self.features)

    @property
    def total_examples(self):
        if self._total_examples is None:
            self._total_examples = np.sum(self.counts)
        return self._total_examples

    @property
    def max_len(self):
        return self.features[0].shape[1]

    @property
    def flat_features(self):
        if self._flat_features is None:
            self._flat_features = np.dstack(self.features)
        return self._flat_features

    @property
    def flat_features_cat(self):
        return np.dstack(
            [f for f, a in zip(self.features, self.attribute_types) if a == AttributeType.CATEGORICAL])

    @property
    def flat_features_2d(self):
        if self._flat_features_2d is None:
            self._flat_features_2d = Dataset.get_2d(self.flat_features)
        return self._flat_features_2d

    @property
    def onehot_features(self):
        if self._onehot_features is None:
            from keras.utils import to_categorical
            self._onehot_features = [to_categorical(f)[:, :, 1:] if a == AttributeType.CATEGORICAL else f[:, :, None]
                                     for f, a in zip(self.features, self.attribute_types)]
        return self._onehot_features

    @property
    def flat_onehot_features(self):
        if self._flat_onehot_features is None:
            self._flat_onehot_features = np.concatenate(self.onehot_features, axis=2)
        return self._flat_onehot_features

    @property
    def flat_onehot_features_2d(self):
        if self._flat_onehot_features_2d is None:
            self._flat_onehot_features_2d = Dataset.get_2d(self.flat_onehot_features)
        return self._flat_onehot_features_2d

    @property
    def train_targets(self):
        if self._train_targets is None:
            self._train_targets = [f if a == AttributeType.NUMERICAL
                                   else np.pad(f[:, 1:], ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
                                   for f, a in zip(self.onehot_features, self.attribute_types)]
        return self._train_targets

    def unpack_event_log_data(self, data):
        (
            self._features,
            self._targets,
            self._labels,
            self._trace_lens,
            self._attribute_dims,
            self.attribute_types,
            self.inverse_indices,
            self.counts
        ) = data

    def load(self, dataset_name):
        el_file = EventLogFile(dataset_name)
        self.dataset_name = el_file.name

        # check for pickle file and load from it
        if el_file.cache_file.exists():
            s = time.time()

            # Load from cached file
            with gzip.open(el_file.cache_file, 'rb') as f:
                self.unpack_event_log_data(pickle.load(f))

            self.logger.info(
                'Dataset "{}" loaded from cache in {:.4f}s'.format(dataset_name, time.time() - s))

        # else generate from event log
        elif el_file.path.exists():
            s = time.time()

            # Load data from event log
            self._event_log = EventLog.load(el_file.path)
            data = self.from_event_log(self._event_log)
            self.unpack_event_log_data(data)

            # Cache to disk
            with gzip.open(el_file.cache_file, 'wb') as f:
                pickle.dump(data, f)

            self.logger.info(
                'Dataset "{}" loaded and cached from event log in {:.4f}s'.format(dataset_name, time.time() - s))

    @property
    def event_log(self):
        if self._event_log is None and self.dataset_name is not None:
            self._event_log = EventLog.load(self.dataset_name)
        return self._event_log

    @property
    def text_labels(self):
        if self._text_labels is None:
            self._text_labels = np.array(['Normal' if l == 'normal' else l['anomaly'] for l in self.labels])
        return self._text_labels

    @property
    def unique_text_labels(self):
        if self._unique_text_labels is None:
            self._unique_text_labels = sorted(set(self.text_labels))
        return self._unique_text_labels

    @property
    def unique_anomaly_text_labels(self):
        if self._unique_anomaly_text_labels is None:
            self._unique_anomaly_text_labels = [l for l in self.unique_text_labels if l != 'Normal']
        return [l for l in self.unique_text_labels if l != 'Normal']

    @staticmethod
    def from_event_log(event_log):
        # Get features from event log
        features, trace_lens, attribute_types = event_log.to_feature_columns()

        if 'compress' in event_log.attributes:
            # If event log is saved in compressed format, retrieve fields from event log
            counts = event_log.attributes['compress']['counts']
            inverse_indices = event_log.attributes['compress']['inverse_indices']
        else:
            # If event log is not compressed, compress data, we only need unique variants and their counts
            _, indices, inverse_indices, counts = np.unique(np.dstack(features), axis=0, return_counts=True,
                                                            return_index=True, return_inverse=True)
            features = [f[indices] for f in features]
            labels = event_log.labels[indices]
            targets = event_log.get_targets()[indices]
            trace_lens = trace_lens[indices]

        # Calculate attribute dimensions
        attribute_dims = None
        if 'attr_dims' in event_log.attributes:
            d = event_log.attr['attr_dims']
            attribute_dims = np.array([d[k] for k in sorted(d.keys())])
        return features, targets, labels, trace_lens, attribute_dims, attribute_types, inverse_indices, counts
