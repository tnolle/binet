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

import _pickle as pickle
import datetime
import inspect
import logging
import os
import sys

import numpy as np

from binet.constants import MODEL_EXT
from binet.datasets import Dataset
from binet.folders import MODEL_DIR
from binet.utils import ModelFile


class AnomalyDetector(object):
    """Abstract base anomaly detector class.

    This is a boilerplate anomaly detector that only provides simple serialization and deserialization methods
    using pickle. Other classes can inherit the behavior. They will have to implement both the fit and the predict
    method.
    """

    def __init__(self, model=None):
        """Initialize base anomaly detector.

        :param model: Path to saved model file. Defaults to None.
        :type model: str
        """
        self.model = None

        self.dataset = None

        self.logger = logging.getLogger(self.__class__.__name__)

        if model is not None:
            self.load(model)

    def load(self, model):
        """
        Load a class instance from a pickle file. If no extension or absolute path are given the method assumes the
        file to be located inside the models dir. It will also add the model extension (see constants.MODEL_EXT).

        :param model: Path to saved model file. Defaults to None.
        :return: None
        """

        # load model file
        model = ModelFile(model)

        # load model
        self.model = pickle.load(open(model.path, 'rb'))

    def save(self, name):
        """Save the class instance using pickle.

        The filename will have the following structure:
        <file_name>_<self.abbreviation>_<current_datetime>.<constants.MODEL_EXT>

        :param name: Custom file name
        :return: None
        """
        if self.model is not None:
            if os.path.isabs(name):
                name = os.path.basename(name).split('.')[0]
            date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')
            file_name = '{}_{}_{}{}'.format(name, self.abbreviation, date, MODEL_EXT)
            with open(os.path.join(MODEL_DIR, file_name), 'wb') as f:
                pickle.dump(self.model, f)
        else:
            raise Exception('Saving not possible. No model has been trained yet.')

    def load_data(self, dataset):
        """Call load method on the dataset object."""
        self.dataset = dataset

    def fit(self, dataset):
        """Train the anomaly detector on an event log.

        This method must be implemented by the subclasses.

        Event log can be passed as a string as an absolute path, or as a file name with or without extension.
        If a file name is passed, event log is loaded from the event log dir.

        :param dataset:
        :type dataset: Dataset
        :return: None
        """
        raise NotImplementedError()

    def detect(self, dataset):
        """Detect anomalies on an event log.

        This method must be implemented by the subclasses.

        Detects anomalies on a given event log. Event log can be passed in the same way as the fit method.
        Returns an array containing an anomaly score for each attribute in each event in each trace. The <BOS> symbol
        is not classified; hence, the shape is [number of traces, maximum trace length - 1, number of attributes]

        :param dataset:
        :type dataset: Dataset
        :return: Array of anomaly scores: Shape is [number of traces, maximum trace length - 1, number of attributes]
        :rtype: numpy.ndarray
        """
        raise NotImplementedError()


class PerfectAnomalyDetector(AnomalyDetector):
    """Implements a random baseline anomaly detector.

    The random anomaly detector randomly chooses anomaly scores between 0 and 1.
    """

    abbreviation = 'perfect'

    def __init__(self, model=None):
        super().__init__(model=model)

    def fit(self, dataset):
        self.model = False

    def detect(self, dataset):
        return (dataset.targets - 1) / -2


class RandomAnomalyDetector(AnomalyDetector):
    """Implements a random baseline anomaly detector.

    The random anomaly detector randomly chooses anomaly scores between 0 and 1.
    """

    abbreviation = 'random'

    def __init__(self, model=None):
        super().__init__(model=model)

    def fit(self, dataset):
        self.model = False

    def detect(self, dataset):
        self.dataset = dataset
        anomaly_scores = np.ones(self.dataset.targets.shape)
        anomaly_scores *= np.random.randint(0, 2, self.dataset.targets.shape[0])[:, None, None]
        return anomaly_scores


class TStidePlusAnomalyDetector(AnomalyDetector):
    """Implements a sliding window anomaly detection algorithm.

    This method is based on the t-Stide implementation. Anomaly scores are based on frequencies of n-grams.
    Size of n-grams is controlled via the k parameter.
    """

    abbreviation = 't-stide+'

    def __init__(self, model=None, k=2):
        """Initialize sliding window anomaly detector.

        :param model: Path to saved model file. Defaults to None.
        :type model: str
        :param k: N-gram size. Defaults to 2.
        :type k: int
        """
        super().__init__(model=model)

        self.k = k
        self.model = {}
        self.get_anomaly_scores = np.vectorize(lambda x: self.model[x] if x in self.model.keys() else np.infty)

    def load(self, model):
        super().load(model)
        idx = np.array(list(self.model.keys())[0].split(','))[::2]
        self.k = int(idx.size / np.unique(idx).size)

    def load_data(self, dataset):
        self.dataset = dataset
        return self.dataset.flat_features_cat

    def fit(self, dataset):
        features = self.load_data(dataset)

        # create ngrams for all features like "0:1,1:2|0:2,1:2"
        n = self.preprocess(features, flatten=True)
        ngrams = [n]
        num_ngrams = len(n)

        # add ngrams per attribute, e.g. "0:1,1:2|0:2" and ""0:1,1:2|1:2""
        if self.dataset.num_attributes > 1:
            for i in range(self.dataset.num_attributes):
                n = self.preprocess(features, flatten=True, attribute_index=i)
                ngrams.append(n)
        keys, counts = np.unique(ngrams, return_counts=True)
        counts = -np.log(counts / num_ngrams)

        self.model = dict(zip(keys, counts))

    def detect(self, dataset):
        features = self.load_data(dataset)

        ngrams = []
        if self.dataset.num_attributes > 1:
            for i in range(self.dataset.num_attributes):
                ngrams.append(self.preprocess(features, flatten=False, attribute_index=i))
        else:
            ngrams = [self.preprocess(features, flatten=False)]
        ngram_matrix = np.dstack(ngrams)
        anomaly_scores = self.get_anomaly_scores(ngram_matrix)
        return anomaly_scores

    def preprocess(self, features, flatten=False, attribute_index=None):
        """Generate n-grams from features.

        Traces are split into n-grams where n is equal to k.

        If flatten is set to True traces are flattened. Each row in the return array will contain exactly one n-gram.
        If flatten is set to False traces are not flattened. Each row in the return array will contain all n-grams
        for each trace. For example, if a trace is [A, B, D, C], then all 2-grams are [[A, B], [B, D], [D, C]].

        :param features: Array of encoded traces.
        :param flatten: If set to True, ngrams are flattened.
        :param attribute_index: Index of the attribute to encode. Defaults to None; so all attributes are included.
        :return: Array of n-grams.
        """

        num_attributes = features.shape[-1]

        if self.k > 2:
            pad = np.zeros((features.shape[0], self.k - 2, features.shape[2]))
            pad_features = np.hstack((pad, features))
        else:
            pad_features = features

        windows = [pad_features]

        for i in np.arange(1, self.k):
            z = np.zeros((features.shape[0], i, features.shape[2]))
            windows.append(np.hstack((pad_features[:, i:], z)))

        if self.k > 1:
            ngrams = np.dstack(windows)[:, :-(self.k - 1)]
        else:
            ngrams = np.array(windows[0])[:, 1:]  # remove BOS, i.e., the first window

        del windows

        if attribute_index is not None:
            remaining_attr = np.array([i for i in range(num_attributes) if i != attribute_index])
            idx = remaining_attr + (self.k - 1) * num_attributes
            ngrams[:, :, idx] = -1

        # Convert to bytes
        # ngrams = np.apply_along_axis(lambda x: x.tobytes(), 2, ngrams)

        # Flatten
        if flatten:
            ngrams = ngrams

        return self.get_keys(ngrams, flatten=flatten, num_attributes=num_attributes)

    def get_keys(self, ngrams, flatten=False, num_attributes=None):
        """Get keys for the n-grams.

        Event attributes are represented by string keys. The string key is an enumerated representation of the event.
        For example, if the event has `name='A'` and the attribute `user='0'`, and the encoding of these two is
        3 and 4, respectively, then the key is going to be "0:3,1:4". The key indicates that the first attribute
        (the name) is encoded by 3, whereas the second attribute (the user in this case) is encoded by 4.

        This function iterates over all n-grams and converts them to key representation.

        :param ngrams: List of list of n-grams.
        :param flatten: If set to True, output is flattened so that each row only contains a single n-gram.
        :return: Array of n-grams.
        """
        new_ngrams = ngrams.reshape((*ngrams.shape[:-1], self.k, int(ngrams.shape[-1] / self.k)))
        enumeration = np.zeros_like(new_ngrams) + np.arange(num_attributes)
        new_ngrams = np.stack((enumeration, new_ngrams), axis=4)
        new_ngrams = new_ngrams.reshape((*ngrams.shape[:-1], self.k * 2 * num_attributes))
        new_ngrams = new_ngrams.astype(int).astype(str)

        if flatten:
            return np.array([','.join(ngram) for ngram_list in new_ngrams for ngram in ngram_list])
        else:
            return np.array([[','.join(ngram) for ngram in ngram_list] for ngram_list in new_ngrams])


class TStideAnomalyDetector(TStidePlusAnomalyDetector):
    abbreviation = 't-stide'

    def __init__(self, model=None, k=2):
        super().__init__(model=model, k=k)

    def preprocess(self, features, flatten=False, attribute_index=None, num_attributes=None):
        """Only use the first attribute; i.e., the activity name."""
        features = features[:, :, 0:1]
        return super().preprocess(features, flatten=flatten, attribute_index=None)


class NaiveAnomalyDetector(AnomalyDetector):
    """Implement the naive frequency based approach.

    Anomaly scores per trace are based on the frequency of the specific variant the trace follows.
    This is ignoring all attributes except the activity name.
    """

    abbreviation = 'naive'

    def __init__(self, model=None):
        super().__init__(model=model)
        self.get_anomaly_scores = np.vectorize(lambda x: self.model[x] if x in self.model.keys() else np.infty)

    def fit(self, dataset):
        self.load_data(dataset)
        keys, counts = np.unique(self.traces, return_counts=True)
        self.model = dict(zip(keys, - np.log(counts / self.traces.shape[0])))

    def detect(self, dataset):
        self.load_data(dataset)
        anomaly_scores = np.ones(self.dataset.targets.shape)
        anomaly_scores *= self.get_anomaly_scores(self.traces)[:, None, None]
        return anomaly_scores

    @property
    def traces(self):
        return np.array([','.join(t[t != 0].astype(str)) for t in self.dataset.features[0][:, :]])


class OneClassSVMAnomalyDetector(AnomalyDetector):
    """Implements a one-class SVM to detect anomalies."""

    abbreviation = 'one-class-svm'

    def __init__(self, model=None, nu=.5):
        super().__init__(model=model)
        self.nu = nu

    def load_data(self, dataset):
        super().load_data(dataset)
        return self.dataset.flat_features_2d

    def fit(self, dataset):
        features = self.load_data(dataset)

        from sklearn.svm import OneClassSVM
        self.model = OneClassSVM(nu=self.nu)
        self.model.fit(features)

    def detect(self, dataset):
        features = self.load_data(dataset)

        input_size = self.model.support_vectors_.shape[1]
        features_size = features.shape[1]
        if input_size > features_size:
            features = np.pad(features, [(0, 0), (0, input_size - features_size), (0, 0)][:len(features.shape)],
                              mode='constant')
        elif input_size < features_size:
            features = features[:, :input_size]

        anomaly_scores = np.zeros(self.dataset.targets.shape, dtype=int)
        pred = self.model.predict(features)
        pred = (pred == -1)[:, np.newaxis, np.newaxis]
        return anomaly_scores + pred


class NNAnomalyDetector(AnomalyDetector):
    """Abstract neural network based anomaly detector.

    Save and load methods are different for Keras based models.
    """

    def __init__(self, model=None, epochs=None, batch_size=None):
        super().__init__(model=model)
        self.epochs = epochs if epochs is not None else 50
        self.batch_size = batch_size if batch_size is not None else 500

    def load(self, model):
        # load model file
        model = ModelFile(model)

        # load model
        from keras.models import load_model
        self.model = load_model(model.path)

    def save(self, name=None):
        if self.model:
            date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')
            file_name = '{}_{}_{}{}'.format(name, self.abbreviation, date, MODEL_EXT)
            self.model.save(str(MODEL_DIR / file_name))
        else:
            raise Exception('No net has been trained yet.')

    def fit(self, dataset):
        raise NotImplementedError()

    def detect(self, dataset):
        raise NotImplementedError()


class DAEAnomalyDetector(NNAnomalyDetector):
    """Implements a denoising autoencoder based anomaly detection algorithm."""

    abbreviation = 'dae'

    def __init__(self, model=None, epochs=None, batch_size=None, hidden_layers=2, hidden_size_factor=.2, noise=None):
        """Initialize DAE model.

        Size of hidden layers is based on input size. The size can be controlled via the hidden_size_factor parameter.
        This can be float or a list of floats (where len(hidden_size_factor) == hidden_layers). The input layer size is
        multiplied by the respective factor to get the hidden layer size.

        :param model: Path to saved model file. Defaults to None.
        :param hidden_layers: Number of hidden layers. Defaults to 2.
        :param hidden_size_factor: Size factors for hidden layer base don input layer size.
        :param epochs: Number of epochs to train.
        :param batch_size: Mini batch size.
        """
        super().__init__(model=model, epochs=epochs, batch_size=batch_size)

        self.hidden_layers = hidden_layers
        self.hidden_size_factor = hidden_size_factor
        self.noise = noise

    def load_data(self, dataset):
        super().load_data(dataset)
        return self.dataset.flat_onehot_features_2d

    def fit(self, dataset):
        # Import keras locally
        from keras.layers import Input, Dense, Dropout, GaussianNoise
        from keras.models import Model
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping

        # Get features
        features = self.load_data(dataset)

        # Parameters
        input_size = features.shape[1]

        # Input layer
        input_layer = Input(shape=(input_size,), name='input')
        x = input_layer

        # Noise layer
        if self.noise is not None:
            x = GaussianNoise(self.noise)(x)

        # Hidden layers
        for i in range(self.hidden_layers):
            if isinstance(self.hidden_size_factor, list):
                factor = self.hidden_size_factor[i]
            else:
                factor = self.hidden_size_factor
            x = Dense(int(input_size * factor), activation='relu', name=f'hid{i + 1}')(x)
            x = Dropout(0.5)(x)

        # Output layer
        output_layer = Dense(input_size, activation='sigmoid', name='output')(x)

        # Build model
        self.model = Model(inputs=input_layer, outputs=output_layer)

        # Compile model
        self.model.compile(
            optimizer=Adam(lr=0.0001, beta_2=0.99),
            loss='mean_squared_error',
        )

        # Fit model
        self.model.fit(
            features,
            features,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=10)]
        )

    def detect(self, dataset):
        """
        Calculate the anomaly score for each event attribute in each trace.
        Anomaly score here is the mean squared error.

        :param traces: traces to predict
        :return:
            anomaly_scores: anomaly scores for each attribute;
                            shape is (#traces, max_trace_length - 1, #attributes)

        """
        # Get features
        features = self.load_data(dataset)

        # Parameters
        input_size = int(self.model.input.shape[1])
        features_size = int(features.shape[1])
        if input_size > features_size:
            features = np.pad(features, [(0, 0), (0, input_size - features_size), (0, 0)][:len(features.shape)],
                              mode='constant')
        elif input_size < features_size:
            features = features[:, :input_size]

        # get event length
        event_len = np.sum(self.dataset.attribute_dims).astype(int)

        # init anomaly scores array
        anomaly_scores = np.zeros(self.dataset.targets.shape)

        # get predictions
        predictions = self.model.predict(features)

        # Only use errors from 1s in the input
        # predictions = predictions * features

        # Calculate error
        errors = np.power(features - predictions, 2)

        # remove the BOS event
        errors = errors[:, event_len:]

        # split the errors according to the attribute dims
        split = np.cumsum(np.tile(self.dataset.attribute_dims, [self.dataset.max_len - 1]), dtype=int)[:-1]
        errors = np.split(errors, split, axis=1)
        errors = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in errors])

        for i in range(len(self.dataset.attribute_dims)):
            error = errors[i::len(self.dataset.attribute_dims)]
            anomaly_scores[:, :, i] = error.T

        return anomaly_scores


class RNNAnomalyDetector(NNAnomalyDetector):
    """Implements an LSTM based anomaly detection algorithm."""

    abbreviation = 'binet'

    def __init__(self, model=None, embedding=False, epochs=None, batch_size=None, recurrent_attr=False):
        super().__init__(model, epochs=epochs, batch_size=batch_size)
        self.embedding = embedding
        self.recurrent_attr = recurrent_attr

    def load(self, model):
        super().load(model)
        if 'Embedding' in [l.__class__.__name__ for l in self.model.layers]:
            self.embedding = True
        else:
            self.embedding = False

    def load_data(self, dataset):
        self.dataset = dataset
        if self.embedding:
            activity_features = self.dataset.features
            shape = ((0, 0), (0, 1))
        else:
            activity_features = self.dataset.onehot_features
            shape = ((0, 0), (0, 1), (0, 0))
        if self.dataset.attribute_dims.size == 1:
            attribute_features = []
        else:
            attribute_features = [np.pad(f[:, 1:], shape, 'constant', constant_values=0) for f in activity_features]
            attribute_features = attribute_features[:1]
        return activity_features, attribute_features

    def fit(self, dataset):
        from keras.models import Model
        from keras.layers import GRU, Dense, TimeDistributed, Input, Masking
        from keras.layers import BatchNormalization, concatenate, Embedding
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping
        from binet.utils import AttributeType

        # Load data
        activity_features, attribute_features = self.load_data(dataset)
        d = self.dataset
        targets = d.train_targets
        attr_data = [d.attribute_dims, d.attribute_types, d.event_log.get_attribute_names()]

        # Combine features
        features = activity_features + attribute_features

        rnn_size = int(activity_features[0].shape[1] * 2)
        # rnn_size = 128

        # Activity input layers
        inputs = []
        activity_layers = []
        for t, attr_dim, attr_type, attr_name in zip(activity_features, *attr_data):
            attr_name = attr_name.replace(':', '_').replace(' ', '_')
            i = Input(shape=(None, *t.shape[2:]), name=f'act_in_{attr_name}')

            # Embedding
            if self.embedding and attr_type == AttributeType.CATEGORICAL:
                voc_size = np.array(attr_dim).astype(int) + 1  # we start at 1, hence +1
                emb_size = np.floor(voc_size * 0.1).astype(int)
                x = Embedding(input_dim=voc_size, output_dim=emb_size, input_length=t.shape[1], mask_zero=True)(i)
            else:
                x = Masking(mask_value=0)(i)

            inputs.append(i)
            activity_layers.append(x)

        # merge layers
        if len(activity_layers) > 1:
            act_in = concatenate(activity_layers)
        else:
            act_in = activity_layers[0]

        # Activity RNN layer
        act_in = BatchNormalization()(act_in)
        act_out = GRU(rnn_size, implementation=2, return_sequences=True)(act_in)

        if len(attribute_features) > 0:
            attribute_layers = []
            for t, attr_dim, attr_type, attr_name in zip(attribute_features, *attr_data):
                attr_name = attr_name.replace(':', '_').replace(' ', '_')
                i = Input(shape=(None, *t.shape[2:]), name=f'attr_in_{attr_name}')

                # Embedding
                if self.embedding and attr_type == AttributeType.CATEGORICAL:
                    voc_size = np.array(attr_dim).astype(int) + 1  # we start at 1, hence +1
                    emb_size = np.floor(voc_size * 0.1).astype(int)
                    x = Embedding(input_dim=voc_size, output_dim=emb_size, input_length=t.shape[1], mask_zero=True)(i)
                else:
                    x = Masking(mask_value=0)(i)

                inputs.append(i)
                attribute_layers.append(x)

            # Merge layers
            if len(attribute_layers) > 1:
                attr_in = concatenate(attribute_layers)
            else:
                attr_in = attribute_layers[0]

            if self.recurrent_attr:
                attr_in = concatenate([act_out, attr_in])

            # Autoencoder
            attr_in = BatchNormalization()(attr_in)
            attr_out = GRU(rnn_size, implementation=2, return_sequences=True)(attr_in)

        outputs = []
        losses = []
        for l, attr_dim, attr_type, attr_name in zip(targets, *attr_data):
            attr_name = attr_name.replace(':', '_').replace(' ', '_')
            x = act_out if attr_name == 'name' else attr_out

            if attr_type == AttributeType.NUMERICAL:
                activation = 'linear'
                losses.append('mean_squared_error')
            else:
                activation = 'softmax'
                losses.append('categorical_crossentropy')

            # Batch norm
            x = BatchNormalization()(x)

            # Output layer
            o = TimeDistributed(Dense(l.shape[2], activation=activation), name=f'out_{attr_name}')(x)

            # Append to outputs array
            outputs.append(o)

        # Build model
        self.model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        self.model.compile(
            optimizer=Adam(),
            loss=losses
        )

        # Train model
        self.model.fit(
            features,
            targets,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=20)]
        )

    def detect(self, dataset):
        """
        Calculate the anomaly score and the probability distribution for each event in each trace.
        Anomaly score here is the probability of that event occurring given all events before.

        :param dataset: The dataset.
        :return: anomaly_scores: anomaly scores for each attribute;
            shape is (#traces, max_trace_length - 1, #attributes).
        :return: distributions: probability distributions for each event and attribute;
            list of np.arrays with shape (#traces, max_trace_length - 1, #attribute_classes),
            one np.array for each attribute, hence list len is #attributes.
        """
        from binet.utils import AttributeType

        # load data
        activity_features, attribute_features = self.load_data(dataset)
        features = activity_features + attribute_features

        # Remove the last time step, it holds no information
        predictions = self.model.predict(features)

        if len(self.model.output_layers) == 1:
            predictions = [predictions]

        # Sigmoid
        # errors = [np.mean((p - f) ** 2, axis=2) for p, f in zip(predictions, d.onehot_features)]
        # anomaly_scores = np.dstack(errors)[:, 1:]

        # Softmax
        anomaly_scores = np.zeros(self.dataset.targets.shape, dtype=np.float64)
        for idx, i in np.ndenumerate(self.dataset.flat_features[:, 1:]):
            if i > 0:
                if self.dataset.attribute_types[idx[2]] == AttributeType.NUMERICAL:
                    anomaly_scores[idx] = i - predictions[idx[-1]][idx[:-1]][0]  # error
                else:
                    p = predictions[idx[-1]][idx[:-1]]
                    # anomaly_scores[idx] = 1 - p[int(i - 1)]  # inverse probability
                    anomaly_scores[idx] = np.max(p) - p[int(i - 1)]  # difference from max

        distributions = [1 - p for p in predictions]

        return anomaly_scores, distributions


# Lookup dict for AD abbreviations
AD = dict((ad.abbreviation, ad) for _, ad in inspect.getmembers(sys.modules[__name__], inspect.isclass)
          if hasattr(ad, 'abbreviation'))
