# Copyright 2018 Timo Nolle
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

import pickle as pickle

import numpy as np

from april.anomalydetection.utils.result import AnomalyDetectionResult
from april.enums import Base
from april.enums import Heuristic
from april.enums import Mode
from april.enums import Normalization
from april.enums import Strategy
from april.fs import ModelFile


class AnomalyDetector(object):
    """Abstract base anomaly detector class.

    This is a boilerplate anomaly detector that only provides simple serialization and deserialization methods
    using pickle. Other classes can inherit the behavior. They will have to implement both the fit and the predict
    method.
    """
    abbreviation = None
    name = None
    supported_binarization = []
    supported_heuristics = []
    supported_strategies = []
    supported_normalizations = [Normalization.MINMAX]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.SCORES]
    supports_attributes = False

    def __init__(self, model=None):
        """Initialize base anomaly detector.

        :param model: Path to saved model file. Defaults to None.
        :type model: str
        """
        self._model = None
        if model is not None:
            self.load(model)

    @property
    def model(self):
        return self._model

    def load(self, file_name):
        """
        Load a class instance from a pickle file. If no extension or absolute path are given the method assumes the
        file to be located inside the models dir. Model extension can be omitted in the file name.

        :param file_name: Path to saved model file.
        :return: None
        """
        # load model file
        model_file = ModelFile(file_name)

        # load model
        self._model = pickle.load(open(model_file.path, 'rb'))

    def _save(self, file_name):
        """The function to save a model. Subclasses that do not use pickle must override this method."""
        with open(file_name, 'wb') as f:
            pickle.dump(self._model, f)

    def save(self, file_name=None):
        """Save the class instance using pickle.

        :param file_name: Custom file name
        :return: the file path
        """
        if self._model is not None:
            model_file = ModelFile(file_name)
            self._save(model_file.str_path)
            return model_file
        else:
            raise RuntimeError(
                'Saving not possible. No model has been trained yet.')

    def fit(self, dataset):
        """Train the anomaly detector on a dataset.

        This method must be implemented by the subclasses.

        :param dataset: Must be passed as a Dataset object
        :type dataset: Dataset
        :return: None
        """
        raise NotImplementedError()

    def detect(self, dataset):
        """Detect anomalies on an event log.

        This method must be implemented by the subclasses.

        Detects anomalies on a given dataset. Dataset can be passed as in the fit method.
        Returns an array containing an anomaly score for each attribute in each event in each case.

        :param dataset:
        :type dataset: Dataset
        :return: Array of anomaly scores: Shape is [number of cases, maximum case length, number of attributes]
        :rtype: numpy.ndarray
        """
        raise NotImplementedError()


class PerfectAnomalyDetector(AnomalyDetector):
    """Implements a random baseline anomaly detector.

    The random anomaly detector randomly chooses anomaly scores between 0 and 1.
    """

    abbreviation = 'perfect'
    name = 'Perfect'

    supported_strategies = [Strategy.SINGLE]
    supported_heuristics = [Heuristic.DEFAULT]
    supports_attributes = True

    def __init__(self, model=None):
        super(PerfectAnomalyDetector, self).__init__(model=model)

    def fit(self, dataset):
        self._model = False

    def detect(self, dataset):
        return AnomalyDetectionResult(scores=dataset.binary_targets)


class RandomAnomalyDetector(AnomalyDetector):
    """Implements a random baseline anomaly detector.

    The random anomaly detector randomly chooses anomaly scores between 0 and 1.
    """

    abbreviation = 'random'
    name = 'Random'

    supported_strategies = [Strategy.SINGLE]
    supported_heuristics = [Heuristic.DEFAULT]
    supports_attributes = True

    def __init__(self, model=None):
        super(RandomAnomalyDetector, self).__init__(model=model)

    def fit(self, dataset):
        self._model = False

    def detect(self, dataset):
        scores = np.ones(dataset.binary_targets.shape)
        scores *= np.random.randint(0, 2, dataset.binary_targets.shape[0])[:, np.newaxis, np.newaxis]
        return AnomalyDetectionResult(scores=scores)


class OneClassSVM(AnomalyDetector):
    """Implements a one-class SVM to detect anomalies."""

    abbreviation = 'one-class-svm'
    name = 'OC-SVM'

    supported_strategies = [Strategy.SINGLE]
    supported_heuristics = [Heuristic.DEFAULT]
    supports_attributes = True

    def __init__(self, model=None, nu=.5):
        super(OneClassSVM, self).__init__(model=model)
        self.nu = nu

    def fit(self, dataset):
        features = dataset.flat_features_2d

        from sklearn.svm import OneClassSVM
        self._model = OneClassSVM(nu=self.nu)
        self._model.fit(features)

    def detect(self, dataset):
        features = dataset.flat_features_2d

        input_size = self._model.support_vectors_.shape[1]
        features_size = features.shape[1]
        if input_size > features_size:
            features = np.pad(features, [(0, 0), (0, input_size - features_size), (0, 0)][:len(features.shape)],
                              mode='constant')
        elif input_size < features_size:
            features = features[:, :input_size]

        anomaly_scores = np.zeros(dataset.binary_targets.shape, dtype=int)
        pred = self._model.predict(features)
        pred = (pred == -1)[:, np.newaxis, np.newaxis]
        anomaly_scores += pred

        return AnomalyDetectionResult(scores=anomaly_scores)
