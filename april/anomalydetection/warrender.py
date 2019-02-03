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

from april.anomalydetection import AnomalyDetectionResult
from april.anomalydetection import AnomalyDetector
from april.enums import Base
from april.enums import Heuristic
from april.enums import Mode
from april.enums import Strategy


class TStidePlus(AnomalyDetector):
    """Implements a sliding window anomaly detection algorithm.

    This method is based on the t-Stide implementation. Anomaly scores are based on frequencies of n-grams.
    Size of n-grams is controlled via the k parameter.
    """

    abbreviation = 't-stide+'
    name = 't-STIDE+'

    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.LEGACY, Base.SCORES]
    supports_attributes = True

    def __init__(self, model=None, k=2):
        """Initialize sliding window anomaly detector.

        :param model: Path to saved model file. Defaults to None.
        :type model: str
        :param k: N-gram size. Defaults to 2.
        :type k: int
        """
        super(TStidePlus, self).__init__(model=model)

        self.k = k
        self._model = None
        self.score = None
        self.get_anomaly_scores = np.vectorize(lambda x: self.score[x] if x in self.score.keys() else np.infty)

    def load(self, file_name):
        super(TStidePlus, self).load(file_name)
        self.score = self._model['score']
        self.k = self._model['k']

    def fit(self, dataset):
        # Create ngrams for all features like "0:1,1:2|0:2,1:2"
        n = self.get_ngrams(dataset.flat_features)
        num_ngrams = len(n)

        # Add ngrams per attribute, e.g. "0:1,1:2|0:2" and ""0:1,1:2|1:2""
        ngrams = [n]
        if dataset.num_attributes > 1:
            for i in range(dataset.num_attributes):
                m = np.copy(n)
                m[:, :, i] = -1
                ngrams.append(m)
            ngrams = np.hstack(ngrams)
        else:
            ngrams = n

        # Flatten ngrams
        ngrams = ngrams.reshape(np.product(ngrams.shape[:-2]), np.product(ngrams.shape[2:]))
        ngrams = np.apply_along_axis(lambda x: hash(x.tostring()), -1, ngrams)

        # Count ngrams
        keys, counts = np.unique(ngrams, return_counts=True, axis=0)
        counts = -np.log(counts / num_ngrams)

        self._model = dict(k=self.k, score=dict(zip(keys, counts)))

    def detect(self, dataset):
        n = self.get_ngrams(dataset.flat_features)
        ngrams = []
        if dataset.num_attributes > 1:
            for i in range(dataset.num_attributes):
                m = np.copy(n)
                m[:, :, i] = -1
                m = m.reshape(*m.shape[:-2], np.product(m.shape[2:]))
                m = np.apply_along_axis(lambda x: hash(x.tostring()), -1, m)
                ngrams.append(m)
            ngrams = np.dstack(ngrams)
        else:
            n.reshape(*n.shape[:-2], np.product(n.shape[2:]))
            n = np.apply_along_axis(lambda x: hash(x.tostring()), -1, n)
            ngrams = n

        scores = self.get_anomaly_scores(ngrams)

        return AnomalyDetectionResult(scores=scores)

    def get_ngrams(self, features):
        if self.k > 1:
            pad_features = np.pad(features, ((0, 0), (self.k - 1, 0), (0, 0)), mode='constant')
            ngrams = [pad_features]
            for i in np.arange(1, self.k):
                ngrams.append(np.pad(pad_features[:, i:], ((0, 0), (0, i), (0, 0)), mode='constant'))
            ngrams = np.stack(ngrams, axis=-1)[:, :-(self.k - 1)]
            return ngrams


class TStide(TStidePlus):
    abbreviation = 't-stide'
    name = 't-STIDE'

    supported_strategies = [Strategy.SINGLE, Strategy.POSITION]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.LEGACY, Base.SCORES]
    supports_attributes = False

    def __init__(self, model=None, k=2):
        super(TStide, self).__init__(model=model, k=k)

    def get_ngrams(self, features, flatten=False, attribute_index=None, num_attributes=None):
        """Only use the first attribute; i.e., the activity name."""
        features = features[:, :, :1]
        return super(TStide, self).get_ngrams(features)
