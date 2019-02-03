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
from april.enums import Heuristic
from april.enums import Strategy


class NaiveAnomalyDetector(AnomalyDetector):
    """Implements the Naive algorithm from Bezerra et al.

    Anomaly scores per trace are based on the frequency of the specific variant the trace follows.
    This is ignoring all attributes except the activity name.
    """

    abbreviation = 'naive'
    name = 'Naive'

    supported_heuristics = [Heuristic.DEFAULT]
    supported_strategies = [Strategy.SINGLE]

    def __init__(self, model=None):
        super(NaiveAnomalyDetector, self).__init__(model=model)
        self.get_anomaly_scores = np.vectorize(lambda x: self._model[x] if x in self._model.keys() else np.infty)

    def fit(self, dataset):
        keys, counts = np.unique(self.traces(dataset), return_counts=True)
        self._model = dict(zip(keys, -np.log(counts / dataset.num_cases)))

    def detect(self, dataset):
        scores = np.zeros_like(dataset.binary_targets)
        scores[:, :, 0] = self.get_anomaly_scores(self.traces(dataset))[:, np.newaxis]
        scores = scores > -np.log(0.02)
        return AnomalyDetectionResult(scores=scores)

    def traces(self, dataset):
        return np.array([hash(t[t != 0].tostring()) for t in dataset.features[0]])


class NaivePlusAnomalyDetector(NaiveAnomalyDetector):
    abbreviation = 'naive+'
    name = 'Naive+'

    supported_strategies = [Strategy.SINGLE]
    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]

    def __init__(self, model=None):
        super(NaivePlusAnomalyDetector, self).__init__(model=model)

    def detect(self, dataset):
        scores = np.zeros_like(dataset.binary_targets)
        scores[:, :, 0] = self.get_anomaly_scores(self.traces(dataset))[:, np.newaxis]
        return AnomalyDetectionResult(scores=scores)


class SamplingAnomalyDetector(AnomalyDetector):
    """Implements the Sampling method from Bezerra et al. (Algorithm 3)

    A HeuristicsMiner is used to mine a process model based on a sample of the event log. Then every non-matching
    trace is marked as an anomaly.
    """

    abbreviation = 'sampling'
    name = 'Sampling'

    supported_heuristics = [Heuristic.DEFAULT]
    supported_strategies = [Strategy.SINGLE]

    def __init__(self, model=None):
        super(SamplingAnomalyDetector, self).__init__(model=model)

    def fit(self, dataset, s=0.7):
        from april.processmining import HeuristicsMiner
        miner = HeuristicsMiner()
        idx = np.random.choice(np.arange(dataset.num_cases), int(dataset.num_cases * s))
        miner.mine(dataset.features[0][idx])
        self._model = miner.adj_mat

    def detect(self, dataset):
        from april.processmining import HeuristicsMiner
        keys, inverse, counts = np.unique(dataset.features[0], return_counts=True, return_inverse=True, axis=0)
        probs = -np.log(counts / dataset.num_cases)[inverse]
        candidates = probs > -np.log(0.02)
        miner = HeuristicsMiner(adj_mat=self._model)
        scores = np.zeros_like(dataset.binary_targets)
        scores[candidates, :, 0] = ~miner.conformance_check(dataset.features[0][candidates])
        return AnomalyDetectionResult(scores=scores)
