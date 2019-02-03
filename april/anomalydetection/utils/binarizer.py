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

import numpy as np

from april.anomalydetection.utils import label_collapse
from april.anomalydetection.utils import max_collapse
from april.anomalydetection.utils.heuristic import best_heuristic
from april.anomalydetection.utils.heuristic import elbow_heuristic
from april.anomalydetection.utils.heuristic import ratio_heuristic
from april.anomalydetection.utils.heuristic import lowest_plateau_heuristic
from april.enums import Base
from april.enums import Class
from april.enums import Heuristic
from april.enums import Strategy


class Binarizer(object):
    def __init__(self, result, mask, features, targets=None):
        self.result = result
        self._mask = mask
        self.mask_ = mask
        self.features = features
        self._targets = targets

        # Try to fix dimensions
        if self.mask_.shape != self.result.scores.shape:
            if len(self.mask_) != len(self.result.scores.shape):
                self.mask_ = np.expand_dims(self.mask_, axis=-1)
            self.mask_ = np.repeat(self.mask_, self.result.scores.shape[-1], axis=-1)

        self.targets = None

        if self._targets is not None:
            self.targets = dict((a, self.mask(label_collapse(self._targets, axis=a))) for a in [0, 1, 2])

    def mask(self, a):
        if len(a.shape) == 1:
            m = self.mask_[:, 0, 0]
        elif len(a.shape) == 2:
            m = self.mask_[:, :, 0]
        else:
            m = self.mask_
        return np.ma.array(a, mask=m)

    def get_targets(self, axis=2):
        return self.targets.get(axis)

    def correct_shape(self, tau, strategy):
        tau = np.asarray(tau)
        if strategy == Strategy.POSITION:
            tau = tau[:, None]
        if strategy == Strategy.POSITION_ATTRIBUTE:
            tau = tau.reshape(*self.result.scores.shape[1:])
        return tau

    def split_by_strategy(self, a, strategy):
        if strategy == Strategy.SINGLE:
            return [a]
        elif isinstance(a, list):
            if strategy == Strategy.POSITION:
                return [[_a[:, i:i + 1] for _a in a] for i in range(len(a[0][0]))]
            elif strategy == Strategy.ATTRIBUTE:
                return [[_a] for _a in a]
            elif strategy == Strategy.POSITION_ATTRIBUTE:
                return [[_a[:, i:i + 1]] for i in range(len(a[0][0])) for _a in a]
        else:
            if strategy == Strategy.POSITION:
                return [a[:, i:i + 1, :] for i in range(a.shape[1])]
            elif strategy == Strategy.ATTRIBUTE:
                return [a[:, :, i:i + 1] for i in range(a.shape[2])]
            elif strategy == Strategy.POSITION_ATTRIBUTE:
                return [a[:, i:i + 1, j:j + 1] for i in range(a.shape[1]) for j in range(a.shape[2])]

    def get_grid_candidate_taus(self, a, steps=20, axis=0):
        """G in the paper."""
        return np.linspace(max_collapse(a, axis=axis).min() - .001, a.max(), steps)

    def get_candidate_taus(self, a, axis=0):
        a = max_collapse(a, axis=axis).compressed()
        a_min = a.min()
        a_max = a.max()
        if a_max > a_min:
            a = (a_max - a) / (a_max - a_min)
        a = 2 * (a / 2).round(2)
        if a_max > a_min:
            a = a_max - a * (a_max - a_min)
        a = np.sort(np.unique(a))
        a[0] -= .001
        if len(a) < 5:
            a = np.linspace(a_min - .001, a_max, 5)
        return a

    def get_legacy_tau(self, scores, heuristic=Heuristic.DEFAULT, strategy=Strategy.SINGLE, axis=0):
        if heuristic == Heuristic.DEFAULT:
            return np.array([0.5])

        if not isinstance(scores, np.ma.MaskedArray):
            scores = self.mask(scores)

        alpha = None
        if strategy == Strategy.SINGLE:
            alpha = np.array([scores.mean()])
        elif strategy == Strategy.ATTRIBUTE:
            alpha = scores.mean(axis=1).mean(axis=0).data
        elif strategy == Strategy.POSITION:
            alpha = scores.mean(axis=2).mean(axis=0).data[:, None]
        elif strategy == Strategy.POSITION_ATTRIBUTE:
            alpha = scores.mean(axis=0).data

        taus = self.get_grid_candidate_taus(scores / alpha, axis=axis)
        tau = None
        if heuristic == Heuristic.BEST:
            y_true = self.get_targets(axis=axis)
            tau = best_heuristic(taus=taus, theta=self.legacy_binarize, y_true=y_true, alpha=alpha, scores=scores,
                                 axis=axis)

        if heuristic == Heuristic.RATIO:
            tau = ratio_heuristic(taus=taus, theta=self.legacy_binarize, scores=scores, axis=axis, alpha=alpha)

        if heuristic in [Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP]:
            tau = elbow_heuristic(taus=taus, theta=self.legacy_binarize, scores=scores, axis=axis,
                                  alpha=alpha)[heuristic]

        if heuristic in [Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT]:
            tau = lowest_plateau_heuristic(taus=taus, theta=self.legacy_binarize, scores=scores, axis=axis,
                                           alpha=alpha)[heuristic]

        return tau * alpha

    def get_tau(self, scores, heuristic=Heuristic.DEFAULT, strategy=Strategy.SINGLE, axis=0, taus=None):
        if heuristic == Heuristic.DEFAULT:
            return np.array([0.5])

        if not isinstance(scores, np.ma.MaskedArray):
            scores = self.mask(scores)

        scores = self.split_by_strategy(scores, strategy)

        if heuristic in [Heuristic.MEAN, Heuristic.MEDIAN]:
            scores = [max_collapse(s, axis=axis) for s in scores]
            if heuristic == Heuristic.MEAN:
                return self.correct_shape([np.mean(s[np.round(s, 1) > 0]) for s in scores], strategy)
            elif heuristic == Heuristic.MEDIAN:
                return self.correct_shape([np.median(s[np.round(s, 1) > 0]) for s in scores], strategy)

        if taus is None:
            taus = [self.get_candidate_taus(s, axis=axis) for s in scores]
        else:
            taus = [taus] * len(scores)

        tau = None
        if heuristic == Heuristic.BEST:
            y_trues = self.split_by_strategy(self.get_targets(axis=2), strategy)
            y_trues = [label_collapse(y, axis=axis) for y in y_trues]
            tau = [best_heuristic(taus=t, theta=self.threshold_binarize, y_true=y, scores=s, axis=axis)
                   for s, t, y in zip(scores, taus, y_trues)]

        if heuristic == Heuristic.RATIO:
            tau = [ratio_heuristic(taus=t, scores=s, theta=self.threshold_binarize, axis=axis)
                   for s, t in zip(scores, taus)]

        if heuristic in [Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP]:
            tau = [elbow_heuristic(taus=t, scores=s, theta=self.threshold_binarize, axis=axis)[heuristic]
                   for s, t in zip(scores, taus)]

        if heuristic in [Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT]:
            tau = [lowest_plateau_heuristic(taus=t, scores=s, theta=self.threshold_binarize, axis=axis)[heuristic]
                   for s, t in zip(scores, taus)]

        return self.correct_shape(tau, strategy)

    def legacy_binarize(self, scores, tau, alpha, axis=0):
        # Apply the threshold function (Theta in the paper) using alpha as a scaling factor
        return self.threshold_binarize(tau=tau * alpha, scores=scores, axis=axis)

    def threshold_binarize(self, tau, scores, axis=0):
        # Apply the threshold function (Theta in the paper)
        predictions = np.array(scores.data > tau, dtype=int)

        # Apply mask
        predictions = np.ma.array(predictions, mask=scores.mask)

        # Positive axis flatten predictions
        if axis in [0, 1]:
            predictions = label_collapse(predictions, axis=axis)

        return predictions

    def binarize(self, scores=None, tau=None, base=None, heuristic=None, strategy=None, go_backwards=False,
                 return_parameters=False, axis=2, heuristic_axis=None):

        if heuristic_axis is None:
            heuristic_axis = axis

        if scores is None:
            if go_backwards:
                scores = self.result.scores_backward
            else:
                scores = self.result.scores

        if not isinstance(scores, np.ma.MaskedArray):
            scores = self.mask(scores)

        # Get baseline threshold (tau in the paper)
        if tau is None or heuristic != Heuristic.MANUAL:
            if base == Base.LEGACY:
                tau = self.get_legacy_tau(scores=scores, heuristic=heuristic, strategy=strategy, axis=heuristic_axis)
            else:
                tau = self.get_tau(scores=scores, heuristic=heuristic, strategy=strategy, axis=heuristic_axis)

        # Apply the threshold function (Theta in the paper)
        predictions = self.threshold_binarize(scores=scores, tau=tau, axis=axis)

        if return_parameters:
            return predictions, tau

        return predictions

    @staticmethod
    def get_scores(probabilities):
        scores = np.zeros_like(probabilities)
        for i in range(scores.shape[2]):
            p = probabilities[:, :, i:i + 1]
            _p = np.copy(probabilities)
            _p[_p <= p] = 0
            scores[:, :, i] = _p.sum(axis=2)
        return scores

    def classify(self, tau, features, predictions):
        def mask(a, mask):
            b = np.copy(a)
            b[mask == 1] = 0
            c = np.copy(a)
            c[mask == 0] = 0
            return b, c

        classification = np.zeros_like(predictions)
        c_cf = classification[:, :, 0]
        c_data = classification[:, :, 1:]
        predictions_cf = predictions[:, :, 0]
        predictions_data = predictions[:, :, 1:]

        # Attribute heuristic
        c_data[predictions_data == 1] = Class.ATTRIBUTE
        c_data[predictions_data == 0] = Class.NORMAL_ATTRIBUTE

        # Insert and Skip heuristics
        if self.result.predictions is not None:
            f = features[0]

            # Top-1 predictions
            # p = np.argmax(self.result.predictions[0], axis=2) + 1

            # Top-n predictions according to threshold (tau)
            _p = self.get_scores(self.result.predictions[0])
            p = np.zeros_like(_p) + np.arange(_p.shape[-1]) + 1
            p[_p > tau[0]] = -1

            # Mask padding
            p[self._mask] = -1

            # Helper objects
            pfht = np.zeros_like(predictions_cf)
            pfhf = np.zeros_like(predictions_cf)
            pftt = np.zeros_like(predictions_cf)
            pftf = np.zeros_like(predictions_cf)
            ppht = np.zeros_like(predictions_cf)
            pphf = np.zeros_like(predictions_cf)
            pptt = np.zeros_like(predictions_cf)
            pptf = np.zeros_like(predictions_cf)
            ffht = np.zeros_like(predictions_cf)
            ffhf = np.zeros_like(predictions_cf)
            fftt = np.zeros_like(predictions_cf)
            fftf = np.zeros_like(predictions_cf)
            fpht = np.zeros_like(predictions_cf)
            fphf = np.zeros_like(predictions_cf)
            fptt = np.zeros_like(predictions_cf)
            fptf = np.zeros_like(predictions_cf)

            for j in np.arange(f.shape[1]):
                # Current prediction and feature
                _p = p[:, j:j + 1]
                _f = f[:, j:j + 1]

                # Top-1 Predictions
                ph = p[:, :j]
                phf, pht = mask(ph, predictions_cf[:, :j])
                pt = p[:, j + 1:]
                ptf, ptt = mask(pt, predictions_cf[:, j + 1:])

                # Actual case features
                fh = f[:, :j]
                fhf, fht = mask(fh, predictions_cf[:, :j])
                ft = f[:, j + 1:]
                ftf, ftt = mask(ft, predictions_cf[:, j + 1:])

                # Prediction appears elsewhere in case
                pfht[:, j] = np.any(np.any(_p == fht[:, :, np.newaxis], axis=-1), axis=-1)
                pfhf[:, j] = np.any(np.any(_p == fhf[:, :, np.newaxis], axis=-1), axis=-1)
                pftt[:, j] = np.any(np.any(_p == ftt[:, :, np.newaxis], axis=-1), axis=-1)
                pftf[:, j] = np.any(np.any(_p == ftf[:, :, np.newaxis], axis=-1), axis=-1)

                # Prediction appears elsewhere in predictions
                ppht[:, j] = np.any(np.any(_p == pht, axis=-1), axis=-1)
                pphf[:, j] = np.any(np.any(_p == phf, axis=-1), axis=-1)
                pptt[:, j] = np.any(np.any(_p == ptt, axis=-1), axis=-1)
                pptf[:, j] = np.any(np.any(_p == ptf, axis=-1), axis=-1)

                # Event appears elsewhere in case
                ffht[:, j] = np.any(_f == fht, axis=-1)
                ffhf[:, j] = np.any(_f == fhf, axis=-1)
                fftt[:, j] = np.any(_f == ftt, axis=-1)
                fftf[:, j] = np.any(_f == ftf, axis=-1)

                # Event appears elsewhere in predictions
                fpht[:, j] = np.any(np.any(_f[:, :, np.newaxis] == pht, axis=-1), axis=-1)
                fphf[:, j] = np.any(np.any(_f[:, :, np.newaxis] == phf, axis=-1), axis=-1)
                fptt[:, j] = np.any(np.any(_f[:, :, np.newaxis] == ptt, axis=-1), axis=-1)
                fptf[:, j] = np.any(np.any(_f[:, :, np.newaxis] == ptf, axis=-1), axis=-1)

            # Classification rules
            skips = np.logical_and(predictions_cf == 1, ~np.logical_or(pfhf, pftf))
            inserts = np.logical_and(predictions_cf == 1, np.logical_or(pfhf, pftf))
            reworks = np.logical_and(predictions_cf == 1, ffhf)

            shifts = np.logical_and(predictions_cf == 1, np.logical_xor(pfht, pftt))
            lates = np.logical_and(predictions_cf == 1, fpht)
            earlies = np.logical_and(predictions_cf == 1, fptt)

            # Set the labels
            c_cf[inserts] = Class.INSERT
            c_cf[skips] = Class.SKIP
            c_cf[shifts] = Class.SHIFT
            c_cf[lates] = Class.LATE
            c_cf[earlies] = Class.EARLY
            c_cf[reworks] = Class.REWORK

        return self.mask(classification)
