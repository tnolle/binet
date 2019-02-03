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
from sklearn import metrics

from april.anomalydetection.utils import anomaly_ratio
from april.enums import Heuristic


def best_heuristic(taus, theta, y_true, **kwargs):
    """h_best in the paper."""
    f1s = [metrics.f1_score(y_true.compressed(), theta(tau=tau, **kwargs).compressed()) for tau in taus]
    return taus[np.argmax(f1s)]


def elbow_heuristic(taus, theta, **kwargs):
    """h_elbow in the paper."""
    if len(taus) < 4:
        return taus[-1]
    r = np.array([anomaly_ratio(theta(tau=tau, **kwargs)) for tau in taus])
    step = taus[1:] - taus[:-1]
    r_prime_prime = (r[2:] - 2 * r[1:-1] + r[:-2]) / (step[1:] * step[:-1])
    return {
        Heuristic.ELBOW_DOWN: taus[np.argmax(r_prime_prime) + 1],
        Heuristic.ELBOW_UP: taus[np.argmin(r_prime_prime) + 1]
    }


def lowest_plateau_heuristic(taus, theta, **kwargs):
    if len(taus) < 4:
        return taus[-1]
    r = np.array([anomaly_ratio(theta(tau=tau, **kwargs)) for tau in taus])
    r_prime = (r[1:] - r[:-1]) / (taus[1:] - taus[:-1])
    stable_region = r_prime > np.mean(r_prime) / 2
    regions = np.split(np.arange(len(stable_region)), np.where(~stable_region)[0])
    regions = [taus[idx[1:]] for idx in regions if len(idx) > 1]
    if len(regions) == 0:
        regions = [taus[-2:]]
    return {
        Heuristic.LP_LEFT: regions[-1].min(),
        Heuristic.LP_MEAN: regions[-1].mean(),
        Heuristic.LP_RIGHT: regions[-1].max()
    }


def ratio_heuristic(taus, theta, nu=0.3, **kwargs):
    for tau in taus:
        r = anomaly_ratio(theta(tau=tau, **kwargs))
        if r < nu:
            return tau
    return 0
