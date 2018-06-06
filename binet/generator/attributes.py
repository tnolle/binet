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

import numpy as np

DECIMALS = 2


class Attribute(object):
    def __init__(self, name):
        self.name = name

    def random_value(self):
        pass


class CategoricalAttribute(Attribute):
    def __init__(self, name, values=10, domain=None, min_group=1, max_group=None):
        super().__init__(name=name)

        if isinstance(values, int):
            values = list(range(1, values + 1))
        elif not isinstance(values, list):
            raise TypeError('Incompatible values type, must be list.')

        self.values = sorted(values)
        if domain is None:
            self.domain = values
        else:
            self.domain = domain

        self.min_group = min_group
        if max_group is None or max_group > len(self.domain):
            self.max_group = len(self.domain)
        else:
            self.max_group = max_group

    def random_value(self):
        return str(np.random.choice(self.values))

    def incorrect_value(self):
        values = [x for x in self.domain if x not in self.values]
        if len(values) == 0:
            raise AttributeError('No incorrect values possible.')
        return str(np.random.choice(values))

    def to_json(self):
        return {'name': self.name, 'values': self.values, 'type': 'categorical'}


class NumericAttribute(Attribute):
    def __init__(self, name):
        super().__init__(name=name)


class UniformAttribute(NumericAttribute):
    def __init__(self, name, low=0, high=100):
        super().__init__(name=name)
        self.low = float(low)
        self.high = float(high)

    def random_value(self):
        return np.round(np.random.uniform(self.low, self.high), DECIMALS).astype(float)

    def incorrect_value(self):
        diff = np.abs(self.high - self.low)
        smaller = np.random.uniform(self.low - diff, self.low)
        greater = np.random.uniform(self.high, self.high + diff)
        return np.round(np.random.choice([smaller, greater]), DECIMALS).astype(float)

    def to_json(self):
        return {'name': self.name, 'low': self.low, 'high': self.high, 'type': 'numeric'}


class NormalAttribute(NumericAttribute):
    def __init__(self, name, sigma=1.0, mu=0.0):
        super().__init__(name=name)
        self.sigma = float(sigma)
        self.mu = float(mu)

    def random_value(self):
        return np.round(np.random.normal(loc=self.mu, scale=self.sigma), DECIMALS).astype(float)

    def incorrect_value(self):
        smaller = np.random.normal(loc=self.mu - self.sigma * 10, scale=self.sigma)
        greater = np.random.normal(loc=self.mu + self.sigma * 10, scale=self.sigma)
        return np.round(np.random.choice([smaller, greater]), DECIMALS).astype(float)

    def to_json(self):
        return {'name': self.name, 'sigma': self.sigma, 'mu': self.mu, 'type': 'numeric'}
