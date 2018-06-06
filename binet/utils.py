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

from enum import Enum
from pathlib import Path

import numpy as np

from binet.constants import MODEL_EXT, EVAL_EXT
from binet.folders import MODEL_DIR, EVENTLOG_DIR, EVAL_DIR, EVENTLOG_CACHE_DIR


class File(object):
    ext = None

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        if path.suffix != self.ext:
            path = Path(str(path) + self.ext)
        if not path.is_absolute():
            path = MODEL_DIR / path.name

        self.path = path
        self.file = self.path.name
        self.name = self.path.stem

        self.event_log_name, self.ad, self.date = self.name.split('_')
        from dateutil import parser
        self.date = parser.parse(self.date)
        s = self.event_log_name.split('-')
        self.model = s[0]
        self.p = float(s[1])
        self.id = int(s[2])

    def remove(self):
        import os
        if self.path.exists():
            os.remove(self.path)


class ModelFile(File):
    ext = MODEL_EXT

    @property
    def eval_file(self):
        return EVAL_DIR / (self.name + EVAL_EXT)


class EvaluationFile(File):
    ext = EVAL_EXT

    @property
    def model_file(self):
        return MODEL_DIR / (self.name + MODEL_EXT)


class EventLogFile(object):
    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        if '.json' not in path.suffixes:
            path = Path(str(path) + '.json.gz')
        if not path.is_absolute():
            path = EVENTLOG_DIR / path.name

        self.path = path
        self.file = self.path.name
        self.name = self.path.stem
        if len(self.path.suffixes) > 1:
            self.name = Path(self.path.stem).stem
        s = self.name.split('-')
        self.model = s[0]
        self.p = float(s[1])
        self.id = int(s[2])

    @property
    def cache_file(self):
        return EVENTLOG_CACHE_DIR / (self.name + '.pkl.gz')

    def remove(self):
        import os
        if self.path.exists():
            os.remove(self.path)


def get_event_logs(path=None):
    if path is None:
        path = EVENTLOG_DIR
    for f in path.glob('*.json*'):
        yield EventLogFile(f)


def get_models():
    for f in MODEL_DIR.glob(f'*{MODEL_EXT}'):
        yield ModelFile(f)


def get_evaluations():
    for f in EVAL_DIR.glob(f'*{EVAL_EXT}'):
        yield EvaluationFile(f)


class AttributeType(Enum):
    CATEGORICAL = 0
    NUMERICAL = 1


class Label(object):
    ANOMALY = True
    NORMAL = False

    @staticmethod
    def values():
        return [Label.NORMAL, Label.ANOMALY]


def label_collapse(a, axis=2):
    if axis < 0:
        axis += 2
    if axis < 2:
        a = np.any(a, axis=2)
        if axis == 0:
            a = np.any(a, axis=1)
    return a


def mean_collapse(a, axis=-2):
    if axis < 0:
        a = np.mean(a, axis=2)
        if axis == -2:
            a = np.mean(a, axis=1)
    return a
