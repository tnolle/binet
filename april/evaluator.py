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

import gzip
import pickle as pickle

import numpy as np
import sklearn.metrics as metrics
from sklearn.exceptions import UndefinedMetricWarning

from april.anomalydetection import AD
from april.anomalydetection.utils import label_collapse
from april.anomalydetection.utils.binarizer import Binarizer
from april.dataset import Dataset
from april.enums import Axis
from april.enums import Base
from april.enums import Class
from april.enums import Heuristic
from april.enums import Mode
from april.enums import Strategy
from april.fs import ModelFile
from april.fs import PLOT_DIR
from april.generation import prettify_label
from april.processmining import Case
from april.processmining import Event
from april.processmining.log import EventLog


class Evaluator(object):
    def __init__(self, model):
        if not isinstance(model, ModelFile):
            self.model = ModelFile(model)
        else:
            self.model = model
        self.model_file = self.model.path

        self.model_name = self.model.name
        self.eventlog_name = self.model.event_log_name
        self.process_model_name = self.model.model
        self.noise = self.model.p
        self.dataset_id = self.model.id
        self.model_date = self.model.date
        self.ad_ = AD.get(self.model.ad)()

        self._dataset = None
        self._result = None
        self._binarizer = None
        self._event_log_df = None
        self._classification = None

        import warnings
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    @staticmethod
    def _cache_result(path, anomaly_scores):
        with gzip.open(path, 'wb') as f:
            pickle.dump(anomaly_scores, f, protocol=4)

    @staticmethod
    def _load_result_from_cache(file):
        return pickle.load(gzip.open(file, 'rb'))

    def cache_result(self):
        return self.result

    @property
    def ad(self):
        if self.ad_.model is None:
            self.ad_.load(self.model_file)
        return self.ad_

    @property
    def event_log(self):
        return self.dataset.event_log

    @property
    def event_log_df(self):
        df = self.event_log.dataframe
        df = df.set_index(['case_id', 'event_position']).unstack()
        df = df.swaplevel(0, 1, axis=1).sort_index(level=0, axis=1)
        df = df.reindex(self.dataset.attribute_keys, axis=1, level=1)
        return df

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = Dataset(self.eventlog_name)
        return self._dataset

    @property
    def binarizer(self):
        if self.result is not None:
            if self._binarizer is None:
                self._binarizer = Binarizer(result=self.result, mask=self.dataset.mask,
                                            features=self.dataset.flat_features, targets=self.dataset.binary_targets)
        return self._binarizer

    @property
    def result(self):
        if self._result is None:
            if self.model.result_file.exists():
                self._result = self._load_result_from_cache(self.model.result_file)
            else:
                self._result = self.ad.detect(self.dataset)
                from april.anomalydetection import NNAnomalyDetector
                if isinstance(self.ad, NNAnomalyDetector):
                    import keras as ks
                    ks.backend.clear_session()
                self._cache_result(self.model.result_file, self.result)
        return self._result

    def evaluate(self, base=None, mode=None, strategy=None, heuristic=None, normalization=None, go_backwards=False,
                 tau=None, return_parameters=False):
        if normalization is not None:
            self.result.normalization = normalization

        if mode is None:
            mode = Mode.BINARIZE
        if strategy is None:
            strategy = Strategy.SINGLE
        if heuristic is None:
            heuristic = Heuristic.DEFAULT

        # Get predictions and targets
        y_pred = None
        y_true = None
        tau_f = None
        tau_b = None
        if mode == Mode.BINARIZE:
            if go_backwards is None and self.result.scores_backward is not None:
                y_pred_f, tau_f = self.binarizer.binarize(base=base, heuristic=heuristic, strategy=strategy, tau=tau,
                                                          return_parameters=True, go_backwards=False)
                y_pred_b, tau_b = self.binarizer.binarize(base=base, heuristic=heuristic, strategy=strategy, tau=tau,
                                                          return_parameters=True, go_backwards=True)
                y_pred = y_pred_f + y_pred_b
                y_pred[y_pred == Class.ANOMALY * 2] = Class.ANOMALY
            else:
                y_pred, tau_f = self.binarizer.binarize(base=base, heuristic=heuristic, strategy=strategy, tau=tau,
                                                        return_parameters=True, go_backwards=go_backwards)
            y_true = self.binarizer.get_targets()
        elif mode == Mode.CLASSIFY:
            y_pred, tau_f = self.binarizer.binarize(base=base, heuristic=heuristic, strategy=strategy, tau=tau,
                                                    return_parameters=True, go_backwards=False)
            y_pred = self.binarizer.classify(tau_f, self.dataset.features, y_pred)
            y_true = self.binarizer.mask(self.dataset.classes)

        # Confusion matrix
        if y_pred.shape[2] > 1:
            classes = self.binarizer.mask(self.dataset.classes)
            classes[:, :, 1:][classes[:, :, 1:] == Class.NORMAL] = Class.NORMAL_ATTRIBUTE
            classes = classes.compressed()
        elif y_pred.shape[2] > 0:
            classes = self.binarizer.mask(self.dataset.classes)[:, :, 0].compressed()
        else:
            classes = y_true

        unique_y = np.unique(np.concatenate((y_pred.compressed(), classes)))
        _cm = metrics.confusion_matrix(classes, y_pred.compressed(), labels=unique_y)
        cm = {}
        for (_i, _j), x in np.ndenumerate(_cm):
            i = int(unique_y[_i])
            j = int(unique_y[_j])
            ignored_classes = [c for c in Class.keys() if c not in [Class.NORMAL, Class.ANOMALY]]
            if mode == Mode.BINARIZE and (i == Class.ANOMALY or j in ignored_classes):
                continue
            if Class.values()[i] not in cm:
                cm[Class.values()[i]] = {}
            cm[Class.values()[i]][Class.values()[j]] = int(x)

        def evaluate(y_true, y_pred):
            evaluation = {}
            axes = Axis.keys()[:y_pred.ndim + 1]
            for axis in axes:
                yt = label_collapse(y_true, axis=axis)
                yp = label_collapse(y_pred, axis=axis)
                p, r, f, s = metrics.precision_recall_fscore_support(yt.compressed(), yp.compressed(), average='macro')
                evaluation[axis] = dict(precision=p, recall=r, f1=f, support=s)
            return evaluation

        # Split control flow and data perspective
        y_pred_cf = y_pred[:, :, :1]
        y_true_cf = y_true[:, :, :1]
        y_pred_data = y_pred[:, :, 1:]
        y_true_data = y_true[:, :, 1:]

        evaluation = dict(cm=cm, combined=evaluate(y_true, y_pred))
        if self.dataset.num_attributes > 1:
            evaluation = dict(**evaluation, cf=evaluate(y_true_cf, y_pred_cf))
            evaluation = dict(**evaluation, data=evaluate(y_true_data, y_pred_data))

        if return_parameters:
            if tau_b is not None:
                return evaluation, tau_f, tau_b
            return evaluation, tau_f

        # Evaluate
        return evaluation

    def get_indices(self, start=0, num_cases=20, reduction=None, anomaly_type=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if anomaly_type is not None and len(self.dataset.anomaly_indices) > 0:
            if anomaly_type in ['Anomalous', 'Anomaly']:
                indices = self.dataset.anomaly_indices
            else:
                indices = self.dataset.get_indices_for_type(anomaly_type)
        else:
            indices = range(self.dataset.num_cases)

        if reduction is None:
            if start is None or start > len(indices):
                start = 0
            if num_cases is None or start + num_cases > len(indices):
                end = len(indices)
            else:
                end = start + num_cases
            indices = indices[start:end]
        elif reduction == 'sample':
            if num_cases < len(indices):
                indices = sorted(np.random.choice(indices, min(len(indices), num_cases), replace=False))
        elif reduction == 'uniform':
            if self.dataset.text_labels is not None and len(self.dataset.text_labels) > 0:
                if num_cases < len(indices):
                    labels = self.dataset.unique_text_labels
                    indices = np.concatenate(
                        [np.random.choice(np.where(self.dataset.text_labels == label)[0], int(num_cases / len(labels)))
                         for
                         label in labels])
        return indices

    def check_parameters(self, base, mode, heuristic, strategy, tau):
        if base is None:
            base = Base.SCORES
        if mode is None:
            heuristic = None
            strategy = None
            tau = None
        elif mode in [Mode.BINARIZE, Mode.CLASSIFY]:
            if heuristic != Heuristic.MANUAL:
                tau = None
        return base, heuristic, strategy, tau

    def plot_heatmap(self, start=0, num_cases=20, reduction=None, anomaly_type=None, seed=None,
                     base=Base.SCORES, mode=None, strategy=None, heuristic=None,
                     normalization=None, tau=None, go_backwards=False,
                     min_event=None, max_event=None, short_labels=False, file_name=None, figsize=None, indices=None,
                     prettify_fn=None):

        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        def flatten(a):
            return a.reshape((a.shape[0], np.product(a.shape[1:])))

        # Number of attributes
        n = self.dataset.num_attributes

        # Get indices
        if indices is None:
            indices = self.get_indices(start, num_cases, reduction, anomaly_type, seed)

        # Check parameters
        base, heuristic, strategy, tau = self.check_parameters(base, mode, heuristic, strategy, tau)

        # Event positions
        if min_event is None:
            min_event = 0
        if max_event is None:
            max_event = self.dataset.max_len

        # Figure size
        if figsize is None:
            width = int((max_event - min_event) * n * 2)
            height = int(num_cases * 0.8)
            figsize = (width, height)

        # Normalization
        self.result.normalization = normalization

        # Binarize
        if mode == Mode.BINARIZE:
            scores = self.binarizer.binarize(base=base, strategy=strategy, heuristic=heuristic, tau=tau,
                                             go_backwards=go_backwards, axis=2)
        else:
            if go_backwards:
                scores = self.result.scores_backward
            else:
                scores = self.result.scores
            scores = self.binarizer.mask(scores)

        # Classify
        if mode == Mode.CLASSIFY:
            y_pred, tau = self.binarizer.binarize(base=base, strategy=strategy, heuristic=heuristic, tau=tau,
                                             go_backwards=False, return_parameters=True, axis=2)
            scores = self.binarizer.classify(tau=tau, features=self.dataset.features, predictions=y_pred)
            scores[scores == Class.NORMAL_ATTRIBUTE] = Class.NORMAL

        vmin = scores.min()
        vmax = scores.max()
        cmap = 'Blues'
        cbar = True
        if mode is not None:
            cmap = sns.color_palette(Class.colors())
            vmin = min(Class.keys())
            vmax = max(Class.keys())
            cbar = False

        # Event log data frame
        el_df = self.event_log_df
        if prettify_fn is not None:
            el_df = prettify_fn(el_df)

        # Labels
        if short_labels:
            labels = np.array([l.split(' ')[0].replace('Sequence', '') for l in self.dataset.pretty_labels])
        else:
            labels = np.array([f'Case {c.id}\n{l}' for c, l in zip(self.event_log.cases, self.dataset.pretty_labels)])

        # Class labels
        classes_df = None
        legend_handles = []
        legend_labels = []
        if self.dataset.classes is not None and len(self.dataset.classes) > 0:
            classes_df = pd.DataFrame(flatten(self.dataset.classes), index=el_df.index, columns=el_df.columns,
                                      dtype=str)

            from matplotlib.patches import Patch
            for key, value in Class.items().items():
                if 'Normal' in value:
                    classes_df = classes_df.replace(str(key), '')
                if np.any(scores == key) and mode is not None:
                    legend_handles.append(Patch(facecolor=Class.color(key), edgecolor=Class.color(key)))
                    legend_labels.append(value)
                classes_df = classes_df.replace(str(key), '\n' + value)

        # Scores
        scores_df = pd.DataFrame(flatten(scores), index=el_df.index, columns=el_df.columns)

        # Annotations
        annot_df = el_df
        if mode is None:
            # Scores for annotations
            scores_str_df = '\n' + scores_df.round(2).astype(str)
            scores_str_df = scores_str_df.replace('\nnan', '')
            annot_df += scores_str_df

        if classes_df is not None:
            annot_df += classes_df

        # Prepare heatmap data frames
        data = scores_df.set_index(labels).iloc[indices, min_event * n:max_event * n]
        annot = annot_df.iloc[indices, min_event * n:max_event * n]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(data=data, annot=annot, cmap=cmap, cbar=cbar, fmt='', linewidths=0.0, rasterized=True,
                         vmin=vmin, vmax=vmax)

        # Move x-axis to top and remove x-axis label
        ax.set_xlabel('')
        ax.xaxis.set_ticks_position('top')

        if mode is not None:
            ax.legend(legend_handles, legend_labels, bbox_to_anchor=(0, 1.1), loc=3, frameon=False,
                      ncol=len(legend_labels), borderaxespad=0.)

        # Save to disk or show
        if file_name is not None:
            fig.tight_layout()
            fig.savefig(str(PLOT_DIR / file_name))
        return fig, ax
