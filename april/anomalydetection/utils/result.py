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


class AnomalyDetectionResult(object):
    def __init__(self,
                 scores,
                 predictions=None,
                 attentions=None,
                 scores_backward=None,
                 predictions_backward=None,
                 attentions_backward=None):
        self.scores_forward = scores
        self.scores_backward = scores_backward

        self.predictions = predictions
        self.predictions_backward = predictions_backward

        self.attentions = attentions
        self.attentions_backward = attentions_backward

    @property
    def scores(self):
        return self.scores_forward

    @staticmethod
    def minmax_normalize(scores):
        return (scores - scores.min()) / (scores.max() - scores.min())
