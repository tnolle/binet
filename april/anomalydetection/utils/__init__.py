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


def label_collapse(a, axis=0):
    if a.ndim > 1 and axis < 2:
        a = a.any(-1)
        if a.ndim > 1 and axis < 1:
            a = a.any(-1)
    return a.astype(int)


def max_collapse(a, axis=0):
    if a.ndim > 1 and axis < 2:
        a = a.max(-1)
        if a.ndim > 1 and axis < 1:
            a = a.max(-1)
    return a


def anomaly_ratio(a):
    """r in the paper"""
    if a.max() == 0:
        return 0.
    elif a.min() == 1:
        return 1.
    else:
        return a.mean()
