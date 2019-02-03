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

from april.anomalydetection.binet.core import BINet


class BINetv0(BINet):
    version = 0
    abbreviation = 'binetv0'
    name = 'BINetv0'
    supports_attributes = False
    config = dict(use_attributes=False, encode=True, decode=True)

    def __init__(self, model=None):
        super(BINetv0, self).__init__(model)


class BINetv1(BINet):
    version = 1
    abbreviation = 'binetv1'
    name = 'BINetv1'
    config = dict(use_attributes=True, encode=True, decode=True)

    def __init__(self, model=None):
        super(BINetv1, self).__init__(model)


class BINetv2(BINet):
    version = 2
    abbreviation = 'binetv2'
    name = 'BINetv2'
    config = dict(use_attributes=True, use_present_activity=True, encode=True, decode=True)

    def __init__(self, model=None):
        super(BINetv2, self).__init__(model)


class BINetv3(BINet):
    version = 3
    abbreviation = 'binetv3'
    name = 'BINetv3'
    config = dict(use_attributes=True, use_present_attributes=True, encode=True, decode=True)

    def __init__(self, model=None):
        super(BINetv3, self).__init__(model)

