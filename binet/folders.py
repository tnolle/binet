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

from pathlib import Path

# root dirs
ROOT_DIR = Path(__file__).parent.parent
OUT_DIR = ROOT_DIR / '.out'
RES_DIR = ROOT_DIR / '.res'

# dataset dirs
EVENTLOG_DIR = RES_DIR / 'eventlogs'
EVENTLOG_CACHE_DIR = RES_DIR / '.cached_eventlogs'
PROCESS_MODEL_DIR = RES_DIR / 'process_models'
BPIC_DIR = RES_DIR / 'bpic'

# output dirs
MODEL_DIR = OUT_DIR / 'models'

# config dirs
CONFIG_DIR = ROOT_DIR / '.config'


def generate():
    """Generate directories."""
    dirs = [
        ROOT_DIR,
        OUT_DIR,
        RES_DIR,
        EVENTLOG_CACHE_DIR,
        MODEL_DIR,
        EVENTLOG_DIR
    ]
    for d in dirs:
        if not d.exists():
            d.mkdir()


if __name__ == '__main__':
    generate()
