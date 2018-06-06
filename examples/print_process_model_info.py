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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
import pandas as pd

from binet.processmining import Flowchart


def main():
    process_models = ['p2p', 'small', 'medium', 'large', 'huge', 'wide']

    frames = []
    for process_model in process_models:
        model = Flowchart.from_plg(process_model)

        g = model.graph

        variants = model.generate_valid_cases()
        num_var = len(variants.traces)
        max_len = max(len(v) for v in variants)

        nodes = g.number_of_nodes()
        edges = g.number_of_edges()
        dens = nx.density(g)
        in_degree = np.mean(list(g.in_degree().values()))
        out_degree = np.mean(list(g.out_degree().values()))

        frames.append([nodes, edges, num_var, max_len, dens, in_degree, out_degree])

    df = pd.DataFrame(frames, index=process_models,
                      columns=['nodes', 'edges', 'num_var', 'max_len', 'density', 'in_deg', 'out_deg'])

    print(df)


if __name__ == '__main__':
    main()
