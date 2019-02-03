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
import seaborn as sns

from april.enums import Axis
from april.enums import Heuristic
from april.enums import Strategy

microsoft_colors = sns.color_palette(['#01b8aa', '#374649', '#fd625e', '#f2c80f', '#5f6b6d',
                                      '#8ad4eb', '#fe9666', '#a66999', '#3599b8', '#dfbfbf'])

datasets = dict(paper='Paper', p2p='P2P', huge='Huge', small='Small', medium='Medium', large='Large', wide='Wide',
                gigantic='Gigantic', testing='Testing',
                bpic12='BPIC12', bpic13='BPIC13', bpic15='BPIC15', bpic17='BPIC17')


def prettify_dataframe(base_df):
    df = base_df.copy()

    lookup = {
        'process_model': datasets,
        'axis': Axis.items(),
        'heuristic': Heuristic.items(),
        'strategy': Strategy.items(),
    }

    for key, value in lookup.items():
        if key in df:
            df[key] = df[key].replace(value)

    return df


def get_cd(k, n, alpha):
    q = {
        0.01: [
            2.575829491, 2.913494192, 3.113250443, 3.254685942, 3.363740192, 3.452212685, 3.526470918, 3.590338924,
            3.646291577, 3.696020982, 3.740733465, 3.781318566, 3.818450865, 3.852654327, 3.884343317, 3.913850176,
            3.941446432, 3.967356946, 3.991769808, 4.014841995, 4.036709272, 4.057487605, 4.077275281, 4.096160689,
            4.114219489, 4.131518856, 4.148118188, 4.164069103, 4.179419684, 4.194212358, 4.208483894, 4.222268941,
            4.235598611, 4.248501188, 4.261002129, 4.273124768, 4.284891024, 4.296319991, 4.307430053, 4.31823818,
            4.328759929, 4.339009442, 4.348999447, 4.358743378, 4.368251843, 4.377536155, 4.386605506, 4.395470504,
            4.404138926, 4.412619258, 4.42091857, 4.429046055, 4.437006664, 4.444807466, 4.452454825, 4.4599544,
            4.467311139, 4.474529992, 4.481617323, 4.488575961, 4.495411562, 4.50212837, 4.508729212, 4.51521833,
            4.521599969, 4.527876956, 4.53405212, 4.540129702, 4.546111826, 4.552002025, 4.557802422, 4.563515138,
            4.569143708, 4.574690253, 4.580156896, 4.585545757, 4.590859664, 4.596099325, 4.601267569, 4.606365809,
            4.611396874, 4.616481678, 4.621261013, 4.626098331, 4.63087413, 4.635590532, 4.64024683, 4.644847267,
            4.649391842, 4.65388197, 4.658319065, 4.662703834, 4.667037692, 4.671322759, 4.675558329, 4.679746522,
            4.683888754, 4.687985023, 4.692036745
        ],
        0.05: [
            1.959964233, 2.343700476, 2.569032073, 2.727774717, 2.849705382, 2.948319908, 3.030878867, 3.10173026,
            3.16368342, 3.218653901, 3.268003591, 3.312738701, 3.353617959, 3.391230382, 3.426041249, 3.458424619,
            3.488684546, 3.517072762, 3.543799277, 3.569040161, 3.592946027, 3.615646276, 3.637252631, 3.657860551,
            3.677556303, 3.696413427, 3.71449839, 3.731869175, 3.748578108, 3.764671858, 3.780192852, 3.795178566,
            3.809663649, 3.823679212, 3.837254248, 3.850413505, 3.863181025, 3.875578729, 3.887627121, 3.899344587,
            3.910747391, 3.921852503, 3.932673359, 3.943224099, 3.953518159, 3.963566147, 3.973379375, 3.98296845,
            3.992343271, 4.001512325, 4.010484803, 4.019267776, 4.02786973, 4.036297029, 4.044556036, 4.05265453,
            4.060596753, 4.068389777, 4.076037844, 4.083547318, 4.090921028, 4.098166044, 4.105284488, 4.112282016,
            4.119161458, 4.125927056, 4.132582345, 4.139131568, 4.145576139, 4.151921008, 4.158168297, 4.164320833,
            4.170380738, 4.176352255, 4.182236797, 4.188036487, 4.19375486, 4.199392622, 4.204952603, 4.21043763,
            4.215848411, 4.221187067, 4.22645572, 4.23165649, 4.236790793, 4.241859334, 4.246864943, 4.251809034,
            4.256692313, 4.261516196, 4.266282802, 4.270992841, 4.275648432, 4.280249575, 4.284798393, 4.289294885,
            4.29374188, 4.298139377, 4.302488791
        ],
        0.1: [
            1.64485341, 2.05229258, 2.291341341, 2.459516082, 2.588520643, 2.692731919, 2.779883537, 2.854606339,
            2.919888558, 2.977768077, 3.029694463, 3.076733328, 3.1196936, 3.159198949, 3.195743642, 3.229723658,
            3.261461439, 3.29122427, 3.31923277, 3.345675735, 3.370711558, 3.39447671, 3.417089277, 3.438651085,
            3.459252641, 3.478971727, 3.497877641, 3.516032608, 3.533492489, 3.550305367, 3.566516497, 3.58216477,
            3.597287662, 3.611916995, 3.626083879, 3.639814478, 3.653134249, 3.666065818, 3.678630398, 3.690847789,
            3.702736375, 3.714311713, 3.725589359, 3.736584163, 3.747309558, 3.757777567, 3.767999503, 3.777987386,
            3.787749702, 3.797297058, 3.806637939, 3.815781537, 3.824734923, 3.833505168, 3.842100758, 3.850526642,
            3.858790599, 3.866897579, 3.874853237, 3.882663231, 3.890333219, 3.897866734, 3.905268728, 3.912543443,
            3.919695828, 3.926729419, 3.933647045, 3.940452947, 3.94715137, 3.953744433, 3.960235674, 3.966628626,
            3.972924705, 3.979128153, 3.985240384, 3.991264934, 3.997203923, 4.003058768, 4.008833001, 4.01452804,
            4.02014671, 4.02568972, 4.031159898, 4.036558658, 4.041887415, 4.047148997, 4.052344817, 4.057475584,
            4.062543418, 4.067549734, 4.072495239, 4.077382761, 4.082213008, 4.086986686, 4.091705209, 4.0963707,
            4.100983157, 4.105544703, 4.110055337
        ]
    }
    return q[alpha][k - 2] * np.sqrt(k * (k + 1) / (6 * n))


def cd_plot(ranks, cd=None, lowv=None, highv=None, width=6, textspace=1, reverse=False):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        ranks (list of float): average ranks of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
    """
    try:
        import matplotlib
        import math
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    avranks = ranks.values
    sums = ranks.values
    names = ranks.index.values

    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    if lowv is None:
        lowv = min(1, int(np.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(np.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    if cd is not None:
        def get_lines(sums):
            lsums = len(sums)
            allpairs = [(i, j) for i, j in mxrange([[lsums], [lsums]]) if j > i]
            notSig = [(i, j) for i, j in allpairs if abs(sums[i] - sums[j]) <= cd]

            def no_longer(ij_tuple, notSig):
                i, j = ij_tuple
                for i1, j1 in notSig:
                    if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                        return False
                return True

            longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]

            return longest

        lines = get_lines(ssums)
        linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

        # add scale
        distanceh = 0.25
        cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    k = len(ssums)

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i], ha="left", va="center")

    if cd is not None:
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)
        line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        line([(begin, distanceh + bigtick / 2), (begin, distanceh - bigtick / 2)], linewidth=0.7)
        line([(end, distanceh + bigtick / 2), (end, distanceh - bigtick / 2)], linewidth=0.7)
        text((begin + end) / 2, distanceh - 0.05, "CD", ha="center", va="bottom")

    start = cline + 0.2
    for l, r in lines:
        line([(rankpos(ssums[l]) - 0.05, start),
              (rankpos(ssums[r]) + 0.05, start)],
             linewidth=2.5)
        start += 0.1

    return fig
