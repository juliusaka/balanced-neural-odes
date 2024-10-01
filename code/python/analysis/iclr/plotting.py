import matplotlib
import pathlib
def settings():
    matplotlib.rcParams.update({
                            'font.family': 'serif',
                            'font.sans-serif': 'Times New Roman',
                            'font.size': 7,
                            'text.usetex': True,
                            'pgf.rcfonts': False,
                            #'pgf.texsystem': 'lualatex',
                            'svg.fonttype': 'none',
                            'lines.linewidth': 2.5,
                            'lines.markersize': 10,
                            'lines.markeredgewidth': 2.5,

                            'text.latex.preamble': r'\usepackage{amsmath,amsfonts,bm,times}',
                            })
# see https://blog.timodenk.com/exporting-matplotlib-plots-to-latex/
# and https://jwalton.info/Matplotlib-latex-PGF/


# \usepackage{layouts}

# [...]

# \printinunitsof{in}\prntlen{\textwidth}

def fig_size(height_inches, width_inches=None, n_figs_per_width=1):
    max_width = 5.50107
    if width_inches is None:
        width_inches = max_width
    if width_inches > max_width:
        width_inches = max_width
    width_inches = width_inches/n_figs_per_width
    return (width_inches, height_inches)

# line cyclers adapted to colourblind people, see https://ranocha.de/blog/colors/#gsc.tab=0
from cycler import cycler
cb_line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
cb_line_cycler_solid = cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"])

cb_marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

cmap = matplotlib.colormaps.get_cmap('plasma')

# custom save routine
def savefig(fig, _path, pgf = True, **kwargs):
    _path = pathlib.Path(_path)
    for file_type in ['png', 'svg']:
        if file_type == 'png':
            kwargs['dpi'] = 300
        if file_type == 'pgf' and not pgf:
            continue
        else:
            fig.savefig(_path.with_suffix(f".{file_type}"), **kwargs, bbox_inches='tight', pad_inches=0)
            # resolve file path and print it
            print(_path.with_suffix(f".{file_type}").resolve())
