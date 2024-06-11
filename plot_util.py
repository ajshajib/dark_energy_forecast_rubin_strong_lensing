import seaborn as sns
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import numpy as np


emerald = sns.xkcd_rgb["emerald"]
orange = sns.xkcd_rgb["bright orange"]
purple = sns.xkcd_rgb["light purple"]

# from cb2
cb2_emerald = "#66c2a5"
cb2_orange = "#fc8d62"
cb2_blue = "#8da0cb"

# from cb2 bright
cb_red = "#e41a1c"
cb_blue = "#377eb8"
cb_green = "#4daf4a"
cb_purple = "#984ea3"
cb_orange = "#ff7f00"
cb_grey = "#404040"

page_width = 384.1122  # in pt
page_width_in = page_width / 72.27


def set_fontscale(font_scale=1.0):
    """
    Set font scale for matplotlib plots.

    Parameters
    ----------
    fontscale : float
        Font scale to use for matplotlib plots.
    """
    sns.set_theme(
        style="ticks",
        context=None,
        font="Segoe UI Symbol",
        rc={  # "text.usetex": True,
            # "font.family": 'serif',
            # "font.serif": 'Times New Roman',
            # "mathtext.rm": 'serif',
            # "mathtext.it": 'serif:italic',
            # "mathtext.bf": 'serif:bold',
            # "mathtext.fontset": 'custom',
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 1,
            "axes.labelsize": 12 * font_scale,
            "font.size": 12 * font_scale,
            "axes.titlesize": 12 * font_scale,
            "legend.fontsize": 12 * font_scale,
            "xtick.labelsize": 11 * font_scale,
            "ytick.labelsize": 11 * font_scale,
        },
    )
