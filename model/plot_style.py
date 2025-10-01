import matplotlib as mpl

mpl.style.use('seaborn-v0_8-paper')

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'mathtext.tt': 'Arial',
    'mathtext.cal': 'Arial',
    'mathtext.fallback': None,
    'savefig.dpi': 600,
    'figure.dpi': 600,
    'font.size': 12
})

MIN_WIDTH = 1578    # in pixels at 600 dpi
MAX_WIDTH = 4500
MAX_HEIGHT = 5250

inh_blue = "#7879ff"
excit_red = "#F58E89"