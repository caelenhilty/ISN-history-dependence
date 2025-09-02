import numpy as np

data_dir = 'figures/figure5/psycho_data'
data = np.load(data_dir + '/p_curves.npy', allow_pickle=True)
reliabilities = np.load(data_dir + '/reliabilities.npy', allow_pickle=True)
good_p_curves = data[reliabilities > 0.73]

# fit a linear and logistic function to each p_curve, record RSS
cue_count = np.arange(len(good_p_curves.shape[1]))
linear_rss = []
logistic_rss = []
for p_curve in good_p_curves:
    # Fit linear model
    A = np.vstack([cue_count, np.ones(len(cue_count))]).T
    m, c = np.linalg.lstsq(A, p_curve, rcond=None)[0]
    linear_fit = m * cue_count + c
    linear_rss.append(np.sum((p_curve - linear_fit) ** 2))