import pickle
import numpy as np
import matplotlib.pylab as plt

model = 'best_models/GRU15_seed1probe_seed10_500.h5'
results_all = pickle.load(open('all_results_GRU15_probe.pickle', 'rb'))

result = results_all[model]

width = 0.018
width_all = 0.04
bin_start = (width_all-width)/2

xticks = ['9L', '9R', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ranges = width_all*np.arange(len(xticks))+width_all/2

ilmse = result['intermediate_locally_mean_squared_error'].values()
irmse = result['intermediate_recursively_mean_squared_error'].values()
sba = result['subtracting_binary_accuracy'].values()

fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.bar(ranges, ilmse, width=width)
ax.set_xticks(ranges)
ax.set_xticklabels(xticks)
plt.ylabel('mean squared error')

plt.plot()



