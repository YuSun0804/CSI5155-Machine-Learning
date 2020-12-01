from scipy import stats
import numpy as np
a = np.array([-1, -1, 0, 0, 1, 1, 1])
b = np.arange(7)
print(stats.pointbiserialr(a, b))
