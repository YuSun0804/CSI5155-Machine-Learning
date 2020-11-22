import pandas as pd
import numpy as np
num_series = pd.Series(np.random.randint(1, 10, 9))
print("Original Series:")
print(num_series)
result = np.argwhere(num_series)
print("Positions of numbers that are multiples of 5:")
print(result)
