import numpy as np
import pandas as pd
import sys

N = 10000
p0 = 100
p1 = 10

np.random.seed(98756761)

columns = dict(("x{}".format(i), np.random.randn(N)) for i in range(p0))

for i in range(p1):
    x = int(i * p1 / p0)
    columns["x{}".format(x)] += i + 1

columns["y"] = 0.5 * p1 * (p1 - 1) + np.random.randn(N)

pd.DataFrame(columns).to_csv(sys.stdout, index=False)

