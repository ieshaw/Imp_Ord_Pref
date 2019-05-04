import os
import numpy as np
import pandas as pd

output_dir = 'Data/random/'
filename = 'S.csv'
num_seekers = 40
num_jobs = 40
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
df = pd.DataFrame(np.random.randn(num_jobs, num_seekers), 
        index = ['Job_{}'.format(i) for i in range(num_jobs)],
        columns = ['Seeker_{}'.format(i) for i in range(num_seekers)])
df = df.rank(axis=0,method='first')
df = df.astype(int)
df.to_csv(output_dir + filename, header=True, index=True)
