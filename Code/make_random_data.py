import os
import numpy as np
import pandas as pd

output_dir = 'Data/random/'
filename = 'S.csv'
num_seekers = 40
num_jobs = 40
if not os.path.exists(output_dir):
    os.make_dir(output_dir):

