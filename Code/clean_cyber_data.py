import pandas as pd

filename = 'Data/cyber/commands/O.csv'
#filename = 'Data/cyber/sailors/S.csv'
df = pd.read_csv(filename,index_col=0)
df = df.rank(axis=0,method='first')
temp = filename.split('.')
output_filename = temp[0] + '_clean.' + temp[1]
df.to_csv(output_filename,header=True,index=True)
