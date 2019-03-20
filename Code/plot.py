import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from iop import iop

def grab_input():
    #pref_filename = input('Pref file name: ') 
    pref_filename = 'Data/army/army_s2.csv' 

    temp = pref_filename.split('.')
    output_filename = temp[0] + '_complete.' + temp[1]
    df = pd.read_csv(pref_filename, header=0, index_col=0)
    df.index = df.index.map(str)
    path_list = pref_filename.split('/')
    output_dir = ''
    for i in path_list[:-1]:
        output_dir += i + '/'
    experiment_dir = output_dir + 'experiment_output/'
    output_dir += 'results/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return df,output_dir,experiment_dir

def agg_exps(experiment_dir):
    num_exp = 0
    for filename in os.listdir(experiment_dir):
        filename = experiment_dir + filename
        if num_exp == 0:
            exp_df = pd.read_csv(filename)
        else:
            exp_df = exp_df.append(pd.read_csv(filename),ignore_index=True)
        num_exp += 1
    return exp_df, num_exp

def expand_df(exp_iop,df):
    df['perc_dropout'] = 1 - df['Pref_Coverage']/exp_iop.pref_coverage
    return df

def plot_df(num_exp,exp_df):
    exp_plt = exp_df.plot(x='perc_dropout', 
                        y=['cosine','euclidean','weighted_euclidean','random'])
    return expl_plt
    #TODO: Caption original coverage, and num experiments

def main():
    df,output_dir,experiment_dir = grab_input()
    exp_df,num_exp = agg_exps(experiment_dir)
    exp_iop = iop(df)
    exp_df = expand_df(exp_iop,exp_df)
    print('---------------------------')
    print('Num_exp: {}'.format(num_exp))
    print('---------------------------')
    print(exp_df)

if __name__ == '__main__':
    main()
