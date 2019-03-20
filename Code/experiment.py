import numpy as np
import os
import pandas as pd
import sys
import time

from iop import iop

def grab_input():
    pref_filename = input('Pref file name: ') 
    num_experiments = int(input('Number of Experiments: '))
    dropout_level_count = int(input('Dropout Level Count: '))

    temp = pref_filename.split('.')
    output_filename = temp[0] + '_complete.' + temp[1]
    df = pd.read_csv(pref_filename, header=0, index_col=0)
    df.index = df.index.map(str)
    path_list = pref_filename.split('/')
    output_dir = ''
    for i in path_list[:-1]:
        output_dir += i + '/'
    output_dir += 'experiment_output/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return df,output_dir,dropout_level_count,num_experiments

def dropout_array(true_iop,dropout_level_count):
    m,n = true_iop.pref_df.shape
    mn = float(1)/float(n)
    gap = true_iop.pref_coverage - mn 
    dropout_diff = gap/float(dropout_level_count)
    return list(mn + dropout_diff *np.arange(dropout_level_count))

def rmse_by_measure(true_iop,dropout_ratio,measures):
    out_dict = {}
    drop_iop = iop(true_iop.dropout(perc=dropout_ratio))
    for measure in measures:
        out_dict[measure] = true_iop.rmse(drop_iop.complete_prefs(measure=measure))
    return out_dict

def experiment_iteration(true_iop,dropout_levels):
    measures = ['cosine','euclidean','weighted_euclidean','random']
    out_dict = {}
    for dropout_ratio in dropout_levels:
        out_dict[dropout_ratio] = rmse_by_measure(true_iop,dropout_ratio,measures)
    df = pd.DataFrame.from_dict(out_dict,orient='index',dtype=float,
                                columns=measures)
    df.index.name = 'Pref_Coverage'
    return df 

def experiment_df_to_file(df,output_dir):
    df.to_csv(output_dir + 'exp_{}.csv'.format(round(1000 *time.time())),
            header=True,index=True)

def run_experiment():
    df,output_dir,dropout_level_count,num_experiments = grab_input()
    true_iop = iop(df)
    dropout_levels = dropout_array(true_iop,dropout_level_count) 
    for i in range(num_experiments):
        tic = time.time()
        print('Running: Experiment iteration {} of {}'.format(i+1,num_experiments))
        experiment_df_to_file(experiment_iteration(true_iop,dropout_levels),
                                output_dir)
        print('Iteration {} took {:.2f} seconds.'.format(i+1,time.time()-tic)) 

def main():
    run_experiment()

if __name__ == '__main__':
    main()
