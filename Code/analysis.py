import os
import pandas as pd
import sys

def results_to_analysis(pref_filename, analysis_dir, results_dir):
    pref_df = pd.read_csv(pref_filename, header=0, index_col=0)
    m,n = pref_df.shape
    results_df = pd.read_csv(results_dir + "results.csv", header=0, index_col=0)
    analysis_df = results_df.copy()
    analysis_df.drop("weighted_euclidean", axis=1, inplace=True)
    analysis_df.reset_index(inplace=True)

def data_dirs(filename):
    temp = filename.split('.')
    path_list = filename.split('/')
    output_dir = ''
    for i in path_list[:-1]:
        output_dir += i + '/'
    analysis_dir = output_dir + 'analysis/'
    results_dir = output_dir + 'results/'
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    return analysis_dir, results_dir

def gen_results_dir(pref_filename):
    path_list = pref_filename.split('/')
    output_dir = ''
    for i in path_list[:-1]:
        output_dir += i + '/'
    experiment_dir = output_dir + 'experiment_output/'
    output_dir += 'results/'
    return output_dir

def pref_coverage_plot(results_dir):
    results_df = pd.read_csv(results_dir + "results.csv", header=0, index_col=0)
    results_df.drop("weighted_euclidean", axis=1, inplace=True)
    results_df.reset_index(inplace=True)
    exp_plt = results_df.plot(x='Pref_Coverage', 
                        y=['cosine','euclidean','random'])
    exp_plt.set_xlabel('Ratio of Expressed Preferences')
    exp_plt.set_ylabel('RMSE')
    fig = exp_plt.get_figure()
    fig.savefig(results_dir + 'pref_coverage.png')


def main():
    pref_csvs = ["army/army_s2.csv" , "cyber/commands/O_clean.csv",
                "cyber/sailors/S_clean.csv", "eod/sailors/eod_s.csv",
                "new_med/Hospitals/O.csv", "new_med/Residents/S.csv",
                "random/S.csv"]
    for pref_csv in pref_csvs:
        pref_filename = "Data/" + pref_csv
        results_dir = gen_results_dir(pref_filename)
        pref_coverage_plot(results_dir)

if __name__ == '__main__':
    main()
