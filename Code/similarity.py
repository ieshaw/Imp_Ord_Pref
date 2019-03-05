import pandas as pd

def sim_weighted_euclidean(df):
    '''
    input: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            preference ranking of decider j of slate option i
    output: sim_df, Pandas DataFrame with row index deciders, column headers deciders
            the entries are the similarities. Entry at row i, column j is the 
            similarity of decider j of decider i
    '''
    sim_df = pd.DataFrame(data=-1,index=df.columns, columns=df.columns, dtype=float)
    n_d = len(df.columns)
    nn_d = 2 * n_d
    sim_coeff = (1/((n_d - 1) **2))*(1/(nn_d))
    for i in df:
        for j in df:
            if i == j:
                sim_df[i][i] = 1
            elif sim_df[i][j] == -1:
                s = 1 - sim_coeff * ((nn_d - (df[i] + df[j])) * ((df[i] - df[j])**2)).sum()
                sim_df[i][j] = s
                sim_df[j][i] = s
    return sim_df

def main():
    print_to_screen = True
    save_to_file = False

    filename = 'Data/med/clean/S.csv'
    temp = filename.split('.')
    output_filename = temp[0] + '_similarity' + temp[1]
    pref_df = pd.read_csv(filename, header=0, index_col=0)
    pref_df.index = pref_df.index.map(str)
    sim_df = sim_weighted_euclidean(pref_df)

    if print_to_screen:
        print('-----------------------------------')
        print('Seeker Similarity')
        print('-----------------------------------')
        print(sim_df.head())
        print('-----------------------------------')

    if save_to_file:
        sim_df.to_csv('Data/job_similarity.csv', header=True, index=True)

if __name__ == '__main__':
    main()
