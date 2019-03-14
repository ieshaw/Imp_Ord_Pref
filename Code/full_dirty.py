import numpy as np
import pandas as pd
import sys

def load_pref_df(filename):
    '''
    input filename: string, filename of csv with ordinal preferences of columns for 
            rows; NaN indicates unexpressed preference
    output: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    '''
    df = pd.read_csv(filename, header=0, index_col=0)
    df.index = df.index.map(str)
    m,n = df.shape
    df.fillna(m + 1, inplace=True)
    df = (m + 1) - df
    ## Error Checking
    if df.values.max() > m or df.values.min() < 0:
        raise ValueError('''
                Preference CSV should have ordinal preferences
                of columns for rows. Somthing is wrong.
                ''')
    return df

def sim_weighted_euclidean(df):
    '''
    input: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    output: sim_df, Pandas DataFrame with row index deciders, column headers deciders
            the entries are the similarities. Entry at row i, column j is the 
            similarity of decider j of decider i
    '''
    sim_df = pd.DataFrame(data=-1,index=df.columns, columns=df.columns, dtype=float)
    m,n = df.shape
    sim_coeff = 1/(2*(m**4))
    for i in df:
        for j in df:
            if i == j:
                sim_df[i][i] = 1
            elif sim_df[i][j] == -1:
                s = 1 - sim_coeff * ((df[i] + df[j]) * ((df[i] - df[j])**2)).sum()
                sim_df[i][j] = s
                sim_df[j][i] = s
    ## Error Checking
    mx = sim_df.values.max()
    mn = sim_df.values.min()
    if mx > 1:
        raise ValueError('''
                Weighted Euclidean Similarity values should be less than 1.
                Current Max Similarity: {}
                '''.format(mx))
    if mn < 0:
        raise ValueError('''
                Weighted Euclidean Similarity values should be greater than 1.
                Current Min Similarity: {}
                '''.format(mn))
    return sim_df

def sim_cosine(df):
    '''
    input: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    output: sim_df, Pandas DataFrame with row index deciders, column headers deciders
            the entries are the similarities. Entry at row i, column j is the 
            similarity of decider j of decider i
    '''
    sim_df = pd.DataFrame(data=-1,index=df.columns, columns=df.columns, dtype=float)
    for i in df:
        for j in df:
            if sim_df[i][j] == -1:
                denom = df[i].sum() + df[j].sum()
                if denom == 0:
                    s = 0
                else:
                    s = ((df[i] * df[j]).sum())/denom
                sim_df[i][j] = s
                sim_df[j][i] = s
    mn = sim_df.values.min()
    if mn < 0:
        raise ValueError('''
                Cosine Similarity values should be greater than or equal to 0.
                Current Min Similarity: {}
                '''.format(mn))
    return sim_df

def sim_euclidean(df):
    '''
    input: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    output: sim_df, Pandas DataFrame with row index deciders, column headers deciders
            the entries are the similarities. Entry at row i, column j is the 
            similarity of decider j of decider i
    '''
    sim_df = pd.DataFrame(data=-1,index=df.columns, columns=df.columns, dtype=float)
    for i in df:
        for j in df:
            if i == j:
                sim_df[i][i] = 1
            elif sim_df[i][j] == -1:
                denom = (df[i]**2).sum() + (df[j]**2).sum()
                if denom == 0:
                    s = 0
                else:
                    s = 1 - (((df[i] - df[j])**2).sum())/denom
                sim_df[i][j] = s
                sim_df[j][i] = s
    sim_df.fillna(0, inplace=True)
    ## Error Checking
    mx = sim_df.values.max()
    mn = sim_df.values.min()
    if mx > 1:
        raise ValueError('''
                Euclidean Similarity values should be less than or equal to 1.
                Current Max Similarity: {}
                '''.format(mx))
    if mn < 0:
        raise ValueError('''
                Euclidean Similarity values should be greater than or equal to 0.
                Current Min Similarity: {}
                '''.format(mn))
    return sim_df

def score(pref_df,sim_df):
    '''
    input pref_df: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    input sim_df: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    output: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            score  of decider j of slate option i
    '''
    # Higher score, means higher preference
    score_matrix = np.matmul(pref_df.to_numpy(copy=True), sim_df.to_numpy(copy=True))
    #Filter 1 indicating no score (due to no similarity with a person who ranked it), 0 otherwise
    temp_score_df = pd.DataFrame(score_matrix, index=pref_df.index,columns=pref_df.columns)
    filter_score_matrix = (1 - temp_score_df.mask(temp_score_df !=0, 1)).to_numpy(copy=True)
    print("Missing Score Count: {}".format(filter_score_matrix.sum()))
    if filter_score_matrix.sum() > 0:
        random_matrix = np.random.uniform(0,1,size=score_matrix.shape)
        #If no score, randomly give a score between 0 and 1
        # add 2 to existing score matrix to ensure all scores above 
        # previously unscored
        score_matrix += 2*(1 - filter_score_matrix) + random_matrix
    #Filter 1 indicating unexpressed preference, 0 means preference was expressed
    filter_matrix = (1 - pref_df.mask(pref_df !=0, 1)).to_numpy(copy=True)
    #Make sure already known preferences have lowest score 0 so they aer below the 
    # preferences we are trying to infer
    score_matrix *= filter_matrix
    return pd.DataFrame(score_matrix, index=pref_df.index,
                    columns=pref_df.columns)

def implied_prefs(pref_df, score_df):
    '''
    input pref_df: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    input score_df: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            score  of decider j of slate option i
    output: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            ordinal preference (true, implied, or random)  of decider j of slate option i
    '''
    #Filter 1 indicating unexpressed preference, 0 means preference was expressed
    filter_df = (1 - pref_df.mask(pref_df !=0, 1))
    rank_df = score_df.rank(axis=0,ascending=True)
    iop_df = pref_df + filter_df * (rank_df - (1 - filter_df).sum(axis=0))
    m,n = iop_df.shape
    return (m+1 -iop_df)

def main():
    if len(sys.argv) != 2:
        raise ValueError('''
                Provide the filename of preference csv.
                Example:
                        python scoring.py path/to/pref.csv 
                ''')

    pref_filename = sys.argv[1] 

    print_to_screen = True
    save_to_file = True

    temp = pref_filename.split('.')
    pref_filename = sys.argv[1] 
    output_filename = temp[0] + '_complete.' + temp[1]
    pref_df = load_pref_df(pref_filename) 
    sim_df = sim_weighted_euclidean(pref_df)
    #sim_df = sim_euclidean(pref_df)
    #sim_df = sim_cosine(pref_df)
    score_df = score(pref_df, sim_df)   
    iop_df = implied_prefs(pref_df, score_df)
    
    if print_to_screen:
        print('-----------------------------------')
        print('Complete Preferences')
        print('-----------------------------------')
        print(iop_df.head())
        print('-----------------------------------')

    if save_to_file:
        iop_df.to_csv(output_filename, header=True, index=True)

if __name__ == '__main__':
    main()
