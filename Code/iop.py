import numpy as np
import pandas as pd

class iop:

    def __init__(self, pref_df):
        '''
        input pref_df: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                complement ordinal preference ranking of decider j of slate option i
                1 is best, m is worst, NaN or 0 for unexpressed preference
        '''
        input_df = pref_df.copy(deep=True)
        m,n = input_df.shape
        input_df.fillna(0,inplace=True)
        mn = input_df.values.min()
        mx = input_df.values.max()
        if mn < 0 or mx > m:
            raise ValueError('''
                    Ordinal Prefernces should be between 1 and {}
                    0 allowed for min if there are unexpressed prefernces.
                    Current Min Preferece: {}
                    Current Max Preference: {}
                    '''.format(m,mn,mx))
        #Filter 0 indicating unexpressed preference, 1 means preference was expressed
        input_df.sort_index(axis=0).sort_index(axis=1, inplace=True)
        iop.check_prefs(input_df)
        self.filter_df = input_df.mask(input_df !=0, 1)
        self.pref_coverage = (float(self.filter_df.sum().sum())/float(m*n))
        self.pref_df = input_df.copy(deep=True)
        self.pref_df_c = (m - self.pref_df)*self.filter_df 

    def check_prefs(input_df):
        '''
        input pref_df: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                complement ordinal preference ranking of decider j of slate option i
                1 is best, m is worst, 0 for unexpressed preference
        '''
        non_expressed = []
        for col in input_df.columns:
            sorted_prefs = input_df[col].sort_values(ascending=True).to_numpy()
            if sorted_prefs.max() == 0:
                non_expressed.append(col)
                continue
            sorted_prefs = sorted_prefs[np.nonzero(sorted_prefs)]
            #If only submitted one ranking, move on
            if len(sorted_prefs) == 1 and sorted_prefs[0] == 1:
                continue
            pref_diff = set(sorted_prefs[1:] - sorted_prefs[:-1])
            if set([1]) != pref_diff:
                raise ValueError('''
                        {} did not express contiguous preferences.
                        At least one preference is skipped or repeated.
                        Their sorted expressed preferences are:
                        {}'''.format(col, sorted_prefs))
        if len(non_expressed) > 0:
            raise ValueError('{} did not express any preferences.'.format(non_expressed))

    def dropout(self,perc=0.5):
        '''
        input perc: float between 0 and 1, percent of expressed preferences
        output: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                complement ordinal preference ranking of decider j of slate option i
                1 is best, m is worst, 0 for unexpressed preference, with
                self.pref_coverage<= perc 
        '''
        m,n = self.pref_df.shape
        drop_df = self.pref_df.copy(deep=True)
        min_perc = float(1)/float(n)
        if perc < min_perc or perc > self.pref_coverage:
            raise ValueError('''
                    Requested Percent of expressed preferences: {}
                    Must be between the existing pref_coverage and
                        at least 1 prefernece per columns.
                        Between {} and {}.'''.format(
                            perc,min_perc,self.pref_coverage))
        drop_count = np.ceil(m*n*(self.pref_coverage - perc))
        counter = 0
        columns = list(drop_df.columns)
        r = len(columns)
        while counter < drop_count:
            if r > 1:
                col = columns[np.random.randint(0,r-1)]
            else:
                col = columns[0]
            mx = drop_df[col].max()
            if mx > 1: 
                drop_df.at[drop_df[col].idxmax(),col] = 0
                counter += 1
            else:
                columns.remove(col)
                r = len(columns)
            if r ==0:
                break
        return drop_df

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
                    denom = np.sqrt((df[i]**2).sum()) * np.sqrt((df[j]**2).sum())
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
                    denom = float(np.sqrt((df[i]**2).sum())*np.sqrt((df[j]**2).sum()))
                    if denom == 0:
                        s = 0
                    else:
                        s = 1 -np.sqrt(((df[i] - df[j])**2).sum())/denom
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

    def similarity(self,measure='cosine'):
        if measure == 'cosine':
            return iop.sim_cosine(self.pref_df_c)
        elif measure == 'euclidean':
            return iop.sim_euclidean(self.pref_df_c)
        elif measure == 'weighted_euclidean':
            return iop.sim_weighted_euclidean(self.pref_df_c)
        elif measure == 'random':
            return pd.DataFrame(0,index=self.pref_df_c.columns,columns=self.pref_df_c.columns,
                                dtype=float)
        else:
            raise ValueError(''''
                Arg 'measure' must be one of the following:
                    cosine , euclidean , weighted_euclidean, random.
                    ''')
            
    def score(self,sim_df):
        '''
        input sim_df: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                complement ordinal preference ranking of decider j of slate option i
        output: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                score  of decider j of slate option i
        '''
        # Higher score, means higher preference
        score_matrix = np.matmul(self.pref_df_c.to_numpy(copy=True), sim_df.to_numpy(copy=True))
        #Filter 1 indicating no score (due to no similarity with a person who ranked it), 0 otherwise
        temp_score_df = pd.DataFrame(score_matrix)
        filter_score_matrix = (1 - temp_score_df.mask(temp_score_df !=0, 1)).to_numpy(copy=True)
        if filter_score_matrix.sum() > 0:
            random_matrix = np.random.uniform(0,1,size=score_matrix.shape)
            #If no score, randomly give a score between 0 and 1
            # add 2 to existing score matrix to ensure all scores above 
            # previously unscored
            score_matrix += 2*(1 - filter_score_matrix) + filter_score_matrix * random_matrix
        #Make sure already known preferences have lowest score 0 so they are below the 
        # preferences we are trying to infer
        score_matrix *= (1 - self.filter_df.to_numpy(copy=True))
        return pd.DataFrame(score_matrix, index=self.pref_df_c.index,
                        columns=self.pref_df_c.columns)

    def implied_prefs(self,score_df):
        '''
        input score_df: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                score  of decider j of slate option i
        output: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                ordinal preference (true, implied, or random)  of decider j of slate option i
        '''
        rank_df = score_df.rank(axis=0,method='first',ascending=True)
        iop_df = self.pref_df_c + (1 - self.filter_df) * (rank_df - 1 - self.filter_df.sum(axis=0))
        m,n = iop_df.shape
        return (m -iop_df)

    def complete_prefs(self,measure='cosine'):
        return self.implied_prefs(self.score(self.similarity(measure=measure)))

    def rmse(self,other_df):
        '''
        input pref_df: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                complement ordinal preference ranking of decider j of slate option i
                1 is best, m is worst 
        output: float
        '''
        other_df.sort_index(axis=0).sort_index(axis=1, inplace=True)
        iop.check_prefs(other_df)
        squared_error =(((self.pref_df - other_df)*self.filter_df)**2).sum().sum()
        total_prefs = self.filter_df.sum().sum()
        return np.sqrt(float(squared_error)/float(total_prefs))

    def __del__(self): 
        pass
