import pandas as pd
import termcolor


def match_id(sample, zoo):
    sample_type1 = sample
    sample_type2 = sample
    output = sample.copy()
    output['objid1']= pd.np.nan
    output['objid2']= pd.np.nan
    limit = 0.0001
    # cnt = 0
    for i in sample_type1.index:
        gal = sample_type1.ix[i]
        ra, dec = gal.RA1, gal.DEC1
        match = zoo[(abs((zoo.ra - ra)) < limit) & (abs((zoo.dec - dec)) < limit)]
        # print(match)
        if not match.empty:
            output.at[i, 'objid1'] = match.dr7objid
            # cnt += 1
    # ratio_type1 = '{:.3f}'.format(cnt / len(sample_type1) * 100)

    # cnt = 0
    for i in sample_type2.index:
        gal = sample_type2.ix[i]
        ra, dec = gal.RA2, gal.DEC2
        match = zoo[(abs((zoo.ra - ra)) < limit) & (abs((zoo.dec - dec)) < limit)]
        if not match.empty:
            output.at[i, 'objid2'] = match.dr7objid
            # cnt += 1
    # ratio_type2 = '{:.3f}'.format(cnt / len(sample_type2) * 100)
    # print(ratio_type1, ratio_type2, float(ratio_type1) / float(ratio_type2))
    output.to_csv('type12.csv', columns=['NAME1', 'objid1', 'RA1', 'DEC1', 'LOGL1_5007', 'Z1'
                                         'NAME2', 'objid2', 'RA2', 'DEC2', 'LOGL2_5007', 'Z2'],index=None)


my_sample_init = pd.read_csv('type12.csv',header=0)
my_sample = my_sample_init[my_sample_init.Z1 < 0.05]
zoo = pd.read_csv('zoo2MainSpecz.csv',index_col='dr7objid')

barriverse = pd.read_csv('zoo2bar.csv')
match_id(my_sample,barriverse)