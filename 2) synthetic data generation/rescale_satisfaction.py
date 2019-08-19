import pandas as pd
from sklearn.preprocessing import RobustScaler

file_name = 'data/satisfaction/synthpop_sat_5000.pkl'
data = pd.read_pickle(file_name)


# rescale continous data to be between -1 and 1
continous_cols = [x for x in data.columns if x not in ['ind_var1_0', 'ind_var1', 'ind_var5_0', 'ind_var5', 'ind_var6_0', 'ind_var6', 'ind_var8_0', 'ind_var8',
        'ind_var12_0', 'ind_var12', 'ind_var13_0', 'ind_var13_corto_0', 'ind_var13_corto', 'ind_var13_largo_0',
        'ind_var13_largo', 'ind_var13_medio_0', 'ind_var13', 'ind_var14_0', 'ind_var14', 'ind_var17_0', 'ind_var17',
        'ind_var18_0', 'ind_var19', 'ind_var20_0', 'ind_var20', 'ind_var24_0', 'ind_var24', 'ind_var25_cte',
        'ind_var26_0', 'ind_var26_cte', 'ind_var25_0', 'ind_var30_0', 'ind_var30', 'ind_var31_0', 'ind_var31',
        'ind_var32_cte', 'ind_var32_0', 'ind_var33_0', 'ind_var33', 'ind_var34_0', 'ind_var37_cte', 'ind_var37_0',
        'ind_var39_0', 'ind_var40_0', 'ind_var40', 'ind_var41_0', 'ind_var44_0', 'ind_var44', 'ind_var7_emit_ult1',
        'ind_var7_recib_ult1', 'ind_var10_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var9_ult1',
        'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'class']]

rob_scaler = RobustScaler()
for i in continous_cols:
    data[i] = rob_scaler.fit_transform(data[i].values.reshape(-1, 1))

data.to_pickle(file_name)