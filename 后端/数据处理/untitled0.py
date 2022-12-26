# -*- coding: utf-8 -*-
"""
Welcome to EoHcenter
"""
import pandas as pd
def st1():
    df = pd.read_csv("pd600050.csv")
    df = df[['date', 'p_change', 'pred']]
    df.to_csv("data_need1.csv")
#st1()

def st2():
    data=pd.read_csv("data_need1.csv", index_col=0)
    for i in range(len(data)):
        data.loc[i,'pred_result'] = function1(data.loc[i, 'p_change'], data.loc[i, 'pred'])
    data.to_csv("data_need2.csv")
    #print(data)
def function1(a, c):
    if c in [0, 1, 2]:
        if a <= 0:
            return 1
        return 0
    if c in [3, 4, 5]:
        if a >= 0:
            return 1
        return 0
    return None
st2()