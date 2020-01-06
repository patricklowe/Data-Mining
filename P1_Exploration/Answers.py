# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os.path
import pandas as pd

if not os.path.exists( "AutoMpg_question1.csv" ):
    print("Missing dataset file - AutoMpg_question1.csv")
    
df_full = pd.read_csv( "AutoMpg_question1.csv" )

df_full["horsepower"].unique()
df_full.head(5)
horsepower_avg = df_full["horsepower"].mean()
df_full["horsepower"] = df_full["horsepower"].fillna(horsepower_avg)
df_full["horsepower"]
df_full["origin"]
origin_min = df_full["origin"].min()
df_full["origin"] = df_full["origin"].fillna(origin_min)
df_full["origin"]
df_full.to_csv(r'question1 out.csv')
if not os.path.exists( "AutoMpg_question2_b.csv" ):
    print("Missing dataset file - AutoMpg_question2_b.csv")

df_2b = pd.read_csv( "AutoMpg_question2_b.csv" )
df_2b.head(5)
df_2b.rename(columns={'name': 'car name'}, inplace=True)
if not os.path.exists( "AutoMpg_question2_a.csv" ):
    print("Missing dataset file - AutoMpg_question2_a.csv")

df_2a = pd.read_csv( "AutoMpg_question2_a.csv" )
df_2a.head(5)
other = 1
df_2a["other"] = other
df_2a.head(5)
df_combined = pd.concat([df_2a, df_2b], axis=0)
df_combined.reset_index(drop=True)
df_combined

df_combined.to_csv(r'question2_out.csv')

