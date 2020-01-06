import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

# Load the data
df_1 = pd.read_csv( "specs/gpa_question1.csv", header=None)
#print(df_1.head(5))

"""
1.1 Filter out the column "Count"
"""
df_1 = df_1.drop(4, axis=1)
#print(df_1.head(15))

"""
1.2 Use the Apriori algorithm to generate frequent itemsets from the input
data. When doing so, only select frequent itemsets with a support of at
least 15% (so, the minimum support should be 0.15). How many frequent
itemsets are produced? How big are they? Include this information in
your report
"""
num_records = len(df_1)
records = []
for i in range(0, num_records):
    records.append([str(df_1.values[i,j]) for j in range(0,4)])

te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets= apriori(df,min_support=0.15,use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
#print(frequent_itemsets)
# 20 itemsets are produced, 13 of length 1 and 7 of length 2

"""
1.3 Save the generated itemsets in ./output/question1 out apriori.csv,
making sure to include the support column
"""
frequent_itemsets.to_csv('output/question1_out_apriori.csv', index=False)

"""
1.4 Using these frequent itemsets, generate a first batch of association rules
with a minimum confidence of 0.9. How many rules are produced? For
each rule, include a short description in your report
"""
assoc_rules_1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
assoc_rules_1 = assoc_rules_1[['antecedents','consequents','support','confidence']]
#print(assoc_rules_1)
# 1 rule is produced, it implies the students aged 21 to 25 are guaranteed to be a junior, with 15.4% of the data supporting this

"""
1.5 Save the generated rules in ./output/question1 out rules9.csv, making sure to include the support and confidence columns.
"""
assoc_rules_1.to_csv('output/question1_out_rules9.csv', index=False)

"""
1.6 Generate a second batch of association rules, but this time use a minimum
confidence of 0.7. How many rules are produced this time? Again, shortly
describe the outcome in your report
"""
assoc_rules_2 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
assoc_rules_2 = assoc_rules_2[['antecedents','consequents','support','confidence']]
#print(assoc_rules_2)
# 3 rules are produced
#   21 to 25 yr olds are 100% likely to be a junior, with 15.4% of the data supporting it
#   Ph.D students are 80% likely to be 26 to 30 years old, 15.4% of data supports this
#   finally, philosophy students are 71.42% likely to be 26 to 30 yrs old with 19.23% of data supporting this

"""
1.7 Save the generated rules in ./output/question1 out rules7.csv in the
same format as the previous rule batch
"""
assoc_rules_2.to_csv('output/question1_out_rules7.csv', index=False)

"""
Question 2: Association Rules with FP-Growth
"""
# Load the data
df_2 = pd.read_csv( "specs/bank_data_question2.csv")#, header=None)

"""
2.1 Filter out the column "id"
"""
df_2 = df_2.drop('id', axis=1)
#print(df_2.head(5))

"""
2.2 Discretize the numeric attributes into 3 bins of equal width, the filter out
the original attributes.
"""
# bin Ages
df_2['age_bin'] = pd.cut(x=df_2['age'], bins=3)
df_2 = df_2.drop('age', axis=1)

# bin income
df_2['income_bin'] = pd.cut(x=df_2['income'], bins=3)
df_2 = df_2.drop('income', axis=1)
#print(df_2['income_bin'].unique())

# bin children
df_2['children_bin'] = pd.cut(x=df_2['children'], bins=3)
df_2 = df_2.drop('children', axis=1)

"""
2.3 Use the FP-Growth algorithm to generate frequent itemsets from the
data. When doing so, only select frequent itemsets with a support of at
least 20% (so, the minimum support should be 0.2). How many frequent
itemsets are produced? How big are they? Include this information in
your report
"""

df_2.head(5)

#  replace yes/no columns to make more sense
df_2['married'] = df_2['married'].replace('NO','Not Married')
df_2['married'] = df_2['married'].replace('YES','Is Married')
df_2['car'] = df_2['car'].replace('NO','No Car')
df_2['car'] = df_2['car'].replace('YES','Has Car')
df_2['save_act'] = df_2['save_act'].replace('NO','No save_act')
df_2['save_act'] = df_2['save_act'].replace('YES','Yes save_act')
df_2['current_act'] = df_2['current_act'].replace('NO','No current_act')
df_2['current_act'] = df_2['current_act'].replace('YES','Has current_act')
df_2['mortgage'] = df_2['mortgage'].replace('NO','No mortgage')
df_2['mortgage'] = df_2['mortgage'].replace('YES','Has mortgage')
df_2['pep'] = df_2['pep'].replace('NO','No pep')
df_2['pep'] = df_2['pep'].replace('YES','Has pep')

num_records2 = len(df_2)
records2 = []
for i in range(0, num_records2):
    records2.append([str(df_2.values[i,j]) for j in range(0,11)])

te_ary2 = te.fit(records2).transform(records2)
df2 = pd.DataFrame(te_ary2, columns=te.columns_)
frequent_itemsets2 = fpgrowth(df2,min_support=0.2,use_colnames=True)
frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets2)
# 98 itemsets are produced, anywhere from 1 to 4 in length

"""
2.4 Save the generated itemsets in ./output/question1 out fpgrowth.csv
"""
frequent_itemsets2.to_csv('output/question2_out_fpgrowth.csv', index=False)

"""
2.5 Using the obtained frequent itemsets, generate association rules. Experiment with different confidence values, selecting a value that produces at
least 10 rules. What is this value? Include it in your report.
"""
assoc_rules_1_fp = association_rules(frequent_itemsets2, metric="confidence", min_threshold=0.791)
assoc_rules_1_fp = assoc_rules_1_fp[['antecedents','consequents','support','confidence']]
#print(assoc_rules_1_fp)
"""
2.6 Save the generated rules in ./output/question2 out rules.csv
"""
assoc_rules_1_fp.to_csv('output/question2_out_rules.csv', index=False)

"""
2.7
Select the top 2 most interesting rules
"""
#({'Yes save_act', 'No mortgage', 'No pep'})	({'Is Married'})	0.2	0.845070423
#({'No save_act'})	({'Has current_act'})	0.226666667	0.731182796


















