import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# QUESTION 1 
df_full = pd.read_csv( "specs/SensorData_question1.csv" )

"""
1. Generate a new attribute called Original Input3 which is a copy of the
attribute Input3. Do the same with the attribute Input12 and copy it into
Original Input12
"""
df_full['Original Input3'] = df_full['Input3']
df_full['Original Input12'] = df_full['Input12']

"""
2. Normalise the attribute Input3 using the z-score transformation method
"""
#Z-Score= (X - Mean) / Standard Dev
Input3_Mean= df_full['Input3'].mean()
Input3_STD = df_full['Input3'].std()

df_full['Input3'] = (df_full['Input3']-Input3_Mean)/Input3_STD

"""
3. Normalise the attribute Input12 in the range [0.0, 1.0].
"""
#Normalized Data = (x-min(x))/(max(x)-min(x))
Input12_Min = df_full['Input12'].min()
Input12_Max = df_full['Input12'].max()

df_full['Input12'] = (df_full['Input12'] - Input12_Min)/(Input12_Max-Input12_Min)

"""
4. Generate a new attribute called Average Input, which is the average of all
the attributes from Input1 to Input12. This average should include the
normalised attributes values but not the copies that were made of these.
"""
df_full.loc[:,'Average Input'] = df_full.drop(['Original Input3','Original Input12'], axis=1).sum(axis=1)/len(df_full.drop(['Original Input3','Original Input12'], axis=1).columns)

"""
5. Save the newly generated dataset to ./output/question1 out.csv.
"""
df_full.to_csv('output/question1_out.csv', index=False)

# QUESTION 2
"""
1. Reduce the number of attributes using Principal Component Analysis
(PCA), making sure at least 95% of all the variance is explained.
"""
df_full2 = pd.read_csv( "specs/DNAData_question2.csv" )
# Get headings of columns
col_head= df_full2.columns.values
# Get values of each column
x = df_full2.loc[:, col_head].values
print(x.shape) 
# normalize the columns
x = StandardScaler().fit_transform(x)
print(x.shape) 
# Get the header of each columm that is equal to the shape of X on axis 1 (column header)
header_cols = [col_head[i] for i in range(x.shape[1])] 
# Store the normalised data in a new df
normalised_data = pd.DataFrame(x,columns=header_cols)
print(normalised_data.shape) 
# Allow PCA to determine the minimum number of components required to have a 95% explained variance
pca_data = PCA(.95)
# Z-Score the PCA data using x, the standardized data
PC_data = pca_data.fit_transform(x)
# Convert PC_data into a dataframe 
PCA_DF = pd.DataFrame(data = PC_data)
# Print the resulting explained variance for each column
#---print('Explained Variance: {}'.format(pca_data.explained_variance_ratio_))

"""
2. Discretise the PCA-generated attribute subset into 10 bins, using bins of
equal width. For each component X that you discretise, generate a new
column in the original dataset named pcaX width. For example, the first
discretised principal component will correspond to a new column called
pca1 width.
"""

# For each 'new' attribute of PCA, do the following (skipping index column)
for i in range(1,len(PCA_DF.columns)):
    # store the values of each column in col_values
    col_values = PCA_DF[i].values
    # bin each column into equal widths of 10 depending on that columns values
    bins = pd.cut(col_values, 10)
    # Get the value of each row entry we are working on (i.e value by value of the column)
    labels = bins.codes
    # Get each bin range of the current value
    bin_range = bins.categories
    # create a blank series
    pca_width_values = pd.Series([]) 
    # for each value in the column get its bin label/range
    for j in range(len(col_values)):
        # get the label of the current bin range
        bin_label = labels[j]
        # create a new column and add the bin label for each value
        pca_width_values[j] = bin_range[bin_label]
    # Create a new column for the new bin width labels
    df_full2['pca' + str(i) + '_width'] = pca_width_values
    
    
"""
3. Discretise PCA-generated attribute subset into 10 bins, using bins of equal
frequency (they should all contain the same number of points). For each
component X that you discretise, generate a new column in the original
1
dataset named pcaX freq. For example, the first discretised principal
component will correspond to a new column called pca1 width.
"""
# does the same as above but replace cut with qcut for quartile cutting, bin set to 10.
for i in range(1,len(PCA_DF.columns)):
    col_values = PCA_DF[i].values
    bins2 = pd.qcut(col_values, 10)
    labels2 = bins2.codes
    bin_range2 = bins2.categories
    pca_width_values2 = pd.Series([]) 
    for j in range(len(col_values)):
        bin_label2 = labels2[j]
        pca_width_values2[j] = bin_range2[bin_label2]
    df_full2['pca' + str(i) + '_freq'] = pca_width_values2
    
# Export to a CSV
df_full2.to_csv('output/question2_out.csv', index=False)

#print(pd.qcut(col_values, 10).value_counts())