import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

# Load the data
df_1 = pd.read_csv( "specs/marks_question1.csv", index_col = False) #, header=None)

X = df_1.iloc[:, :-1]
Y = df_1.iloc[:, 1]

#plt.plot(X,Y)
# Those doing better in midterm, do better in the finals

"""
Use linear regression
"""
# Split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/12, random_state = 0)

reg = LinearRegression()
reg.fit(X_train,Y_train)
X_test = X_test.replace(59,86)
Y_pred = reg.predict(X_test)
print('Midterm: 86. E(Final):',Y_pred)

"""
Question 2
https://www.datacamp.com/community/tutorials/decision-tree-classification-python
"""
# Load the data
df_2 = pd.read_csv( "specs/borrower_question2.csv") #, index_col = False) #, header=None)
df_2 = df_2.drop('TID', axis=1)

X2 = df_2.iloc[:, :-1] # attributes to build off, everything but last column
Y2 = df_2.iloc[:, 3] # defaulted borrower, label

One_Hot_Data = pd.get_dummies(X2)

X_train, X_test, Y_train, Y_test = train_test_split(One_Hot_Data, Y2, test_size=0.3, random_state=1)

clf = tree.DecisionTreeClassifier(criterion="entropy", min_impurity_decrease = 0.1)
clf_train = clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

dot_data = StringIO()
graph_labels = list(One_Hot_Data.columns)

export_graphviz(clf, out_file=dot_data, feature_names = graph_labels)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('test.png')
Image(graph.create_png())
