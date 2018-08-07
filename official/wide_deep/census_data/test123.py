import pandas as pd

test = pd.read_csv("adult.test")
print(test.count())

above = test.iloc[:,14] == "<=50K"
print(test[above].count())

below= test.iloc[:,14] == ">50K"
print(test[below].count())
