import pandas as pd

#read the 'csv' format file

list = pd.read_csv('D:\\RNN\\unix.csv')
#print(list.shape)
#print(list.columns)
info = list.info()
pic = list.describe()
#print(pic)
#print(list.dropna(axis=0,how='any'))
#print(list.replace(to_replace='heiheihei',value="haha"))
print(pd.isnull(list))