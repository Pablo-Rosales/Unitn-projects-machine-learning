import pandas as pd
from sklearn.model_selection import train_test_split

#Opening the csv data file
file = pd.read_csv('leukemia.dat')

#Randomly splitting the data file into train and test subsets
train, test = train_test_split(file, test_size = 0.2) #80% train, 20% test

print('TRAIN SET')
print(train)
print('TEST SET')
print(test)

#Saving the new files with the subsets
train.to_csv('leukemia_train.dat', index=False)
test.to_csv('leukemia_test.dat', index=False)