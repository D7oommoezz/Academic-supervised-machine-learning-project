from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

#Import dataset

filename = "../data/glass.csv"
names = ["id", "refractive index", "Na", "Mg", "Al",
         "Si", "K", "Ca", "Ba", "Fe", "Type"]
d = pd.read_csv(filename, names=names)
print(d.shape)

# check missing values
bool_series = pd.isnull(d[:]).values.any()
d[bool_series]

# Split the dataset into the Training set and Test set
d1 = d.iloc[:, 1:10].values
d2 = d.iloc[:, -1].values
d1_train, d1_test, d2_train, d2_test = train_test_split(
    d1, d2, test_size=0.20, random_state=0)



# Scaling
s = StandardScaler()
d1_train = s.fit_transform(d1_train)
d1_test = s.transform(d1_test)

# Train the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(d1_train, d2_train)

# Predict the Test set results
d2_pred = classifier.predict(d1_test)


#Check how many rows was predicted wrong ,"there are 27 rows were wrong" 
diff = (d2_pred == d2_test)
np.size(np.where(diff == False))

# Data Visualization
plt.scatter(d2_test, d2_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
