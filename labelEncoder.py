from sklearn import preprocessing

le = preprocessing.LabelEncoder()
l=["paris", "paris", "tokyo", "amsterdam"]
l=le.fit_transform(l)

# list(le.classes_)
l
