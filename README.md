# FA595_SKLearn
This repository contains the scripts SKLearn_CHD.py and SKLearn_Iris.py which are the two scripts required for the SKLearn Basics homework assignment in FA595: Financial Technology at Stevens Institute of Technology.

# SKLearn_CHD.py
This script is the solution to the following problem:

Using the California Housing Data set available here (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html (Links to an external site.)) create a linear regression model that maps the features onto the target, the predicted housing price. Once the model is trained, create a list of the features sorted by how much they impact the housing price. Note: some might have a negative impact, you will need to sort on absolute value, not just on the value from the model.

# SKLearn_Iris.py
This script is the solution to the following problem:

Using the Iris (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris) data set, create a script that trains multiple KMeans clustering models. Each KMeans model should have a different number of clusters, starting at 1 ending at 7. Use this information to graph the number of clusters vs the "inertia" (the average distance to the centroids) and confirm, using the elbow heuristic, how many clusters should be used to best describe this dataset.