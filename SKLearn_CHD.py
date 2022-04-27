# SKLearn Basics HW: Script 1
# Using the California Housing Data set create a linear regression model that maps the features onto the target,
# the predicted housing price. Create a list of the features sorted by how much they impact the housing price.
# Sort on absolute value.

# Import packages
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing  # Imports dataset

# Initialize dataset and view head
chd = fetch_california_housing(as_frame=True)
ch_data = chd.data
ch_target = chd.target
ch_data.head()


# Function to perform linear regression and sort features on absolute value
def feature_impact(data, target):
    # Regression
    lr = LinearRegression()
    lin_reg = lr.fit(data, target)
    coefs = lin_reg.coef_

    # Put results into dataframe
    feat = pd.DataFrame()
    feat['Features'] = data.columns
    feat['Coefficients'] = coefs

    # Sort on absolute value
    feat['Coefficients'] = feat['Coefficients'].abs()
    feat = feat.sort_values(by=['Coefficients'], ascending=False)

    # Create and print ordered features list
    imp_feats = list(feat['Features'])
    print('Features in order of how much they impact housing price: ', imp_feats)


if __name__ == '__main__':
    feature_impact(ch_data, ch_target)
