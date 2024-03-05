import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(filename):
    #loads data
    data = pd.read_csv(filename)

    #drop columns with a lot of missing values
    data.drop(['koi_teq_err1', 'koi_teq_err2'], axis=1, inplace=True)

    #replaces missing values with the mean of the colomn
    imputer = SimpleImputer(strategy = 'mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data.select_dtypes(include=['float64', 'int64'])))
    data_imputed.columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[data_imputed.columns] = data_imputed

    #drop non important identifier columns (i.e. names and such)
    data.drop(['rowid', 'kepid', 'kepoi_name', 'kepler_name'], axis = 1, inplace = True)

    #one-hot encode categorical columns
    data = pd.get_dummies(data, columns=['koi_disposition', 'koi_pdisposition', 'koi_tce_delivname'], drop_first = True)

    #standardizes features
    scaler = StandardScaler()
    
    #'koi_disposition_CONFIRMED' is the target after encoding
    features = data.drop('koi_disposition_CONFIRMED', axis=1)  
    scaled_features = scaler.fit_transform(features)
    
    #split data into training and testing
    X = scaled_features # X holds the input features of the dataset after scaling
    y = data['koi_disposition_CONFIRMED'].to_numpy()  #convert the Series to a numpy array
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

__all__ = ['preprocess_data']
