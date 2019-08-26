import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def unpack_model(model_name):
    path = '1) classification algorithms/random forest/bioresponse/' + model_name
    with open(path, 'rb') as file:
        model = pickle.load(file)
        return model

real_model = unpack_model('model_forest_unbalanced_bio.pkl')


# load the original data
file_name = 'data/bioresponse/bio_clean.pkl'
real_data = pd.read_pickle(file_name)

# unbalanced data
X = real_data.drop('class', axis=1)
y = real_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


def get_feature_importance(model, x_train):
    feature_importance = pd.DataFrame(model.feature_importances_, index=x_train.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)
    return feature_importance

FI = get_feature_importance(real_model, X_train)

row_list = list(FI.index.values)

# save only column names for features which contribute to random forest training
cols_to_keep = row_list[:265]
cols_to_keep.append('class')

data = real_data[cols_to_keep]
print(data)

data.to_pickle('data/bioresponse/bio_short_266.pkl')
