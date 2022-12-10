# extra imports

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import GridSearchCV

from utils.helper_functions import *

# load config and data
X_train, y_train = load_train_data()

# random forest
regressor = RandomForestRegressor(random_state=0)

# cross validation method
cv = KFold(n_splits=5, random_state=0, shuffle=True)

# set params
params = {
	'criterion': ['squared_error'],  # absolute_error
	'n_estimators': np.arange(100, 1000, 100),
	'bootstrap': [False],  # True is not optimal here
	# 'max_depth': np.arange(10, 100, 10),
	'max_features': [0.1, 0.3, 0.5, 1, 'sqrt', 'log2'],
	'min_samples_leaf': np.arange(1, 10, 1),
	'min_samples_split': np.arange(2, 10, 2)
}

# RandomizedSearch for best hyper params
reg_cv = RandomizedSearchCV(
	regressor,
	params,
	cv=cv,
	n_iter=200,
	scoring='neg_root_mean_squared_error',
	# error_score=0,
	verbose=1,
	refit=True,
	n_jobs=14
)
reg_cv.fit(X_train, y_train.values.ravel())

# best score
ret_val = {
	'model': 'random_forest',
	'validation_best_RMSE': -reg_cv.best_score_,
	'validation_best_params': reg_cv.best_params_
}
print(ret_val)

save_result(ret_val)
save_model(reg_cv, model_name='random_forest')
