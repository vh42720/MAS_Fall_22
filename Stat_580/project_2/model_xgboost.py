# extra imports
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBRegressor

from utils.helper_functions import *

# load config and data
X_train, y_train = load_train_data()
X_test, y_test = load_test_data()

# XGBoost
regressor = XGBRegressor()

# cross validation method
cv = KFold(n_splits=5, random_state=0, shuffle=True)

# set params
params = {
	'booster': ['gbtree'],  # gblinear
	'n_estimators': np.arange(100, 1500, 100),
	'max_depth': np.arange(2, 20, 2),
	'learning_rate': np.arange(0, 1, 0.1),
	'min_child_weight': np.arange(0, 60, 10)
}

# RandomizedSearch for best hyper params
reg_cv = RandomizedSearchCV(
	regressor,
	params,
	cv=cv,
	n_iter=100,
	scoring='neg_root_mean_squared_error',
	error_score=0,
	verbose=1,
	refit=True,
	n_jobs=12
)
reg_cv.fit(X_train, y_train.values.ravel())

# best score
ret_val = {
	'model': 'xgboost',
	'validation_best_RMSE': -reg_cv.best_score_,
	'validation_best_params': reg_cv.best_params_
}
print(ret_val)

save_result(ret_val)
save_model(reg_cv, model_name='xgboost')
