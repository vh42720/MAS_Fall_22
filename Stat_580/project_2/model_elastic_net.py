# extra imports
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, RandomizedSearchCV

from utils.helper_functions import *

# load config and data
X_train, y_train = load_train_data()

# elastic net
regressor = ElasticNet()

# cross validation method
cv = KFold(n_splits=5, random_state=0, shuffle=True)

# set params
params = {
	'alpha': np.logspace(-5, 5, 100, endpoint=True),  # between e^-5 and e^5
	'l1_ratio': np.arange(0, 1, 0.01)  # between 0 and 1
}

# RandomizedSearch for best hyper params
reg_cv = RandomizedSearchCV(
	regressor,
	params,
	n_iter=500,
	scoring='neg_root_mean_squared_error',
	cv=cv,
	verbose=1,
	refit=True,
	n_jobs=14
)
reg_cv.fit(X_train, y_train.values.ravel())

# best score
ret_val = {
	'model': 'elastic_net',
	'validation_best_RMSE': -reg_cv.best_score_,
	'validation_best_params': reg_cv.best_params_
}
print(ret_val)

save_result(ret_val)
save_model(reg_cv, model_name='elastic_net')
