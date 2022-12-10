from utils.helper_functions import *

np.set_printoptions(suppress=True)

# load config and data
config = load_config()
X_train, y_train = load_train_data()
X_test, y_test = load_test_data()

# make prediction
best_reg = load_model('xgboost')
best_reg.fit(X_train, y_train.values.ravel())
y_pred = best_reg.predict(X_test)

# save prediction
prediction = pd.DataFrame({'uniqueID': y_test.values.ravel(), 'SalePrice': y_pred})
prediction.to_csv(os.path.join(config['path']['result'], 'prediction.csv'))
