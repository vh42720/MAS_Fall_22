from sklearn.utils import all_estimators
import inspect

has_n_jobs = []
for est in all_estimators():
    s = inspect.signature(est[1])
    if 'n_jobs' in s.parameters:
        has_n_jobs.append(est)
print(has_n_jobs)