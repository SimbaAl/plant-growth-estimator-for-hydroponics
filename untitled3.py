# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:37:23 2021

@author: 36450057
"""

X_for_training = X_for_RF #This is our X input to RF

## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.25, 0.30, 0.4, 0.5, 0.7] ,
 "max_depth"        : [5, 6, 8, 10, 12],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.4, 0.5, 0.6, 0.7 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=1,n_jobs=-1,cv=3,verbose=3)

start_time = timer(None) 
random_search.fit(X_for_training,y_train)
timer(start_time)


random_search.best_estimator_
random_search.best_params_


classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7,
              enable_categorical=False, gamma=0.6, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.5, max_delta_step=0, max_depth=8,
              min_child_weight=7, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


