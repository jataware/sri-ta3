try:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
except:
    print('!! no cuda support')

import numpy as np
from vizier.service import clients
from vizier.service import pyvizier as vz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class HPO_RandomForestClassifier:
    def __init__(self, n_estimators=100, verbose=True, cuda=False):
        self.n_estimators = n_estimators
        self.verbose      = verbose
        self.cuda         = cuda
    
    def tune(self, X, y, score_fn, n_iters=10):
        # self.res = []
        
        if self.cuda:
            # !! Stupidly, cuML doesn't support oob_score?
            #    So we have to do manual train / valid split ...
            #    Should do proper cross-validation ...
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75)
            _X_train, _y_train = cp.array(X_train), cp.array(y_train)
            _X_valid, _y_valid = cp.array(X_valid), cp.array(y_valid)
            
        def _run_one(**params):
            params = {
                "criterion"         : str(params['criterion']),
                "max_features"      : float(params['max_features']),
                "min_samples_split" : int(params['min_samples_split']),
                "min_samples_leaf"  : int(params['min_samples_leaf']),
            }
            
            if not self.cuda:
                model = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1, verbose=1, oob_score=True, **params)
                model = model.fit(X, y)
                oob_scores = model.oob_decision_function_[:,1]
                return score_fn(y, oob_scores)
            else:
                params['split_criterion'] = params.pop('criterion')
                
                model        = cuRandomForestClassifier(n_estimators=self.n_estimators, verbose=1, **params)
                model        = model.fit(_X_train, _y_train)
                valid_scores = model.predict_proba(_X_valid)[:,1].get()
                return score_fn(y_valid, valid_scores)
        
        # --
        # Setup
        
        study_config = vz.StudyConfig(algorithm='GAUSSIAN_PROCESS_BANDIT', observation_noise=vz.ObservationNoise.HIGH)
        root = study_config.search_space.root
        
        _ = root.add_categorical_param('criterion', ['gini', 'entropy'])
        _ = root.add_float_param('max_features', 1e-3, 1)
        _ = root.add_int_param('min_samples_split', 2, 20)
        _ = root.add_int_param('min_samples_leaf', 1, 20)
        
        _ = study_config.metric_information.append(vz.MetricInformation('acc', goal=vz.ObjectiveMetricGoal.MAXIMIZE))
        
        # --
        # Run
        
        self.study = clients.Study.from_study_config(study_config, owner='bkj', study_id=str(np.random.choice(2 ** 16)))
        for iter in range(n_iters):
            suggestion = self.study.suggest(count=1)[0]
            params     = suggestion.parameters
            objective  = _run_one(**params)
            suggestion.complete(vz.Measurement({'acc': objective}))
            # self.res.append(objective)
            print(iter, params, objective)
        
        return self
    
    def best_params(self):
        
        self.study.set_state(vz.StudyState.COMPLETED)
        optimal_trials = self.study.optimal_trials()
        for optimal_trial in optimal_trials:
            optimal_trial = optimal_trial.materialize(include_all_measurements=True)
            print(optimal_trial.parameters)
        
        oparams = optimal_trial.parameters.as_dict()
        
        oparams = {
            "criterion"          : str(oparams['criterion']),
            "max_features"       : float(oparams['max_features']),
            "min_samples_split"  : int(oparams['min_samples_split']),
            "min_samples_leaf"   : int(oparams['min_samples_leaf']),
        }
        
        if self.cuda:
            oparams['split_criterion'] = oparams.pop('criterion')
        
        self._oparams = oparams
        return oparams
    
    def fit(self, X, y, oparams, n_estimators=2048):
        if not self.cuda:
            model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, verbose=1, **oparams)
            model = model.fit(X, y)
        else:
            model = cuRandomForestClassifier(n_estimators=self.n_estimators, verbose=1, **oparams)
            model = model.fit(cp.array(X), cp.array(y))
            
        self._model = model
        return self
            
    def predict_proba(self, X):
        return self._model.predict_proba(X)[:,1]

# Usage:
# rf_hpo  = HPO_RandomForestClassifier()
# rf_hpo  = rf_hpo.tune(X_train, y_train)
# oparams = rf_hpo.best_params()
# rf_hpo  = rf_hpo.fit(X_train, y_train, oparams)
# p_hpo   = rf_hpo.predict_proba(X_valid)