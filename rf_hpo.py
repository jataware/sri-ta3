import numpy as np
from vizier.service import clients
from vizier.service import pyvizier as vz
from sklearn.ensemble import RandomForestClassifier

def r_auc(target, scores, n=None, p=None):
    n_pos = target.sum()
    
    curve = np.cumsum(target[np.argsort(-scores)])
    
    if p:
        n = np.where(curve > (n_pos * p))[0][0]
    
    if n:
        curve = curve[:n]
    
    metrics = np.trapz(
        y = curve / target.sum(),
        x = np.linspace(0, 1, curve.shape[0])
    )
    
    return curve, metrics


class HPO_RandomForestClassifier:
    def __init__(self, n_estimators=100, n_metric=100, verbose=True):
        self.n_estimators = n_estimators
        self.n_metric     = n_metric
        self.verbose      = verbose
    
    def tune(self, X, y, n_iters=10):
        self.res = []
        
        def _run_one(**params):
            params = {
                "criterion"         : str(params['criterion']),
                "max_features"      : float(params['max_features']),
                "min_samples_split" : int(params['min_samples_split']),
                "min_samples_leaf"  : int(params['min_samples_leaf']),
            }
            
            model = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1, verbose=1, oob_score=True, **params)
            model = model.fit(X, y)
            
            oob_scores = model.oob_decision_function_[:,1]
            
            _, score = r_auc(y, oob_scores, n=self.n_metric)
            
            if self.verbose:
                print(params, score)
            
            return score
        
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
        for _ in range(n_iters):
            suggestion = self.study.suggest(count=1)[0]
            params     = suggestion.parameters
            objective  = _run_one(**params)
            suggestion.complete(vz.Measurement({'acc': objective}))
            self.res.append(objective)
        
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
        
        self._oparams = oparams
        return oparams
    
    def fit(self, X, y, oparams, n_estimators=2048):
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, verbose=1, **oparams)
        model = model.fit(X, y)
        self._model = model
        return self
            
    def predict_proba(self, X):
        return self._model.predict_proba(X)[:,1]

# Usage:
# rf_hpo  = HPO_RandomForestClassifier()
# rf_hpo  = rf_hpo.tune(X_train, y_train)
# oparams = rf_hpo.best_params()
# rf_hpo  = rf_hpo.fit(X_train, y_train, oparams)
# p_hpo   = rf_hpo.predict(X_valid)