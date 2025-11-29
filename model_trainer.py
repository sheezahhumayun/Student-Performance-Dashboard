import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def create_baseline_models():
    """Create baseline dummy models"""
    return {
        'dummy_mean': DummyRegressor(strategy='mean'),
        'dummy_median': DummyRegressor(strategy='median')
    }

def create_regression_models():
    """Create various regression models"""
    return {
        'simple_linear': LinearRegression(),
        'multiple_linear': LinearRegression(),
        'polynomial_degree2': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ]),
        'polynomial_degree3': Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('linear', LinearRegression())
        ])
    }

def evaluate_baseline_models(modeling_datasets):
    """Establish performance benchmarks for each RQ"""
    baseline_results = {}
    
    for rq_name, dataset_info in modeling_datasets.items():
        X = dataset_info['data'][dataset_info['features']]
        y = dataset_info['data'][dataset_info['target']]
        
        # Ensure we have enough data for train/test split
        if len(X) < 10:
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        rq_results = {}
        baseline_models = create_baseline_models()
        
        for model_name, model in baseline_models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                rq_results[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'model': model
                }
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        baseline_results[rq_name] = rq_results
    
    return baseline_results

def train_and_evaluate_models(modeling_datasets):
    """Train and evaluate all models for each RQ"""
    all_results = {}
    
    for rq_name, dataset_info in modeling_datasets.items():
        X = dataset_info['data'][dataset_info['features']]
        y = dataset_info['data'][dataset_info['target']]
        
        # Skip if not enough data
        if len(X) < 10:
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        rq_results = {}
        models = create_regression_models()
        
        # For simple linear regression, use best single feature (only for RQ1)
        if rq_name == 'RQ1' and len(dataset_info['features']) > 0:
            try:
                correlations = []
                for feature in dataset_info['features']:
                    if feature in X_train.columns:
                        corr = np.corrcoef(X_train[feature], y_train)[0,1]
                        correlations.append((feature, abs(corr)))
                
                if correlations:
                    best_feature = max(correlations, key=lambda x: x[1])[0]
                    
                    simple_model = LinearRegression()
                    simple_model.fit(X_train[[best_feature]], y_train)
                    y_pred_simple = simple_model.predict(X_test[[best_feature]])
                    
                    rq_results['simple_linear'] = {
                        'MAE': mean_absolute_error(y_test, y_pred_simple),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_simple)),
                        'R2': r2_score(y_test, y_pred_simple),
                        'model': simple_model,
                        'feature_used': best_feature
                    }
            except Exception as e:
                print(f"Error with simple linear: {e}")
        
        # Train other models
        for model_name, model in models.items():
            if model_name == 'simple_linear' and rq_name != 'RQ1':
                continue
                
            try:
                if model_name == 'simple_linear':
                    continue
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                rq_results[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'model': model
                }
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        all_results[rq_name] = rq_results
    
    return all_results

def perform_bootstrapping(modeling_datasets, all_results, n_bootstrap=50):
    """Perform bootstrapping to compute confidence intervals"""
    bootstrap_results = {}
    
    for rq_name, dataset_info in modeling_datasets.items():
        if rq_name not in all_results or not all_results[rq_name]:
            continue
            
        X = dataset_info['data'][dataset_info['features']]
        y = dataset_info['data'][dataset_info['target']]
        
        # Skip if not enough data
        if len(X) < 20:
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        rq_bootstrap = {}
        
        # Focus on best performing non-dummy model
        non_dummy_models = {k: v for k, v in all_results[rq_name].items() 
                           if not k.startswith('dummy')}
        if not non_dummy_models:
            continue
            
        best_model_name = min(non_dummy_models.keys(), 
                            key=lambda x: non_dummy_models[x]['MAE'])
        
        model = all_results[rq_name][best_model_name]['model']
        mae_scores = []
        
        for i in range(min(n_bootstrap, 50)):  # Limit to 50 for performance
            try:
                X_bootstrap, y_bootstrap = resample(X_train, y_train, random_state=i)
                
                if best_model_name == 'simple_linear':
                    feature_used = all_results[rq_name][best_model_name]['feature_used']
                    model.fit(X_bootstrap[[feature_used]], y_bootstrap)
                    y_pred = model.predict(X_test[[feature_used]])
                else:
                    model.fit(X_bootstrap, y_bootstrap)
                    y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                mae_scores.append(mae)
            except Exception as e:
                print(f"Bootstrap iteration {i} failed: {e}")
                continue
        
        if mae_scores:
            # Calculate confidence intervals
            alpha = 0.95
            p_lower = (1 - alpha) / 2 * 100
            p_upper = (alpha + (1 - alpha) / 2) * 100
            
            ci_lower = np.percentile(mae_scores, p_lower)
            ci_upper = np.percentile(mae_scores, p_upper)
            mean_mae = np.mean(mae_scores)
            
            rq_bootstrap[best_model_name] = {
                'mean_MAE': mean_mae,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'all_MAE_scores': mae_scores
            }
            
            bootstrap_results[rq_name] = rq_bootstrap
    
    return bootstrap_results

def train_all_models(modeling_datasets):
    """Main function to train all models and return results"""
    if not modeling_datasets:
        return {}, {}, {}
        
    baseline_results = evaluate_baseline_models(modeling_datasets)
    all_results = train_and_evaluate_models(modeling_datasets)
    
    # Add baseline results to all results
    for rq_name in all_results.keys():
        if rq_name in baseline_results:
            all_results[rq_name].update(baseline_results[rq_name])
    
    bootstrap_results = perform_bootstrapping(modeling_datasets, all_results, n_bootstrap=50)
    
    return all_results, baseline_results, bootstrap_results
