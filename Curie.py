import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import scipy.stats

# Set visualization parameters
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = (10, 8)

# Configuration
RANDOM_STATE = 0
TEST_SIZE = 0.1
poly = PolynomialFeatures(degree=3, include_bias=False)


def load_and_preprocess_data(data_path, ionic_params=None):
    """Load and preprocess the dataset with optional ionic radius parameters"""
    with open(data_path) as f:
        dataset = [[float(x) for x in line.split()] for line in f]

    data_array = np.array(dataset)
    data_transpose = data_array.transpose()

    # Calculate tolerance factor if ionic parameters provided
    if ionic_params:
        ir_bi, ir_o = ionic_params
        tf_factor = [(ir_bi + ir_o) / ((data_transpose[12][i] + ir_o) * (2.0 ** 0.5))
                     for i in range(len(data_transpose[12]))]
        data_transpose = np.vstack((data_transpose, tf_factor))

        # Calculate ionic displacement
        ionic_disp = [(data_transpose[3][i] ** 2 + data_transpose[4][i] ** 2) ** 0.5
                      for i in range(len(data_transpose[3]))]
        data_transpose = np.vstack((data_transpose, ionic_disp))

        # Calculate polarization factor PX
        px_factor = [(1 - data_transpose[0][i]) * 4.5 * 0.75 + (data_transpose[0][i]) * 3.6 * 0.45 +
                     (data_transpose[0][i]) * 6.3 * 0.28 +
                     (1 - data_transpose[0][i]) * data_transpose[1][i] * data_transpose[3][i] * data_transpose[5][i] +
                     (1 - data_transpose[0][i]) * data_transpose[2][i] * data_transpose[4][i] * data_transpose[6][i]
                     for i in range(len(data_transpose[3]))]
        data_transpose = np.vstack((data_transpose, px_factor))

    return data_transpose


def create_dataframe(data, columns=None):
    """Create a DataFrame from the data array with specified columns"""
    data_dict = {str(i + 1): data[i] for i in range(data.shape[0])}
    if columns is None:
        columns = [str(i + 1) for i in range(data.shape[0])]
    return pd.DataFrame(data_dict, columns=columns)


def train_evaluate_model(model, X_train, X_test, y_train, y_test, transform_poly=True):
    """Train a model and evaluate its performance"""
    # Store the original data for later use
    X_train_orig = X_train.copy()
    X_test_orig = X_test.copy()

    if transform_poly:
        # Important: fit on training data only, then transform both
        poly_transformer = PolynomialFeatures(degree=3, include_bias=False)
        X_train = poly_transformer.fit_transform(X_train)
        X_test = poly_transformer.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Calculate metrics
    test_error = mean_absolute_error(y_test, y_pred)
    train_error = mean_absolute_error(y_train, y_pred_train)

    # Calculate adjusted R^2
    model_linear = LinearRegression()
    model_linear.fit(np.array(y_test).reshape(-1, 1), np.array(y_pred).reshape(-1, 1))
    score = model_linear.score(np.array(y_test).reshape(-1, 1), np.array(y_pred).reshape(-1, 1))
    adj_r2 = 1 - (1 - score) * (len(y_pred) - 1) / (len(y_pred) - 1 - 1)

    results = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_train': y_pred_train,
        'test_error': test_error,
        'train_error': train_error,
        'adj_r2': adj_r2,
        'X_train_orig': X_train_orig,  # Save original data
        'transform_poly': transform_poly,
        'poly_transformer': poly_transformer if transform_poly else None
    }

    return results


def perform_hyperparameter_tuning(base_model, param_grid, X_train, y_train,
                                  cv=10, scoring='neg_mean_squared_error', transform_poly=True):
    """Perform hyperparameter tuning for a model"""
    X_train_orig = X_train.copy()
    poly_transformer = None

    if transform_poly:
        poly_transformer = PolynomialFeatures(degree=3, include_bias=False)
        X_train = poly_transformer.fit_transform(X_train)

    if isinstance(base_model, RandomForestRegressor):
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=100,
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
    else:
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            verbose=1,
            n_jobs=-1
        )

    search.fit(X_train, y_train)

    # Add metadata to the search object
    search.X_train_orig = X_train_orig
    search.transform_poly = transform_poly
    search.poly_transformer = poly_transformer

    return search


def plot_results(y_test, y_pred, y_train=None, y_pred_train=None,
                 validation_y=None, validation_pred=None,
                 test_error=None, train_error=None, validation_error=None,
                 test_r2=None, train_r2=None, validation_r2=None,
                 title="Model Predictions", filename=None):
    """Plot model prediction results"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Plot prediction points with error bars
    if y_train is not None and y_pred_train is not None:
        ax.errorbar(y_train, y_pred_train, yerr=train_error, fmt='ro',
                    label=f'Training data MAE = {round(train_error, 2)}'
                          f', adjusted R² = {round(train_r2, 2)}')

    ax.errorbar(y_test, y_pred, yerr=test_error, fmt='co',
                label=f'Test data MAE = {round(test_error, 2)}'
                      f', adjusted R² = {round(test_r2, 2)}')

    if validation_y is not None and validation_pred is not None:
        ax.errorbar(validation_y, validation_pred, yerr=validation_error, fmt='go',
                    label=f'Validation MAE = {round(validation_error, 2)}'
                          f', adjusted R² = {round(validation_r2, 2)}')

    # Plot identity line
    xmin, xmax = ax.get_xlim()
    x_line = np.linspace(max(0, xmin), xmax, 100)
    ax.plot(x_line, x_line, '-k', linewidth=2.0)

    # Format plot
    ax.set_xlabel('True Tc (K)', fontsize=15)
    ax.set_ylabel('Predicted Tc (K)', fontsize=15)
    ax.legend(loc='upper left', fontsize='medium')

    if title:
        ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=15,
                verticalalignment='top')

    # Save plot if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def compare_models(results_dict, metric='test_error', labels=None, title="Model Comparison",
                   ylabel="MAE (K)", filename=None):
    """Compare multiple models using bar chart"""
    if labels is None:
        labels = list(results_dict.keys())

    values = [results_dict[model][metric] for model in results_dict.keys()]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x, values, width)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel('Method', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Save plot if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # 1. Load and preprocess data
    main_data = load_and_preprocess_data(r'C:/Users/Downloads/datasetVfii.txt',
                                         ionic_params=(1.38, 1.4))
    test_data = load_and_preprocess_data(r'C:/Users/Downloads/datasetVftesti.txt',
                                         ionic_params=(1.38, 1.4))

    # 2. Create dataframes
    df = create_dataframe(main_data)
    df_test = create_dataframe(test_data)

    # 3. Define feature sets
    X_full = df[['1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '16']]
    y = df['10']

    X_reduced = df[['1', '2', '3', '4', '5', '6', '7', '11', '12', '13', '14', '15', '16']]

    X_test_full = df_test[['1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '16']]
    y_test = df_test['10']

    X_test_reduced = df_test[['1', '2', '3', '4', '5', '6', '7', '11', '12', '13', '14', '15', '16']]

    # 4. Split data
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
        X_reduced, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 5. Define models
    models = {
        'RF': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'SVR': SVR(kernel='rbf', gamma='auto'),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }

    # 6. Define hyperparameter grids
    rf_param_grid = {
        'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=10)],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=5)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    svr_param_grid = {
        'C': [0.1, 1, 100, 1000],
        'epsilon': [0.0001, 0.001, 0.01, 0.1, 1],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    }

    # 7. Train and evaluate base models
    results = {}
    for name, model in models.items():
        # Full feature set
        results[f"{name}_full"] = train_evaluate_model(
            model.deepcopy() if hasattr(model, 'deepcopy') else model.__class__(**model.get_params()),
            X_train_full, X_test_full, y_train_full, y_test_full)

        # Reduced feature set
        results[f"{name}_reduced"] = train_evaluate_model(
            model.deepcopy() if hasattr(model, 'deepcopy') else model.__class__(**model.get_params()),
            X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced)

    # 8. Hyperparameter tuning for selected models
    tuned_rf = perform_hyperparameter_tuning(
        RandomForestRegressor(random_state=RANDOM_STATE),
        rf_param_grid,
        X_train_full,
        y_train_full
    )

    tuned_svr = perform_hyperparameter_tuning(
        SVR(kernel='rbf'),
        svr_param_grid,
        X_train_reduced,
        y_train_reduced
    )

    # 9. Evaluate tuned models
    results['RF_tuned_full'] = train_evaluate_model(
        tuned_rf.best_estimator_, X_train_full, X_test_full, y_train_full, y_test_full,
        transform_poly=tuned_rf.transform_poly)

    results['SVR_tuned_reduced'] = train_evaluate_model(
        tuned_svr.best_estimator_, X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced,
        transform_poly=tuned_svr.transform_poly)

    # 10. Evaluate on validation data
    validation_results = {}
    for name, result in results.items():
        # Get correct validation feature set
        if 'full' in name:
            X_val = X_test_full.copy()
        else:
            X_val = X_test_reduced.copy()

        # Use the same transformation as during training
        if result['transform_poly'] and result['poly_transformer'] is not None:
            X_val_transformed = result['poly_transformer'].transform(X_val)
        else:
            X_val_transformed = X_val

        # Predict on validation set
        y_pred_val = result['model'].predict(X_val_transformed)

        # Make sure all arrays are the same length
        min_len = min(len(y_test_full), len(y_pred_val))
        y_val_trimmed = y_test_full[:min_len]
        y_pred_val_trimmed = y_pred_val[:min_len]

        val_error = mean_absolute_error(y_val_trimmed, y_pred_val_trimmed)

        # Calculate validation R^2
        model_linear = LinearRegression()
        model_linear.fit(
            np.array(y_val_trimmed).reshape(-1, 1),
            np.array(y_pred_val_trimmed).reshape(-1, 1)
        )
        score = model_linear.score(
            np.array(y_val_trimmed).reshape(-1, 1),
            np.array(y_pred_val_trimmed).reshape(-1, 1)
        )
        val_r2 = 1 - (1 - score) * (len(y_pred_val_trimmed) - 1) / (len(y_pred_val_trimmed) - 1 - 1)

        validation_results[name] = {
            'y_pred_val': y_pred_val_trimmed,
            'y_val': y_val_trimmed,
            'val_error': val_error,
            'val_r2': val_r2
        }

    # 11. Visualize results
    # Compare model errors
    compare_models(
        {name: {'test_error': results[name]['test_error']} for name in results},
        title="Model Comparison - Test Error",
        filename="model_comparison_error.svg"
    )

    # Plot best model predictions
    best_model_name = min(results, key=lambda x: results[x]['test_error'])
    best_model = results[best_model_name]
    best_validation = validation_results[best_model_name]

    plot_results(
        y_test_full if 'full' in best_model_name else y_test_reduced,
        best_model['y_pred'],
        y_train_full if 'full' in best_model_name else y_train_reduced,
        best_model['y_pred_train'],
        best_validation['y_val'],
        best_validation['y_pred_val'],
        best_model['test_error'],
        best_model['train_error'],
        best_validation['val_error'],
        best_model['adj_r2'],
        best_model['adj_r2'],  # Using same R2 for train
        best_validation['val_r2'],
        title=f"Best Model: {best_model_name}",
        filename="best_model_predictions.svg"
    )
