import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('data/iris.csv')
X = data.drop(columns=['target'])
y = data['target']

# Define model and hyperparameter grid
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X, y)

# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'models/best_model.pkl')
print("Best model saved to 'models/best_model.pkl'.")

# Log results
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)
