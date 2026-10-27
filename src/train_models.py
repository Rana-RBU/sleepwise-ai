import os
import json
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from data_preprocessing import load_and_preprocess_data, get_features_and_target

def train_and_evaluate():
    df = load_and_preprocess_data('data/Sleep_health_and_lifestyle_dataset.csv')
    X, y = get_features_and_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    cat_cols = ['Gender', 'Occupation', 'BMI Category']
    num_cols = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    models = {
        'KNN': Pipeline([('preprocessor', preprocessor), ('classifier', KNeighborsClassifier())]),
        'Decision Tree': Pipeline([('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(random_state=42))]),
        'Random Forest': Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))]),
        'SVM': Pipeline([('preprocessor', preprocessor), ('classifier', SVC(probability=True, random_state=42))])
    }
    
    params = {
        'KNN': {'classifier__n_neighbors': [3, 5, 7]},
        'Decision Tree': {'classifier__max_depth': [3, 5, 10]},
        'Random Forest': {'classifier__n_estimators': [50, 100]},
        'SVM': {'classifier__C': [1, 10], 'classifier__kernel': ['linear', 'rbf']}
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_f1, best_model_name = -1, None
    
    for name, model in models.items():
        grid = GridSearchCV(model, params[name], cv=cv, scoring='f1_macro', return_train_score=True, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_est = grid.best_estimator_
        y_pred = best_est.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        results[name] = {
            'cv_macro_f1': grid.best_score_,
            'train_score': grid.cv_results_['mean_train_score'][grid.best_index_],
            'test_macro_f1': macro_f1
        }
        
        joblib.dump(best_est, f'models/{name.replace(" ", "_")}_model.pkl')
        if macro_f1 > best_f1:
            best_f1, best_model_name = macro_f1, name
            
    results['best_model'] = best_model_name
    with open('reports/model_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    train_and_evaluate()