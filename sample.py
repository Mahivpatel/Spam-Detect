import pandas as pd
import numpy as np
import re
import os
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def extract_additional_features(df):
    """
    Extract additional features that might be useful for detection
    """
    df['digit_to_url_length_ratio'] = df['number_of_digits_in_url'] / df['url_length']
    df['special_to_url_length_ratio'] = df['number_of_special_char_in_url'] / df['url_length']
    df['domain_to_url_length_ratio'] = df['domain_length'] / df['url_length']
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    df['high_entropy_url'] = (df['entropy_of_url'] > 4.5).astype(int)
    df['high_entropy_domain'] = (df['entropy_of_domain'] > 4.0).astype(int)

    df['url_risk_score'] = (
        df['having_repeated_digits_in_url'] * 2 +
        (df['number_of_special_char_in_url'] > 3).astype(int) * 1.5 +
        (df['number_of_hyphens_in_url'] > 1).astype(int) * 1.2 +
        (df['number_of_dots_in_url'] > 3).astype(int) * 1.8 +
        df['having_special_characters_in_domain'] * 2.5 +
        df['having_digits_in_domain'] * 1.3 +
        df['having_repeated_digits_in_domain'] * 2 +
        (df['number_of_subdomains'] > 2).astype(int) * 1.7 +
        df['having_hyphen_in_subdomain'] * 1.5 +
        df['having_special_characters_in_subdomain'] * 2.2 +
        df['having_repeated_digits_in_subdomain'] * 1.8 +
        df['high_entropy_url'] * 1.5 +
        df['high_entropy_domain'] * 1.5
    )
    
    return df

def preprocess_dataset(df):
    """
    Preprocess the dataset and prepare it for model training
    """
    df = extract_additional_features(df)
    
    X = df.drop('Type', axis=1)
    y = df['Type']
    
    if y.dtype == 'object':
        y = y.map({'phishing': 1, 'legitimate': 0})
    
    return X, y

def apply_rules(prediction, features):
    """
    Apply heuristic rules to enhance model predictions
    """
    if features['number_of_special_char_in_url'] > 4 and features['number_of_digits_in_url'] > 5:
        return 1
    
    if features['having_special_characters_in_domain'] == 1 and features['having_digits_in_domain'] == 1:
        return 1
    
    if features['number_of_at_in_url'] > 0:
        return 1
    
    if features['having_repeated_digits_in_domain'] == 1 and features['number_of_hyphens_in_domain'] > 0:
        return 1
    
    if features['entropy_of_url'] > 4.5 and features['number_of_special_char_in_url'] > 3:
        return 1
    
    if features['number_of_subdomains'] > 3 and features['having_special_characters_in_subdomain'] == 1:
        return 1
    
    if features.get('url_risk_score', 0) > 8:
        return 1
    
    return prediction

def egpd_dnr_prediction(model, X, features_df):
    """
    Make predictions using the EGPD-DNR approach:
    1. Get predictions from the ML model
    2. Apply rules to enhance predictions
    """
    raw_predictions = model.predict(X)
    
    enhanced_predictions = []
    for i, pred in enumerate(raw_predictions):
        instance_features = features_df.iloc[i].to_dict()
        
        enhanced_pred = apply_rules(pred, instance_features)
        enhanced_predictions.append(enhanced_pred)
    
    return np.array(enhanced_predictions)

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance
    """
    print(f"\n{model_name} Performance:")
    print(classification_report(y_true, y_pred))
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    
    return accuracy

def main():
    df = load_dataset('Dataset.csv')
    if df is None:
        return
    
    print("\nDataset summary:")
    print(df['Type'].value_counts())
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nPreprocessing the dataset...")
    X, y = preprocess_dataset(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nNumber of features after preprocessing: {X.shape[1]}")
    
    print("\nTraining models...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        verbose=True,
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)
    
    print("\nEvaluating base models...")
    
    rf_predictions = rf_model.predict(X_test_scaled)
    rf_accuracy = evaluate_model(y_test, rf_predictions, "Random Forest")
    
    gb_predictions = gb_model.predict(X_test_scaled)
    gb_accuracy = evaluate_model(y_test, gb_predictions, "Gradient Boosting")
    
    mlp_predictions = mlp_model.predict(X_test_scaled)
    mlp_accuracy = evaluate_model(y_test, mlp_predictions, "MLP Neural Network")
    
    best_accuracy = max(rf_accuracy, gb_accuracy, mlp_accuracy)
    if best_accuracy == rf_accuracy:
        best_model = rf_model
        best_model_name = "Random Forest"
    elif best_accuracy == gb_accuracy:
        best_model = gb_model
        best_model_name = "Gradient Boosting"
    else:
        best_model = mlp_model
        best_model_name = "MLP Neural Network"
    
    print(f"\nBest base model: {best_model_name} with accuracy {best_accuracy:.4f}")
    
    print("\nApplying rule-based enhancement...")
    egpd_predictions = egpd_dnr_prediction(best_model, X_test_scaled, X_test)
    egpd_accuracy = evaluate_model(y_test, egpd_predictions, "EGPD-DNR")
    
    improvement = (egpd_accuracy - best_accuracy) * 100
    print(f"\nAccuracy improvement: {improvement:.2f}%")
    
    if best_model_name in ["Random Forest", "Gradient Boosting"]:
        importances = best_model.feature_importances_
        feature_names = X.columns
        
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 15 most important features:")
        for i in range(min(15, len(feature_names))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances ({best_model_name})')
        plt.bar(range(min(15, len(feature_names))), importances[indices[:15]], align='center')
        plt.xticks(range(min(15, len(feature_names))), [feature_names[i] for i in indices[:15]], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        plt.show()
    
    import joblib
    joblib.dump(best_model, f'best_model_{best_model_name.lower().replace(" ", "_")}.pkl')
    print(f"\nBest model saved as 'best_model_{best_model_name.lower().replace(' ', '_')}.pkl'")
    
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")
    
    print("\nExample of using the model for prediction:")
    print("Load the model and scaler:")
    print("model = joblib.load('best_model_{}.pkl')".format(best_model_name.lower().replace(" ", "_")))
    print("scaler = joblib.load('scaler.pkl')")
    print("\nPredict on new data:")
    print("X_new_scaled = scaler.transform(X_new)")
    print("raw_predictions = model.predict(X_new_scaled)")
    print("enhanced_predictions = egpd_dnr_prediction(model, X_new_scaled, X_new)")

if __name__ == "__main__":
    main()