import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as StandardScaler3
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score as ACCU_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_dataset(file_path):
    try:
        csvf = pd.read_csv(file_path)
        print(f"CSV file loaded")
        return csvf
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def extract_additional_features(csvf):
    
    csvf['digit_to_url_length_ratio'] = csvf['number_of_digits_in_url'] / csvf['url_length']
    csvf['special_to_url_length_ratio'] = csvf['number_of_special_char_in_url'] / csvf['url_length']
    csvf['domain_to_url_length_ratio'] = csvf['domain_length'] / csvf['url_length']
    csvf['url_length'] = csvf['url_length'].replace(0, np.nan)
    csvf = csvf.replace([np.inf, -np.inf], 0).fillna(0)
    
    csvf['high_entropy_url'] = (csvf['entropy_of_url'] > 4.5).astype(int)
    csvf['high_entropy_domain'] = (csvf['entropy_of_domain'] > 4.0).astype(int)

    csvf['url_risk_score'] = (
        csvf['having_repeated_digits_in_url'] * 2 +
        (csvf['number_of_special_char_in_url'] > 3).astype(int) * 1.5 +
        (csvf['number_of_hyphens_in_url'] > 1).astype(int) * 1.2 +
        (csvf['number_of_dots_in_url'] > 3).astype(int) * 1.8 +
        csvf['having_special_characters_in_domain'] * 2.5 +
        csvf['having_digits_in_domain'] * 1.3 +
        csvf['having_repeated_digits_in_domain'] * 2 +
        (csvf['number_of_subdomains'] > 2).astype(int) * 1.7 +
        csvf['having_hyphen_in_subdomain'] * 1.5 +
        csvf['having_special_characters_in_subdomain'] * 2.2 +
        csvf['having_repeated_digits_in_subdomain'] * 1.8 +
        csvf['high_entropy_url'] * 1.5 +
        csvf['high_entropy_domain'] * 1.5
    )
    
    return csvf

def preprocess_dataset(csvf):
    
    csvf = extract_additional_features(csvf)
    
    X = csvf.drop('Type', axis=1)
    y = csvf['Type']
    
    if y.dtype == 'object':
        y = y.map({'phishing': 1, 'legitimate': 0})
    
    return X, y

def apply_rules(prediction, features):
    
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
    
    raw_predictions = model.predict(X)
    
    enhanced_predictions = []
    for i, pred in enumerate(raw_predictions):
        instance_features = features_df.iloc[i].to_dict()
        
        enhanced_pred = apply_rules(pred, instance_features)
        enhanced_predictions.append(enhanced_pred)
    
    return np.array(enhanced_predictions)

def evaluate_model(y_true, y_pred, model_name="Model"):
    
    print(f"\n Performance:")
    print(classification_report(y_true, y_pred))
    
    ACCU = ACCU_score(y_true, y_pred)
    print(f"ACCU: {ACCU:.4f}")
    
    plt.figure(figsize=(6, 5))
    cmatrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix')
    plt.xlabel('PredictedClass')
    plt.ylabel('ActualClass')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    plt.show()
    
    return ACCU

def main():
    csvf = load_dataset('Dataset.csv')
    if csvf is None:
        return
    
    print("\nDataset summary:")
    print(csvf['Type'].value_counts())
    print("\nFirst few rows:")
    print(csvf.head())
    
    print("\nPreprocessing the dataset...")
    X, y = preprocess_dataset(csvf)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler3 = StandardScaler3()
    X_train_scaled = scaler3.fit_transform(X_train)
    X_test_scaled = scaler3.transform(X_test)
    
    print(f"\nFeatures after preprocessing of the Data: {X.shape[1]}")
    
    print("\nTraining models...")
    
    rfc_model = RFC(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rfc_model.fit(X_train_scaled, y_train)
    
    gbc_model = GBC(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gbc_model.fit(X_train_scaled, y_train)
    
    mlpc_model = MLPC(
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
    mlpc_model.fit(X_train_scaled, y_train)
    
    print("\nEvaluating base models...")
    
    rfc_pred = rfc_model.predict(X_test_scaled)
    rf_ACCU = evaluate_model(y_test, rfc_pred, "RFC")
    
    gbc_pred = gbc_model.predict(X_test_scaled)
    gb_ACCU = evaluate_model(y_test, gbc_pred, "GBC")
    
    mlpc_pred = mlpc_model.predict(X_test_scaled)
    mlp_ACCU = evaluate_model(y_test, mlpc_pred, "MLPC")
    
    best_ACCU = max(rf_ACCU, gb_ACCU, mlp_ACCU)
    if best_ACCU == rf_ACCU:
        bestmodel3 = rfc_model
        TopModel = "Random Forest"
    elif best_ACCU == gb_ACCU:
        bestmodel3 = gbc_model
        TopModel = "Gradient Boosting"
    else:
        bestmodel3 = mlpc_model
        TopModel = "MLP Neural Network"
    
    print(f"\nBest base model: {TopModel} with ACCU {best_ACCU:.4f}")
    
    print("\nApplying rule-based enhancement...")
    egpd_predictions = egpd_dnr_prediction(bestmodel3, X_test_scaled, X_test)
    egpd_ACCU = evaluate_model(y_test, egpd_predictions, "EGPD-DNR")
    
    improvement = (egpd_ACCU - best_ACCU) * 100
    print(f"\nACCU improvement: {improvement:.2f}%")
    
    if TopModel in ["Random Forest", "Gradient Boosting"]:
        Impmodel = bestmodel3.feature_importances_
        Implabel = X.columns
        
        indices = np.argsort(Impmodel)[::-1]
        
        print("\nTop 15 imp label:")
        for i in range(min(15, len(Implabel))):
            print(f"{Implabel[indices[i]]}: {Impmodel[indices[i]]:.4f}")
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Impmodel ({TopModel})')
        plt.bar(range(min(15, len(Implabel))), Impmodel[indices[:15]], align='center')
        plt.xticks(range(min(15, len(Implabel))), [Implabel[i] for i in indices[:15]], rotation=90)
        plt.tight_layout()
        plt.savefig('labels_Impmodel.png')
        plt.show()
    
    import joblib
    joblib.dump(bestmodel3, f'bestmodel3_{TopModel.lower()}.pkl')
    print(f"\nModel is saved as 'bestmodel3_{TopModel.lower()}.pkl'")
    
    joblib.dump(scaler3, 'scaler3.pkl')
    print("Scaler3 is saved as 'scaler3.pkl'")
    
    print("\nExample of using the model for prediction:")
    print("Load the model and scaler3:")
    print("model = joblib.load('bestmodel3_{}.pkl')".format(TopModel.lower().replace(" ", "_")))
    print("scaler3 = joblib.load('scaler3.pkl')")
    print("\nPredict on new data:")
    print("X_new_scaled = scaler3.transform(X_new)")
    print("raw_predictions = model.predict(X_new_scaled)")
    print("enhanced_predictions = egpd_dnr_prediction(model, X_new_scaled, X_new)")

if __name__ == "__main__":
    main()