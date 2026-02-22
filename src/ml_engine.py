import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

class MLEngine:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.models = {} # Dictionary of trained models
        self.label_encoders = {}
        
        # Features calculated by AutoAnnotator (must match export_dataset.py)
        self.feature_columns = [
            'feat_z1_h_ratio', 'feat_z2_h_ratio', 'feat_z3_h_ratio',
            'feat_z1_w_ratio', 'feat_z3_w_ratio', 'feat_face_wh_ratio',
            'feat_eye_spacing_ratio', 'feat_eye_size_ratio',
            'feat_nose_width_ratio', 'feat_nose_icd_ratio',
            'feat_mouth_width_ratio', 'feat_upper_lip_ratio', 'feat_lower_lip_ratio',
            'feat_jaw_cheek_ratio'
        ]
        
        # Load existing models if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def load_data(self, csv_path):
        """Load training data from exported CSV"""
        if not os.path.exists(csv_path):
            print(f"Data file not found: {csv_path}")
            return None
            
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} records from {csv_path}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def train(self, data_path, target_column='target_face_shape', model_type='classification'):
        """
        Train a model for a specific target column.
        model_type: 'classification' or 'regression'
        """
        df = self.load_data(data_path)
        if df is None or df.empty:
            return False
            
        # Check if features exist
        missing_feats = [col for col in self.feature_columns if col not in df.columns]
        if missing_feats:
            print(f"Missing feature columns in CSV: {missing_feats}")
            return False
            
        if target_column not in df.columns:
            print(f"Target column '{target_column}' not found in CSV.")
            return False
            
        # Drop rows with missing target values
        df_clean = df.dropna(subset=[target_column] + self.feature_columns)
        
        if len(df_clean) < 10:
            print("Not enough data to train (minimum 10 samples required).")
            return False
            
        X = df_clean[self.feature_columns]
        y = df_clean[target_column]
        
        print(f"Training {model_type} model for '{target_column}' with {len(df_clean)} samples...")
        
        if model_type == 'classification':
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoders[target_column] = le
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc:.2f}")
            
            # Save model
            self.models[target_column] = clf
            
        elif model_type == 'regression':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_train, y_train)
            
            y_pred = reg.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"MAE: {mae:.4f}, R2: {r2:.4f}")
            
            self.models[target_column] = reg
            
        return True

    def predict(self, features_dict, target_column='target_face_shape'):
        """Predict a specific target using provided features"""
        if target_column not in self.models:
            print(f"Model for '{target_column}' not found.")
            return None
            
        # Prepare input vector
        try:
            vector = []
            for col in self.feature_columns:
                # Remove 'feat_' prefix if input dict doesn't have it
                key = col
                if key not in features_dict:
                    short_key = key.replace('feat_', '')
                    if short_key in features_dict:
                        key = short_key
                
                vector.append(features_dict.get(key, 0.0))
            
            X_new = np.array([vector])
            model = self.models[target_column]
            
            if isinstance(model, RandomForestClassifier):
                le = self.label_encoders[target_column]
                pred_idx = model.predict(X_new)[0]
                pred_label = le.inverse_transform([pred_idx])[0]
                probs = model.predict_proba(X_new)[0]
                confidence = float(np.max(probs))
                
                return {'value': pred_label, 'confidence': confidence}
                
            elif isinstance(model, RandomForestRegressor):
                pred_val = model.predict(X_new)[0]
                return {'value': float(pred_val), 'confidence': 1.0} # No confidence score for regression usually
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def save_model(self, path):
        """Saves the entire MLEngine state."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        
        versioned_name = f"{name}_v{timestamp}{ext}"
        versioned_path = os.path.join(dirname, versioned_name)
        
        data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'version': timestamp
        }
        
        joblib.dump(data, versioned_path)
        joblib.dump(data, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Loads the MLEngine state."""
        try:
            data = joblib.load(path)
            self.models = data['models']
            self.label_encoders = data.get('label_encoders', {})
            self.feature_columns = data.get('feature_columns', [])
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
