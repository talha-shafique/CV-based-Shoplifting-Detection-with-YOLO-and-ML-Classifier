
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# --- Configuration ---
FEATURES_CSV = r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\features.csv'
MODEL_OUTPUT_PATH = r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\shoplifting_model.joblib'

def train_model():
    """Loads features, trains a model, evaluates it, and saves it."""
    print(f"Loading features from {FEATURES_CSV}...")
    try:
        df = pd.read_csv(FEATURES_CSV)
    except FileNotFoundError:
        print(f"Error: Features file not found at {FEATURES_CSV}. Please run the feature_extractor.py script first.")
        return

    # Define features (X) and target (y)
    # We drop task_name and frame_id as they are identifiers, not features.
    features = [
        'num_people', 
        'num_products', 
        'num_bags',
        'min_person_product_dist',
        'product_in_bag_occluded'
    ]
    target = 'is_shoplifting_event'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    # We use stratify=y to ensure the proportion of shoplifting events is the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # Initialize and train the RandomForestClassifier
    print("Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Make predictions on the test data
    print("Evaluating the model...")
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=['Actual Normal', 'Actual Shoplifting'], columns=['Predicted Normal', 'Predicted Shoplifting']))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Shoplifting']))

    # Save the trained model
    print(f"Saving the trained model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print("Model saved successfully.")

if __name__ == '__main__':
    train_model()
