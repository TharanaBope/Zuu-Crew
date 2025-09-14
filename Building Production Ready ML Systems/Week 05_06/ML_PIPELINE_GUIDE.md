# Complete ML Pipeline Guide: From Data to Deployment

This guide explains the machine learning pipeline workflow in the correct order, focusing on what each file does step by step.

## 📋 Overview of the Complete ML Pipeline

The ML pipeline consists of 6 main files that work together in sequence:

1. **Data Processing** (covered in previous weeks)
2. **Model Building** (`model_building.py`)
3. **Model Training** (`model_training.py`)
4. **Model Evaluation** (`model_evaluation.py`)
5. **Training Pipeline** (`training_pipeline.py`) - orchestrates everything
6. **Model Inference** (`model_inference.py`)
7. **Streaming Inference Pipeline** (`streaming_inference_pipeline.py`)

---

## 🏗️ File 1: `model_building.py` - Creating Model Architectures

**Purpose**: This file defines different types of machine learning models we can use.

### What it does:
```python
# Think of this as a "model factory" - it creates different types of models
```

### Key Components:

#### 1. Base Model Builder (Abstract Class)
```python
class BaseModelBuilder(ABC):
    # This is like a template that all model builders must follow
    # It ensures every model builder has the same basic functions
```

**What this means for beginners:**
- It's like a blueprint that says "every model builder must have these functions"
- You can't use this directly - it's just a template

#### 2. Random Forest Model Builder
```python
class RandomForestModelBuilder(BaseModelBuilder):
    # Creates a Random Forest model with default settings
    default_params = {
        'max_depth': 10,        # How deep each tree can grow
        'n_estimators': 100,    # How many trees to create
        'random_state': 42      # For reproducible results
    }
```

**What Random Forest does:**
- Creates many decision trees
- Each tree votes on the prediction
- Final prediction = majority vote from all trees
- Good for: General classification tasks, handles mixed data types well

#### 3. XGBoost Model Builder
```python
class XGboostModelBuilder(BaseModelBuilder):
    # Creates an XGBoost model (more advanced than Random Forest)
    default_params = {
        'max_depth': 10,        # How deep each tree can grow
        'n_estimators': 100,    # How many boosting rounds
        'random_state': 42      # For reproducible results
    }
```

**What XGBoost does:**
- Creates trees one by one, each learning from previous mistakes
- Very powerful and often wins competitions
- Good for: Complex patterns, usually better accuracy than Random Forest

### How to use this file:
```python
# Create a Random Forest model
rf_builder = RandomForestModelBuilder(n_estimators=200)  # Use 200 trees
rf_model = rf_builder.build_model()

# Create an XGBoost model
xgb_builder = XGboostModelBuilder(max_depth=5)  # Limit tree depth
xgb_model = xgb_builder.build_model()
```

---

## 🎯 File 2: `model_training.py` - Training Models with Data

**Purpose**: Takes a model and teaches it using training data.

### What it does:
```python
# This is like a "trainer" that teaches models using examples
# Input: Untrained model + training data
# Output: Trained model that can make predictions
```

### Key Components:

#### 1. ModelTrainer Class
```python
class ModelTrainer:
    def __init__(self):
        # Sets up the trainer with logging
        logger.info("ModelTrainer initialized")
```

#### 2. Training Process
```python
def train(self, model, X_train, Y_train):
    # Step 1: Validate the data
    if X_train is None or Y_train is None:
        raise ValueError("Training data cannot be None")
    
    # Step 2: Check data consistency
    if len(X_train) != len(Y_train):
        raise ValueError("Feature and target length mismatch")
    
    # Step 3: Log training information
    logger.info(f"Model Type: {type(model).__name__}")
    logger.info(f"Training Samples: {len(X_train):,}")
    logger.info(f"Features: {X_train.shape[1]}")
    
    # Step 4: Actually train the model
    model.fit(X_train, Y_train)  # This is where the magic happens!
    
    # Step 5: Calculate how well it learned
    train_score = model.score(X_train, Y_train)
    
    return model, train_score
```

**What happens during `model.fit()`:**
1. Model looks at each example in X_train and its correct answer in Y_train
2. Model adjusts its internal parameters to minimize mistakes
3. Process repeats until model converges or reaches max iterations

#### 3. Saving and Loading Models
```python
def save_model(self, model, filepath):
    # Saves the trained model to disk so you can use it later
    joblib.dump(model, filepath)

def load_model(self, filepath):
    # Loads a previously saved model from disk
    model = joblib.load(filepath)
    return model
```

**Why save models?**
- Training takes time and computational resources
- Once trained, you can reuse the model many times
- You can share trained models with others

---

## 📊 File 3: `model_evaluation.py` - Measuring Model Performance

**Purpose**: Tests how well the trained model performs on unseen data.

### What it does:
```python
# This is like giving an exam to the model
# Uses test data the model has never seen before
# Calculates various scores to measure performance
```

### Key Components:

#### 1. ModelEvaluator Class
```python
class ModelEvaluator:
    def __init__(self, model, model_name):
        self.model = model           # The trained model to test
        self.model_name = model_name # Name for identification
        self.evaluation_results = {} # Will store all the scores
```

#### 2. Evaluation Process
```python
def evaluate(self, X_test, Y_test):
    # Step 1: Make predictions on test data
    Y_pred = self.model.predict(X_test)
    
    # Step 2: Compare predictions with actual answers
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
```

### Understanding the Metrics:

#### Confusion Matrix
```
                Predicted
Actual    |  No Churn  |  Churn  |
----------|------------|---------|
No Churn  |    TN      |   FP    |  (True Negative, False Positive)
Churn     |    FN      |   TP    |  (False Negative, True Positive)
```

#### Accuracy
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
# What percentage of predictions were correct?
# Example: 85% accuracy means 85 out of 100 predictions were right
```

#### Precision
```python
precision = TP / (TP + FP)
# Of all customers predicted to churn, how many actually churned?
# High precision = fewer false alarms
```

#### Recall
```python
recall = TP / (TP + FN)
# Of all customers who actually churned, how many did we catch?
# High recall = we catch most of the churners
```

#### F1 Score
```python
f1 = 2 * (precision * recall) / (precision + recall)
# Balanced measure between precision and recall
# Good when you need both to be high
```

**Real-world example:**
- Model predicts 100 customers will churn
- 80 actually churn (Precision = 80%)
- But there were 120 total churners, so we missed 40 (Recall = 67%)
- F1 Score balances these: 2 * (0.8 * 0.67) / (0.8 + 0.67) = 73%

---

## 🔄 File 4: `training_pipeline.py` - Orchestrating Everything

**Purpose**: The conductor of the orchestra - coordinates all steps in the correct order.

### What it does:
```python
# This file puts everything together:
# 1. Gets the data ready
# 2. Builds the model
# 3. Trains the model
# 4. Evaluates the model
# 5. Saves everything for later use
```

### Step-by-Step Process:

#### Step 1: Data Preparation
```python
# Load the processed data (from previous data pipeline steps)
X_train = pd.read_csv('artifacts/data/X_train.csv')
Y_train = pd.read_csv('artifacts/data/Y_train.csv')
X_test = pd.read_csv('artifacts/data/X_test.csv')
Y_test = pd.read_csv('artifacts/data/Y_test.csv')

logger.info(f"Data loaded - Training: {X_train.shape}, Test: {X_test.shape}")
```

#### Step 2: Model Building
```python
# Create the model using our model builder
model_builder = XGboostModelBuilder(**model_params)
model = model_builder.build_model()

logger.info(f"Model created: {type(model).__name__}")
```

#### Step 3: Model Training
```python
# Train the model using our trainer
trainer = ModelTrainer()
trained_model, training_score = trainer.train(model, X_train, Y_train)

logger.info(f"Training completed - Score: {training_score:.4f}")
```

#### Step 4: Model Evaluation
```python
# Evaluate the model's performance
evaluator = ModelEvaluator(trained_model, 'XGboost')
evaluation_results = evaluator.evaluate(X_test, Y_test)

logger.info(f"Accuracy: {evaluation_results['accuracy']:.4f}")
logger.info(f"F1 Score: {evaluation_results['f1']:.4f}")
```

#### Step 5: Model Saving
```python
# Save the trained model for later use
model_path = 'artifacts/models/churn_analysis.joblib'
trainer.save_model(trained_model, model_path)

logger.info(f"Model saved to: {model_path}")
```

### Output Summary:
```
✓ Data loaded - Training: (7990, 10), Test: (1998, 10)
✓ Model training completed in 3.85 seconds
✓ Training Score: 1.0000
✓ Model saved to: artifacts/models/churn_analysis.joblib
✓ Model Performance - Accuracy: 0.8529, F1 Score: 0.5859
```

---

## 🔮 File 5: `model_inference.py` - Making Predictions

**Purpose**: Takes a trained model and uses it to predict new, unseen data.

### What it does:
```python
# This is like using your trained model in the real world
# Input: New customer data
# Output: Prediction (will they churn or not?)
```

### Key Components:

#### 1. ModelInference Class
```python
class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.encoders = {}
        
        # Load the trained model
        self.load_model()
```

#### 2. Loading the Trained Model
```python
def load_model(self):
    # Load the saved model from training
    self.model = joblib.load(self.model_path)
    logger.info(f"Model loaded: {type(self.model).__name__}")
```

#### 3. Data Preprocessing for Prediction
```python
def preprocess_input(self, data):
    # New data needs the same preprocessing as training data
    
    # Step 1: Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Step 2: Apply encoders (convert text to numbers)
    for col, encoder in self.encoders.items():
        if col in df.columns:
            df[col] = df[col].map(encoder)
    
    # Step 3: Apply feature binning (group credit scores)
    if 'CreditScore' in df.columns:
        binning = CustomBinningStratergy(self.binning_config)
        df = binning.bin_feature(df, 'CreditScore')
    
    # Step 4: Apply ordinal encoding
    ordinal_strategy = OrdinalEncodingStratergy(self.encoding_config)
    df = ordinal_strategy.encode(df)
    
    # Step 5: Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Firstname', 'Lastname'])
    
    return df
```

#### 4. Making Predictions
```python
def predict(self, data):
    # Step 1: Preprocess the input data
    processed_data = self.preprocess_input(data)
    
    # Step 2: Make prediction
    y_pred = self.model.predict(processed_data)        # 0 or 1
    y_proba = self.model.predict_proba(processed_data) # probability
    
    # Step 3: Format results
    prediction = int(y_pred[0])
    probability = float(y_proba[0][1])  # probability of churn
    
    status = 'Churn' if prediction == 1 else 'Retain'
    confidence = round(probability * 100, 2)
    
    return {
        "Status": status,
        "Confidence": f"{confidence}%"
    }
```

### Example Usage:
```python
# New customer data
customer_data = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Balance": 0,
    "NumOfProducts": 1,
    # ... other features
}

# Make prediction
inference = ModelInference('artifacts/models/churn_analysis.joblib')
result = inference.predict(customer_data)
print(result)  # {"Status": "Retain", "Confidence": "23.45%"}
```

---

## 🌊 File 6: `streaming_inference_pipeline.py` - Real-Time Predictions

**Purpose**: Handles continuous, real-time predictions with tracking and monitoring.

### What it does:
```python
# This simulates a real-world system where predictions happen continuously
# Like a customer service system that checks churn risk for every interaction
# Tracks performance and batches results for analysis
```

### Key Components:

#### 1. InferenceTracker Class
```python
class InferenceTracker:
    def __init__(self):
        self.predictions_batch = []     # Store predictions temporarily
        self.batch_size = 100          # Process 100 predictions at a time
```

#### 2. Tracking Individual Predictions
```python
def track_prediction(self, input_data, prediction_result, inference_time):
    # Create a record for each prediction
    prediction_record = {
        'timestamp': datetime.now().isoformat(),
        'input_data': input_data,
        'prediction': prediction_result,
        'inference_time_ms': inference_time * 1000,
        'churn_probability': float(prediction_result['Confidence'].replace('%', '')) / 100,
        'predicted_class': 1 if prediction_result['Status'] == 'Churn' else 0
    }
    
    self.predictions_batch.append(prediction_record)
```

#### 3. Batch Processing
```python
def _log_prediction_batch(self):
    # When we have 100 predictions, analyze them together
    batch_stats = self._calculate_batch_stats()
    
    # Calculate metrics like:
    # - Average inference time
    # - How many churn predictions
    # - Distribution of confidence scores
    
    # Save batch to file for later analysis
    batch_file = f"inference_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(batch_file, 'w') as f:
        json.dump(self.predictions_batch, f)
```

#### 4. Streaming Inference Function
```python
def streaming_inference(inference, data):
    # Time the prediction
    start_time = time.time()
    
    # Make prediction
    prediction_result = inference.predict(data)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Track this prediction
    inference_tracker.track_prediction(data, prediction_result, inference_time)
    
    return prediction_result
```

### Real-World Simulation:
```python
# Example: Processing customer service calls
customers_data = [
    {"CreditScore": 619, "Age": 42, ...},  # Customer 1
    {"CreditScore": 720, "Age": 35, ...},  # Customer 2
    # ... more customers
]

for customer in customers_data:
    result = streaming_inference(inference, customer)
    print(f"Customer risk: {result['Status']} ({result['Confidence']})")
    
    # Every 100 customers, the system automatically:
    # 1. Calculates batch statistics
    # 2. Saves predictions to file
    # 3. Monitors system performance
```

---

## 🔄 Complete Workflow Summary

Here's how everything works together in order:

### Phase 1: Development & Training
1. **Data Preprocessing** (previous weeks) → Clean, prepared data
2. **`model_building.py`** → Create model architecture
3. **`model_training.py`** → Train the model with data
4. **`model_evaluation.py`** → Test model performance
5. **`training_pipeline.py`** → Orchestrate steps 2-4

### Phase 2: Production & Inference
6. **`model_inference.py`** → Make individual predictions
7. **`streaming_inference_pipeline.py`** → Handle continuous predictions

### Data Flow:
```
Raw Data → Data Processing → Training Data (X_train, Y_train, X_test, Y_test)
                                      ↓
Model Builder → Untrained Model → Model Training → Trained Model
                                      ↓
Model Evaluation → Performance Metrics → Saved Model
                                      ↓
New Customer Data → Model Inference → Prediction Results
                                      ↓
Streaming Pipeline → Batch Processing → Performance Monitoring
```

### Key Files Output:
- **Training**: `artifacts/models/churn_analysis.joblib` (trained model)
- **Evaluation**: Performance metrics (accuracy, precision, recall, F1)
- **Inference**: Prediction results `{"Status": "Churn/Retain", "Confidence": "X%"}`
- **Streaming**: Batch files with prediction history and statistics

This pipeline represents a complete machine learning system from development to production deployment!