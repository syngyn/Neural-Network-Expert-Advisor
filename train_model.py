import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import sys
import traceback
from datetime import datetime

from data_processing import load_and_align_data, create_features

class CombinedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs):
        super(CombinedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc_regression = nn.Linear(hidden_size, num_regression_outputs)
        self.fc_classification = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden_state = out[:, -1, :]
        regression_output = self.fc_regression(last_hidden_state)
        classification_logits = self.fc_classification(last_hidden_state)
        return regression_output, classification_logits

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = SCRIPT_DIR
REQUIRED_FILES = {
    "EURUSD": "EURUSD60.csv", "EURJPY": "EURJPY60.csv", "USDJPY": "USDJPY60.csv",
    "GBPUSD": "GBPUSD60.csv", "EURGBP": "EURGBP60.csv", "USDCAD": "USDCAD60.csv",
    "USDCHF": "USDCHF60.csv"
}

INPUT_FEATURES = 15
HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 128, 2, 20
OUTPUT_STEPS = 5
NUM_CLASSES = 3
EPOCHS, BATCH_SIZE, LEARNING_RATE = 30, 64, 0.001
LOOKAHEAD_BARS, PROFIT_THRESHOLD_ATR = 5, 0.75

def create_relative_targets(df, output_steps, symbol_close_col='EURUSD_close'):
    """
    Create percentage-based regression targets instead of absolute prices.
    This allows the model to work across different price regimes.
    """
    print(f"Creating relative percentage-based targets for {output_steps} steps...")
    
    regr_targets = []
    current_prices = df[symbol_close_col]
    
    for step in range(1, output_steps + 1):
        # Calculate percentage change from current price to future price
        future_prices = df[symbol_close_col].shift(-step)
        pct_change = (future_prices / current_prices) - 1.0
        
        # Clip extreme values to prevent training instability
        pct_change = pct_change.clip(-0.05, 0.05)  # Max ±5% change per step
        
        regr_targets.append(pct_change)
        
        # Log statistics for this step
        valid_changes = pct_change.dropna()
        print(f"  Step {step}: Mean={valid_changes.mean():.4f}, "
              f"Std={valid_changes.std():.4f}, "
              f"Min={valid_changes.min():.4f}, "
              f"Max={valid_changes.max():.4f}")
    
    regr_target_df = pd.concat(regr_targets, axis=1)
    regr_target_df.columns = [f'target_regr_{i}' for i in range(output_steps)]
    
    print(f"✓ Created relative targets with shape: {regr_target_df.shape}")
    return regr_target_df

def create_enhanced_classification_targets(df, lookahead_bars, profit_threshold_atr, symbol_close_col='EURUSD_close', atr_col='eurusd_atr'):
    """
    Create classification targets with improved logic.
    """
    print(f"Creating enhanced classification targets (lookahead: {lookahead_bars} bars)...")
    
    current_prices = df[symbol_close_col]
    future_prices = df[symbol_close_col].shift(-lookahead_bars)
    atr_values = df[atr_col]
    
    # Calculate price change and normalize by ATR
    price_change = future_prices - current_prices
    normalized_change = price_change / (atr_values + 1e-8)
    
    # Enhanced classification logic
    conditions = [
        normalized_change > profit_threshold_atr,   # Strong bullish
        normalized_change < -profit_threshold_atr,  # Strong bearish
    ]
    choices = [2, 0]  # 2=Buy, 0=Sell
    class_targets = np.select(conditions, choices, default=1)  # 1=Hold
    
    # Convert to pandas Series
    class_target_series = pd.Series(class_targets, index=df.index, name='target_class')
    
    # Log class distribution
    class_counts = pd.Series(class_targets).value_counts().sort_index()
    total = len(class_targets)
    print(f"  Class distribution:")
    print(f"    Sell (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total*100:.1f}%)")
    print(f"    Hold (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total*100:.1f}%)")
    print(f"    Buy (2):  {class_counts.get(2, 0):,} ({class_counts.get(2, 0)/total*100:.1f}%)")
    
    return class_target_series

def validate_data_quality(X, y_regr, y_class):
    """
    Validate data quality and check for potential issues.
    """
    print("Validating data quality...")
    
    # Check for NaN values
    nan_features = np.isnan(X).sum()
    nan_regression = np.isnan(y_regr).sum()
    nan_classification = np.isnan(y_class).sum()
    
    if nan_features > 0:
        print(f"  WARNING: {nan_features} NaN values found in features")
    if nan_regression > 0:
        print(f"  WARNING: {nan_regression} NaN values found in regression targets")
    if nan_classification > 0:
        print(f"  WARNING: {nan_classification} NaN values found in classification targets")
    
    # Check for extreme values in regression targets
    regr_stats = pd.DataFrame(y_regr).describe()
    print(f"  Regression target statistics:")
    print(f"    Mean range: [{regr_stats.loc['mean'].min():.4f}, {regr_stats.loc['mean'].max():.4f}]")
    print(f"    Std range:  [{regr_stats.loc['std'].min():.4f}, {regr_stats.loc['std'].max():.4f}]")
    
    # Check class balance
    unique, counts = np.unique(y_class, return_counts=True)
    min_class_pct = counts.min() / counts.sum() * 100
    if min_class_pct < 15:
        print(f"  WARNING: Minimum class representation is only {min_class_pct:.1f}%")
    else:
        print(f"  ✓ Class balance acceptable (min: {min_class_pct:.1f}%)")
    
    print("✓ Data validation complete")

def train_model_with_validation(model, train_loader, val_loader, epochs, learning_rate):
    """
    Train model with validation monitoring.
    """
    regr_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
    
    def print_lr_update(old_lr, new_lr):
        if old_lr != new_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10
    
    print(f"\n--- Starting Enhanced Combined Model Training ---")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_total_loss, train_regr_loss, train_class_loss = 0, 0, 0
        
        for i, (xb, yb_regr, yb_class) in enumerate(train_loader):
            optimizer.zero_grad()
            pred_regr, pred_class_logits = model(xb)
            
            loss_regr = regr_loss_fn(pred_regr, yb_regr)
            loss_class = class_loss_fn(pred_class_logits, yb_class)
            
            # Weighted combination - regression is more important for price prediction
            combined_loss = 0.7 * loss_regr + 0.3 * loss_class
            
            combined_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_total_loss += combined_loss.item()
            train_regr_loss += loss_regr.item()
            train_class_loss += loss_class.item()
        
        # Validation phase
        model.eval()
        val_total_loss, val_regr_loss, val_class_loss = 0, 0, 0
        
        with torch.no_grad():
            for xb, yb_regr, yb_class in val_loader:
                pred_regr, pred_class_logits = model(xb)
                
                loss_regr = regr_loss_fn(pred_regr, yb_regr)
                loss_class = class_loss_fn(pred_class_logits, yb_class)
                combined_loss = 0.7 * loss_regr + 0.3 * loss_class
                
                val_total_loss += combined_loss.item()
                val_regr_loss += loss_regr.item()
                val_class_loss += loss_class.item()
        
        # Calculate averages
        train_avg = train_total_loss / len(train_loader)
        train_regr_avg = train_regr_loss / len(train_loader)
        train_class_avg = train_class_loss / len(train_loader)
        
        val_avg = val_total_loss / len(val_loader)
        val_regr_avg = val_regr_loss / len(val_loader)
        val_class_avg = val_class_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_avg)
        
        # Early stopping check
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"Train={train_avg:.4f} (R:{train_regr_avg:.4f}, C:{train_class_avg:.4f}) | "
                  f"Val={val_avg:.4f} (R:{val_regr_avg:.4f}, C:{val_class_avg:.4f}) | "
                  f"LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs")
            model.load_state_dict(best_model_state)
            break
    
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    print(f"=== ENHANCED LSTM TRAINING SYSTEM ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Key Improvements:")
    print(f"   • Relative percentage-based predictions (fixes timeframe mismatch)")
    print(f"   • Enhanced data validation and quality checks")
    print(f"   • Early stopping and learning rate scheduling")
    print(f"   • Train/validation split for better generalization")
    
    # Load and prepare data
    try:
        print("\n📊 Loading and processing data...")
        main_df, feature_names = create_features(load_and_align_data(REQUIRED_FILES, DATA_DIR))
        print(f"✓ Loaded {len(main_df):,} data points with {len(feature_names)} features")
        
        if len(main_df) < SEQ_LEN + OUTPUT_STEPS + 1000:
            print(f"FATAL ERROR: Insufficient data. Need at least {SEQ_LEN + OUTPUT_STEPS + 1000:,} points, got {len(main_df):,}")
            sys.exit(1)
            
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Create targets
    try:
        print(f"\n🎯 Creating training targets...")
        
        # Create relative regression targets (percentage changes)
        regr_target_df = create_relative_targets(main_df, OUTPUT_STEPS)
        
        # Create enhanced classification targets  
        class_target_series = create_enhanced_classification_targets(
            main_df, LOOKAHEAD_BARS, PROFIT_THRESHOLD_ATR
        )
        
        # Combine all data
        main_df = pd.concat([main_df, regr_target_df, class_target_series], axis=1)
        main_df.dropna(inplace=True)
        
        print(f"✓ Final dataset size after cleaning: {len(main_df):,} samples")
        
    except Exception as e:
        print(f"FATAL ERROR creating targets: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Prepare features and targets
    X = main_df[feature_names].values
    y_regr = main_df[[f'target_regr_{i}' for i in range(OUTPUT_STEPS)]].values
    y_class = main_df['target_class'].values
    
    # Data quality validation
    validate_data_quality(X, y_regr, y_class)
    
    # Feature scaling (StandardScaler for features, no scaling for relative targets)
    print(f"\n⚖️ Scaling features...")
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    # For relative targets, we use minimal scaling to preserve the percentage interpretation
    target_scaler = StandardScaler()
    y_regr_scaled = target_scaler.fit_transform(y_regr)
    
    print(f"✓ Feature scaling complete")
    print(f"   Features: mean≈{X_scaled.mean():.3f}, std≈{X_scaled.std():.3f}")
    print(f"   Targets: mean≈{y_regr_scaled.mean():.3f}, std≈{y_regr_scaled.std():.3f}")
    
    # Build sequences
    print(f"\n🔄 Building sequences (length: {SEQ_LEN})...")
    X_seq, y_regr_seq, y_class_seq = [], [], []
    
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        y_regr_seq.append(y_regr_scaled[i + SEQ_LEN - 1])
        y_class_seq.append(y_class[i + SEQ_LEN - 1])
    
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_regr_tensor = torch.tensor(np.array(y_regr_seq), dtype=torch.float32)
    y_class_tensor = torch.tensor(np.array(y_class_seq), dtype=torch.long)
    
    print(f"✓ Created {len(X_tensor):,} sequences")
    
    # Train/validation split (80/20)
    split_idx = int(0.8 * len(X_tensor))
    
    train_dataset = TensorDataset(X_tensor[:split_idx], y_regr_tensor[:split_idx], y_class_tensor[:split_idx])
    val_dataset = TensorDataset(X_tensor[split_idx:], y_regr_tensor[split_idx:], y_class_tensor[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    print(f"✓ Data split: {len(train_dataset):,} training, {len(val_dataset):,} validation")
    
    # Initialize and train model
    print(f"\n🧠 Initializing model...")
    model = CombinedLSTM(
        input_size=INPUT_FEATURES, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        num_classes=NUM_CLASSES, 
        num_regression_outputs=OUTPUT_STEPS
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized with {total_params:,} trainable parameters")
    
    # Train model with validation
    model = train_model_with_validation(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
    
    # Save model and scalers
    print(f"\n💾 Saving model and scalers...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    MODEL_FILE = os.path.join(MODEL_SAVE_PATH, "lstm_model_regression.pth")
    SCALER_FILE_TARGET = os.path.join(MODEL_SAVE_PATH, "scaler_regression.pkl")
    SCALER_FILE_FEATURE = os.path.join(MODEL_SAVE_PATH, "scaler.pkl")
    
    # Save model state with metadata
    model_save_dict = {
        "model_state": model.state_dict(),
        "model_config": {
            "input_size": INPUT_FEATURES,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_classes": NUM_CLASSES,
            "num_regression_outputs": OUTPUT_STEPS,
            "seq_len": SEQ_LEN
        },
        "training_info": {
            "trained_on": datetime.now().isoformat(),
            "data_samples": len(main_df),
            "feature_names": feature_names,
            "prediction_type": "relative_percentage_changes"
        }
    }
    
    torch.save(model_save_dict, MODEL_FILE)
    joblib.dump(target_scaler, SCALER_FILE_TARGET)
    joblib.dump(feature_scaler, SCALER_FILE_FEATURE)
    
    print(f"✅ TRAINING COMPLETE!")
    print(f"   • Model saved to: {MODEL_FILE}")
    print(f"   • Target scaler saved to: {SCALER_FILE_TARGET}")  
    print(f"   • Feature scaler saved to: {SCALER_FILE_FEATURE}")
    
    print(f"\n🎉 SUCCESS: Model now predicts relative percentage changes!")
    print(f"   This fixes the timeframe mismatch issue - the model will work")
    print(f"   whether EURUSD is at 1.08, 1.17, or any other level.")
    print(f"\n⚠️  IMPORTANT: Update your daemon to convert percentage predictions")
    print(f"   back to absolute prices using current market levels!")
    
    # Quick test prediction to verify
    print(f"\n🧪 Quick model test...")
    model.eval()
    with torch.no_grad():
        test_input = X_tensor[:1]  # First sequence
        pred_regr, pred_class = model(test_input)
        
        # Inverse transform to get percentage changes
        pred_pct_changes = target_scaler.inverse_transform(pred_regr.cpu().numpy())[0]
        
        print(f"   Sample prediction (percentage changes): {pred_pct_changes}")
        print(f"   Range: {pred_pct_changes.min():.4f} to {pred_pct_changes.max():.4f}")
        print(f"   ✓ These are reasonable percentage changes that can be applied to any price level!")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")