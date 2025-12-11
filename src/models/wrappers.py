"""
Model Wrapper Classes

Unified wrapper classes for all generative models, providing consistent
interfaces for training and sampling with robust preprocessing and error handling.

Code migrated from setup.py CHUNK_002, CHUNK_003
"""

import pandas as pd
import numpy as np
import os
import tempfile

# Import model classes from imports module
from src.models.imports import (
    CTABGANSynthesizer,
    CTABGAN_AVAILABLE,
    CTABGANPLUS_AVAILABLE
)

# Import preprocessing function from data module (Phase 2 migration complete)
from src.data.preprocessing import clean_and_preprocess_data


class CTABGANModel:
    def __init__(self, epochs=100, batch_size=500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.loss_history = None  # For training loss tracking
        
    def fit(self, data, categorical_columns=None, target_column=None):
        """Train CTAB-GAN model with robust data preprocessing"""
        try:
            # Check if CTAB-GAN is actually available
            if not CTABGAN_AVAILABLE:
                raise ImportError("CTAB-GAN is not available in this environment")

            # Store original column names for later use in generate()
            self.original_columns = list(data.columns)

            # CRITICAL FIX: Clean and preprocess data before training
            print("[CTABGAN] Applying comprehensive data preprocessing...")
            cleaned_data, categorical_cols, self.encoders = clean_and_preprocess_data(
                data, categorical_columns
            )

            # Store preprocessing info for reverse transformation
            self.categorical_columns = categorical_cols
            print(f"[CTABGAN] Using categorical columns: {categorical_cols}")
            print(f"[CTABGAN] Data shape after preprocessing: {cleaned_data.shape}")

            # Initialize CTAB-GAN with basic parameters only
            self.model = CTABGANSynthesizer(
                epochs=self.epochs,
                batch_size=self.batch_size
            )

            # Train the model with preprocessed data
            print(f"[CTABGAN] Training CTAB-GAN for {self.epochs} epochs...")
            if categorical_cols:
                self.model.fit(cleaned_data, categorical_columns=categorical_cols)
            else:
                self.model.fit(cleaned_data)
            print("[OK] CTAB-GAN training completed successfully")

            # Capture loss history if available
            try:
                if hasattr(self.model, 'loss_values'):
                    self.loss_history = self.model.loss_values
                elif hasattr(self.model, 'loss'):
                    self.loss_history = self.model.loss
                elif hasattr(self.model, '_loss_values'):
                    self.loss_history = self.model._loss_values
                # Note: Some GAN models don't expose loss publicly
            except Exception as e:
                print(f"[INFO] Could not capture loss history: {e}")

        except Exception as e:
            print(f"[ERROR] CTAB-GAN training failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Full error: {str(e)}")
            raise RuntimeError(f"CTAB-GAN training error: {str(e)}")

    def _apply_smart_reverse_encoding(self, synthetic_data):
        """
        Apply smart reverse encoding based on encoding type used during preprocessing.
        Handles binary, one-hot, and label encoding types.
        """
        try:
            for col, encoder_info in self.encoders.items():
                if isinstance(encoder_info, dict):
                    # New smart encoding format
                    encoding_type = encoder_info.get('type', 'label')
                    original_col = encoder_info.get('original_column', col)
                    encoder = encoder_info.get('encoder')

                    if encoding_type == 'binary':
                        # Binary encoding: simple label encoder reverse
                        if original_col in synthetic_data.columns:
                            try:
                                # Ensure values are integers and in valid range
                                synthetic_data[original_col] = synthetic_data[original_col].round().astype(int)
                                valid_range = range(len(encoder.classes_))
                                synthetic_data[original_col] = synthetic_data[original_col].clip(
                                    lower=min(valid_range), upper=max(valid_range)
                                )
                                # Apply inverse transform
                                synthetic_data[original_col] = encoder.inverse_transform(synthetic_data[original_col])
                                print(f"[CTABGAN] Binary reverse encoded column '{original_col}'")
                            except Exception as e:
                                print(f"[WARNING] Binary reverse encoding failed for '{original_col}': {e}")

                    elif encoding_type == 'onehot':
                        # One-hot encoding: reconstruct original categorical column
                        onehot_columns = encoder_info.get('onehot_columns', [])
                        if all(oh_col in synthetic_data.columns for oh_col in onehot_columns):
                            try:
                                # Get one-hot data and find the maximum value column for each row
                                onehot_data = synthetic_data[onehot_columns].values

                                # Convert to probabilities (handle values outside 0-1 range)
                                onehot_data = np.clip(onehot_data, 0, 1)

                                # Find the column with maximum value for each row
                                max_indices = np.argmax(onehot_data, axis=1)

                                # Map back to original categorical values
                                original_values = encoder_info.get('original_values', [])
                                if len(original_values) == len(onehot_columns):
                                    # Use original values if available
                                    reconstructed_values = [original_values[i] for i in max_indices]
                                else:
                                    # Use encoder classes as fallback
                                    reconstructed_values = [encoder.classes_[i] for i in max_indices]

                                # Add reconstructed column and remove one-hot columns
                                synthetic_data[original_col] = reconstructed_values
                                synthetic_data = synthetic_data.drop(columns=onehot_columns)
                                print(f"[CTABGAN] One-hot reverse encoded column '{original_col}' (removed {len(onehot_columns)} one-hot columns)")

                            except Exception as e:
                                print(f"[WARNING] One-hot reverse encoding failed for '{original_col}': {e}")

                    elif encoding_type in ['label', 'label_fallback']:
                        # Label encoding: standard reverse transform
                        if original_col in synthetic_data.columns:
                            try:
                                # Ensure values are integers and in valid range
                                synthetic_data[original_col] = synthetic_data[original_col].round().astype(int)
                                valid_range = range(len(encoder.classes_))
                                synthetic_data[original_col] = synthetic_data[original_col].clip(
                                    lower=min(valid_range), upper=max(valid_range)
                                )
                                # Apply inverse transform
                                synthetic_data[original_col] = encoder.inverse_transform(synthetic_data[original_col])
                                print(f"[CTABGAN] Label reverse encoded column '{original_col}'")
                            except Exception as e:
                                print(f"[WARNING] Label reverse encoding failed for '{original_col}': {e}")

                else:
                    # Legacy encoding format (backward compatibility)
                    if col in synthetic_data.columns:
                        try:
                            # Ensure values are integers for label encoder
                            synthetic_data[col] = synthetic_data[col].round().astype(int)

                            # Handle out-of-range values by clipping to valid range
                            valid_range = range(len(encoder_info.classes_))
                            synthetic_data[col] = synthetic_data[col].clip(
                                lower=min(valid_range), upper=max(valid_range)
                            )

                            # Apply inverse transform
                            synthetic_data[col] = encoder_info.inverse_transform(synthetic_data[col])
                            print(f"[CTABGAN] Legacy reverse encoded column '{col}'")
                        except Exception as enc_error:
                            print(f"[WARNING] Legacy reverse encoding failed for '{col}': {enc_error}")

            return synthetic_data

        except Exception as e:
            print(f"[ERROR] Smart reverse encoding failed: {e}")
            return synthetic_data

    def generate(self, n_samples):
        """Generate synthetic samples with reverse preprocessing"""
        if self.model is None:
            raise ValueError("Model must be fitted before generating samples")

        try:
            # Generate raw synthetic data
            synthetic_data = self.model.sample(n_samples)
            print(f"[CTABGAN] Generated {len(synthetic_data)} raw synthetic samples")

            # Convert to DataFrame if it's a numpy array
            if hasattr(synthetic_data, 'shape') and not hasattr(synthetic_data, 'columns'):
                if hasattr(self, 'original_columns'):
                    synthetic_data = pd.DataFrame(synthetic_data, columns=self.original_columns)
                else:
                    synthetic_data = pd.DataFrame(synthetic_data, columns=[f'feature_{i}' for i in range(synthetic_data.shape[1])])
                    print("[WARNING] Using generic column names - original column names not preserved")

            # Apply smart reverse encoding for categorical columns
            if hasattr(self, 'encoders') and self.encoders:
                print("[CTABGAN] Applying smart reverse encoding to categorical columns...")
                synthetic_data = self._apply_smart_reverse_encoding(synthetic_data)

            print(f"[OK] CTAB-GAN generation completed: {synthetic_data.shape}")
            return synthetic_data

        except Exception as e:
            print(f"[ERROR] CTAB-GAN generation failed: {e}")
            raise RuntimeError(f"CTAB-GAN generation error: {str(e)}")


class CTABGANPlusModel:
    def __init__(self, epochs=100, batch_size=500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.has_plus_features = False
        self.temp_csv_path = None
        self.original_data = None
        self.loss_history = None  # For training loss tracking
        
    def _check_plus_features(self):
        """Check if CTAB-GAN+ features are available"""
        try:
            # Try to import CTAB-GAN+ - correct path and class name
            import sys
            if './CTAB-GAN-Plus' not in sys.path:
                sys.path.append('./CTAB-GAN-Plus')
            from model.ctabgan import CTABGAN
            self.has_plus_features = True
            return CTABGAN
        except ImportError:
            print("WARNING: CTAB-GAN+ features not available, falling back to regular CTAB-GAN parameters")
            self.has_plus_features = False
            return CTABGANSynthesizer
    
    def fit(self, data, categorical_columns=None, target_column=None):
        """Train CTAB-GAN+ model with robust data preprocessing"""
        try:
            # Check for CTAB-GAN+ availability
            CTABGANClass = self._check_plus_features()

            if self.has_plus_features:
                # Store original data for later reference
                self.original_data = data.copy()
                self.original_columns = list(data.columns)

                # CRITICAL FIX: Clean and preprocess data before training
                print("[CTABGAN+] Applying comprehensive data preprocessing...")
                cleaned_data, categorical_cols, self.encoders = clean_and_preprocess_data(
                    data, categorical_columns
                )

                # Store preprocessing info for potential reverse transformation
                self.categorical_columns = categorical_cols
                print(f"[CTABGAN+] Using categorical columns: {categorical_cols}")
                print(f"[CTABGAN+] Data shape after preprocessing: {cleaned_data.shape}")

                # CTAB-GAN+ requires CSV file, so save cleaned DataFrame to temp file
                import tempfile
                import os

                temp_dir = tempfile.mkdtemp()
                self.temp_csv_path = os.path.join(temp_dir, "temp_cleaned_data.csv")
                cleaned_data.to_csv(self.temp_csv_path, index=False)

                # Automatically detect integer columns from cleaned data
                integer_columns = cleaned_data.select_dtypes(include=['int64', 'int32']).columns.tolist()

                # Determine problem type (using main branch logic)
                if target_column and target_column in cleaned_data.columns:
                    target_col = target_column
                else:
                    target_col = cleaned_data.columns[-1]  # Assume last column is target

                # Smart problem_type logic from main branch
                if cleaned_data[target_col].nunique() <= 10:
                    problem_type = {"Classification": target_col}
                    print(f"[CTABGAN+] Using Classification with target: {target_col} ({cleaned_data[target_col].nunique()} unique values)")
                else:
                    problem_type = {None: None}  # Avoid stratification issues for continuous targets
                    print(f"[CTABGAN+] Using regression mode (target has {cleaned_data[target_col].nunique()} unique values)")

                # Use default test_ratio since stratification is handled by problem_type logic
                test_ratio = 0.20

                # Add data validation for CTAB-GAN+ robustness
                print("[CTABGAN+] Validating data for robust training...")

                # Validate data dimensions and content
                if cleaned_data.shape[0] < 100:
                    print(f"[WARNING] Small dataset ({cleaned_data.shape[0]} rows) may cause layer dimension issues")

                if cleaned_data.shape[1] < 2:
                    raise ValueError("Dataset must have at least 2 columns for CTAB-GAN+ training")

                # Validate no infinite or extremely large values that could cause layer issues
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if np.isinf(cleaned_data[col]).any():
                        print(f"[WARNING] Infinite values detected in {col}, replacing with finite values")
                        cleaned_data[col] = cleaned_data[col].replace([np.inf, -np.inf],
                                                                      [cleaned_data[col].max(), cleaned_data[col].min()])

                    # Check for extremely large values that might cause numeric issues
                    max_val = abs(cleaned_data[col]).max()
                    if max_val > 1e6:
                        print(f"[WARNING] Very large values in {col} (max: {max_val}), normalizing...")
                        cleaned_data[col] = cleaned_data[col] / max_val

                # Re-save the validated data
                cleaned_data.to_csv(self.temp_csv_path, index=False)

                # Initialize CTAB-GAN+ with proper parameters using validated data
                print("[CTABGAN+] Initializing CTAB-GAN+ with validated parameters...")
                try:
                    self.model = CTABGANClass(
                        raw_csv_path=self.temp_csv_path,
                        categorical_columns=categorical_cols,
                        integer_columns=integer_columns,
                        problem_type=problem_type,
                        test_ratio=test_ratio
                    )

                    print(f"[CTABGAN+] Training CTAB-GAN+ (Enhanced) for {self.epochs} epochs...")
                    self.model.fit()
                    print("[OK] CTAB-GAN+ training completed successfully")

                    # Capture loss history if available
                    try:
                        if hasattr(self.model, 'loss_values'):
                            self.loss_history = self.model.loss_values
                        elif hasattr(self.model, 'loss'):
                            self.loss_history = self.model.loss
                        elif hasattr(self.model, '_loss_values'):
                            self.loss_history = self.model._loss_values
                    except Exception as e:
                        print(f"[INFO] Could not capture loss history: {e}")

                except Exception as model_error:
                    print(f"[ERROR] CTAB-GAN+ initialization failed: {model_error}")
                    if "NoneType" in str(model_error) and "int" in str(model_error):
                        print("[RECOVERY] Attempting fallback with adjusted parameters...")
                        # Try with simpler categorical configuration
                        try:
                            self.model = CTABGANClass(
                                raw_csv_path=self.temp_csv_path,
                                categorical_columns=[],  # Remove categorical columns as fallback
                                integer_columns=integer_columns,
                                problem_type={None: None},  # Simplify problem type
                                test_ratio=0.15  # Use smaller test ratio
                            )
                            self.model.fit()
                            print("[OK] CTAB-GAN+ training completed with fallback parameters")

                            # Capture loss history if available
                            try:
                                if hasattr(self.model, 'loss_values'):
                                    self.loss_history = self.model.loss_values
                                elif hasattr(self.model, 'loss'):
                                    self.loss_history = self.model.loss
                                elif hasattr(self.model, '_loss_values'):
                                    self.loss_history = self.model._loss_values
                            except Exception as e:
                                print(f"[INFO] Could not capture loss history: {e}")
                        except Exception as fallback_error:
                            print(f"[ERROR] Fallback also failed: {fallback_error}")
                            raise model_error
                    else:
                        raise model_error
                
            else:
                # Fallback to regular CTAB-GAN with preprocessing
                print("[CTABGAN+] Falling back to regular CTAB-GAN with preprocessing...")
                self.original_columns = list(data.columns)

                # Apply same preprocessing for consistency
                cleaned_data, categorical_cols, self.encoders = clean_and_preprocess_data(
                    data, categorical_columns
                )
                self.categorical_columns = categorical_cols

                self.model = CTABGANClass(
                    epochs=self.epochs,
                    batch_size=self.batch_size
                )

                if categorical_cols:
                    self.model.fit(cleaned_data, categorical_columns=categorical_cols)
                else:
                    self.model.fit(cleaned_data)
                print("[OK] CTAB-GAN (fallback) training completed successfully")

                # Capture loss history if available
                try:
                    if hasattr(self.model, 'loss_values'):
                        self.loss_history = self.model.loss_values
                    elif hasattr(self.model, 'loss'):
                        self.loss_history = self.model.loss
                    elif hasattr(self.model, '_loss_values'):
                        self.loss_history = self.model._loss_values
                except Exception as e:
                    print(f"[INFO] Could not capture loss history: {e}")

        except Exception as e:
            print(f"[ERROR] CTAB-GAN+ training failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Full error: {str(e)}")
            # Clean up temp file on error
            if hasattr(self, 'temp_csv_path') and self.temp_csv_path and os.path.exists(self.temp_csv_path):
                os.remove(self.temp_csv_path)
            raise RuntimeError(f"CTAB-GAN+ training error: {str(e)}")

    def _apply_smart_reverse_encoding(self, synthetic_data):
        """
        Apply smart reverse encoding based on encoding type used during preprocessing.
        Handles binary, one-hot, and label encoding types.
        """
        try:
            for col, encoder_info in self.encoders.items():
                if isinstance(encoder_info, dict):
                    # New smart encoding format
                    encoding_type = encoder_info.get('type', 'label')
                    original_col = encoder_info.get('original_column', col)
                    encoder = encoder_info.get('encoder')

                    if encoding_type == 'binary':
                        # Binary encoding: simple label encoder reverse
                        if original_col in synthetic_data.columns:
                            try:
                                # Ensure values are integers and in valid range
                                synthetic_data[original_col] = synthetic_data[original_col].round().astype(int)
                                valid_range = range(len(encoder.classes_))
                                synthetic_data[original_col] = synthetic_data[original_col].clip(
                                    lower=min(valid_range), upper=max(valid_range)
                                )
                                # Apply inverse transform
                                synthetic_data[original_col] = encoder.inverse_transform(synthetic_data[original_col])
                                print(f"[CTABGAN+] Binary reverse encoded column '{original_col}'")
                            except Exception as e:
                                print(f"[WARNING] Binary reverse encoding failed for '{original_col}': {e}")

                    elif encoding_type == 'onehot':
                        # One-hot encoding: reconstruct original categorical column
                        onehot_columns = encoder_info.get('onehot_columns', [])
                        if all(oh_col in synthetic_data.columns for oh_col in onehot_columns):
                            try:
                                # Get one-hot data and find the maximum value column for each row
                                onehot_data = synthetic_data[onehot_columns].values

                                # Convert to probabilities (handle values outside 0-1 range)
                                onehot_data = np.clip(onehot_data, 0, 1)

                                # Find the column with maximum value for each row
                                max_indices = np.argmax(onehot_data, axis=1)

                                # Map back to original categorical values
                                original_values = encoder_info.get('original_values', [])
                                if len(original_values) == len(onehot_columns):
                                    # Use original values if available
                                    reconstructed_values = [original_values[i] for i in max_indices]
                                else:
                                    # Use encoder classes as fallback
                                    reconstructed_values = [encoder.classes_[i] for i in max_indices]

                                # Add reconstructed column and remove one-hot columns
                                synthetic_data[original_col] = reconstructed_values
                                synthetic_data = synthetic_data.drop(columns=onehot_columns)
                                print(f"[CTABGAN+] One-hot reverse encoded column '{original_col}' (removed {len(onehot_columns)} one-hot columns)")

                            except Exception as e:
                                print(f"[WARNING] One-hot reverse encoding failed for '{original_col}': {e}")

                    elif encoding_type in ['label', 'label_fallback']:
                        # Label encoding: standard reverse transform
                        if original_col in synthetic_data.columns:
                            try:
                                # Ensure values are integers and in valid range
                                synthetic_data[original_col] = synthetic_data[original_col].round().astype(int)
                                valid_range = range(len(encoder.classes_))
                                synthetic_data[original_col] = synthetic_data[original_col].clip(
                                    lower=min(valid_range), upper=max(valid_range)
                                )
                                # Apply inverse transform
                                synthetic_data[original_col] = encoder.inverse_transform(synthetic_data[original_col])
                                print(f"[CTABGAN+] Label reverse encoded column '{original_col}'")
                            except Exception as e:
                                print(f"[WARNING] Label reverse encoding failed for '{original_col}': {e}")

                else:
                    # Legacy encoding format (backward compatibility)
                    if col in synthetic_data.columns:
                        try:
                            # Ensure values are integers for label encoder
                            synthetic_data[col] = synthetic_data[col].round().astype(int)

                            # Handle out-of-range values by clipping to valid range
                            valid_range = range(len(encoder_info.classes_))
                            synthetic_data[col] = synthetic_data[col].clip(
                                lower=min(valid_range), upper=max(valid_range)
                            )

                            # Apply inverse transform
                            synthetic_data[col] = encoder_info.inverse_transform(synthetic_data[col])
                            print(f"[CTABGAN+] Legacy reverse encoded column '{col}'")
                        except Exception as enc_error:
                            print(f"[WARNING] Legacy reverse encoding failed for '{col}': {enc_error}")

            return synthetic_data

        except Exception as e:
            print(f"[ERROR] Smart reverse encoding failed: {e}")
            return synthetic_data

    def generate(self, n_samples):
        """Generate synthetic samples using CTAB-GAN+ with reverse preprocessing"""
        if self.model is None:
            raise ValueError("Model must be fitted before generating samples")

        try:
            if self.has_plus_features:
                # CTAB-GAN+ generates all samples at once
                synthetic_data = self.model.generate_samples()
                print(f"[CTABGAN+] Generated {len(synthetic_data)} raw synthetic samples")

                # If we need fewer samples, take a random subset
                if len(synthetic_data) > n_samples:
                    synthetic_data = synthetic_data.sample(n=n_samples, random_state=42).reset_index(drop=True)
                elif len(synthetic_data) < n_samples:
                    print(f"[WARNING] CTAB-GAN+ generated {len(synthetic_data)} samples, requested {n_samples}")

            else:
                # Use regular CTAB-GAN generation with preprocessing support
                synthetic_data = self.model.sample(n_samples)
                print(f"[CTABGAN+] Generated {len(synthetic_data)} raw synthetic samples (fallback mode)")

                # Convert to DataFrame if it's a numpy array (fallback mode)
                if hasattr(synthetic_data, 'shape') and not hasattr(synthetic_data, 'columns'):
                    if hasattr(self, 'original_columns'):
                        synthetic_data = pd.DataFrame(synthetic_data, columns=self.original_columns)
                    else:
                        synthetic_data = pd.DataFrame(synthetic_data, columns=[f'feature_{i}' for i in range(synthetic_data.shape[1])])

            # Apply smart reverse encoding for categorical columns
            if hasattr(self, 'encoders') and self.encoders:
                print("[CTABGAN+] Applying smart reverse encoding to categorical columns...")
                synthetic_data = self._apply_smart_reverse_encoding(synthetic_data)

            print(f"[OK] CTAB-GAN+ generation completed: {synthetic_data.shape}")

            # Clean up temp file after successful generation
            if hasattr(self, 'temp_csv_path') and self.temp_csv_path and os.path.exists(self.temp_csv_path):
                os.remove(self.temp_csv_path)

            return synthetic_data

        except Exception as e:
            print(f"[ERROR] CTAB-GAN+ generation failed: {e}")
            # Clean up temp file on error
            if hasattr(self, 'temp_csv_path') and self.temp_csv_path and os.path.exists(self.temp_csv_path):
                os.remove(self.temp_csv_path)
            raise RuntimeError(f"CTAB-GAN+ generation error: {str(e)}")
