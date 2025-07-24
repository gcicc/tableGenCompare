#!/usr/bin/env python3
"""
Clinical Synthetic Data Generation Demo
Run this script to see all visualizations without Jupyter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')

# Import our framework components
from preprocessing.clinical_preprocessor import ClinicalDataPreprocessor
from models.mock_models import MockCTGANModel, MockTVAEModel, MockCopulaGANModel, MockGANerAidModel

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Set random seeds for reproducibility
np.random.seed(42)

def main():
    print("🏥 Clinical Synthetic Data Generation Demo")
    print("=" * 60)
    
    # Load the liver disease dataset
    DATA_FILE = "doc/liver_train.csv"
    TARGET_COLUMN = "result"
    
    try:
        df_original = pd.read_csv(DATA_FILE, encoding='ISO-8859-1')
        print(f"✅ Dataset loaded: {df_original.shape}")
        
        # Find target column
        target_candidates = ['Result', 'result', 'RESULT']
        target_col_found = None
        for candidate in target_candidates:
            if candidate in df_original.columns:
                target_col_found = candidate
                break
        
        if target_col_found:
            TARGET_COLUMN = target_col_found
        else:
            TARGET_COLUMN = df_original.columns[-1]
        
        print(f"🎯 Target column: {TARGET_COLUMN}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {DATA_FILE}")
        return
    
    # Basic dataset visualization
    print("\n📊 Creating initial data visualizations...")
    
    # 1. Dataset overview
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)
    
    if numeric_cols:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('Original Data - Numeric Feature Distributions', fontsize=16)
        
        if n_rows * n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                df_original[col].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    # 2. Correlation heatmap
    if len(numeric_cols) > 1:
        cols_for_corr = numeric_cols.copy()
        if df_original[TARGET_COLUMN].dtype in ['int64', 'float64']:
            cols_for_corr.append(TARGET_COLUMN)
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = df_original[cols_for_corr].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, fmt='.2f')
        plt.title('Correlation Matrix of Numeric Variables', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # 3. Preprocessing
    print("\n🔧 Applying preprocessing...")
    preprocessor = ClinicalDataPreprocessor(random_state=42)
    df_processed = preprocessor.fit_transform(df_original, target_col=TARGET_COLUMN)
    print(f"✅ Processed shape: {df_processed.shape}")
    
    discrete_columns = preprocessor.get_discrete_columns(df_processed)
    
    # 4. Generate synthetic data
    print("\n🤖 Generating synthetic data...")
    models = {
        'CTGAN': MockCTGANModel(random_state=42),
        'TVAE': MockTVAEModel(random_state=42),
        'CopulaGAN': MockCopulaGANModel(random_state=42),
        'GANerAid': MockGANerAidModel(random_state=42)
    }
    
    # Use subset for demo
    n_samples = min(1000, len(df_processed))
    df_demo = df_processed.sample(n=n_samples, random_state=42)
    
    synthetic_datasets = {}
    model_performance = []
    
    for model_name, model in models.items():
        try:
            # Create mock parameters
            param_space = model.get_param_space()
            mock_params = {}
            for param, config in param_space.items():
                if config[0] == 'int':
                    mock_params[param] = config[1] + (config[2] - config[1]) // 2
                elif config[0] == 'float':
                    mock_params[param] = (config[1] + config[2]) / 2
                elif config[0] == 'categorical':
                    mock_params[param] = config[1][0]
            
            # Train and generate
            model.model = model.create_model(mock_params)
            model.fit(df_demo, discrete_columns)
            synthetic_data = model.generate(len(df_demo))
            synthetic_datasets[model_name] = synthetic_data
            
            model_info = model.get_model_info()
            model_performance.append({
                'Model': model_name,
                'Training_Time': f"{model_info['training_time']:.3f}s",
                'Generation_Time': f"{model_info['generation_time']:.3f}s",
                'Samples': len(synthetic_data)
            })
            
            print(f"✅ {model_name}: {len(synthetic_data)} samples generated")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    # 5. Compare distributions
    print("\n📊 Creating distribution comparisons...")
    numeric_cols_for_plot = [col for col in df_demo.columns if df_demo[col].dtype in ['int64', 'float64']]
    numeric_cols_for_plot = [col for col in numeric_cols_for_plot if col != TARGET_COLUMN.lower()][:4]
    
    if numeric_cols_for_plot and synthetic_datasets:
        n_cols = len(numeric_cols_for_plot)
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols))
        if n_cols == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(synthetic_datasets) + 1))
        
        for i, col in enumerate(numeric_cols_for_plot):
            # Plot original data
            axes[i].hist(df_demo[col], alpha=0.6, density=True, 
                        label='Original', color=colors[0], bins=30, edgecolor='black')
            
            # Plot synthetic data
            for j, (model_name, synth_data) in enumerate(synthetic_datasets.items()):
                if col in synth_data.columns:
                    axes[i].hist(synth_data[col], alpha=0.6, density=True, 
                               label=f'{model_name}', 
                               color=colors[j+1], bins=30, histtype='step', linewidth=2)
            
            axes[i].set_title(f'Distribution Comparison: {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Original vs Synthetic Data Distributions', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # 6. Mock evaluation results
    print("\n📈 Creating mock evaluation results...")
    mock_results = []
    
    for i, model_name in enumerate(synthetic_datasets.keys()):
        base_similarity = 0.75 + np.random.normal(0, 0.1)
        base_utility = 0.80 + np.random.normal(0, 0.08)
        
        mock_result = {
            'Model': model_name,
            'Overall_Similarity': np.clip(base_similarity, 0.5, 0.95),
            'Average_Utility': np.clip(base_utility, 0.6, 0.95),
            'Training_Time_Sec': np.random.uniform(2, 15),
            'Combined_Score': 0.6 * np.clip(base_similarity, 0.5, 0.95) + 0.4 * np.clip(base_utility, 0.6, 0.95)
        }
        mock_results.append(mock_result)
    
    mock_results_df = pd.DataFrame(mock_results)
    mock_results_df = mock_results_df.sort_values('Combined_Score', ascending=False)
    
    # 7. Final comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Clinical Synthetic Data Generation - Model Comparison', fontsize=16)
    
    # Combined scores
    bars = axes[0, 0].bar(mock_results_df['Model'], mock_results_df['Combined_Score'], 
                         color=['gold', 'silver', '#CD7F32', 'lightblue'][:len(mock_results_df)])
    axes[0, 0].set_title('Overall Model Performance')
    axes[0, 0].set_ylabel('Combined Score')
    axes[0, 0].set_ylim(0, 1)
    
    for bar, value in zip(bars, mock_results_df['Combined_Score']):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Similarity vs Utility
    axes[0, 1].scatter(mock_results_df['Overall_Similarity'], mock_results_df['Average_Utility'], 
                      s=150, alpha=0.7, c=range(len(mock_results_df)), cmap='viridis')
    
    for i, model in enumerate(mock_results_df['Model']):
        axes[0, 1].annotate(model, 
                           (mock_results_df.iloc[i]['Overall_Similarity'], 
                            mock_results_df.iloc[i]['Average_Utility']),
                           xytext=(5, 5), textcoords='offset points')
    
    axes[0, 1].set_xlabel('Overall Similarity')
    axes[0, 1].set_ylabel('Average Utility')
    axes[0, 1].set_title('Similarity vs Utility Trade-off')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    
    # Training times
    axes[1, 0].bar(mock_results_df['Model'], mock_results_df['Training_Time_Sec'])
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].set_ylabel('Time (seconds)')
    
    # Performance summary table (as text)
    axes[1, 1].axis('off')
    table_text = "Model Performance Summary:\n\n"
    for i, (_, row) in enumerate(mock_results_df.iterrows()):
        table_text += f"{i+1}. {row['Model']}: {row['Combined_Score']:.3f}\n"
    
    axes[1, 1].text(0.1, 0.9, table_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Rankings')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 60)
    print(f"📊 Dataset: {df_original.shape[0]} samples, {df_original.shape[1]} features")
    print(f"🤖 Models: {len(synthetic_datasets)} synthetic data generators")
    print(f"📈 Visualizations: Distribution comparisons, correlation analysis, performance metrics")
    print(f"🏆 Best model: {mock_results_df.iloc[0]['Model']} (Score: {mock_results_df.iloc[0]['Combined_Score']:.3f})")
    
    print("\n💡 To see more detailed analysis, run the Jupyter notebook:")
    print("   jupyter notebook notebooks/clinical_synth_demo.ipynb")

if __name__ == "__main__":
    main()