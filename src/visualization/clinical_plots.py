"""
Clinical Visualization Framework

Comprehensive visualization module for clinical synthetic data analysis.
Includes EDA plots, model comparisons, and clinical decision support visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ClinicalPlotter:
    """Clinical-focused plotting utilities for synthetic data analysis."""
    
    def __init__(self, style='default', figsize=(12, 8), dpi=100):
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.setup_style()
    
    def setup_style(self):
        """Setup matplotlib style for clinical plots."""
        plt.style.use(self.style)
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_clinical_distributions(self, real_data, synthetic_data, columns=None, 
                                  clinical_ranges=None, save_path=None):
        """Plot distribution comparisons with clinical reference lines."""
        if columns is None:
            columns = real_data.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6 for readability
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot histograms
            real_values = real_data[col].dropna()
            synth_values = synthetic_data[col].dropna()
            
            ax.hist(real_values, bins=30, alpha=0.7, label='Real Data', density=True, color='blue')
            ax.hist(synth_values, bins=30, alpha=0.7, label='Synthetic Data', density=True, color='orange')
            
            # Add clinical reference lines if provided
            if clinical_ranges and col in clinical_ranges:
                ranges = clinical_ranges[col]
                if 'normal_range' in ranges:
                    ax.axvline(ranges['normal_range'][0], color='green', linestyle='--', alpha=0.7, label='Normal Range')
                    ax.axvline(ranges['normal_range'][1], color='green', linestyle='--', alpha=0.7)
                if 'critical_value' in ranges:
                    ax.axvline(ranges['critical_value'], color='red', linestyle=':', alpha=0.7, label='Critical Value')
            
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.set_title(f'{col} Distribution Comparison')
            ax.legend()
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_comparison(self, real_data, synthetic_data, save_path=None):
        """Plot correlation matrix comparison."""
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Real data correlation
        real_corr = real_data[numeric_cols].corr()
        mask = np.triu(np.ones_like(real_corr, dtype=bool))
        sns.heatmap(real_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, ax=ax1, fmt='.2f')
        ax1.set_title('Real Data Correlations')
        
        # Synthetic data correlation
        synth_corr = synthetic_data[numeric_cols].corr()
        sns.heatmap(synth_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, ax=ax2, fmt='.2f')
        ax2.set_title('Synthetic Data Correlations')
        
        # Difference heatmap
        corr_diff = np.abs(real_corr - synth_corr)
        sns.heatmap(corr_diff, mask=mask, annot=True, cmap='Reds', 
                   square=True, linewidths=0.5, ax=ax3, fmt='.2f')
        ax3.set_title('Absolute Correlation Differences')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_target_analysis(self, data, target_column, clinical_labels=None, save_path=None):
        """Plot enhanced target variable analysis with clinical context."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution pie chart
        target_counts = data[target_column].value_counts()
        labels = target_counts.index
        if clinical_labels:
            labels = [clinical_labels.get(label, str(label)) for label in labels]
        
        wedges, texts, autotexts = ax1.pie(target_counts.values, labels=labels, autopct='%1.1f%%',
                                          explode=[0.05]*len(target_counts), shadow=True, startangle=90)
        ax1.set_title(f'{target_column} Distribution', fontsize=14, fontweight='bold')
        
        # Target distribution bar chart
        ax2.bar(range(len(target_counts)), target_counts.values, 
               color=plt.cm.Set3(np.linspace(0, 1, len(target_counts))))
        ax2.set_xticks(range(len(target_counts)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Count')
        ax2.set_title(f'{target_column} Counts', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, v in enumerate(target_counts.values):
            ax2.text(i, v + max(target_counts.values)*0.01, str(v), ha='center', va='bottom')
        
        # Target vs numeric features (correlation analysis)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_column][:5]  # Limit to 5
        
        if len(numeric_cols) > 0:
            correlations = []
            for col in numeric_cols:
                corr = data[target_column].corr(data[col])
                correlations.append(corr)
            
            bars = ax3.barh(numeric_cols, correlations, color='lightblue')
            ax3.set_xlabel('Correlation with Target')
            ax3.set_title(f'Feature Correlations with {target_column}', fontsize=14, fontweight='bold')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add correlation values on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                        f'{correlations[i]:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        # Clinical risk assessment (if applicable)
        if len(target_counts) == 2:  # Binary classification
            positive_rate = target_counts.iloc[1] / target_counts.sum()
            risk_assessment = self.assess_clinical_risk(positive_rate)
            
            ax4.text(0.5, 0.7, 'Clinical Risk Assessment', ha='center', va='center', 
                    fontsize=16, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.5, 0.5, f'Positive Rate: {positive_rate:.1%}', ha='center', va='center',
                    fontsize=14, transform=ax4.transAxes)
            ax4.text(0.5, 0.3, f'Risk Level: {risk_assessment}', ha='center', va='center',
                    fontsize=14, transform=ax4.transAxes)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance_dashboard(self, results_df, save_path=None):
        """Create comprehensive model performance dashboard."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overall Performance Ranking
        ax1 = plt.subplot(3, 3, 1)
        models = results_df['Model'].tolist()
        scores = results_df['Overall_Score'].tolist()
        colors = sns.color_palette("husl", len(models))
        
        bars = ax1.bar(models, scores, color=colors, alpha=0.8)
        ax1.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Composite Score')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Similarity vs Performance Trade-off
        ax2 = plt.subplot(3, 3, 2)
        scatter = ax2.scatter(results_df['Statistical_Similarity'], 
                             results_df['Classification_Ratio'],
                             c=results_df['Overall_Score'], 
                             cmap='viridis', s=100, alpha=0.7)
        
        for i, model in enumerate(models):
            ax2.annotate(model, (results_df['Statistical_Similarity'].iloc[i], 
                               results_df['Classification_Ratio'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Statistical Similarity')
        ax2.set_ylabel('Classification Performance Ratio')
        ax2.set_title('Similarity vs Performance Trade-off', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax2, label='Overall Score')
        
        # 3. Radar Chart for Top 3 Models
        ax3 = plt.subplot(3, 3, 3, projection='polar')
        top_3_models = results_df.head(3)
        
        categories = ['Overall\nScore', 'Statistical\nSimilarity', 'Classification\nRatio', 'Correlation\nPreservation']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, (_, row) in enumerate(top_3_models.iterrows()):
            values = [row['Overall_Score'], row['Statistical_Similarity'], 
                     row['Classification_Ratio'], row.get('Correlation_Preservation', 0)]
            values += values[:1]
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=row['Model'], alpha=0.7)
            ax3.fill(angles, values, alpha=0.1)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories, fontsize=10)
        ax3.set_ylim(0, 1)
        ax3.set_title('Top 3 Models Comparison', fontsize=14, fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 4. Clinical Risk Assessment
        ax4 = plt.subplot(3, 3, 4)
        if 'Clinical_Recommendation' in results_df.columns:
            risk_counts = results_df['Clinical_Recommendation'].value_counts()
            colors_risk = {'Evaluate': 'green', 'Consider': 'orange', 'Caution': 'red', 
                          'Failed': 'darkred', 'Not Available': 'gray'}
            pie_colors = [colors_risk.get(level, 'gray') for level in risk_counts.index]
            
            wedges, texts, autotexts = ax4.pie(risk_counts.values, labels=risk_counts.index,
                                              colors=pie_colors, autopct='%1.0f%%', startangle=90)
            ax4.set_title('Clinical Risk Assessment', fontsize=14, fontweight='bold')
        
        # 5. Performance Metrics Heatmap
        ax5 = plt.subplot(3, 3, (5, 6))
        metrics_cols = ['Overall_Score', 'Statistical_Similarity', 'Classification_Ratio']
        if 'Correlation_Preservation' in results_df.columns:
            metrics_cols.append('Correlation_Preservation')
        
        metrics_data = results_df[['Model'] + metrics_cols].set_index('Model')
        sns.heatmap(metrics_data.T, annot=True, cmap='viridis', fmt='.3f',
                   cbar_kws={'label': 'Score'}, ax=ax5)
        ax5.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        ax5.set_xlabel('')
        
        # 6. TRTS Framework Results (if available)
        ax6 = plt.subplot(3, 3, 7)
        if 'TRTR_Accuracy' in results_df.columns:
            trts_metrics = ['TRTR_Accuracy', 'TSTR_Accuracy', 'TRTS_Accuracy', 'TSTS_Accuracy']
            available_metrics = [col for col in trts_metrics if col in results_df.columns]
            
            if available_metrics:
                x = np.arange(len(models))
                width = 0.2
                
                for i, metric in enumerate(available_metrics):
                    ax6.bar(x + i*width, results_df[metric], width, label=metric.replace('_', ' '))
                
                ax6.set_xlabel('Models')
                ax6.set_ylabel('Accuracy')
                ax6.set_title('TRTS Framework Results', fontsize=14, fontweight='bold')
                ax6.set_xticks(x + width * (len(available_metrics)-1) / 2)
                ax6.set_xticklabels(models, rotation=45, ha='right')
                ax6.legend()
        
        # 7. Computational Efficiency
        ax7 = plt.subplot(3, 3, 8)
        if 'Optimization_Time_min' in results_df.columns:
            ax7.scatter(results_df['Optimization_Time_min'], results_df['Overall_Score'],
                       c=colors, s=100, alpha=0.7)
            
            for i, model in enumerate(models):
                ax7.annotate(model, (results_df['Optimization_Time_min'].iloc[i], 
                                   results_df['Overall_Score'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax7.set_xlabel('Optimization Time (minutes)')
            ax7.set_ylabel('Overall Score')
            ax7.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
        
        # 8. Clinical Utility Summary
        ax8 = plt.subplot(3, 3, 9)
        if len(results_df) > 0:
            best_model = results_df.iloc[0]['Model']
            best_score = results_df.iloc[0]['Overall_Score']
            
            ax8.text(0.5, 0.8, 'Clinical Utility Summary', ha='center', va='center',
                    fontsize=16, fontweight='bold', transform=ax8.transAxes)
            ax8.text(0.5, 0.6, f'Best Model: {best_model}', ha='center', va='center',
                    fontsize=14, transform=ax8.transAxes)
            ax8.text(0.5, 0.4, f'Best Score: {best_score:.3f}', ha='center', va='center',
                    fontsize=14, transform=ax8.transAxes)
            
            if best_score > 0.8:
                recommendation = "Ready for Clinical Use"
                color = 'green'
            elif best_score > 0.6:
                recommendation = "Needs Further Validation"
                color = 'orange'
            else:
                recommendation = "Not Recommended"
                color = 'red'
            
            ax8.text(0.5, 0.2, recommendation, ha='center', va='center',
                    fontsize=14, color=color, fontweight='bold', transform=ax8.transAxes)
            ax8.set_xlim(0, 1)
            ax8.set_ylim(0, 1)
            ax8.axis('off')
        
        plt.tight_layout()
        plt.suptitle('Clinical Synthetic Data Generation - Model Assessment Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def assess_clinical_risk(self, positive_rate):
        """Assess clinical risk based on positive rate."""
        if positive_rate < 0.1:
            return "Low Prevalence"
        elif positive_rate < 0.3:
            return "Moderate Prevalence"
        elif positive_rate < 0.5:
            return "High Prevalence"
        else:
            return "Very High Prevalence"
    
    def plot_feature_importance(self, feature_names, importances, save_path=None):
        """Plot feature importance from clinical perspective."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        indices = np.argsort(importances)[::-1]
        
        ax.bar(range(len(importances)), importances[indices], color='skyblue', alpha=0.8)
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title('Clinical Feature Importance Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(importances[indices]):
            ax.text(i, v + max(importances)*0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()