# Enhanced visualizations
numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COLUMN in numeric_cols:
    numeric_cols.remove(TARGET_COLUMN)

if numeric_cols:
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'Distribution of Features - {DATASET_NAME}', fontsize=16, fontweight='bold')
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Enhanced histogram
            axes[i].hist(original_data[col], bins=30, alpha=0.7, color='skyblue', 
                        edgecolor='black', density=True)
            
            axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save figure if enabled
    if EXPORT_FIGURES:
        plt.savefig(RESULTS_DIR / f'feature_distributions.{FIGURE_FORMAT}', 
                   dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"📊 Figure saved: {RESULTS_DIR / f'feature_distributions.{FIGURE_FORMAT}'}")
    
    plt.show()
    
# Enhanced correlation analysis
if len(numeric_cols) > 1:
    # Include target in correlation if numeric
    cols_for_corr = numeric_cols.copy()
    if original_data[TARGET_COLUMN].dtype in ['int64', 'float64']:
        cols_for_corr.append(TARGET_COLUMN)
    
    correlation_matrix = original_data[cols_for_corr].corr()
    
    # Enhanced correlation heatmap
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r',
                center=0, 
                square=True, 
                linewidths=0.5,
                fmt='.3f')
    
    plt.title(f'Feature Correlation Matrix - {DATASET_NAME}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if EXPORT_FIGURES:
        plt.savefig(RESULTS_DIR / f'correlation_matrix.{FIGURE_FORMAT}', 
                   dpi=FIGURE_DPI, bbox_inches='tight')
    
    plt.show()
    
    # Correlation with target analysis
    if TARGET_COLUMN in correlation_matrix.columns:
        print("\n🔍 CORRELATIONS WITH TARGET VARIABLE")
        print("="*45)
        
        target_corrs = correlation_matrix[TARGET_COLUMN].abs().sort_values(ascending=False)
        target_corrs = target_corrs[target_corrs.index != TARGET_COLUMN]
        
        corr_analysis = pd.DataFrame({
            'Feature': target_corrs.index,
            'Absolute_Correlation': target_corrs.values,
            'Raw_Correlation': [correlation_matrix.loc[feat, TARGET_COLUMN] for feat in target_corrs.index],
            'Strength': ['Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak' 
                        for corr in target_corrs.values]
        })
        
        display(corr_analysis)
        
        if EXPORT_TABLES:
            correlation_matrix.to_csv(RESULTS_DIR / 'correlation_matrix.csv')
            corr_analysis.to_csv(RESULTS_DIR / 'target_correlations.csv', index=False)