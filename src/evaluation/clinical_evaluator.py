"""
Clinical Synthetic Data Evaluation Framework

Comprehensive evaluation module for clinical synthetic data generation models.
Includes statistical similarity, classification performance, and clinical utility metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
import warnings
warnings.filterwarnings('ignore')


class ClinicalModelEvaluator:
    """Comprehensive evaluation for clinical synthetic data models."""
    
    def __init__(self, real_data, target_column=None, random_state=42):
        self.real_data = real_data.copy()
        self.target_column = target_column
        self.random_state = random_state
        self.numeric_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in self.numeric_columns:
            self.numeric_columns.remove(target_column)
    
    def evaluate_similarity(self, synthetic_data):
        """Calculate comprehensive statistical similarity metrics."""
        similarities = {}
        
        # Univariate similarities
        univariate_scores = []
        column_details = {}
        
        for col in self.numeric_columns:
            if col in synthetic_data.columns:
                real_col = self.real_data[col].dropna()
                synth_col = synthetic_data[col].dropna()
                
                if len(real_col) == 0 or len(synth_col) == 0:
                    continue
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = ks_2samp(real_col, synth_col)
                
                # Wasserstein distance (normalized)
                ws_dist = wasserstein_distance(real_col, synth_col)
                max_range = max(real_col.max() - real_col.min(), synth_col.max() - synth_col.min())
                ws_normalized = ws_dist / max_range if max_range > 0 else 0
                
                # Jensen-Shannon divergence
                try:
                    hist_real, bins = np.histogram(real_col, bins=50, density=True)
                    hist_synth, _ = np.histogram(synth_col, bins=bins, density=True)
                    hist_real = hist_real / hist_real.sum() if hist_real.sum() > 0 else hist_real
                    hist_synth = hist_synth / hist_synth.sum() if hist_synth.sum() > 0 else hist_synth
                    js_div = jensenshannon(hist_real, hist_synth)
                except:
                    js_div = 1.0
                
                # Composite similarity score
                similarity_score = (1 - ks_stat) * (1 - ws_normalized) * (1 - js_div)
                univariate_scores.append(similarity_score)
                
                column_details[col] = {
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pval,
                    'wasserstein_distance': ws_dist,
                    'jensen_shannon_divergence': js_div,
                    'similarity_score': similarity_score
                }
        
        similarities['univariate_avg'] = np.mean(univariate_scores) if univariate_scores else 0
        similarities['column_details'] = column_details
        
        # Correlation similarity
        try:
            real_corr = self.real_data[self.numeric_columns].corr()
            synth_corr = synthetic_data[self.numeric_columns].corr()
            
            # Handle NaN values
            real_corr = real_corr.fillna(0)
            synth_corr = synth_corr.fillna(0)
            
            corr_diff = np.abs(real_corr - synth_corr).mean().mean()
            similarities['correlation'] = max(0, 1 - corr_diff)
        except:
            similarities['correlation'] = 0
        
        # Overall similarity (weighted average)
        similarities['overall'] = (
            0.7 * similarities['univariate_avg'] + 
            0.3 * similarities['correlation']
        )
        
        return similarities
    
    def evaluate_classification(self, synthetic_data):
        """Evaluate classification performance using TRTS framework."""
        if not self.target_column or self.target_column not in synthetic_data.columns:
            return {'accuracy_ratio': 0, 'f1_ratio': 0, 'trtr_accuracy': 0, 'tstr_accuracy': 0}
        
        try:
            # Prepare data
            feature_cols = [col for col in self.numeric_columns if col in synthetic_data.columns]
            
            X_real = self.real_data[feature_cols]
            y_real = self.real_data[self.target_column]
            X_synth = synthetic_data[feature_cols]
            y_synth = synthetic_data[self.target_column]
            
            # Handle missing values
            X_real = X_real.fillna(X_real.median())
            X_synth = X_synth.fillna(X_synth.median())
            
            # Train/test splits
            X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
                X_real, y_real, test_size=0.3, random_state=self.random_state, stratify=y_real
            )
            X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
                X_synth, y_synth, test_size=0.3, random_state=self.random_state
            )
            
            # Standardize features
            scaler = StandardScaler()
            X_real_train_scaled = scaler.fit_transform(X_real_train)
            X_real_test_scaled = scaler.transform(X_real_test)
            X_synth_train_scaled = scaler.transform(X_synth_train)
            X_synth_test_scaled = scaler.transform(X_synth_test)
            
            # Train models
            rf = RandomForestClassifier(random_state=self.random_state, n_estimators=50)
            
            # TRTR (baseline)
            rf.fit(X_real_train_scaled, y_real_train)
            y_pred_trtr = rf.predict(X_real_test_scaled)
            trtr_acc = accuracy_score(y_real_test, y_pred_trtr)
            trtr_f1 = f1_score(y_real_test, y_pred_trtr, average='weighted')
            
            # TSTR (synthetic training)
            rf.fit(X_synth_train_scaled, y_synth_train)
            y_pred_tstr = rf.predict(X_real_test_scaled)
            tstr_acc = accuracy_score(y_real_test, y_pred_tstr)
            tstr_f1 = f1_score(y_real_test, y_pred_tstr, average='weighted')
            
            # TRTS (real training, synthetic testing)
            rf.fit(X_real_train_scaled, y_real_train)
            y_pred_trts = rf.predict(X_synth_test_scaled)
            trts_acc = accuracy_score(y_synth_test, y_pred_trts)
            trts_f1 = f1_score(y_synth_test, y_pred_trts, average='weighted')
            
            # TSTS (synthetic training, synthetic testing)
            rf.fit(X_synth_train_scaled, y_synth_train)
            y_pred_tsts = rf.predict(X_synth_test_scaled)
            tsts_acc = accuracy_score(y_synth_test, y_pred_tsts)
            tsts_f1 = f1_score(y_synth_test, y_pred_tsts, average='weighted')
            
            # Calculate ratios
            acc_ratio = tstr_acc / trtr_acc if trtr_acc > 0 else 0
            f1_ratio = tstr_f1 / trtr_f1 if trtr_f1 > 0 else 0
            
            return {
                'accuracy_ratio': acc_ratio,
                'f1_ratio': f1_ratio,
                'trtr_accuracy': trtr_acc,
                'tstr_accuracy': tstr_acc,
                'trts_accuracy': trts_acc,
                'tsts_accuracy': tsts_acc,
                'trtr_f1': trtr_f1,
                'tstr_f1': tstr_f1,
                'trts_f1': trts_f1,
                'tsts_f1': tsts_f1
            }
        except Exception as e:
            print(f"Classification evaluation error: {e}")
            return {'accuracy_ratio': 0, 'f1_ratio': 0, 'trtr_accuracy': 0, 'tstr_accuracy': 0}
    
    def evaluate_clinical_utility(self, synthetic_data):
        """Evaluate clinical utility and regulatory compliance."""
        utility_scores = {}
        
        # Data completeness
        completeness = 1 - (synthetic_data.isnull().sum().sum() / synthetic_data.size)
        utility_scores['completeness'] = completeness
        
        # Distribution preservation
        similarity = self.evaluate_similarity(synthetic_data)
        utility_scores['distribution_preservation'] = similarity['overall']
        
        # Classification utility
        classification = self.evaluate_classification(synthetic_data)
        utility_scores['classification_utility'] = classification['accuracy_ratio']
        
        # Privacy assessment (simplified)
        privacy_score = self.assess_privacy_risk(synthetic_data)
        utility_scores['privacy_protection'] = privacy_score
        
        # Overall clinical utility
        utility_scores['overall_utility'] = np.mean([
            utility_scores['completeness'],
            utility_scores['distribution_preservation'],
            utility_scores['classification_utility'],
            utility_scores['privacy_protection']
        ])
        
        return utility_scores
    
    def assess_privacy_risk(self, synthetic_data):
        """Simplified privacy risk assessment."""
        # This is a basic implementation - in practice, more sophisticated privacy metrics would be used
        try:
            # Check for exact matches (simplified)
            exact_matches = 0
            for idx, synth_row in synthetic_data.iterrows():
                for _, real_row in self.real_data.iterrows():
                    if (synth_row[self.numeric_columns] == real_row[self.numeric_columns]).all():
                        exact_matches += 1
                        break
            
            privacy_score = 1 - (exact_matches / len(synthetic_data))
            return max(0, privacy_score)
        except:
            return 0.5  # Neutral score if assessment fails
    
    def generate_clinical_report(self, synthetic_data, model_name="Unknown"):
        """Generate comprehensive clinical evaluation report."""
        similarity = self.evaluate_similarity(synthetic_data)
        classification = self.evaluate_classification(synthetic_data)
        utility = self.evaluate_clinical_utility(synthetic_data)
        
        report = {
            'model_name': model_name,
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_size': {
                'real_samples': len(self.real_data),
                'synthetic_samples': len(synthetic_data),
                'features': len(self.numeric_columns)
            },
            'similarity_metrics': similarity,
            'classification_metrics': classification,
            'clinical_utility': utility,
            'regulatory_assessment': self.assess_regulatory_compliance(similarity, classification, utility),
            'clinical_recommendations': self.generate_clinical_recommendations(similarity, classification, utility)
        }
        
        return report
    
    def assess_regulatory_compliance(self, similarity, classification, utility):
        """Assess regulatory compliance readiness."""
        assessment = {}
        
        # Statistical adequacy
        statistical_adequacy = "High" if similarity['overall'] > 0.8 else "Medium" if similarity['overall'] > 0.6 else "Low"
        assessment['statistical_adequacy'] = statistical_adequacy
        
        # Predictive validity
        predictive_validity = "High" if classification['accuracy_ratio'] > 0.9 else "Medium" if classification['accuracy_ratio'] > 0.7 else "Low"
        assessment['predictive_validity'] = predictive_validity
        
        # Privacy protection
        privacy_protection = "High" if utility['privacy_protection'] > 0.8 else "Medium" if utility['privacy_protection'] > 0.6 else "Low"
        assessment['privacy_protection'] = privacy_protection
        
        # Overall readiness
        high_scores = sum(1 for score in [statistical_adequacy, predictive_validity, privacy_protection] if score == "High")
        if high_scores >= 2:
            assessment['regulatory_readiness'] = "Ready"
        elif high_scores >= 1:
            assessment['regulatory_readiness'] = "Needs Review"
        else:
            assessment['regulatory_readiness'] = "Not Ready"
        
        return assessment
    
    def generate_clinical_recommendations(self, similarity, classification, utility):
        """Generate clinical use case recommendations."""
        recommendations = {}
        
        overall_score = utility['overall_utility']
        
        if overall_score > 0.8:
            recommendations['primary_use_cases'] = ["Protocol Design", "Sample Size Estimation", "Regulatory Submission"]
            recommendations['risk_level'] = "Low"
        elif overall_score > 0.6:
            recommendations['primary_use_cases'] = ["Exploratory Analysis", "Method Development"]
            recommendations['risk_level'] = "Medium"
        elif overall_score > 0.4:
            recommendations['primary_use_cases'] = ["Internal Testing", "Proof of Concept"]
            recommendations['risk_level'] = "High"
        else:
            recommendations['primary_use_cases'] = ["Not Recommended"]
            recommendations['risk_level'] = "Very High"
        
        # Specific recommendations
        if similarity['overall'] < 0.6:
            recommendations['warnings'] = ["Low statistical similarity - review model parameters"]
        if classification['accuracy_ratio'] < 0.7:
            recommendations['warnings'] = recommendations.get('warnings', []) + ["Low predictive utility - may not preserve relationships"]
        if utility['privacy_protection'] < 0.7:
            recommendations['warnings'] = recommendations.get('warnings', []) + ["Potential privacy concerns - conduct detailed privacy assessment"]
        
        return recommendations