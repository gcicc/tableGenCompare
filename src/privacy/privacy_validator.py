#!/usr/bin/env python3
"""
Data privacy and compliance validation system.
Ensures synthetic data meets privacy requirements and regulatory compliance.
"""

import logging
import json
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import warnings

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PrivacyLevel(Enum):
    """Privacy protection levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    FERPA = "ferpa"

@dataclass
class PrivacyMetric:
    """Individual privacy metric result."""
    name: str
    value: float
    threshold: float
    passed: bool
    severity: str
    description: str
    recommendation: str

@dataclass
class ComplianceResult:
    """Compliance validation result."""
    framework: str
    overall_score: float
    passed: bool
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    risk_level: str

@dataclass
class PrivacyReport:
    """Comprehensive privacy validation report."""
    
    # Overall assessment
    overall_privacy_score: float
    privacy_level: str
    risk_assessment: str
    
    # Individual metrics
    anonymity_metrics: Dict[str, PrivacyMetric]
    utility_preservation: Dict[str, float]
    disclosure_risks: Dict[str, float]
    
    # Compliance results
    compliance_results: Dict[str, ComplianceResult]
    
    # Recommendations
    recommendations: List[str]
    required_actions: List[str]
    
    # Metadata
    validation_timestamp: str
    dataset_fingerprint: str

class PrivacyValidator:
    """Comprehensive privacy validation system."""
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM):
        self.privacy_level = privacy_level
        self.thresholds = self._initialize_thresholds()
        self.compliance_rules = self._initialize_compliance_rules()
        
        logger.info(f"PrivacyValidator initialized with {privacy_level.value} privacy level")
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize privacy metric thresholds based on privacy level."""
        
        base_thresholds = {
            "k_anonymity": {"low": 2.0, "medium": 5.0, "high": 10.0, "critical": 25.0},
            "l_diversity": {"low": 1.5, "medium": 2.0, "high": 3.0, "critical": 5.0},
            "t_closeness": {"low": 0.2, "medium": 0.15, "high": 0.1, "critical": 0.05},
            "distance_to_closest": {"low": 0.01, "medium": 0.05, "high": 0.1, "critical": 0.2},
            "membership_inference": {"low": 0.6, "medium": 0.55, "high": 0.52, "critical": 0.51},
            "attribute_inference": {"low": 0.7, "medium": 0.6, "high": 0.55, "critical": 0.52},
            "mutual_information": {"low": 0.5, "medium": 0.3, "high": 0.2, "critical": 0.1}
        }
        
        level = self.privacy_level.value
        return {metric: values[level] for metric, values in base_thresholds.items()}
    
    def _initialize_compliance_rules(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Initialize compliance framework rules."""
        
        return {
            ComplianceFramework.GDPR: {
                "required_anonymization": True,
                "min_k_anonymity": 5,
                "pseudonymization_required": True,
                "right_to_erasure": True,
                "data_minimization": True,
                "purpose_limitation": True
            },
            ComplianceFramework.HIPAA: {
                "required_anonymization": True,
                "min_k_anonymity": 10,
                "safe_harbor_method": True,
                "expert_determination": True,
                "phi_removal": True,
                "limited_data_set": False
            },
            ComplianceFramework.CCPA: {
                "required_anonymization": False,
                "deidentification_required": True,
                "consumer_rights": True,
                "data_sale_restrictions": True,
                "opt_out_required": True
            },
            ComplianceFramework.PCI_DSS: {
                "cardholder_data_protection": True,
                "encryption_required": True,
                "access_control": True,
                "network_security": True,
                "vulnerability_management": True
            }
        }
    
    def validate_privacy(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_columns: Optional[List[str]] = None,
        quasi_identifiers: Optional[List[str]] = None,
        compliance_frameworks: Optional[List[ComplianceFramework]] = None
    ) -> PrivacyReport:
        """
        Comprehensive privacy validation of synthetic data.
        
        Args:
            original_data: Original sensitive dataset
            synthetic_data: Generated synthetic dataset
            sensitive_columns: List of sensitive attribute columns
            quasi_identifiers: List of quasi-identifier columns
            compliance_frameworks: Compliance frameworks to validate against
            
        Returns:
            Comprehensive privacy report
        """
        
        logger.info("Starting comprehensive privacy validation")
        
        # Auto-detect sensitive columns if not provided
        if sensitive_columns is None:
            sensitive_columns = self._detect_sensitive_columns(original_data)
        
        if quasi_identifiers is None:
            quasi_identifiers = self._detect_quasi_identifiers(original_data)
        
        # Calculate anonymity metrics
        anonymity_metrics = self._calculate_anonymity_metrics(
            original_data, synthetic_data, sensitive_columns, quasi_identifiers
        )
        
        # Calculate utility preservation
        utility_preservation = self._calculate_utility_preservation(
            original_data, synthetic_data
        )
        
        # Calculate disclosure risks
        disclosure_risks = self._calculate_disclosure_risks(
            original_data, synthetic_data, sensitive_columns
        )
        
        # Validate compliance
        compliance_results = {}
        if compliance_frameworks:
            for framework in compliance_frameworks:
                compliance_results[framework.value] = self._validate_compliance(
                    original_data, synthetic_data, framework, anonymity_metrics
                )
        
        # Calculate overall privacy score
        overall_score = self._calculate_overall_privacy_score(
            anonymity_metrics, disclosure_risks
        )
        
        # Generate recommendations
        recommendations, required_actions = self._generate_recommendations(
            anonymity_metrics, disclosure_risks, compliance_results
        )
        
        # Create report
        report = PrivacyReport(
            overall_privacy_score=overall_score,
            privacy_level=self._assess_privacy_level(overall_score),
            risk_assessment=self._assess_risk_level(disclosure_risks),
            anonymity_metrics={name: metric for name, metric in anonymity_metrics.items()},
            utility_preservation=utility_preservation,
            disclosure_risks=disclosure_risks,
            compliance_results=compliance_results,
            recommendations=recommendations,
            required_actions=required_actions,
            validation_timestamp=pd.Timestamp.now().isoformat(),
            dataset_fingerprint=self._calculate_dataset_fingerprint(synthetic_data)
        )
        
        logger.info(f"Privacy validation completed: Score={overall_score:.3f}")
        return report
    
    def _detect_sensitive_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect potentially sensitive columns."""
        
        sensitive_patterns = [
            r'.*name.*', r'.*id.*', r'.*ssn.*', r'.*social.*',
            r'.*phone.*', r'.*email.*', r'.*address.*', r'.*zip.*',
            r'.*credit.*', r'.*salary.*', r'.*income.*', r'.*medical.*',
            r'.*diagnosis.*', r'.*patient.*', r'.*account.*'
        ]
        
        sensitive_columns = []
        
        for col in data.columns:
            col_lower = col.lower()
            for pattern in sensitive_patterns:
                if re.match(pattern, col_lower):
                    sensitive_columns.append(col)
                    break
        
        # Also consider high-cardinality columns as potentially sensitive
        for col in data.columns:
            if col not in sensitive_columns:
                if data[col].dtype == 'object' and data[col].nunique() > len(data) * 0.5:
                    sensitive_columns.append(col)
        
        return sensitive_columns
    
    def _detect_quasi_identifiers(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect quasi-identifier columns."""
        
        qi_patterns = [
            r'.*age.*', r'.*birth.*', r'.*date.*', r'.*zip.*',
            r'.*postal.*', r'.*gender.*', r'.*sex.*', r'.*race.*',
            r'.*ethnicity.*', r'.*education.*', r'.*occupation.*',
            r'.*department.*', r'.*location.*', r'.*city.*', r'.*state.*'
        ]
        
        quasi_identifiers = []
        
        for col in data.columns:
            col_lower = col.lower()
            for pattern in qi_patterns:
                if re.match(pattern, col_lower):
                    quasi_identifiers.append(col)
                    break
        
        return quasi_identifiers
    
    def _calculate_anonymity_metrics(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_columns: List[str],
        quasi_identifiers: List[str]
    ) -> Dict[str, PrivacyMetric]:
        """Calculate various anonymity metrics."""
        
        metrics = {}
        
        # K-Anonymity
        k_anon = self._calculate_k_anonymity(synthetic_data, quasi_identifiers)
        metrics['k_anonymity'] = PrivacyMetric(
            name="K-Anonymity",
            value=k_anon,
            threshold=self.thresholds['k_anonymity'],
            passed=k_anon >= self.thresholds['k_anonymity'],
            severity="high" if k_anon < self.thresholds['k_anonymity'] else "low",
            description=f"Minimum group size in dataset: {k_anon}",
            recommendation="Increase k-anonymity by generalizing quasi-identifiers" if k_anon < self.thresholds['k_anonymity'] else "K-anonymity threshold met"
        )
        
        # L-Diversity
        if sensitive_columns:
            l_div = self._calculate_l_diversity(synthetic_data, quasi_identifiers, sensitive_columns[0])
            metrics['l_diversity'] = PrivacyMetric(
                name="L-Diversity",
                value=l_div,
                threshold=self.thresholds['l_diversity'],
                passed=l_div >= self.thresholds['l_diversity'],
                severity="medium" if l_div < self.thresholds['l_diversity'] else "low",
                description=f"Minimum diversity in sensitive attributes: {l_div:.2f}",
                recommendation="Increase diversity in sensitive attribute values" if l_div < self.thresholds['l_diversity'] else "L-diversity threshold met"
            )
        
        # T-Closeness
        if sensitive_columns:
            t_close = self._calculate_t_closeness(original_data, synthetic_data, quasi_identifiers, sensitive_columns[0])
            metrics['t_closeness'] = PrivacyMetric(
                name="T-Closeness",
                value=t_close,
                threshold=self.thresholds['t_closeness'],
                passed=t_close <= self.thresholds['t_closeness'],
                severity="medium" if t_close > self.thresholds['t_closeness'] else "low",
                description=f"Maximum distribution distance: {t_close:.3f}",
                recommendation="Reduce distribution skew in equivalence classes" if t_close > self.thresholds['t_closeness'] else "T-closeness threshold met"
            )
        
        # Distance to Closest Record
        dist_closest = self._calculate_distance_to_closest(original_data, synthetic_data)
        metrics['distance_to_closest'] = PrivacyMetric(
            name="Distance to Closest Record",
            value=dist_closest,
            threshold=self.thresholds['distance_to_closest'],
            passed=dist_closest >= self.thresholds['distance_to_closest'],
            severity="high" if dist_closest < self.thresholds['distance_to_closest'] else "low",
            description=f"Minimum distance to original records: {dist_closest:.3f}",
            recommendation="Increase distance between synthetic and original records" if dist_closest < self.thresholds['distance_to_closest'] else "Distance threshold met"
        )
        
        return metrics
    
    def _calculate_k_anonymity(self, data: pd.DataFrame, quasi_identifiers: List[str]) -> float:
        """Calculate k-anonymity of the dataset."""
        
        if not quasi_identifiers:
            return float('inf')  # No quasi-identifiers means perfect k-anonymity
        
        # Group by quasi-identifiers and find minimum group size
        qi_columns = [col for col in quasi_identifiers if col in data.columns]
        
        if not qi_columns:
            return float('inf')
        
        group_sizes = data.groupby(qi_columns).size()
        return float(group_sizes.min())
    
    def _calculate_l_diversity(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_column: str
    ) -> float:
        """Calculate l-diversity for a sensitive attribute."""
        
        if sensitive_column not in data.columns:
            return 0.0
        
        qi_columns = [col for col in quasi_identifiers if col in data.columns]
        
        if not qi_columns:
            return data[sensitive_column].nunique()
        
        # For each equivalence class, calculate diversity
        diversities = []
        for name, group in data.groupby(qi_columns):
            diversity = group[sensitive_column].nunique()
            diversities.append(diversity)
        
        return float(min(diversities)) if diversities else 0.0
    
    def _calculate_t_closeness(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_column: str
    ) -> float:
        """Calculate t-closeness for a sensitive attribute."""
        
        if sensitive_column not in original_data.columns or sensitive_column not in synthetic_data.columns:
            return 1.0  # Worst case
        
        qi_columns = [col for col in quasi_identifiers if col in synthetic_data.columns]
        
        if not qi_columns:
            # Calculate overall distribution distance
            if synthetic_data[sensitive_column].dtype == 'object':
                orig_dist = original_data[sensitive_column].value_counts(normalize=True)
                synth_dist = synthetic_data[sensitive_column].value_counts(normalize=True)
                
                # Align distributions
                all_values = set(orig_dist.index) | set(synth_dist.index)
                orig_aligned = [orig_dist.get(v, 0) for v in all_values]
                synth_aligned = [synth_dist.get(v, 0) for v in all_values]
                
                # Calculate Earth Mover's Distance (approximate)
                return float(sum(abs(o - s) for o, s in zip(orig_aligned, synth_aligned)) / 2)
            else:
                # Numerical: use Wasserstein distance
                try:
                    return float(stats.wasserstein_distance(
                        original_data[sensitive_column].dropna(),
                        synthetic_data[sensitive_column].dropna()
                    ))
                except:
                    return 1.0
        
        # Calculate t-closeness for each equivalence class
        max_distance = 0.0
        
        for name, group in synthetic_data.groupby(qi_columns):
            if len(group) == 0:
                continue
            
            # Get overall distribution from original data
            if sensitive_column in original_data.columns:
                if original_data[sensitive_column].dtype == 'object':
                    overall_dist = original_data[sensitive_column].value_counts(normalize=True)
                    group_dist = group[sensitive_column].value_counts(normalize=True)
                    
                    # Calculate distance
                    all_values = set(overall_dist.index) | set(group_dist.index)
                    overall_aligned = [overall_dist.get(v, 0) for v in all_values]
                    group_aligned = [group_dist.get(v, 0) for v in all_values]
                    
                    distance = sum(abs(o - g) for o, g in zip(overall_aligned, group_aligned)) / 2
                    max_distance = max(max_distance, distance)
        
        return float(max_distance)
    
    def _calculate_distance_to_closest(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> float:
        """Calculate minimum distance from synthetic records to original records."""
        
        # Select only numerical columns for distance calculation
        num_cols_orig = original_data.select_dtypes(include=[np.number]).columns
        num_cols_synth = synthetic_data.select_dtypes(include=[np.number]).columns
        common_num_cols = list(set(num_cols_orig) & set(num_cols_synth))
        
        if not common_num_cols:
            return 1.0  # Can't calculate meaningful distance
        
        # Normalize the data
        scaler = StandardScaler()
        orig_scaled = scaler.fit_transform(original_data[common_num_cols].fillna(0))
        synth_scaled = scaler.transform(synthetic_data[common_num_cols].fillna(0))
        
        # Calculate distances from each synthetic record to closest original record
        min_distances = []
        
        # Use efficient nearest neighbor search
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nbrs.fit(orig_scaled)
        
        distances, indices = nbrs.kneighbors(synth_scaled)
        min_distances = distances.flatten()
        
        return float(np.min(min_distances))
    
    def _calculate_utility_preservation(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate how well the synthetic data preserves utility."""
        
        utility_metrics = {}
        
        # Statistical similarity
        numerical_cols = original_data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col in synthetic_data.columns]
        
        if numerical_cols:
            correlations = []
            for col in numerical_cols:
                if original_data[col].std() > 0 and synthetic_data[col].std() > 0:
                    corr = np.corrcoef(
                        original_data[col].fillna(original_data[col].mean()),
                        synthetic_data[col].fillna(synthetic_data[col].mean())
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            utility_metrics['statistical_similarity'] = np.mean(correlations) if correlations else 0.0
        
        # Distribution similarity (KS test)
        ks_statistics = []
        for col in numerical_cols:
            try:
                ks_stat, _ = stats.ks_2samp(
                    original_data[col].dropna(),
                    synthetic_data[col].dropna()
                )
                ks_statistics.append(1 - ks_stat)  # Convert to similarity (higher is better)
            except:
                continue
        
        utility_metrics['distribution_similarity'] = np.mean(ks_statistics) if ks_statistics else 0.0
        
        # Correlation structure preservation
        if len(numerical_cols) > 1:
            try:
                orig_corr = original_data[numerical_cols].corr()
                synth_corr = synthetic_data[numerical_cols].corr()
                
                # Flatten correlation matrices and compare
                orig_flat = orig_corr.values[np.triu_indices_from(orig_corr.values, k=1)]
                synth_flat = synth_corr.values[np.triu_indices_from(synth_corr.values, k=1)]
                
                corr_preservation = np.corrcoef(orig_flat, synth_flat)[0, 1]
                utility_metrics['correlation_preservation'] = corr_preservation if not np.isnan(corr_preservation) else 0.0
            except:
                utility_metrics['correlation_preservation'] = 0.0
        
        return utility_metrics
    
    def _calculate_disclosure_risks(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_columns: List[str]
    ) -> Dict[str, float]:
        """Calculate various disclosure risk metrics."""
        
        risks = {}
        
        # Membership inference risk
        risks['membership_inference'] = self._calculate_membership_inference_risk(
            original_data, synthetic_data
        )
        
        # Attribute inference risk
        if sensitive_columns:
            risks['attribute_inference'] = self._calculate_attribute_inference_risk(
                original_data, synthetic_data, sensitive_columns
            )
        
        # Record linkage risk
        risks['record_linkage'] = self._calculate_record_linkage_risk(
            original_data, synthetic_data
        )
        
        return risks
    
    def _calculate_membership_inference_risk(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> float:
        """Calculate membership inference attack risk."""
        
        # Simplified membership inference: measure overlap in exact records
        # In practice, this would use more sophisticated ML-based attacks
        
        common_cols = list(set(original_data.columns) & set(synthetic_data.columns))
        if not common_cols:
            return 0.0
        
        # Convert to string representation for comparison
        orig_strings = set()
        synth_strings = set()
        
        for _, row in original_data[common_cols].iterrows():
            orig_strings.add(str(tuple(row.values)))
        
        for _, row in synthetic_data[common_cols].iterrows():
            synth_strings.add(str(tuple(row.values)))
        
        # Calculate overlap ratio
        overlap = len(orig_strings & synth_strings)
        total_synthetic = len(synth_strings)
        
        return overlap / total_synthetic if total_synthetic > 0 else 0.0
    
    def _calculate_attribute_inference_risk(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_columns: List[str]
    ) -> float:
        """Calculate attribute inference attack risk."""
        
        # Measure how easily sensitive attributes can be inferred
        # This is a simplified version - in practice would use ML models
        
        risks = []
        
        for sensitive_col in sensitive_columns:
            if sensitive_col not in original_data.columns or sensitive_col not in synthetic_data.columns:
                continue
            
            # Calculate mutual information between sensitive attribute and other attributes
            other_cols = [col for col in synthetic_data.columns if col != sensitive_col]
            
            if not other_cols:
                continue
            
            # For categorical sensitive attributes
            if synthetic_data[sensitive_col].dtype == 'object':
                # Encode categorical variables
                from sklearn.preprocessing import LabelEncoder
                
                le_sensitive = LabelEncoder()
                sensitive_encoded = le_sensitive.fit_transform(synthetic_data[sensitive_col].fillna('missing'))
                
                max_mi = 0.0
                for other_col in other_cols[:5]:  # Limit to avoid excessive computation
                    try:
                        if synthetic_data[other_col].dtype == 'object':
                            le_other = LabelEncoder()
                            other_encoded = le_other.fit_transform(synthetic_data[other_col].fillna('missing'))
                        else:
                            # Discretize numerical variables
                            other_encoded = pd.cut(synthetic_data[other_col].fillna(0), bins=10, labels=False)
                        
                        mi = mutual_info_score(sensitive_encoded, other_encoded)
                        max_mi = max(max_mi, mi)
                    except:
                        continue
                
                risks.append(max_mi)
        
        return np.mean(risks) if risks else 0.0
    
    def _calculate_record_linkage_risk(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> float:
        """Calculate record linkage attack risk."""
        
        # Measure how easily records can be linked back to original data
        # Based on uniqueness of attribute combinations
        
        common_cols = list(set(original_data.columns) & set(synthetic_data.columns))
        if len(common_cols) < 2:  # Need at least 2 columns for linkage
            return 0.0
        
        # Calculate uniqueness ratios for different column combinations
        uniqueness_scores = []
        
        # Test different combinations of columns (up to 3 for efficiency)
        from itertools import combinations
        
        for r in range(2, min(4, len(common_cols) + 1)):
            for col_combo in list(combinations(common_cols, r))[:10]:  # Limit combinations
                try:
                    # Count unique combinations in synthetic data
                    synth_unique = synthetic_data[list(col_combo)].drop_duplicates().shape[0]
                    synth_total = synthetic_data.shape[0]
                    
                    uniqueness = synth_unique / synth_total if synth_total > 0 else 0
                    uniqueness_scores.append(uniqueness)
                except:
                    continue
        
        # Higher uniqueness means higher linkage risk
        return np.mean(uniqueness_scores) if uniqueness_scores else 0.0
    
    def _validate_compliance(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        framework: ComplianceFramework,
        anonymity_metrics: Dict[str, PrivacyMetric]
    ) -> ComplianceResult:
        """Validate compliance with specific regulatory framework."""
        
        rules = self.compliance_rules[framework]
        violations = []
        recommendations = []
        score = 1.0
        
        if framework == ComplianceFramework.GDPR:
            # GDPR compliance checks
            if rules['required_anonymization']:
                k_anon = anonymity_metrics.get('k_anonymity')
                if k_anon and k_anon.value < rules['min_k_anonymity']:
                    violations.append({
                        'rule': 'Article 4(1) - Anonymization requirement',
                        'description': f'K-anonymity ({k_anon.value}) below required minimum ({rules["min_k_anonymity"]})',
                        'severity': 'high'
                    })
                    score -= 0.3
                    recommendations.append('Increase k-anonymity through generalization or suppression')
            
            if rules['data_minimization']:
                # Check if all columns are necessary (simplified check)
                if len(synthetic_data.columns) > len(original_data.columns) * 0.8:
                    recommendations.append('Consider data minimization - remove unnecessary attributes')
        
        elif framework == ComplianceFramework.HIPAA:
            # HIPAA compliance checks
            if rules['required_anonymization']:
                k_anon = anonymity_metrics.get('k_anonymity')
                if k_anon and k_anon.value < rules['min_k_anonymity']:
                    violations.append({
                        'rule': '45 CFR 164.514(b) - Safe Harbor method',
                        'description': f'K-anonymity insufficient for Safe Harbor compliance',
                        'severity': 'critical'
                    })
                    score -= 0.5
                    recommendations.append('Apply Safe Harbor de-identification method')
            
            # Check for PHI identifiers (simplified)
            phi_patterns = ['name', 'address', 'phone', 'email', 'ssn', 'account']
            for col in synthetic_data.columns:
                if any(pattern in col.lower() for pattern in phi_patterns):
                    violations.append({
                        'rule': '45 CFR 164.514(b)(2) - PHI identifiers',
                        'description': f'Column "{col}" may contain PHI identifiers',
                        'severity': 'high'
                    })
                    score -= 0.2
        
        # Determine overall pass/fail and risk level
        passed = len([v for v in violations if v['severity'] == 'critical']) == 0
        
        if score >= 0.9:
            risk_level = 'low'
        elif score >= 0.7:
            risk_level = 'medium'
        elif score >= 0.5:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        return ComplianceResult(
            framework=framework.value,
            overall_score=score,
            passed=passed,
            violations=violations,
            recommendations=recommendations,
            risk_level=risk_level
        )
    
    def _calculate_overall_privacy_score(
        self,
        anonymity_metrics: Dict[str, PrivacyMetric],
        disclosure_risks: Dict[str, float]
    ) -> float:
        """Calculate overall privacy score (0-1, higher is better)."""
        
        score_components = []
        
        # Anonymity metrics (positive contribution)
        for metric in anonymity_metrics.values():
            if metric.name == 'T-Closeness':
                # For t-closeness, lower is better, so invert
                normalized = max(0, 1 - (metric.value / metric.threshold))
            else:
                # For others, higher is better
                normalized = min(1, metric.value / metric.threshold)
            
            score_components.append(normalized)
        
        # Disclosure risks (negative contribution)
        for risk_name, risk_value in disclosure_risks.items():
            # Convert risk to privacy score (invert and normalize)
            privacy_contribution = max(0, 1 - risk_value)
            score_components.append(privacy_contribution)
        
        return np.mean(score_components) if score_components else 0.5
    
    def _assess_privacy_level(self, overall_score: float) -> str:
        """Assess privacy level based on overall score."""
        
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.75:
            return "good"
        elif overall_score >= 0.6:
            return "acceptable"
        elif overall_score >= 0.4:
            return "poor"
        else:
            return "inadequate"
    
    def _assess_risk_level(self, disclosure_risks: Dict[str, float]) -> str:
        """Assess overall risk level based on disclosure risks."""
        
        max_risk = max(disclosure_risks.values()) if disclosure_risks else 0.0
        
        if max_risk < 0.1:
            return "low"
        elif max_risk < 0.3:
            return "medium"
        elif max_risk < 0.5:
            return "high"
        else:
            return "critical"
    
    def _generate_recommendations(
        self,
        anonymity_metrics: Dict[str, PrivacyMetric],
        disclosure_risks: Dict[str, float],
        compliance_results: Dict[str, ComplianceResult]
    ) -> Tuple[List[str], List[str]]:
        """Generate recommendations and required actions."""
        
        recommendations = []
        required_actions = []
        
        # Anonymity-based recommendations
        for metric in anonymity_metrics.values():
            if not metric.passed:
                if metric.severity == "high":
                    required_actions.append(metric.recommendation)
                else:
                    recommendations.append(metric.recommendation)
        
        # Risk-based recommendations
        for risk_name, risk_value in disclosure_risks.items():
            if risk_value > 0.5:
                required_actions.append(f"Address high {risk_name.replace('_', ' ')} risk (score: {risk_value:.3f})")
            elif risk_value > 0.3:
                recommendations.append(f"Monitor {risk_name.replace('_', ' ')} risk (score: {risk_value:.3f})")
        
        # Compliance-based recommendations
        for result in compliance_results.values():
            recommendations.extend(result.recommendations)
            
            for violation in result.violations:
                if violation['severity'] == 'critical':
                    required_actions.append(f"{result.framework.upper()}: {violation['description']}")
                else:
                    recommendations.append(f"{result.framework.upper()}: {violation['description']}")
        
        return recommendations, required_actions
    
    def _calculate_dataset_fingerprint(self, data: pd.DataFrame) -> str:
        """Calculate a fingerprint of the dataset for tracking."""
        
        # Create a hash based on dataset structure and sample
        fingerprint_data = {
            'shape': data.shape,
            'columns': sorted(data.columns.tolist()),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'sample_hash': hashlib.md5(str(data.head().values).encode()).hexdigest()
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    def generate_privacy_report(
        self,
        report: PrivacyReport,
        output_file: Optional[str] = None
    ) -> str:
        """Generate a comprehensive privacy validation report."""
        
        report_lines = [
            "# Privacy Validation Report",
            "",
            f"**Validation Date:** {pd.Timestamp.now().isoformat()}",
            f"**Dataset Fingerprint:** {report.dataset_fingerprint}",
            f"**Privacy Level:** {report.privacy_level.upper()}",
            f"**Overall Score:** {report.overall_privacy_score:.3f}/1.000",
            f"**Risk Assessment:** {report.risk_assessment.upper()}",
            "",
            "## Executive Summary",
            ""
        ]
        
        if report.overall_privacy_score >= 0.75:
            report_lines.append("✅ **ACCEPTABLE PRIVACY LEVEL** - The synthetic data demonstrates adequate privacy protection.")
        else:
            report_lines.append("⚠️ **PRIVACY CONCERNS IDENTIFIED** - The synthetic data requires privacy improvements before use.")
        
        # Anonymity Metrics
        report_lines.extend([
            "",
            "## Anonymity Metrics",
            ""
        ])
        
        for name, metric in report.anonymity_metrics.items():
            status = "✅ PASS" if metric.passed else "❌ FAIL"
            report_lines.extend([
                f"### {metric.name}",
                f"**Status:** {status}",
                f"**Value:** {metric.value:.3f}",
                f"**Threshold:** {metric.threshold:.3f}",
                f"**Description:** {metric.description}",
                f"**Recommendation:** {metric.recommendation}",
                ""
            ])
        
        # Utility Preservation
        if report.utility_preservation:
            report_lines.extend([
                "## Utility Preservation",
                ""
            ])
            
            for metric_name, value in report.utility_preservation.items():
                report_lines.append(f"- **{metric_name.replace('_', ' ').title()}:** {value:.3f}")
            
            report_lines.append("")
        
        # Disclosure Risks
        if report.disclosure_risks:
            report_lines.extend([
                "## Disclosure Risk Analysis",
                ""
            ])
            
            for risk_name, risk_value in report.disclosure_risks.items():
                risk_level = "HIGH" if risk_value > 0.5 else "MEDIUM" if risk_value > 0.3 else "LOW"
                risk_icon = "🔴" if risk_value > 0.5 else "🟡" if risk_value > 0.3 else "🟢"
                
                report_lines.extend([
                    f"### {risk_name.replace('_', ' ').title()}",
                    f"{risk_icon} **Risk Level:** {risk_level}",
                    f"**Risk Score:** {risk_value:.3f}",
                    ""
                ])
        
        # Compliance Results
        if report.compliance_results:
            report_lines.extend([
                "## Regulatory Compliance",
                ""
            ])
            
            for framework, result in report.compliance_results.items():
                status = "✅ COMPLIANT" if result.passed else "❌ NON-COMPLIANT"
                report_lines.extend([
                    f"### {framework.upper()}",
                    f"**Status:** {status}",
                    f"**Score:** {result.overall_score:.3f}",
                    f"**Risk Level:** {result.risk_level.upper()}",
                    ""
                ])
                
                if result.violations:
                    report_lines.append("**Violations:**")
                    for violation in result.violations:
                        severity_icon = "🔴" if violation['severity'] == 'critical' else "🟡" if violation['severity'] == 'high' else "🟠"
                        report_lines.append(f"- {severity_icon} {violation['rule']}: {violation['description']}")
                    report_lines.append("")
        
        # Recommendations
        if report.recommendations or report.required_actions:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            
            if report.required_actions:
                report_lines.extend([
                    "### 🚨 Required Actions (Critical)",
                    ""
                ])
                for action in report.required_actions:
                    report_lines.append(f"1. {action}")
                
                report_lines.append("")
            
            if report.recommendations:
                report_lines.extend([
                    "### 💡 Recommendations (Advisory)",
                    ""
                ])
                for rec in report.recommendations:
                    report_lines.append(f"- {rec}")
                
                report_lines.append("")
        
        # Technical Details
        report_lines.extend([
            "## Technical Details",
            "",
            f"**Validation Timestamp:** {report.validation_timestamp}",
            f"**Privacy Validator Version:** 1.0",
            f"**Anonymity Metrics Count:** {len(report.anonymity_metrics)}",
            f"**Disclosure Risk Metrics:** {len(report.disclosure_risks)}",
            f"**Compliance Frameworks:** {len(report.compliance_results)}",
            ""
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Privacy report saved to {output_file}")
        
        return report_text

def validate_synthetic_data_privacy(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    privacy_level: str = "medium",
    compliance_frameworks: Optional[List[str]] = None,
    sensitive_columns: Optional[List[str]] = None,
    output_dir: str = "privacy_reports"
) -> Dict[str, Any]:
    """
    Convenience function for privacy validation.
    
    Args:
        original_data: Original sensitive dataset
        synthetic_data: Generated synthetic dataset
        privacy_level: Privacy protection level ('low', 'medium', 'high', 'critical')
        compliance_frameworks: List of compliance frameworks to check
        sensitive_columns: List of sensitive columns
        output_dir: Directory to save reports
        
    Returns:
        Dictionary with privacy validation results
    """
    
    # Initialize validator
    privacy_enum = PrivacyLevel(privacy_level)
    validator = PrivacyValidator(privacy_enum)
    
    # Parse compliance frameworks
    frameworks = []
    if compliance_frameworks:
        for framework in compliance_frameworks:
            try:
                frameworks.append(ComplianceFramework(framework.lower()))
            except ValueError:
                logger.warning(f"Unknown compliance framework: {framework}")
    
    # Run validation
    report = validator.validate_privacy(
        original_data=original_data,
        synthetic_data=synthetic_data,
        sensitive_columns=sensitive_columns,
        compliance_frameworks=frameworks
    )
    
    # Generate and save report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(output_dir) / f"privacy_validation_{timestamp}.md"
    
    report_text = validator.generate_privacy_report(report, str(report_file))
    
    return {
        'report': report,
        'privacy_score': report.overall_privacy_score,
        'privacy_level': report.privacy_level,
        'risk_assessment': report.risk_assessment,
        'passed_validation': report.overall_privacy_score >= 0.6,
        'report_text': report_text,
        'report_file': str(report_file),
        'recommendations': report.recommendations,
        'required_actions': report.required_actions
    }