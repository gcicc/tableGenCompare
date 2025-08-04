"""
Pydantic models for API request/response validation.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import re

class TrainingRequest(BaseModel):
    """Request model for training a synthetic data generation model."""
    
    model_name: str = Field(..., description="Name of the model to train (e.g., 'ctgan', 'tvae', 'ganeraid')")
    data_path: Optional[str] = Field(None, description="Path to training data CSV file")
    data_content: Optional[str] = Field(None, description="JSON string of training data")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model hyperparameters")
    output_path: Optional[str] = Field(None, description="Path to save trained model")
    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated training duration")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        valid_models = ['ctgan', 'tvae', 'ganeraid', 'copulagan']
        if v.lower() not in valid_models:
            raise ValueError(f"Invalid model name. Must be one of: {valid_models}")
        return v.lower()
    
    @validator('hyperparameters')
    def validate_hyperparameters(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("Hyperparameters must be a dictionary")
        return v

class TrainingResponse(BaseModel):
    """Response model for training request."""
    
    job_id: str = Field(..., description="Unique identifier for the training job")
    status: str = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable status message")
    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated training duration")

class GenerationRequest(BaseModel):
    """Request model for generating synthetic data."""
    
    num_samples: int = Field(..., gt=0, le=100000, description="Number of synthetic samples to generate")
    model_path: Optional[str] = Field(None, description="Path to trained model file")
    training_job_id: Optional[str] = Field(None, description="ID of training job to use model from")
    output_format: str = Field("csv", description="Output format (csv, json, parquet)")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    
    @validator('output_format')
    def validate_output_format(cls, v):
        valid_formats = ['csv', 'json', 'parquet']
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid output format. Must be one of: {valid_formats}")
        return v.lower()
    
    @validator('model_path', 'training_job_id')
    def validate_model_source(cls, v, values):
        # At least one of model_path or training_job_id must be provided
        model_path = values.get('model_path')
        training_job_id = values.get('training_job_id')
        
        if not model_path and not training_job_id:
            raise ValueError("Either model_path or training_job_id must be provided")
        
        return v

class GenerationResponse(BaseModel):
    """Response model for generation request."""
    
    job_id: str = Field(..., description="Unique identifier for the generation job")
    status: str = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable status message")

class EvaluationRequest(BaseModel):
    """Request model for evaluating synthetic data quality."""
    
    original_data_path: str = Field(..., description="Path to original data CSV file")
    synthetic_data_path: str = Field(..., description="Path to synthetic data CSV file")
    target_column: Optional[str] = Field(None, description="Name of target column for ML evaluation")
    metrics: List[str] = Field(
        default_factory=lambda: ["trts", "similarity", "quality"], 
        description="Evaluation metrics to compute"
    )
    dataset_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Metadata about the dataset"
    )
    
    @validator('metrics')
    def validate_metrics(cls, v):
        valid_metrics = ['trts', 'similarity', 'quality', 'statistical', 'privacy']
        for metric in v:
            if metric.lower() not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Must be one of: {valid_metrics}")
        return [m.lower() for m in v]

class EvaluationResponse(BaseModel):
    """Response model for evaluation request."""
    
    job_id: str = Field(..., description="Unique identifier for the evaluation job")
    status: str = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable status message")

class JobStatus(BaseModel):
    """Model for job status information."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (queued, running, completed, failed, cancelled)")
    job_type: str = Field(..., description="Type of job (training, generation, evaluation)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Job progress percentage")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Job metadata")
    results: Optional[Dict[str, Any]] = Field(None, description="Job results (if completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")

class ModelInfo(BaseModel):
    """Model information for available models."""
    
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (GAN, VAE, etc.)")
    description: str = Field(..., description="Model description")
    available: bool = Field(..., description="Whether model is available")
    hyperparameters: List[str] = Field(default_factory=list, description="Available hyperparameters")
    supports_categorical: bool = Field(False, description="Supports categorical data")
    supports_mixed_types: bool = Field(False, description="Supports mixed data types")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Required dependencies")

class APIError(BaseModel):
    """Standard API error response."""
    
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

class BatchRequest(BaseModel):
    """Request model for batch operations."""
    
    operations: List[Dict[str, Any]] = Field(..., description="List of operations to perform")
    parallel: bool = Field(False, description="Execute operations in parallel")
    max_workers: int = Field(4, gt=0, le=16, description="Maximum parallel workers")
    
    @validator('operations')
    def validate_operations(cls, v):
        if not v:
            raise ValueError("At least one operation must be specified")
        if len(v) > 100:
            raise ValueError("Maximum 100 operations per batch")
        return v

class BatchResponse(BaseModel):
    """Response model for batch operations."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    total_operations: int = Field(..., description="Total number of operations")
    job_ids: List[str] = Field(..., description="Individual job IDs")
    status: str = Field(..., description="Batch status")

class DataUpload(BaseModel):
    """Model for data upload metadata."""
    
    upload_id: str = Field(..., description="Unique upload identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    
    @validator('filename')
    def validate_filename(cls, v):
        # Basic filename validation
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError("Invalid filename. Use only alphanumeric characters, dots, hyphens, and underscores")
        if not v.lower().endswith(('.csv', '.json', '.parquet')):
            raise ValueError("File must be CSV, JSON, or Parquet format")
        return v

class ModelConfig(BaseModel):
    """Model configuration for advanced setups."""
    
    model_name: str = Field(..., description="Model name")
    config: Dict[str, Any] = Field(..., description="Model configuration parameters")
    preprocessing: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Data preprocessing options")
    postprocessing: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Data postprocessing options")
    validation: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validation settings")

class OptimizationRequest(BaseModel):
    """Request model for hyperparameter optimization."""
    
    model_name: str = Field(..., description="Model to optimize")
    data_path: str = Field(..., description="Path to training data")
    target_column: Optional[str] = Field(None, description="Target column for optimization")
    optimization_metric: str = Field("composite", description="Metric to optimize")
    n_trials: int = Field(50, gt=0, le=1000, description="Number of optimization trials")
    timeout_minutes: int = Field(60, gt=0, le=1440, description="Optimization timeout")
    sampler: str = Field("tpe", description="Optimization sampler (tpe, random, cmaes)")
    
    @validator('optimization_metric')
    def validate_metric(cls, v):
        valid_metrics = ['trts', 'similarity', 'quality', 'composite']
        if v.lower() not in valid_metrics:
            raise ValueError(f"Invalid metric. Must be one of: {valid_metrics}")
        return v.lower()
    
    @validator('sampler')
    def validate_sampler(cls, v):
        valid_samplers = ['tpe', 'random', 'cmaes', 'nsgaii']
        if v.lower() not in valid_samplers:
            raise ValueError(f"Invalid sampler. Must be one of: {valid_samplers}")
        return v.lower()

class OptimizationResponse(BaseModel):
    """Response model for optimization request."""
    
    job_id: str = Field(..., description="Optimization job ID")
    status: str = Field(..., description="Optimization status")
    n_trials: int = Field(..., description="Number of trials to run")
    estimated_duration_minutes: int = Field(..., description="Estimated duration")

class SystemStats(BaseModel):
    """System statistics and health information."""
    
    active_jobs: int = Field(..., description="Number of active jobs")
    completed_jobs: int = Field(..., description="Total completed jobs")
    failed_jobs: int = Field(..., description="Total failed jobs")
    total_models_trained: int = Field(..., description="Total models trained")
    total_samples_generated: int = Field(..., description="Total synthetic samples generated")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")
    available_models: Dict[str, bool] = Field(..., description="Available models and their status")