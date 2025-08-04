#!/usr/bin/env python3
"""
Production-ready API service for synthetic data generation.
Provides RESTful endpoints for training models and generating synthetic data.
"""

import os
import sys
import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import traceback

# Third-party imports
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Internal imports
from models.model_factory import ModelFactory
from evaluation.unified_evaluator import UnifiedEvaluator
from api.models import (
    TrainingRequest, GenerationRequest, EvaluationRequest,
    TrainingResponse, GenerationResponse, EvaluationResponse,
    JobStatus, ModelInfo, APIError
)
from api.auth import verify_api_key
from api.job_manager import JobManager
from api.utils import validate_dataframe, sanitize_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Synthetic Data Generation API",
    description="Production-ready API for training generative models and creating synthetic tabular data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize job manager
job_manager = JobManager()

# Security
security = HTTPBearer()

@app.on_event("startup")
async def startup_event():
    """Initialize API service on startup."""
    logger.info("Starting Synthetic Data Generation API")
    
    # Create necessary directories
    os.makedirs("api_data/uploads", exist_ok=True)
    os.makedirs("api_data/outputs", exist_ok=True)
    os.makedirs("api_data/models", exist_ok=True)
    
    # Log available models
    available_models = ModelFactory.list_available_models()
    logger.info(f"Available models: {available_models}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Synthetic Data Generation API")
    await job_manager.cleanup()

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

# Model information endpoints
@app.get("/models", response_model=Dict[str, ModelInfo])
async def list_models():
    """List all available models and their information."""
    try:
        available_models = ModelFactory.list_available_models()
        model_info = {}
        
        for model_name, is_available in available_models.items():
            if is_available:
                try:
                    model = ModelFactory.create(model_name, random_state=42)
                    info = model.get_model_info()
                    hyperparams = model.get_hyperparameter_space()
                    
                    model_info[model_name] = ModelInfo(
                        name=model_name,
                        type=info.get('model_type', 'unknown'),
                        description=info.get('description', ''),
                        available=True,
                        hyperparameters=list(hyperparams.keys()) if hyperparams else [],
                        supports_categorical=info.get('supports_categorical', False),
                        supports_mixed_types=info.get('supports_mixed_types', False)
                    )
                except Exception as e:
                    logger.warning(f"Error getting info for {model_name}: {e}")
                    model_info[model_name] = ModelInfo(
                        name=model_name,
                        type="unknown",
                        description=f"Error: {str(e)}",
                        available=False
                    )
            else:
                model_info[model_name] = ModelInfo(
                    name=model_name,
                    type="unknown", 
                    description="Dependencies not available",
                    available=False
                )
        
        return model_info
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# Training endpoints
@app.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Start training a synthetic data generation model."""
    try:
        # Verify authentication
        await verify_api_key(credentials.credentials)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Validate model availability
        available_models = ModelFactory.list_available_models()
        if not available_models.get(request.model_name, False):
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model_name}' is not available"
            )
        
        # Load and validate training data
        try:
            if request.data_path:
                training_data = pd.read_csv(request.data_path)
            elif request.data_content:
                training_data = pd.read_json(request.data_content)
            else:
                raise HTTPException(status_code=400, detail="No training data provided")
            
            validate_dataframe(training_data)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid training data: {str(e)}")
        
        # Create job
        job_manager.create_job(
            job_id=job_id,
            job_type="training",
            status="queued",
            metadata={
                "model_name": request.model_name,
                "data_shape": training_data.shape,
                "hyperparameters": request.hyperparameters or {},
                "output_path": request.output_path
            }
        )
        
        # Start training in background
        background_tasks.add_task(
            _train_model_background,
            job_id,
            request.model_name,
            training_data,
            request.hyperparameters or {},
            request.output_path
        )
        
        logger.info(f"Training job {job_id} started for model {request.model_name}")
        
        return TrainingResponse(
            job_id=job_id,
            status="queued",
            message=f"Training job started for {request.model_name}",
            estimated_duration_minutes=request.estimated_duration_minutes
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

async def _train_model_background(
    job_id: str,
    model_name: str,
    training_data: pd.DataFrame,
    hyperparameters: Dict[str, Any],
    output_path: Optional[str]
):
    """Background task for model training."""
    try:
        # Update job status
        job_manager.update_job(job_id, status="running")
        
        # Create model
        model = ModelFactory.create(model_name, random_state=42)
        
        # Configure hyperparameters
        if hyperparameters:
            model.set_config(hyperparameters)
        
        # Train model
        logger.info(f"Starting training for job {job_id}")
        training_result = model.train(training_data, **hyperparameters)
        
        # Save model if output path provided
        model_path = None
        if output_path:
            model_path = f"api_data/models/{sanitize_filename(output_path)}"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save_model(model_path)
        
        # Update job with results
        job_manager.update_job(
            job_id,
            status="completed",
            results={
                "training_result": training_result,
                "model_path": model_path,
                "model_info": model.get_model_info()
            }
        )
        
        logger.info(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        job_manager.update_job(
            job_id,
            status="failed",
            error=str(e)
        )

# Generation endpoints
@app.post("/generate", response_model=GenerationResponse)
async def generate_data(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate synthetic data using a trained model."""
    try:
        # Verify authentication
        await verify_api_key(credentials.credentials)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Validate model path or training job
        if request.model_path and not os.path.exists(request.model_path):
            raise HTTPException(status_code=400, detail="Model file not found")
        
        if request.training_job_id:
            training_job = job_manager.get_job(request.training_job_id)
            if not training_job or training_job.status != "completed":
                raise HTTPException(status_code=400, detail="Training job not found or not completed")
        
        # Create generation job
        job_manager.create_job(
            job_id=job_id,
            job_type="generation",
            status="queued",
            metadata={
                "num_samples": request.num_samples,
                "model_path": request.model_path,
                "training_job_id": request.training_job_id,
                "output_format": request.output_format
            }
        )
        
        # Start generation in background
        background_tasks.add_task(
            _generate_data_background,
            job_id,
            request
        )
        
        logger.info(f"Generation job {job_id} started")
        
        return GenerationResponse(
            job_id=job_id,
            status="queued",
            message=f"Generation job started for {request.num_samples} samples"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting generation: {str(e)}")

async def _generate_data_background(job_id: str, request: GenerationRequest):
    """Background task for data generation."""
    try:
        # Update job status
        job_manager.update_job(job_id, status="running")
        
        # Load model
        if request.model_path:
            # Load from file (implementation depends on model type)
            raise NotImplementedError("Loading from file not yet implemented")
        elif request.training_job_id:
            # Use model from training job
            training_job = job_manager.get_job(request.training_job_id)
            model_path = training_job.results.get("model_path")
            if not model_path:
                raise ValueError("No model path found in training job")
            # Load model (implementation needed)
            raise NotImplementedError("Loading from training job not yet implemented")
        else:
            raise ValueError("No model source specified")
        
        # Generate synthetic data
        logger.info(f"Generating {request.num_samples} samples for job {job_id}")
        synthetic_data = model.generate(request.num_samples)
        
        # Save output
        output_path = f"api_data/outputs/{job_id}.{request.output_format}"
        
        if request.output_format == "csv":
            synthetic_data.to_csv(output_path, index=False)
        elif request.output_format == "json":
            synthetic_data.to_json(output_path, orient="records")
        elif request.output_format == "parquet":
            synthetic_data.to_parquet(output_path)
        
        # Update job with results
        job_manager.update_job(
            job_id,
            status="completed",
            results={
                "output_path": output_path,
                "num_samples_generated": len(synthetic_data),
                "data_shape": synthetic_data.shape,
                "column_names": list(synthetic_data.columns)
            }
        )
        
        logger.info(f"Generation job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Generation job {job_id} failed: {e}")
        job_manager.update_job(
            job_id,
            status="failed",
            error=str(e)
        )

# Evaluation endpoints
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_synthetic_data(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Evaluate quality of synthetic data against original data."""
    try:
        # Verify authentication
        await verify_api_key(credentials.credentials)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Load data files
        try:
            original_data = pd.read_csv(request.original_data_path)
            synthetic_data = pd.read_csv(request.synthetic_data_path)
            validate_dataframe(original_data)
            validate_dataframe(synthetic_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")
        
        # Create evaluation job
        job_manager.create_job(
            job_id=job_id,
            job_type="evaluation",
            status="queued",
            metadata={
                "original_data_shape": original_data.shape,
                "synthetic_data_shape": synthetic_data.shape,
                "target_column": request.target_column,
                "metrics": request.metrics
            }
        )
        
        # Start evaluation in background
        background_tasks.add_task(
            _evaluate_data_background,
            job_id,
            original_data,
            synthetic_data,
            request.target_column,
            request.metrics,
            request.dataset_metadata or {}
        )
        
        logger.info(f"Evaluation job {job_id} started")
        
        return EvaluationResponse(
            job_id=job_id,
            status="queued",
            message="Evaluation job started"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting evaluation: {str(e)}")

async def _evaluate_data_background(
    job_id: str,
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: Optional[str],
    metrics: List[str],
    dataset_metadata: Dict[str, Any]
):
    """Background task for data evaluation."""
    try:
        # Update job status
        job_manager.update_job(job_id, status="running")
        
        # Create evaluator
        evaluator = UnifiedEvaluator(random_state=42)
        
        # Prepare metadata
        eval_metadata = {
            'dataset_info': dataset_metadata.get('dataset_info', {'name': f'api_eval_{job_id}'}),
            'target_info': {'column': target_column, 'type': 'auto'} if target_column else None
        }
        
        # Run evaluation
        logger.info(f"Running evaluation for job {job_id}")
        results = evaluator.run_complete_evaluation(
            model=None,  # No model object needed for evaluation-only
            original_data=original_data,
            synthetic_data=synthetic_data,
            dataset_metadata=eval_metadata,
            output_dir=f"api_data/outputs/eval_{job_id}",
            target_column=target_column
        )
        
        # Extract key metrics
        summary = {
            "trts_overall_score": results.get('trts_results', {}).get('overall_score_percent', 0),
            "similarity_score": results.get('similarity_analysis', {}).get('final_similarity', 0),
            "data_quality_score": results.get('data_quality', {}).get('data_type_consistency', 0),
            "statistical_similarity": results.get('statistical_tests', {}).get('overall_similarity', 0)
        }
        
        # Update job with results
        job_manager.update_job(
            job_id,
            status="completed",
            results={
                "evaluation_results": results,
                "summary": summary,
                "output_directory": f"api_data/outputs/eval_{job_id}"
            }
        )
        
        logger.info(f"Evaluation job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation job {job_id} failed: {e}")
        job_manager.update_job(
            job_id,
            status="failed",
            error=str(e)
        )

# Job management endpoints
@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get status of a specific job."""
    try:
        await verify_api_key(credentials.credentials)
        
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatus(
            job_id=job.job_id,
            status=job.status,
            job_type=job.job_type,
            created_at=job.created_at,
            updated_at=job.updated_at,
            progress=job.progress,
            metadata=job.metadata,
            results=job.results if job.status == "completed" else None,
            error=job.error if job.status == "failed" else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")

@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs(
    limit: int = 50,
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """List jobs with optional filtering."""
    try:
        await verify_api_key(credentials.credentials)
        
        jobs = job_manager.list_jobs(limit=limit, job_type=job_type, status=status)
        
        return [
            JobStatus(
                job_id=job.job_id,
                status=job.status,
                job_type=job.job_type,
                created_at=job.created_at,
                updated_at=job.updated_at,
                progress=job.progress,
                metadata=job.metadata,
                results=job.results if job.status == "completed" else None,
                error=job.error if job.status == "failed" else None
            )
            for job in jobs
        ]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {str(e)}")

@app.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Cancel a running job."""
    try:
        await verify_api_key(credentials.credentials)
        
        success = job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {"message": f"Job {job_id} cancelled successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")

# File download endpoints
@app.get("/download/{job_id}")
async def download_results(
    job_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Download results from a completed job."""
    try:
        await verify_api_key(credentials.credentials)
        
        job = job_manager.get_job(job_id)
        if not job or job.status != "completed":
            raise HTTPException(status_code=404, detail="Job not found or not completed")
        
        output_path = job.results.get("output_path")
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Output file not found")
        
        return FileResponse(
            path=output_path,
            filename=os.path.basename(output_path),
            media_type='application/octet-stream'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading results: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading results: {str(e)}")

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )