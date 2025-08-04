"""
Job management system for tracking asynchronous operations.
"""

import asyncio
import logging
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class Job:
    """Represents a single job in the system."""
    job_id: str
    job_type: str  # training, generation, evaluation
    status: str    # queued, running, completed, failed, cancelled
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0  # 0-100
    metadata: Dict[str, Any] = None
    results: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.results is None:
            self.results = {}

class JobManager:
    """Manages job lifecycle and persistence."""
    
    def __init__(self, db_path: str = "api_data/jobs.db"):
        self.db_path = db_path
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()
        
        # Create directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load existing jobs
        self._load_jobs()
        
        logger.info(f"JobManager initialized with {len(self.jobs)} existing jobs")
    
    def _init_database(self):
        """Initialize SQLite database for job persistence."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        progress REAL DEFAULT 0.0,
                        metadata TEXT,
                        results TEXT,
                        error TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _load_jobs(self):
        """Load existing jobs from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM jobs")
                for row in cursor.fetchall():
                    job_id, job_type, status, created_at, updated_at, progress, metadata, results, error = row
                    
                    job = Job(
                        job_id=job_id,
                        job_type=job_type,
                        status=status,
                        created_at=datetime.fromisoformat(created_at),
                        updated_at=datetime.fromisoformat(updated_at),
                        progress=progress or 0.0,
                        metadata=json.loads(metadata) if metadata else {},
                        results=json.loads(results) if results else {},
                        error=error
                    )
                    self.jobs[job_id] = job
                    
        except Exception as e:
            logger.error(f"Error loading jobs from database: {e}")
    
    def _save_job(self, job: Job):
        """Save job to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO jobs 
                    (job_id, job_type, status, created_at, updated_at, progress, metadata, results, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job.job_id,
                    job.job_type,
                    job.status,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                    job.progress,
                    json.dumps(job.metadata) if job.metadata else None,
                    json.dumps(job.results) if job.results else None,
                    job.error
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving job {job.job_id}: {e}")
    
    def create_job(
        self,
        job_id: str,
        job_type: str,
        status: str = "queued",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Job:
        """Create a new job."""
        now = datetime.now(timezone.utc)
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=status,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.jobs[job_id] = job
            self._save_job(job)
        
        logger.info(f"Created {job_type} job {job_id}")
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update a job's status and/or data."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            
            # Update fields
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0.0, min(100.0, progress))
            if metadata is not None:
                job.metadata.update(metadata)
            if results is not None:
                job.results.update(results)
            if error is not None:
                job.error = error
            
            job.updated_at = datetime.now(timezone.utc)
            
            # Save to database
            self._save_job(job)
        
        logger.debug(f"Updated job {job_id}: status={status}, progress={progress}")
        return True
    
    def list_jobs(
        self,
        limit: int = 50,
        job_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Job]:
        """List jobs with optional filtering."""
        with self.lock:
            jobs = list(self.jobs.values())
        
        # Apply filters
        if job_type:
            jobs = [job for job in jobs if job.job_type == job_type]
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return jobs[:limit]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's in a cancellable state."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            
            if job.status in ["queued", "running"]:
                job.status = "cancelled"
                job.updated_at = datetime.now(timezone.utc)
                self._save_job(job)
                logger.info(f"Cancelled job {job_id}")
                return True
            
            return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its data."""
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            # Remove from memory
            del self.jobs[job_id]
            
            # Remove from database
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error deleting job {job_id} from database: {e}")
                return False
        
        logger.info(f"Deleted job {job_id}")
        return True
    
    def cleanup_old_jobs(self, max_age_days: int = 30):
        """Clean up old completed/failed jobs."""
        cutoff_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - max_age_days)
        
        jobs_to_delete = []
        
        with self.lock:
            for job_id, job in self.jobs.items():
                if (job.status in ["completed", "failed", "cancelled"] and 
                    job.updated_at < cutoff_date):
                    jobs_to_delete.append(job_id)
        
        for job_id in jobs_to_delete:
            self.delete_job(job_id)
        
        if jobs_to_delete:
            logger.info(f"Cleaned up {len(jobs_to_delete)} old jobs")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get job statistics."""
        with self.lock:
            jobs = list(self.jobs.values())
        
        stats = {
            "total_jobs": len(jobs),
            "by_status": {},
            "by_type": {},
            "active_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0
        }
        
        for job in jobs:
            # Count by status
            stats["by_status"][job.status] = stats["by_status"].get(job.status, 0) + 1
            
            # Count by type
            stats["by_type"][job.job_type] = stats["by_type"].get(job.job_type, 0) + 1
            
            # Count active/completed/failed
            if job.status in ["queued", "running"]:
                stats["active_jobs"] += 1
            elif job.status == "completed":
                stats["completed_jobs"] += 1
            elif job.status == "failed":
                stats["failed_jobs"] += 1
        
        return stats
    
    async def cleanup(self):
        """Cleanup resources on shutdown."""
        # Cancel any running jobs
        jobs_to_cancel = []
        
        with self.lock:
            for job_id, job in self.jobs.items():
                if job.status == "running":
                    jobs_to_cancel.append(job_id)
        
        for job_id in jobs_to_cancel:
            self.cancel_job(job_id)
        
        logger.info("JobManager cleanup completed")

class JobQueue:
    """Queue system for managing job execution order."""
    
    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.queue = queue.PriorityQueue()
        self.running_jobs = set()
        self.lock = threading.Lock()
        
    def add_job(self, job_id: str, priority: int = 0):
        """Add job to queue with priority (lower number = higher priority)."""
        self.queue.put((priority, job_id))
    
    def get_next_job(self) -> Optional[str]:
        """Get next job to execute if under concurrency limit."""
        with self.lock:
            if len(self.running_jobs) >= self.max_concurrent:
                return None
            
            try:
                priority, job_id = self.queue.get_nowait()
                self.running_jobs.add(job_id)
                return job_id
            except queue.Empty:
                return None
    
    def mark_job_completed(self, job_id: str):
        """Mark job as completed and remove from running set."""
        with self.lock:
            self.running_jobs.discard(job_id)
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get queue status information."""
        with self.lock:
            return {
                "queued": self.queue.qsize(),
                "running": len(self.running_jobs),
                "max_concurrent": self.max_concurrent
            }

class JobScheduler:
    """Advanced job scheduler with priority and resource management."""
    
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.training_queue = JobQueue(max_concurrent=2)  # Training is resource-intensive
        self.generation_queue = JobQueue(max_concurrent=4)  # Generation is lighter
        self.evaluation_queue = JobQueue(max_concurrent=3)  # Evaluation is medium
        
        self.is_running = False
        self.scheduler_task = None
    
    async def start(self):
        """Start the job scheduler."""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Job scheduler started")
    
    async def stop(self):
        """Stop the job scheduler."""
        self.is_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Job scheduler stopped")
    
    def schedule_job(self, job_id: str, priority: int = 0):
        """Schedule a job for execution."""
        job = self.job_manager.get_job(job_id)
        if not job:
            return
        
        if job.job_type == "training":
            self.training_queue.add_job(job_id, priority)
        elif job.job_type == "generation":
            self.generation_queue.add_job(job_id, priority)
        elif job.job_type == "evaluation":
            self.evaluation_queue.add_job(job_id, priority)
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Check each queue for jobs to execute
                for queue_name, job_queue in [
                    ("training", self.training_queue),
                    ("generation", self.generation_queue),
                    ("evaluation", self.evaluation_queue)
                ]:
                    job_id = job_queue.get_next_job()
                    if job_id:
                        # Execute job in background
                        asyncio.create_task(self._execute_job(job_id, job_queue))
                
                # Sleep before next check
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)
    
    async def _execute_job(self, job_id: str, job_queue: JobQueue):
        """Execute a single job."""
        try:
            job = self.job_manager.get_job(job_id)
            if not job:
                return
            
            # Update job status
            self.job_manager.update_job(job_id, status="running")
            
            # Simulate job execution (replace with actual implementation)
            await asyncio.sleep(2)  # Placeholder
            
            # Mark as completed
            self.job_manager.update_job(job_id, status="completed", progress=100.0)
            
        except Exception as e:
            logger.error(f"Error executing job {job_id}: {e}")
            self.job_manager.update_job(job_id, status="failed", error=str(e))
        
        finally:
            job_queue.mark_job_completed(job_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "is_running": self.is_running,
            "training_queue": self.training_queue.get_queue_status(),
            "generation_queue": self.generation_queue.get_queue_status(),
            "evaluation_queue": self.evaluation_queue.get_queue_status()
        }