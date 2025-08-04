#!/usr/bin/env python3
"""
Test script for the Synthetic Data Generation API.
"""

import sys
import requests
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "synthetic-data-api-key-12345"  # Default development key

def create_test_data():
    """Create test dataset for API testing."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.randint(20, 70, 100),
        'income': np.random.normal(50000, 15000, 100),
        'education': np.random.choice(['High School', 'Bachelor', 'Master'], 100),
        'score': np.random.uniform(0, 100, 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    return data

def make_request(method, endpoint, data=None, files=None, timeout=30):
    """Make authenticated request to API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json" if data and not files else None
    }
    
    if headers["Content-Type"] is None:
        del headers["Content-Type"]
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == "POST":
            if files:
                headers_no_content_type = {k: v for k, v in headers.items() if k != "Content-Type"}
                response = requests.post(url, headers=headers_no_content_type, data=data, files=files, timeout=timeout)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def test_health_check():
    """Test health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{API_BASE_URL}/health", timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_list_models():
    """Test listing available models."""
    print("Testing model listing...")
    response = make_request("GET", "/models")
    
    if response and response.status_code == 200:
        models = response.json()
        print(f"✅ Found {len(models)} models:")
        for name, info in models.items():
            status = "✅" if info['available'] else "❌"
            print(f"  {status} {name}: {info['type']} - {info['description']}")
        return True
    else:
        print(f"❌ Model listing failed: {response.status_code if response else 'No response'}")
        return False

def test_training_workflow():
    """Test complete training workflow."""
    print("Testing training workflow...")
    
    # Create test data
    test_data = create_test_data()
    test_file = "test_data.csv"
    test_data.to_csv(test_file, index=False)
    
    try:
        # Upload data and start training
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'text/csv')}
            training_data = {
                'model_name': 'ganeraid',
                'hyperparameters': json.dumps({
                    'epochs': 20,  # Quick training for testing
                    'lr_g': 0.001,
                    'lr_d': 0.001
                })
            }
            
            response = make_request("POST", "/train", data=training_data, files=files)
        
        if not response or response.status_code != 200:
            print(f"❌ Training request failed: {response.status_code if response else 'No response'}")
            return False
        
        training_result = response.json()
        job_id = training_result['job_id']
        print(f"✅ Training started: {job_id}")
        
        # Poll for completion
        max_attempts = 60  # 5 minutes max
        for attempt in range(max_attempts):
            response = make_request("GET", f"/jobs/{job_id}")
            
            if response and response.status_code == 200:
                status_data = response.json()
                status = status_data['status']
                progress = status_data.get('progress', 0)
                
                print(f"  Status: {status}, Progress: {progress:.1f}%")
                
                if status == "completed":
                    print("✅ Training completed successfully!")
                    return True, job_id
                elif status == "failed":
                    error = status_data.get('error', 'Unknown error')
                    print(f"❌ Training failed: {error}")
                    return False
                elif status in ["cancelled"]:
                    print(f"❌ Training was cancelled")
                    return False
            
            time.sleep(5)
        
        print("❌ Training timed out")
        return False
        
    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)

def test_generation_workflow(training_job_id=None):
    """Test data generation workflow."""
    print("Testing generation workflow...")
    
    if not training_job_id:
        print("❌ No training job ID provided for generation test")
        return False
    
    generation_data = {
        'num_samples': 50,
        'training_job_id': training_job_id,
        'output_format': 'csv'
    }
    
    response = make_request("POST", "/generate", data=generation_data)
    
    if not response or response.status_code != 200:
        print(f"❌ Generation request failed: {response.status_code if response else 'No response'}")
        return False
    
    generation_result = response.json()
    job_id = generation_result['job_id']
    print(f"✅ Generation started: {job_id}")
    
    # Poll for completion
    max_attempts = 30  # 2.5 minutes max
    for attempt in range(max_attempts):
        response = make_request("GET", f"/jobs/{job_id}")
        
        if response and response.status_code == 200:
            status_data = response.json()
            status = status_data['status']
            
            print(f"  Status: {status}")
            
            if status == "completed":
                print("✅ Generation completed successfully!")
                return True
            elif status == "failed":
                error = status_data.get('error', 'Unknown error')
                print(f"❌ Generation failed: {error}")
                return False
        
        time.sleep(5)
    
    print("❌ Generation timed out")
    return False

def test_job_management():
    """Test job management endpoints."""
    print("Testing job management...")
    
    # List jobs
    response = make_request("GET", "/jobs?limit=10")
    
    if response and response.status_code == 200:
        jobs = response.json()
        print(f"✅ Found {len(jobs)} jobs")
        
        for job in jobs[:3]:  # Show first 3 jobs
            print(f"  {job['job_id'][:8]}... - {job['job_type']} - {job['status']}")
        
        return True
    else:
        print(f"❌ Job listing failed: {response.status_code if response else 'No response'}")
        return False

def test_error_handling():
    """Test API error handling."""
    print("Testing error handling...")
    
    # Test with invalid API key
    headers = {"Authorization": "Bearer invalid-key"}
    response = requests.get(f"{API_BASE_URL}/models", headers=headers)
    
    if response.status_code == 401:
        print("✅ Invalid API key properly rejected")
    else:
        print(f"❌ Invalid API key not handled correctly: {response.status_code}")
        return False
    
    # Test with invalid model name
    invalid_training_data = {
        'model_name': 'invalid_model',
        'data_content': '{"test": "data"}'
    }
    
    response = make_request("POST", "/train", data=invalid_training_data)
    
    if response and response.status_code == 400:
        print("✅ Invalid model name properly rejected")
        return True
    else:
        print(f"❌ Invalid model name not handled correctly: {response.status_code if response else 'No response'}")
        return False

def run_comprehensive_test():
    """Run comprehensive API test suite."""
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION API TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Health check
    results['health'] = test_health_check()
    print()
    
    # Test 2: Model listing
    results['models'] = test_list_models()
    print()
    
    # Test 3: Error handling
    results['errors'] = test_error_handling()
    print()
    
    # Test 4: Job management
    results['jobs'] = test_job_management()
    print()
    
    # Test 5: Training workflow (takes longest)
    training_result = test_training_workflow()
    if isinstance(training_result, tuple):
        results['training'] = training_result[0]
        training_job_id = training_result[1]
    else:
        results['training'] = training_result
        training_job_id = None
    print()
    
    # Test 6: Generation workflow (if training succeeded)
    if training_job_id:
        results['generation'] = test_generation_workflow(training_job_id)
    else:
        results['generation'] = False
        print("⏭️  Skipping generation test (no training job)")
    print()
    
    # Summary
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():.<20} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! API is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)