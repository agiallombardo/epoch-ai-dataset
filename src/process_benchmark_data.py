#!/usr/bin/env python3
"""
Process benchmark data from Epoch AI and integrate with model data.

Downloads benchmark data from https://epoch.ai/data/benchmark_data.zip
and processes it to create a unified dataset with benchmark scores.
"""

import urllib.request
import zipfile
import csv
import json
import os
import tempfile
import re
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Constants
BENCHMARK_DATA_URL = "https://epoch.ai/data/benchmark_data.zip"
OUTPUT_DIR = "data/processed"
DATE_FORMAT = '%Y-%m-%d'

def normalize_model_name(name: str) -> str:
    """Normalize model name for matching."""
    if not name or pd.isna(name):
        return ""
    # Convert to string and lowercase, then remove common variations
    name = str(name).lower().strip()
    # Remove version suffixes like _none, _low, _medium, _high, _32K, _16K
    name = re.sub(r'_[0-9]+k', '', name)
    name = re.sub(r'_(none|low|medium|high|preview)', '', name)
    # Remove date suffixes like -20251101
    name = re.sub(r'-\d{8}', '', name)
    # Remove common prefixes/suffixes
    name = re.sub(r'^(model|version|v)\s*', '', name)
    return name.strip()

def normalize_organization(org: str) -> str:
    """Normalize organization name for matching."""
    if not org or pd.isna(org):
        return ""
    org = str(org).lower().strip()
    # Common mappings
    mappings = {
        "google deepmind": "google",
        "deepmind": "google",
        "openai": "openai",
        "anthropic": "anthropic",
        "meta": "meta",
        "facebook": "meta",
        "facebook ai research": "meta",
    }
    return mappings.get(org, org)

def load_benchmark_index(zip_path: str) -> pd.DataFrame:
    """Load the epoch_capabilities_index.csv from the zip file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        index_path = os.path.join(temp_dir, "epoch_capabilities_index.csv")
        if not os.path.exists(index_path):
            raise FileNotFoundError("epoch_capabilities_index.csv not found in zip file")
        
        df = pd.read_csv(index_path)
        return df

def load_individual_benchmarks(zip_path: str) -> Dict[str, pd.DataFrame]:
    """Load all individual benchmark CSV files."""
    benchmarks = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find all CSV files except the index
        for file in os.listdir(temp_dir):
            if file.endswith('.csv') and file != 'epoch_capabilities_index.csv':
                benchmark_name = file.replace('_external.csv', '').replace('.csv', '')
                file_path = os.path.join(temp_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    if 'Model version' in df.columns and 'Best score (across scorers)' in df.columns:
                        benchmarks[benchmark_name] = df
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
    
    return benchmarks

def process_benchmark_data(benchmark_dir: Optional[str] = None) -> Dict:
    """
    Download and process benchmark data.
    
    Args:
        benchmark_dir: Optional directory containing already extracted benchmark data.
                       If None, downloads from URL.
    
    Returns:
        Dictionary with processed benchmark data
    """
    if benchmark_dir:
        # Use existing extracted data
        extract_dir = benchmark_dir
        index_df = pd.read_csv(os.path.join(extract_dir, "epoch_capabilities_index.csv"))
        
        # Load individual benchmarks
        benchmarks = {}
        for file in os.listdir(extract_dir):
            if file.endswith('.csv') and file != 'epoch_capabilities_index.csv':
                benchmark_name = file.replace('_external.csv', '').replace('.csv', '')
                file_path = os.path.join(extract_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    if 'Model version' in df.columns:
                        benchmarks[benchmark_name] = df
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
    else:
        # Download and extract
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "benchmark_data.zip")
            urllib.request.urlretrieve(BENCHMARK_DATA_URL, zip_path)
            
            extract_dir = os.path.join(temp_dir, "benchmark_data")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Load index
            index_path = os.path.join(extract_dir, "epoch_capabilities_index.csv")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"epoch_capabilities_index.csv not found in {extract_dir}")
            
            index_df = pd.read_csv(index_path)
            
            # Load individual benchmarks
            benchmarks = {}
            for file in os.listdir(extract_dir):
                if file.endswith('.csv') and file != 'epoch_capabilities_index.csv':
                    benchmark_name = file.replace('_external.csv', '').replace('.csv', '')
                    file_path = os.path.join(extract_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        if 'Model version' in df.columns:
                            benchmarks[benchmark_name] = df
                    except Exception as e:
                        print(f"Warning: Could not load {file}: {e}")
            
            # Process index data (while still in temp directory)
            processed_index = []
            for _, row in index_df.iterrows():
                model_version = row.get('Model version', '') or ''
                eci_score = row.get('ECI Score', None)
                release_date = row.get('Release date', '') or ''
                organization = row.get('Organization', '') or ''
                country = row.get('Country', '') or ''
                model_name = row.get('Model name', '') or ''
                display_name = row.get('Display name', '') or ''
                
                # Normalize for matching
                normalized_model = normalize_model_name(model_name or model_version)
                normalized_org = normalize_organization(organization)
                
                processed_index.append({
                    'model_version': model_version,
                    'model_name': model_name,
                    'normalized_model': normalized_model,
                    'normalized_org': normalized_org,
                    'eci_score': float(eci_score) if pd.notna(eci_score) else None,
                    'release_date': release_date,
                    'organization': organization,
                    'country': country,
                    'display_name': display_name,
                })
            
            # Process individual benchmarks
            processed_benchmarks = {}
            for benchmark_name, df in benchmarks.items():
                benchmark_results = []
                for _, row in df.iterrows():
                    model_version = row.get('Model version', '')
                    best_score = row.get('Best score (across scorers)', None)
                    release_date = row.get('Release date', '')
                    organization = row.get('Organization', '')
                    
                    if pd.notna(best_score):
                        try:
                            score = float(best_score)
                        except (ValueError, TypeError):
                            score = None
                    else:
                        score = None
                    
                    if score is not None:
                        benchmark_results.append({
                            'model_version': model_version,
                            'score': score,
                            'release_date': release_date,
                            'organization': organization,
                        })
                
                if benchmark_results:
                    processed_benchmarks[benchmark_name] = benchmark_results
            
            return {
                'index': processed_index,
                'benchmarks': processed_benchmarks,
            }
    
    # Process index data (for benchmark_dir case)
    processed_index = []
    for _, row in index_df.iterrows():
        model_version = row.get('Model version', '') or ''
        eci_score = row.get('ECI Score', None)
        release_date = row.get('Release date', '') or ''
        organization = row.get('Organization', '') or ''
        country = row.get('Country', '') or ''
        model_name = row.get('Model name', '') or ''
        display_name = row.get('Display name', '') or ''
        
        # Normalize for matching
        normalized_model = normalize_model_name(model_name or model_version)
        normalized_org = normalize_organization(organization)
        
        processed_index.append({
            'model_version': model_version,
            'model_name': model_name,
            'normalized_model': normalized_model,
            'normalized_org': normalized_org,
            'eci_score': float(eci_score) if pd.notna(eci_score) else None,
            'release_date': release_date,
            'organization': organization,
            'country': country,
            'display_name': display_name,
        })
    
    # Process individual benchmarks
    processed_benchmarks = {}
    for benchmark_name, df in benchmarks.items():
        benchmark_results = []
        for _, row in df.iterrows():
            model_version = row.get('Model version', '')
            best_score = row.get('Best score (across scorers)', None)
            release_date = row.get('Release date', '')
            organization = row.get('Organization', '')
            
            if pd.notna(best_score):
                try:
                    score = float(best_score)
                except (ValueError, TypeError):
                    score = None
            else:
                score = None
            
            if score is not None:
                benchmark_results.append({
                    'model_version': model_version,
                    'score': score,
                    'release_date': release_date,
                    'organization': organization,
                })
        
        if benchmark_results:
            processed_benchmarks[benchmark_name] = benchmark_results
    
    return {
        'index': processed_index,
        'benchmarks': processed_benchmarks,
    }

def merge_benchmark_with_models(model_data: List[Dict], benchmark_data: Dict) -> List[Dict]:
    """
    Merge benchmark data with model data by matching model names and organizations.
    
    Args:
        model_data: List of model records from modelsfetch.py
        benchmark_data: Processed benchmark data from process_benchmark_data()
    
    Returns:
        List of model records with benchmark data added
    """
    # Create lookup dictionaries
    index_lookup = {}
    for item in benchmark_data['index']:
        key = (item['normalized_model'], item['normalized_org'])
        if key not in index_lookup:
            index_lookup[key] = []
        index_lookup[key].append(item)
    
    # Create benchmark score aggregations by model
    benchmark_aggregates = {}
    for benchmark_name, results in benchmark_data['benchmarks'].items():
        for result in results:
            model_version = result['model_version']
            score = result['score']
            
            # Try to match with index
            normalized_model = normalize_model_name(model_version)
            normalized_org = normalize_organization(result.get('organization', ''))
            key = (normalized_model, normalized_org)
            
            if key not in benchmark_aggregates:
                benchmark_aggregates[key] = {}
            if benchmark_name not in benchmark_aggregates[key]:
                benchmark_aggregates[key][benchmark_name] = []
            benchmark_aggregates[key][benchmark_name].append(score)
    
    # Merge with model data
    merged_data = []
    for model in model_data:
        model_copy = model.copy()
        
        # Try to match model
        model_name = model.get('model', '')
        company = model.get('company', '')
        normalized_model = normalize_model_name(model_name)
        normalized_org = normalize_organization(company)
        
        # Look for ECI score
        key = (normalized_model, normalized_org)
        if key in index_lookup:
            # Use the most recent match
            matches = index_lookup[key]
            if matches:
                best_match = matches[0]  # Could improve by matching dates
                if best_match['eci_score'] is not None:
                    model_copy['eci_score'] = best_match['eci_score']
        
        # Look for benchmark scores
        if key in benchmark_aggregates:
            benchmark_scores = {}
            for benchmark_name, scores in benchmark_aggregates[key].items():
                # Use the best score
                benchmark_scores[benchmark_name] = max(scores)
            model_copy['benchmark_scores'] = benchmark_scores
        
        merged_data.append(model_copy)
    
    return merged_data

def save_benchmark_data(benchmark_data: Dict, output_path: str):
    """Save processed benchmark data to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

def main():
    """Main function to process benchmark data."""
    print("Downloading benchmark data...")
    benchmark_data = process_benchmark_data()
    
    print(f"Processed {len(benchmark_data['index'])} model entries from index")
    print(f"Processed {len(benchmark_data['benchmarks'])} benchmark datasets")
    
    # Save processed benchmark data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    benchmark_output = os.path.join(OUTPUT_DIR, "benchmark_data.json")
    save_benchmark_data(benchmark_data, benchmark_output)
    print(f"Saved benchmark data to {benchmark_output}")
    
    # Also save as CSV for easier inspection
    index_df = pd.DataFrame(benchmark_data['index'])
    index_csv = os.path.join(OUTPUT_DIR, "benchmark_index.csv")
    index_df.to_csv(index_csv, index=False)
    print(f"Saved benchmark index to {index_csv}")

if __name__ == "__main__":
    main()

