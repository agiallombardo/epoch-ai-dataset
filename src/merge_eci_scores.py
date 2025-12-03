#!/usr/bin/env python3
"""
Merge ECI scores from Epoch AI benchmark data into the notable AI models JSON file.

Downloads benchmark data from https://epoch.ai/data/benchmark_data.zip,
extracts ECI scores, and merges them into model records.
All models will have an eci_score field (null if not available).
"""

import urllib.request
import zipfile
import json
import os
import tempfile
import re
from typing import Dict, List

# Constants
BENCHMARK_DATA_URL = "https://epoch.ai/data/benchmark_data.zip"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
JSON_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def normalize_model_name(name: str) -> str:
    """Normalize model name for matching."""
    if not name:
        return ""
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
    if not org:
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

def extract_eci_scores() -> Dict:
    """
    Download benchmark data and extract ECI scores.
    
    Returns:
        Dictionary mapping (normalized_model, normalized_org) -> max_eci_score
    """
    print("Downloading benchmark data...")
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "benchmark_data.zip")
        urllib.request.urlretrieve(BENCHMARK_DATA_URL, zip_path)
        
        extract_dir = os.path.join(temp_dir, "benchmark_data")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Load only the ECI index file
        index_path = os.path.join(extract_dir, "epoch_capabilities_index.csv")
        if not os.path.exists(index_path):
            raise FileNotFoundError("epoch_capabilities_index.csv not found in zip file")
        
        # Read CSV and extract ECI scores
        import csv
        eci_lookup = {}
        
        with open(index_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get('Model name', '') or row.get('Model version', '')
                organization = row.get('Organization', '')
                eci_score_str = row.get('ECI Score', '')
                
                # Skip if no ECI score
                if not eci_score_str or eci_score_str.strip() == '':
                    continue
                
                try:
                    eci_score = float(eci_score_str)
                except (ValueError, TypeError):
                    continue
                
                # Normalize for matching
                normalized_model = normalize_model_name(model_name)
                normalized_org = normalize_organization(organization)
                
                if normalized_model and normalized_org:
                    key = (normalized_model, normalized_org)
                    # Keep the maximum ECI score if multiple entries exist
                    if key not in eci_lookup:
                        eci_lookup[key] = eci_score
                    else:
                        eci_lookup[key] = max(eci_lookup[key], eci_score)
        
        print(f"Extracted {len(eci_lookup)} unique model-organization pairs with ECI scores")
        return eci_lookup

def load_notable_models() -> tuple[List[Dict], str]:
    """Load the most recent notable AI models JSON file."""
    if not os.path.exists(JSON_OUTPUT_DIR):
        raise FileNotFoundError(f"Directory {JSON_OUTPUT_DIR} does not exist")
    
    # Find all notable JSON files in the processed directory
    json_files = [f for f in os.listdir(JSON_OUTPUT_DIR) if f.endswith('.json') and 'notable' in f.lower()]
    
    if not json_files:
        raise FileNotFoundError(f"No notable JSON files found in {JSON_OUTPUT_DIR}")
    
    # Sort by modification time and use the most recent
    json_files.sort(key=lambda f: os.path.getmtime(os.path.join(JSON_OUTPUT_DIR, f)), reverse=True)
    notable_file = os.path.join(JSON_OUTPUT_DIR, json_files[0])
    
    with open(notable_file, 'r', encoding='utf-8') as f:
        return json.load(f), notable_file

def merge_eci_scores(notable_models: List[Dict], eci_lookup: Dict) -> List[Dict]:
    """Merge ECI scores into notable models data."""
    merged_models = []
    matched_count = 0
    
    for model in notable_models:
        model_copy = model.copy()
        
        # Try to match model
        model_name = model.get('model', '')
        company = model.get('company', '')
        
        normalized_model = normalize_model_name(model_name)
        normalized_org = normalize_organization(company)
        
        key = (normalized_model, normalized_org)
        
        # Always add eci_score field - set to null if not found
        eci_score = None
        
        # Try exact match first
        if key in eci_lookup:
            eci_score = eci_lookup[key]
            matched_count += 1
        else:
            # Fallback: try matching by model name only (ignore organization)
            for (norm_model, _), score in eci_lookup.items():
                if norm_model == normalized_model:
                    eci_score = score
                    matched_count += 1
                    break
        
        # Always set eci_score field (null if not found)
        model_copy['eci_score'] = eci_score
        
        merged_models.append(model_copy)
    
    print(f"Matched ECI scores for {matched_count} out of {len(notable_models)} models")
    return merged_models

def main():
    """Main function to download, extract, and merge ECI scores."""
    # Extract ECI scores from benchmark data
    eci_lookup = extract_eci_scores()
    
    # Load notable models
    print("Loading notable AI models...")
    notable_models, notable_file = load_notable_models()
    print(f"Loaded {len(notable_models)} models from {os.path.basename(notable_file)}")
    
    # Merge ECI scores
    print("Merging ECI scores...")
    merged_models = merge_eci_scores(notable_models, eci_lookup)
    
    # Save the updated JSON file
    print(f"Saving updated models to {os.path.basename(notable_file)}...")
    with open(notable_file, 'w', encoding='utf-8') as f:
        json.dump(merged_models, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully merged ECI scores into {os.path.basename(notable_file)}")
    
    # Count how many models now have ECI scores
    models_with_eci = sum(1 for m in merged_models if m.get('eci_score') is not None)
    models_with_null = sum(1 for m in merged_models if m.get('eci_score') is None)
    print(f"Total models with ECI scores: {models_with_eci}")
    print(f"Total models with eci_score = null: {models_with_null}")

if __name__ == "__main__":
    main()
