import urllib.request
import zipfile
import csv
import json
import os
import tempfile
import re
from datetime import datetime

# Constants
DATA_URL = "https://epoch.ai/data/ai_models.zip"
OUTPUT_DIR = "data/processed"
CSV_FILENAME = "notable_ai_models.csv"
DATE_FORMAT = '%Y-%m-%d'
MIN_DATE = datetime(2018, 1, 1)

# Generate output JSON filename with date
def get_output_json_filename():
    """Generate output JSON filename with current date in MM-DD-YYYY format."""
    today = datetime.now()
    date_str = today.strftime('%m-%d-%Y')
    return f"notable_ai_models_{date_str}.json"

# Country relationship sets (module-level for performance)
# Includes both original variations and normalized names (lowercase for lookup)
_FEYE_COUNTRIES = {
    "united states", "united states of america", "usa", "us",
    "united kingdom", "united kingdom of great britain and northern ireland", 
    "uk", "great britain", "gb",
    "canada",
    "australia",
    "new zealand",
}

_NATO_COUNTRIES = {
    "belgium", "denmark", "france", "germany", "greece", "iceland",
    "italy", "luxembourg", "netherlands", "norway", "portugal", "spain",
    "turkey", "poland", "czech republic", "hungary", "estonia", "latvia",
    "lithuania", "slovakia", "slovenia", "bulgaria", "romania", "albania",
    "croatia", "montenegro", "north macedonia", "finland", "sweden",
}

_CLOSE_ALLIES = {
    "japan", "south korea", "korea (republic of)", "israel", "taiwan",
    "singapore", "philippines", "thailand", "india", "indonesia",
    "switzerland", "austria", "ireland", "malta", "cyprus",
    "hong kong",
    "south africa", "brazil", "mexico", "chile", "colombia",
    "united arab emirates", "qatar", "kuwait", "bahrain", "oman",
    "saudi arabia",  # Strategic partner despite some tensions
}

_ADVERSARIES = {
    "china", "russia", "north korea", "iran", "cuba", "venezuela",
    "belarus", "syria", "myanmar", "sudan",
}

def to_camel_case(s):
    """Convert a string to camelCase."""
    # Extract content from parentheses and include it in the conversion
    # e.g., "Country (of organization)" -> "Country of organization"
    s = re.sub(r'\(([^)]+)\)', r' \1', s)  # Replace parentheses with space and content
    # Remove special characters except spaces
    s = re.sub(r'[^\w\s]', '', s)
    # Split by spaces/underscores and convert to camelCase
    words = re.split(r'[\s_]+', s.strip())
    if not words:
        return s
    # Filter out empty words and convert to camelCase
    words = [w for w in words if w]
    if not words:
        return s
    # First word lowercase, rest capitalized
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:] if word)

def convert_to_number(value):
    """Convert a string value to a number if possible."""
    if not value or not isinstance(value, str):
        return value
    try:
        # Try to convert to float first
        num = float(value)
        # If it's a whole number, return as int
        if num.is_integer():
            return int(num)
        return num
    except (ValueError, TypeError):
        return value

def normalize_single_org(org):
    """Normalize a single organization string.
    - Google DeepMind -> company: "Google", department: "DeepMind"
    - DeepMind -> company: "Google", department: "DeepMind"
    - Google Research -> company: "Google", department: "Google Research"
    - Google Brain -> company: "Google", department: "Google Brain"
    - Google -> company: "Google", department: None
    - Facebook AI Research -> company: "Meta", department: "Facebook AI Research"
    - Facebook AI -> company: "Meta", department: "Facebook AI"
    - Facebook -> company: "Meta", department: None
    - Meta AI -> company: "Meta", department: "Meta AI"
    - Meta -> company: "Meta", department: None
    """
    if not org or not isinstance(org, str):
        return None, None
    
    org_lower = org.lower().strip()
    
    # Google/DeepMind normalization (check DeepMind first before other Google departments)
    if "deepmind" in org_lower:
        return "Google", "DeepMind"
    elif "google research" in org_lower:
        return "Google", "Google Research"
    elif "google brain" in org_lower:
        return "Google", "Google Brain"
    elif org_lower == "google":
        return "Google", None
    
    # Facebook/Meta normalization (check specific Facebook departments first)
    elif "facebook ai research" in org_lower:
        return "Meta", "Facebook AI Research"
    elif "facebook ai" in org_lower:
        return "Meta", "Facebook AI"
    elif org_lower == "facebook":
        return "Meta", None
    elif "meta ai" in org_lower:
        return "Meta", "Meta AI"
    elif org_lower == "meta":
        return "Meta", None
    elif "meta" in org_lower:
        return "Meta", org  # Keep original as department if it's something like "Meta Research"
    
    # No normalization needed
    return org, None

def get_country_relationship(country):
    """
    Determine the relationship category of a country with the United States.
    Returns: "FEYE", "NATO", "Close Allies & Partners", "Adversaries", or "Other"
    
    Args:
        country: Normalized country name (will be converted to lowercase for lookup)
    """
    if not country or not isinstance(country, str):
        return None
    
    # Convert normalized country name to lowercase for lookup
    country_lower = country.lower().strip()
    
    # Check in priority order (using module-level sets for O(1) lookup)
    if country_lower in _FEYE_COUNTRIES:
        return "FEYE"
    elif country_lower in _NATO_COUNTRIES:
        return "NATO"
    elif country_lower in _ADVERSARIES:
        return "Adversaries"
    elif country_lower in _CLOSE_ALLIES:
        return "Close Allies & Partners"
    else:
        # Default for countries not in any category
        return "Other"

def get_country_group(country_value):
    """
    Categorize country/countries into relationship groups.
    Normalizes countries first, then categorizes them.
    Returns "Private Partnership" if:
    - Country is United States only, OR
    - Multiple countries including United States
    Otherwise returns the alliance relationship for single countries.
    """
    if not country_value:
        return None
    
    # Normalize country first before categorizing
    normalized_country = normalize_country(country_value)
    
    if normalized_country is None:
        return None
    
    if isinstance(normalized_country, list):
        # Multiple countries: check if United States is included
        if "United States" in normalized_country:
            return "Private Partnership"
        # Multiple countries without US -> still Private Partnership
        return "Private Partnership"
    elif isinstance(normalized_country, str):
        # Single country: if United States, return Private Partnership
        if normalized_country == "United States":
            return "Private Partnership"
        # Otherwise return the normal alliance relationship
        relationship = get_country_relationship(normalized_country)
        return relationship if relationship else "Other"
    
    return "Other"

def normalize_country(country_value):
    """
    Normalize country names to standard format.
    """
    if not country_value:
        return None
    
    # Country name mappings for common variations
    country_mappings = {
        "united states of america": "United States",
        "united states": "United States",
        "usa": "United States",
        "us": "United States",
        "united kingdom of great britain and northern ireland": "United Kingdom",
        "united kingdom": "United Kingdom",
        "uk": "United Kingdom",
        "great britain": "United Kingdom",
        "gb": "United Kingdom",
        "korea (republic of)": "Korea (Republic of)",
        "south korea": "Korea (Republic of)",
    }
    
    if isinstance(country_value, list):
        # Use set for O(1) membership check during deduplication
        seen = set()
        normalized = []
        for country in country_value:
            if isinstance(country, str):
                country_lower = country.lower().strip()
                mapped = country_mappings.get(country_lower, country)
                # Deduplicate while preserving order
                if mapped not in seen:
                    seen.add(mapped)
                    normalized.append(mapped)
            else:
                if country not in seen:
                    seen.add(country)
                    normalized.append(country)
        return single_or_list(normalized)
    elif isinstance(country_value, str):
        country_lower = country_value.lower().strip()
        return country_mappings.get(country_lower, country_value)
    
    return country_value

def normalize_model_accessibility(accessibility_value):
    """
    Normalize model accessibility values to standard format.
    """
    if not accessibility_value:
        return None
    
    if not isinstance(accessibility_value, str):
        return accessibility_value
    
    accessibility_lower = accessibility_value.lower().strip()
    
    # Standardize common variations
    accessibility_mappings = {
        "api access": "API access",
        "api": "API access",
        "open weights (unrestricted)": "Open weights (unrestricted)",
        "open weights unrestricted": "Open weights (unrestricted)",
        "open weights": "Open weights (unrestricted)",
        "open weights (restricted use)": "Open weights (restricted use)",
        "open weights restricted": "Open weights (restricted use)",
        "open weights (restricted)": "Open weights (restricted use)",
        "hosted access (no api)": "Hosted access (no API)",
        "hosted access": "Hosted access (no API)",
        "unreleased": "Unreleased",
        "closed": "Unreleased",
        "private": "Unreleased",
    }
    
    return accessibility_mappings.get(accessibility_lower, accessibility_value)

def normalize_domain(domain_value):
    """
    Normalize domain values to ensure consistent capitalization and format.
    """
    if not domain_value:
        return None
    
    # Domain name mappings for consistency
    domain_mappings = {
        "image generation": "Image generation",
        "3d modeling": "3D modeling",
        "3d": "3D modeling",
        "earth science": "Earth science",
        "materials science": "Materials science",
    }
    
    if isinstance(domain_value, list):
        normalized = []
        for domain in domain_value:
            if isinstance(domain, str):
                domain_lower = domain.lower().strip()
                # Use mapping if available, otherwise capitalize properly
                mapped = domain_mappings.get(domain_lower)
                if mapped:
                    normalized.append(mapped)
                else:
                    # Capitalize first letter of each word, preserving special cases
                    normalized.append(domain.strip())
            else:
                normalized.append(domain)
        # Deduplicate while preserving order
        unique_domains = deduplicate_list(normalized)
        return single_or_list(unique_domains)
    elif isinstance(domain_value, str):
        domain_lower = domain_value.lower().strip()
        mapped = domain_mappings.get(domain_lower)
        if mapped:
            return mapped
        return domain_value.strip()
    
    return domain_value

def normalize_training_hardware(hardware_value):
    """
    Normalize training hardware values to ensure consistent format.
    Ensures arrays are used for multiple values, strings for single values.
    """
    if not hardware_value:
        return None
    
    if isinstance(hardware_value, list):
        # Clean and deduplicate hardware list using set for O(1) lookup
        seen = set()
        cleaned = []
        for hw in hardware_value:
            if isinstance(hw, str):
                hw_cleaned = hw.strip()
                if hw_cleaned and hw_cleaned not in seen:
                    seen.add(hw_cleaned)
                    cleaned.append(hw_cleaned)
            else:
                if hw not in seen:
                    seen.add(hw)
                    cleaned.append(hw)
        # Return single value if only one, otherwise list
        return single_or_list(cleaned)
    elif isinstance(hardware_value, str):
        # Single string value - just clean it
        cleaned = hardware_value.strip()
        return cleaned if cleaned else None
    
    return hardware_value

def normalize_task(task_value):
    """
    Normalize task values to ensure consistent format and remove duplicates.
    """
    if not task_value:
        return None
    
    # Task name mappings for common variations
    task_mappings = {
        "language modeling": "Language modeling/generation",
        "language generation": "Language modeling/generation",
        "text generation": "Language modeling/generation",
        "question answering": "Question answering",
        "qa": "Question answering",
        "code generation": "Code generation",
        "object detection": "Object detection",
        "image classification": "Image classification",
        "speech recognition": "Speech recognition (ASR)",
        "asr": "Speech recognition (ASR)",
        "speech recognition (asr)": "Speech recognition (ASR)",
    }
    
    if isinstance(task_value, list):
        # Use set for O(1) membership check during deduplication
        seen = set()
        normalized = []
        for task in task_value:
            if isinstance(task, str):
                task_lower = task.lower().strip()
                # Use mapping if available, otherwise keep original
                mapped = task_mappings.get(task_lower)
                if mapped:
                    if mapped not in seen:
                        seen.add(mapped)
                        normalized.append(mapped)
                else:
                    # Keep original but ensure it's not a duplicate
                    task_stripped = task.strip()
                    if task_stripped not in seen:
                        seen.add(task_stripped)
                        normalized.append(task_stripped)
            else:
                if task not in seen:
                    seen.add(task)
                    normalized.append(task)
        # Return single value if only one, otherwise list
        return single_or_list(normalized)
    elif isinstance(task_value, str):
        task_lower = task_value.lower().strip()
        mapped = task_mappings.get(task_lower)
        if mapped:
            return mapped
        return task_value.strip()
    
    return task_value

def normalize_base_model(base_model_value):
    """
    Normalize base model references to ensure consistent format.
    """
    if not base_model_value:
        return None
    
    if isinstance(base_model_value, str):
        # Clean whitespace
        cleaned = base_model_value.strip()
        return cleaned if cleaned else None
    
    return base_model_value

def categorize_model_capacity(parameters):
    """
    Categorize model capacity based on parameters.
    Returns: "Small (<1B)", "Medium (1B-10B)", "Large (10B-100B)", 
             "Very Large (100B-1T)", "Massive (≥1T)", or "Unknown"
    """
    if not parameters or parameters is None:
        return "Unknown"
    
    # Convert to float if it's a string
    if isinstance(parameters, str):
        try:
            parameters = float(parameters)
        except (ValueError, TypeError):
            return "Unknown"
    
    if not isinstance(parameters, (int, float)):
        return "Unknown"
    
    parameters = float(parameters)
    
    if parameters < 1e9:  # < 1B
        return "Small (<1B)"
    elif parameters < 1e10:  # < 10B
        return "Medium (1B-10B)"
    elif parameters < 1e11:  # < 100B
        return "Large (10B-100B)"
    elif parameters < 1e12:  # < 1T
        return "Very Large (100B-1T)"
    else:  # >= 1T
        return "Massive (≥1T)"

def categorize_accessibility(accessibility_value):
    """
    Categorize model accessibility into broader categories.
    This is different from normalize_model_accessibility which standardizes the exact format.
    Returns: "Open Source (Unrestricted)", "Open Source (Restricted)", "Open Source",
             "Closed Source (API)", "Closed Source (Unreleased)", "Closed Source (Hosted)", 
             "Other", or "Unknown"
    """
    if not accessibility_value:
        return "Unknown"
    
    if not isinstance(accessibility_value, str):
        return "Unknown"
    
    accessibility_lower = accessibility_value.lower().strip()
    
    # Open source categories
    if "open weights" in accessibility_lower:
        if "unrestricted" in accessibility_lower:
            return "Open Source (Unrestricted)"
        elif "restricted" in accessibility_lower or "non-commercial" in accessibility_lower:
            return "Open Source (Restricted)"
        else:
            return "Open Source"
    elif "open" in accessibility_lower:
        return "Open Source"
    
    # Closed source categories
    elif "api access" in accessibility_lower or "api" in accessibility_lower:
        return "Closed Source (API)"
    elif "unreleased" in accessibility_lower or "closed" in accessibility_lower:
        return "Closed Source (Unreleased)"
    elif "hosted access" in accessibility_lower:
        return "Closed Source (Hosted)"
    
    return "Other"

def deduplicate_list(items):
    """Deduplicate a list while preserving order."""
    return list(dict.fromkeys(items))

def single_or_list(values):
    """Return single value if only one, otherwise return list."""
    if not values:
        return None
    return values[0] if len(values) == 1 else values

def normalize_organization_field(record):
    """Normalize organization field to company and department. Modifies record in place."""
    if 'organization' not in record:
        return
    
    org_value = record['organization']
    if isinstance(org_value, list):
        # Handle multiple organizations
        companies = []
        departments = []
        for org in org_value:
            company, department = normalize_single_org(org)
            if company:
                companies.append(company)
            if department:
                departments.append(department)
        # Deduplicate while preserving order
        unique_companies = deduplicate_list(companies)
        unique_departments = deduplicate_list(departments)
        # Return single value if only one, otherwise list
        record['company'] = single_or_list(unique_companies)
        record['department'] = single_or_list(unique_departments)
    else:
        # Single organization
        company, department = normalize_single_org(org_value)
        record['company'] = company
        record['department'] = department

def extract_single_value(value):
    """Extract single value from list or return value as-is. Optimized for data cleaning."""
    if not value:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value

def convert_to_numeric(value):
    """Convert value to float if possible, otherwise return None. Optimized for data cleaning."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return None

def categorize_sector(org_cat):
    """Categorize organization into sector (Private/Public/Other)."""
    if not org_cat:
        return "Other"
    if org_cat == "Industry":
        return "Private"
    elif org_cat == "Academia":
        return "Public"
    else:
        return "Other"

def extract_year_month(date_str):
    """Extract year and year-month from date string. Returns (year, year_month) or (None, None)."""
    if not date_str:
        return None, None
    try:
        date_obj = datetime.strptime(date_str, DATE_FORMAT)
        year = date_obj.year
        year_month = date_obj.strftime('%Y-%m')
        return year, year_month
    except (ValueError, TypeError):
        return None, None

def normalize_record_fields(record):
    """Normalize all fields in a record. Modifies record in place."""
    # Normalize organization to company and department
    normalize_organization_field(record)
    
    # Normalize country
    if 'countryOfOrganization' in record:
        normalized_country = normalize_country(record['countryOfOrganization'])
        record['countryOfOrganization'] = normalized_country
        # Extract single country value for easier access
        record['country'] = extract_single_value(normalized_country)
    
    # Special handling for DeepMind based on publication date
    company = record.get('company')
    department = record.get('department')
    publication_date = record.get('publicationDate')
    
    # Check if this is DeepMind (either Google/DeepMind or standalone DeepMind)
    is_deepmind = (company == 'Google' and department == 'DeepMind') or company == 'DeepMind'
    
    if is_deepmind and publication_date:
        try:
            # Parse publication date to get year
            date_obj = datetime.strptime(publication_date, DATE_FORMAT)
            year = date_obj.year
            
            if year < 2014:
                # Before 2014: DeepMind is standalone UK company
                record['company'] = 'DeepMind'
                record['department'] = None
                record['countryOfOrganization'] = 'United Kingdom'
                record['country'] = 'United Kingdom'
                # Let normal country relationship logic apply (UK = FEYE)
            else:
                # 2014 or later: DeepMind is part of Google in United States
                record['company'] = 'Google'
                record['department'] = 'DeepMind'
                record['countryOfOrganization'] = 'United States'
                record['country'] = 'United States'
                # Explicitly set countryRelationship to Private Partnership for DeepMind
                record['countryRelationship'] = 'Private Partnership'
        except (ValueError, TypeError):
            # If date parsing fails, default to post-2014 behavior (Google/DeepMind/US)
            if company == 'Google' and department == 'DeepMind':
                record['countryOfOrganization'] = 'United States'
                record['country'] = 'United States'
                record['countryRelationship'] = 'Private Partnership'
    elif company == 'Google' and department == 'DeepMind':
        # If no publication date but it's Google/DeepMind, assume post-2014 (default)
        record['countryOfOrganization'] = 'United States'
        record['country'] = 'United States'
        record['countryRelationship'] = 'Private Partnership'
    
    # Add country relationship categorization using normalized country (after DeepMind correction)
    if 'countryOfOrganization' in record and 'countryRelationship' not in record:
        record['countryRelationship'] = get_country_group(record['countryOfOrganization'])
    # Remove old countryGroup field if it exists
    if 'countryGroup' in record:
        del record['countryGroup']
    
    # Extract single company value
    if 'company' in record:
        record['company'] = extract_single_value(record['company']) or 'Unknown'
    
    # Normalize model accessibility
    if 'modelAccessibility' in record:
        record['modelAccessibility'] = normalize_model_accessibility(record['modelAccessibility'])
        # Also add categorized accessibility
        record['accessibilityCategory'] = categorize_accessibility(record['modelAccessibility'])
    
    # Categorize model capacity based on parameters
    if 'parameters' in record:
        # Ensure parameters is numeric (decimal format)
        record['parameters'] = convert_to_numeric(record['parameters'])
        record['modelCapacity'] = categorize_model_capacity(record['parameters'])
    
    # Convert trainingComputeFlop to numeric (decimal format)
    if 'trainingComputeFlop' in record:
        record['trainingComputeFlop'] = convert_to_numeric(record['trainingComputeFlop'])
    
    # Normalize domain and extract single value
    if 'domain' in record:
        record['domain'] = normalize_domain(record['domain'])
        record['domain'] = extract_single_value(record['domain'])
    
    # Normalize task and extract single value
    if 'task' in record:
        record['task'] = normalize_task(record['task'])
        record['task'] = extract_single_value(record['task'])
    
    # Normalize training hardware and extract single value
    if 'trainingHardware' in record:
        record['trainingHardware'] = normalize_training_hardware(record['trainingHardware'])
        record['trainingHardware'] = extract_single_value(record['trainingHardware'])
    
    # Normalize base model
    if 'baseModel' in record:
        record['baseModel'] = normalize_base_model(record['baseModel'])
    
    # Convert training compute to numeric
    if 'trainingCompute' in record:
        record['trainingCompute'] = convert_to_numeric(record['trainingCompute'])
    
    # Extract single value for notability criteria
    if 'notabilityCriteria' in record:
        record['notabilityCriteria'] = extract_single_value(record['notabilityCriteria'])
    
    # Extract single value for country relationship
    if 'countryRelationship' in record:
        record['countryRelationship'] = extract_single_value(record['countryRelationship']) or 'Unknown'
    
    # Categorize sector from organization categorization
    if 'organizationCategorization' in record:
        record['sector'] = categorize_sector(record['organizationCategorization'])
    
    # Extract year and year-month from publication date for easier time-based analysis
    if 'publicationDate' in record:
        year, year_month = extract_year_month(record['publicationDate'])
        if year:
            record['year'] = year
            record['yearMonth'] = year_month
    
    # Remove organization field if both company and department exist
    if 'organization' in record and 'company' in record and 'department' in record:
        if record.get('company') and record.get('department'):
            del record['organization']
    
    # Remove country field (keep countryOfOrganization)
    if 'country' in record:
        del record['country']

def download_and_convert():
    """
    Download AI models data from Epoch AI and convert to JSON.
    """
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "ai_models.zip")
        
        # Download the zip file
        print(f"Downloading {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        print("Download complete.")
        
        # Extract the zip file
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print("Extraction complete.")
        
        # Find CSV file in the extracted files
        csv_path = os.path.join(temp_dir, CSV_FILENAME)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{CSV_FILENAME} not found in the zip file")
        
        # Columns to exclude (using set for O(1) lookup)
        columns_to_drop = {
            'Parameters notes',
            'Training compute notes',
            'Training dataset',
            'Training dataset size (gradients)',
            'Dataset size notes',
            'Link',
            'Reference',
            'Citations',
            'Authors',
            'Abstract',
            'Notability criteria notes',
            'Epochs',
            'Training time (hours)',
            'Training time notes',
            'Numerical format',
            'Frontier model',
            'Hardware acquisition cost',
            'Hardware utilization (HFU)',
            'Training compute cost (cloud)',
            'Training compute cost (upfront)',
            'Finetune compute (FLOP)',
            'Finetune compute notes',
            'Batch size',
            'Batch size notes',
            'Hardware quantity',
            'Hardware utilization (MFU)',
            'Training compute cost (2023 USD)',
            'Compute cost notes',
            'Training code accessibility',
            "Training power draw (W)",
            'Inference code accessibility',
            'Accessibility notes'
        }
        
        # Read CSV and convert to JSON
        print("Converting CSV to JSON...")
        data = []
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # Convert multi-value fields to arrays and clean data
                processed_row = {}
                for key, value in row.items():
                    # Skip columns that should be dropped
                    if key in columns_to_drop:
                        continue
                    
                    # Normalize field name to camelCase
                    normalized_key = to_camel_case(key)
                    
                    # Trim whitespace from value
                    if isinstance(value, str):
                        value = value.strip()
                    
                    # Handle empty strings - convert to None (null in JSON)
                    if value == '':
                        processed_row[normalized_key] = None
                        continue
                    
                    # Check if value contains commas (potential multi-value field)
                    if value and ',' in value:
                        # Split by comma and trim whitespace
                        values = [v.strip() for v in value.split(',')]
                        # Filter out empty strings
                        non_empty_values = [v for v in values if v]
                        # Deduplicate while preserving order
                        deduplicated = deduplicate_list(non_empty_values)
                        # Use single_or_list helper to handle single vs list
                        final_value = single_or_list(deduplicated)
                        # Convert Parameters to number if applicable
                        if normalized_key == 'parameters' and final_value:
                            if isinstance(final_value, list):
                                final_value = [convert_to_number(v) for v in final_value]
                            else:
                                final_value = convert_to_number(final_value)
                        processed_row[normalized_key] = final_value
                    else:
                        # Single value - convert Parameters to number if applicable
                        if normalized_key == 'parameters':
                            value = convert_to_number(value)
                        processed_row[normalized_key] = value
                data.append(processed_row)
        
        # Normalize data
        print("Normalizing data...")
        for record in data:
            normalize_record_fields(record)
        
        print(f"Normalized {len(data)} records")
        
        # Filter records by publication date (January 2018 to current date)
        max_date = datetime.now()
        filtered_data = []
        for record in data:
            pub_date_str = record.get('publicationDate')
            if pub_date_str:
                pub_date = datetime.strptime(pub_date_str, DATE_FORMAT)
                if MIN_DATE <= pub_date <= max_date:
                    filtered_data.append(record)
        
        data = filtered_data
        print(f"Filtered to records from Jan 2018 to present: {len(data)} records")
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Generate output JSON filename with date
        output_json_filename = get_output_json_filename()
        output_json_path = os.path.join(OUTPUT_DIR, output_json_filename)
        
        # Save as JSON
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)
        
        print(f"JSON file saved as {output_json_path}")
        print(f"Total records: {len(data)}")

def normalize_existing_json(input_json=None, output_json=None):
    """
    Normalize an existing JSON file by adding company/department fields
    and normalizing countries and model accessibility.
    
    Args:
        input_json: Path to input JSON file (default: looks for latest in output dir or current dir)
        output_json: Path to output JSON file (default: generates new filename with date in output dir)
    
    Returns:
        List of normalized records
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if input_json is None:
        # Try to find the latest JSON file in output directory, or fallback to current directory
        if os.path.exists(OUTPUT_DIR):
            json_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
            if json_files:
                # Sort by modification time and use the most recent
                json_files.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)), reverse=True)
                input_json = os.path.join(OUTPUT_DIR, json_files[0])
            else:
                # Fallback to current directory
                input_json = "notable_ai_models.json"
        else:
            input_json = "notable_ai_models.json"
    
    if output_json is None:
        # Generate new filename with current date
        output_json_filename = get_output_json_filename()
        output_json = os.path.join(OUTPUT_DIR, output_json_filename)
    
    print(f"Loading JSON from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Normalizing {len(data)} records...")
    
    for record in data:
        normalize_record_fields(record)
    
    print(f"Normalized {len(data)} records")
    
    # Save normalized JSON
    print(f"Saving normalized JSON to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Normalized JSON saved to {output_json}")
    print(f"Total records: {len(data)}")
    
    return data

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "normalize":
        # Normalize existing JSON file
        input_file = sys.argv[2] if len(sys.argv) > 2 else None
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        normalize_existing_json(input_file, output_file)
    else:
        # Download and convert
        download_and_convert()

