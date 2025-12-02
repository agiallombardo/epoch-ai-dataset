# Code Summary: modelsfetch.py

## Overview

`modelsfetch.py` is a Python script that downloads AI model data from Epoch AI, processes it, and converts it to a normalized JSON format. The script handles data normalization, field transformations, and filtering to create a clean, structured dataset of notable AI models.

## Main Functionality

The script performs the following operations:
1. **Downloads** AI models data from Epoch AI as a ZIP file
2. **Extracts** and reads the CSV file from the ZIP
3. **Converts** CSV data to JSON format
4. **Normalizes** various fields (organizations, countries, accessibility, domains, tasks, etc.)
5. **Filters** records by publication date (January 2018 to present)
6. **Saves** the processed data as JSON

## Code Structure and Logic Walkthrough

### 1. Constants and Configuration

```python
DATA_URL = "https://epoch.ai/data/ai_models.zip"
OUTPUT_JSON = "notable_ai_models.json"
CSV_FILENAME = "notable_ai_models.csv"
DATE_FORMAT = '%Y-%m-%d'
MIN_DATE = datetime(2018, 1, 1)
```

- Defines the data source URL, output filenames, and date filtering parameters
- Sets minimum date threshold to January 1, 2018

### 2. Country Relationship Sets

The script defines four country relationship categories for geopolitical analysis:

- **`_FEYE_COUNTRIES`**: Five Eyes alliance (US, UK, Canada, Australia, New Zealand)
- **`_NATO_COUNTRIES`**: NATO member countries
- **`_CLOSE_ALLIES`**: Strategic partners and allies (Japan, South Korea, Israel, etc.)
- **`_ADVERSARIES`**: Countries with adversarial relationships (China, Russia, North Korea, etc.)

These sets are used to categorize countries based on their relationship with the United States.

### 3. Utility Functions

#### `to_camel_case(s)`
- Converts field names from various formats to camelCase
- Handles parentheses, special characters, and spaces
- Example: "Country (of organization)" → "countryOfOrganization"

#### `convert_to_number(value)`
- Attempts to convert string values to numbers (int or float)
- Returns original value if conversion fails
- Used for numeric fields like "Parameters"

#### `deduplicate_list(items)`
- Removes duplicates from a list while preserving order
- Uses dictionary keys to maintain insertion order

#### `single_or_list(values)`
- Returns a single value if the list has one item, otherwise returns the list
- Helps maintain consistent data structure (single values vs. arrays)

### 4. Normalization Functions

#### `normalize_single_org(org)`
**Purpose**: Normalizes organization names to separate company and department.

**Logic**:
- Handles Google/DeepMind variations:
  - "Google DeepMind" → company: "Google", department: "DeepMind"
  - "DeepMind" → company: "Google", department: "DeepMind"
  - "Google Research" → company: "Google", department: "Google Research"
- Handles Facebook/Meta variations:
  - "Facebook AI Research" → company: "Meta", department: "Facebook AI Research"
  - "Facebook" → company: "Meta", department: None
  - "Meta AI" → company: "Meta", department: "Meta AI"

**Returns**: Tuple of (company, department)

#### `normalize_country(country_value)`
**Purpose**: Standardizes country name variations.

**Logic**:
- Maps common variations to standard names:
  - "USA", "US", "United States of America" → "United States"
  - "UK", "Great Britain", "GB" → "United Kingdom"
  - "South Korea" → "Korea (Republic of)"
- Handles both single values and lists
- Preserves order and deduplicates

#### `get_country_relationship(country)`
**Purpose**: Categorizes countries by their relationship with the United States.

**Logic**:
1. Converts country name to lowercase for lookup
2. Checks membership in priority order:
   - FEYE (Five Eyes)
   - NATO
   - Adversaries
   - Close Allies & Partners
   - Other (default)

**Returns**: Relationship category string

#### `get_country_group(country_value)`
**Purpose**: Wrapper that normalizes countries first, then categorizes them.

**Logic**:
- Calls `normalize_country()` first
- Then calls `get_country_relationship()` for each country
- For lists, collects unique relationships and orders them by priority
- Returns single value or list based on input

#### `normalize_model_accessibility(accessibility_value)`
**Purpose**: Standardizes model accessibility terminology.

**Mappings**:
- "API", "api access" → "API access"
- "open weights", "open weights unrestricted" → "Open weights (unrestricted)"
- "open weights restricted" → "Open weights (restricted use)"
- "hosted access" → "Hosted access (no API)"
- "closed", "private" → "Unreleased"

#### `normalize_domain(domain_value)`
**Purpose**: Normalizes domain names with consistent capitalization.

**Mappings**:
- "3d", "3d modeling" → "3D modeling"
- "image generation" → "Image generation"
- "earth science" → "Earth science"
- "materials science" → "Materials science"

**Logic**:
- Handles both single values and lists
- Deduplicates while preserving order

#### `normalize_training_hardware(hardware_value)`
**Purpose**: Cleans and normalizes training hardware values.

**Logic**:
- Strips whitespace
- Deduplicates lists
- Returns single value if only one item, otherwise list

#### `normalize_task(task_value)`
**Purpose**: Standardizes task names and consolidates variations.

**Mappings**:
- "language modeling", "language generation", "text generation" → "Language modeling/generation"
- "qa", "question answering" → "Question answering"
- "asr", "speech recognition" → "Speech recognition (ASR)"

**Logic**:
- Handles both single values and lists
- Deduplicates while preserving order

#### `normalize_base_model(base_model_value)`
**Purpose**: Cleans base model references.

**Logic**:
- Strips whitespace
- Returns None for empty strings

### 5. Record Processing Functions

#### `normalize_organization_field(record)`
**Purpose**: Transforms organization field into company and department fields.

**Logic**:
1. Checks if 'organization' field exists
2. If list: processes each organization, collects companies and departments separately
3. If single value: processes single organization
4. Deduplicates and uses `single_or_list()` for consistent structure
5. Modifies record in place, adding 'company' and 'department' fields

#### `normalize_record_fields(record)`
**Purpose**: Master normalization function that processes all fields in a record.

**Processing Order**:
1. **Organization**: Normalizes to company/department
2. **Country**: Normalizes country names and adds `countryRelationship` field
3. **Model Accessibility**: Standardizes accessibility values
4. **Domain**: Normalizes domain names
5. **Task**: Standardizes task names
6. **Training Hardware**: Cleans hardware values
7. **Base Model**: Cleans base model references

**Note**: Modifies record in place (mutates the dictionary)

### 6. Main Processing Functions

#### `download_and_convert(print_json=False)`
**Purpose**: Main function that orchestrates the entire data processing pipeline.

**Step-by-Step Logic**:

1. **Download Phase**:
   - Creates temporary directory
   - Downloads ZIP file from `DATA_URL`
   - Extracts ZIP file to temporary directory

2. **CSV Reading Phase**:
   - Locates `notable_ai_models.csv` in extracted files
   - Defines set of columns to exclude (notes, citations, links, etc.)
   - Reads CSV using `csv.DictReader`

3. **Data Transformation Phase**:
   For each row:
   - Skips excluded columns
   - Converts field names to camelCase using `to_camel_case()`
   - Trims whitespace from values
   - Converts empty strings to `None`
   - Detects multi-value fields (comma-separated):
     - Splits by comma
     - Trims and filters empty values
     - Deduplicates
     - Uses `single_or_list()` for structure
   - Converts "Parameters" field to numbers if applicable
   - Stores processed row

4. **Normalization Phase**:
   - Iterates through all records
   - Calls `normalize_record_fields()` for each record
   - Applies all normalization rules

5. **Date Filtering Phase**:
   - Filters records where `publicationDate` is between `MIN_DATE` (Jan 1, 2018) and current date
   - Parses dates using `DATE_FORMAT` ('%Y-%m-%d')

6. **Output Phase**:
   - Saves filtered and normalized data to JSON file
   - Uses indentation (2 spaces) for readability
   - Optionally prints JSON to console if `print_json=True`

#### `normalize_existing_json(input_json, output_json)`
**Purpose**: Normalizes an existing JSON file without downloading new data.

**Logic**:
1. Loads JSON from input file
2. Applies `normalize_record_fields()` to each record
3. Saves normalized data to output file (or overwrites input if no output specified)
4. Returns normalized data list

### 7. Command-Line Interface

**Main Entry Point** (`if __name__ == "__main__"`):

**Usage Patterns**:
1. **Download and convert** (default):
   ```bash
   python modelsfetch.py
   python modelsfetch.py --print-json  # Also prints JSON to console
   ```

2. **Normalize existing JSON**:
   ```bash
   python modelsfetch.py normalize [input_file] [output_file]
   ```

**Logic**:
- Checks command-line arguments
- If first argument is "normalize": calls `normalize_existing_json()`
- Otherwise: calls `download_and_convert()`
- Supports `--print-json` flag for verbose output

## Data Flow Diagram

```
┌─────────────────┐
│  Download ZIP   │
│  from Epoch AI  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract CSV    │
│  from ZIP       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Read CSV       │
│  (DictReader)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transform      │
│  - camelCase    │
│  - Multi-values │
│  - Numbers      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Normalize      │
│  - Organizations│
│  - Countries    │
│  - Accessibility│
│  - Domains      │
│  - Tasks        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Filter by Date │
│  (2018-present) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save to JSON   │
└─────────────────┘
```

## Key Design Patterns

1. **In-Place Mutation**: Normalization functions modify records directly rather than returning new objects
2. **Single vs. List Consistency**: Uses `single_or_list()` helper to maintain consistent data structure
3. **Set-Based Lookups**: Uses sets for O(1) membership checks in country relationships and column exclusions
4. **Order Preservation**: Deduplication maintains original order using `dict.fromkeys()`
5. **Graceful Degradation**: Functions return `None` or original values when normalization isn't possible

## Output Structure

The final JSON contains records with the following key fields:

- **Basic Info**: `name`, `publicationDate`, `parameters`
- **Organization**: `company`, `department` (derived from `organization`)
- **Geography**: `countryOfOrganization`, `countryRelationship`
- **Technical**: `modelAccessibility`, `domain`, `task`, `trainingHardware`, `baseModel`
- **Other normalized fields**: Various other fields converted to camelCase and cleaned

## Performance Considerations

- Uses sets for O(1) lookups in country relationships and column filtering
- Uses temporary directory for ZIP extraction (automatically cleaned up)
- Processes data in memory (suitable for datasets that fit in RAM)
- Preserves order during deduplication using dictionary keys

## Error Handling

- Raises `FileNotFoundError` if CSV file not found in ZIP
- Handles missing fields gracefully (checks for existence before processing)
- Converts invalid dates/numbers to original values rather than crashing

