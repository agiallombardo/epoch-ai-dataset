# AI Models Data Processing and Visualization

A comprehensive Python toolkit for downloading, processing, and visualizing AI model data from Epoch AI. This project analyzes the AI arms race through various lenses including sector analysis (private vs public), accessibility (open vs closed source), geopolitical relationships, and temporal trends.

## Features

- **Data Fetching**: Downloads and processes AI model data from Epoch AI
- **Data Normalization**: Cleans and standardizes organization names, countries, accessibility types, domains, and tasks
- **Geopolitical Analysis**: Categorizes countries by relationships (Five Eyes, NATO, Adversaries, Close Allies)
- **Comprehensive Visualizations**: Generates 49+ visualizations covering:
  - Sector distribution and trends
  - Model accessibility analysis
  - Country relationship breakdowns
  - Temporal trends and growth rates
  - Parameter and hardware analysis
  - Multi-dimensional comparisons

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd aimodelfinal
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- plotly >= 5.0.0
- kaleido >= 0.2.1

## Usage

### Complete Workflow

The recommended workflow is to run the scripts in the following order:

**Note:** Make sure your virtual environment is activated (if using one):
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Download and process AI model data**:
```bash
python src/modelsfetch.py
```

2. **Merge ECI scores from benchmark data** (runs right after modelsfetch):
```bash
python src/merge_eci_scores.py
```

3. **Generate visualizations**:
```bash
python src/visualize_ai_race_examples.py
```

### Individual Script Usage

#### Downloading and Processing Data

Download AI model data from Epoch AI and convert to normalized JSON:

```bash
python src/modelsfetch.py
```

Print JSON output to console:
```bash
python src/modelsfetch.py --print-json
```

Normalize an existing JSON file:
```bash
python src/modelsfetch.py normalize [input_file] [output_file]
```

#### Merging ECI Scores

Merge ECI (Epoch Capabilities Index) scores from benchmark data into the processed models:

```bash
python src/merge_eci_scores.py
```

This script:
- Downloads benchmark data from Epoch AI
- Extracts ECI scores from the benchmark index
- Matches models by name and organization
- Adds `eci_score` field to each model (null if not available)

#### Generating Visualizations

Run the visualization script to generate all charts:

```bash
python src/visualize_ai_race_examples.py
```

Visualizations will be saved in the `visualizations/` directory.

## Project Structure

```
aimodelfinal/
├── src/
│   ├── modelsfetch.py                # Data fetching and normalization script
│   ├── merge_eci_scores.py           # ECI score merging script
│   └── visualize_ai_race_examples.py # Visualization generation script
├── data/
│   ├── raw/                          # Raw CSV data files (downloaded)
│   │   ├── all_ai_models.csv
│   │   ├── frontier_ai_models.csv
│   │   ├── large_scale_ai_models.csv
│   │   └── notable_ai_models.csv
│   └── processed/                    # Processed JSON data files
│       ├── notable_ai_models_MM-DD-YYYY.json
│       ├── benchmark_data.json
│       └── benchmark_index.csv
├── requirements.txt                  # Python dependencies
└── visualizations/                   # Generated visualization PNG files
    ├── finals/                       # Final selected visualizations
    └── shotgun examples/             # Additional visualization examples
```

## Data Processing Pipeline

### Overview

The complete data processing pipeline consists of three main steps:

1. **`modelsfetch.py`** - Downloads and normalizes AI model data:
   - Downloads AI models data from Epoch AI as a ZIP file
   - Extracts and reads the CSV file from the ZIP
   - Converts CSV data to JSON format
   - Normalizes various fields (organizations, countries, accessibility, domains, tasks, etc.)
   - Filters records by publication date (January 2018 to present)
   - Saves the processed data as JSON

2. **`merge_eci_scores.py`** - Merges ECI scores (runs right after modelsfetch):
   - Downloads benchmark data from Epoch AI
   - Extracts ECI (Epoch Capabilities Index) scores from benchmark index
   - Matches models by normalized name and organization
   - Adds `eci_score` field to each model record
   - Updates the JSON file with merged scores

3. **`visualize_ai_race_examples.py`** - Generates visualizations:
   - Loads processed JSON data with ECI scores
   - Generates 49+ comprehensive visualizations
   - Saves charts as PNG files

### Data Flow

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

## Key Components

### Country Relationship Categories

The script categorizes countries into four geopolitical groups:

- **FEYE (Five Eyes)**: US, UK, Canada, Australia, New Zealand
- **NATO**: NATO member countries
- **Close Allies & Partners**: Strategic partners (Japan, South Korea, Israel, etc.)
- **Adversaries**: Countries with adversarial relationships (China, Russia, North Korea, etc.)

### Normalization Functions

#### Organization Normalization
- Handles Google/DeepMind variations (Google DeepMind → company: "Google", department: "DeepMind")
- Handles Facebook/Meta variations (Facebook AI Research → company: "Meta", department: "Facebook AI Research")

#### Country Normalization
- Standardizes country name variations (USA, US → "United States")
- Maps to geopolitical relationship categories

#### Accessibility Normalization
- Standardizes model accessibility terminology:
  - "API", "api access" → "API access"
  - "open weights", "open weights unrestricted" → "Open weights (unrestricted)"
  - "open weights restricted" → "Open weights (restricted use)"
  - "closed", "private" → "Unreleased"

#### Domain and Task Normalization
- Normalizes domain names with consistent capitalization
- Consolidates task name variations (e.g., "language modeling", "text generation" → "Language modeling/generation")

### Output Structure

The final JSON contains records with the following key fields:

- **Basic Info**: `name`, `publicationDate`, `parameters`
- **Organization**: `company`, `department` (derived from `organization`)
- **Geography**: `countryOfOrganization`, `countryRelationship`
- **Technical**: `modelAccessibility`, `domain`, `task`, `trainingHardware`, `baseModel`
- **Performance**: `eci_score` (Epoch Capabilities Index score, null if not available)
- **Other normalized fields**: Various other fields converted to camelCase and cleaned

## Visualizations

The visualization script generates comprehensive charts organized by category:

### Sector Analysis
- Pie charts and bar charts showing private vs public sector distribution
- Time series showing sector trends over time
- Stacked area charts for cumulative analysis

### Accessibility Analysis
- Distribution of open vs closed source models
- Accessibility trends by alliance
- Detailed breakdowns by accessibility type

### Geopolitical Analysis
- Country relationship pie charts and bar charts
- Alliance comparison visualizations
- Growth rates by alliance
- Market share analysis

### Advanced Visualizations
- Heatmaps (sector vs accessibility, country vs accessibility)
- Sankey diagrams (country-sector-capacity flows)
- Radar charts (multi-dimensional comparisons)
- Bubble charts (parameter vs compute analysis)
- Sunburst charts (hierarchical alliance-sector-capacity)
- Waffle charts (alliance distribution)

### Statistical Analysis
- Parameter distribution over time
- Hardware analysis
- Notability criteria distribution
- Lead indicators dashboard

All visualizations follow Gestalt design principles with a consistent color palette and styling.

## Data Source and Licensing

### Data Source
This project uses data from [Epoch AI](https://epoch.ai/data/ai-models).

### Licensing
Epoch AI's data is free to use, distribute, and reproduce provided the source and authors are credited under the [Creative Commons Attribution license](https://creativecommons.org/licenses/by/4.0/).

### Citation

**Plain Text:**
```
Epoch AI, 'Data on AI Models'. Published online at epoch.ai. Retrieved from 'https://epoch.ai/data/ai-models' [online resource].
```

**BibTeX:**
```bibtex
@misc{EpochAIModels2025},
  title = {Data on AI Models},
  author = {{Epoch AI}},
  year = {2025},
  month = {07},
  url = {https://epoch.ai/data/ai-models}
}
```

## Design Patterns

1. **In-Place Mutation**: Normalization functions modify records directly rather than returning new objects
2. **Single vs. List Consistency**: Uses helper functions to maintain consistent data structure
3. **Set-Based Lookups**: Uses sets for O(1) membership checks in country relationships
4. **Order Preservation**: Deduplication maintains original order using dictionary keys
5. **Graceful Degradation**: Functions return `None` or original values when normalization isn't possible

## Performance Considerations

- Uses sets for O(1) lookups in country relationships and column filtering
- Uses temporary directory for ZIP extraction (automatically cleaned up)
- Processes data in memory (suitable for datasets that fit in RAM)
- Preserves order during deduplication using dictionary keys

## Error Handling

- Raises `FileNotFoundError` if CSV file not found in ZIP
- Handles missing fields gracefully (checks for existence before processing)
- Converts invalid dates/numbers to original values rather than crashing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project uses data from Epoch AI, which is licensed under Creative Commons Attribution 4.0. Please refer to the [Epoch AI licensing information](https://epoch.ai/data/ai-models) for details.

