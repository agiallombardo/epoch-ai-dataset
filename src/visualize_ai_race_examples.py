#!/usr/bin/env python3
"""
AI Arms Race Visualization Script

This script generates comprehensive visualizations analyzing the AI arms race:
- Private vs Public sector (Industry vs Academia)
- Open vs Closed source models
- Country/Allies breakdown
- Trends over time

All visualizations are saved as PNG files.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

from typing import Dict, List, Optional

# Gestalt-compliant styling and color palette
# Following Gestalt Laws: Proximity, Similarity, Closure, Continuity, Figure/Ground, Common Region

# Unified color palette (Law of Similarity - consistent colors for same categories)
# Using purple and blue tones from the provided color scheme
COLOR_PALETTE = {
    # Sector colors
    'Private': '#4A90E2',      # Medium blue - Industry
    'Public': '#6A4C93',      # Dark purple - Academia
    'Other': '#9B9B9B',       # Gray - Other
    
    # Source type colors
    'Open Source': '#87CEEB',  # Light blue - Open
    'Closed Source': '#2C3E8F', # Dark blue - Closed
    'Other Source': '#9B9B9B',  # Gray
    
    # Country relationship colors (consistent across all visualizations)
    'FEYE': '#4A90E2',         # Medium blue - Five Eyes
    'NATO': '#6A4C93',         # Dark purple - NATO
    'Adversaries': '#2C3E8F',  # Dark blue - Adversaries
    'Close Allies & Partners': '#B19CD9',  # Light purple - Allies
    # 'Other' is already defined above for sectors, same color used here
    
    # Capacity colors (gradient from light to dark purple/blue)
    'Small (<1B)': '#E6E6FA',      # Lavender (very light purple)
    'Medium (1B-10B)': '#B19CD9',  # Light purple
    'Large (10B-100B)': '#87CEEB', # Light blue
    'Very Large (100B-1T)': '#4A90E2', # Medium blue
    'Massive (≥1T)': '#2C3E8F',    # Dark blue
    'Unknown': '#D3D3D3'          # Light gray
}

# Parameter capacity colors mapping
PARAM_COLORS = {
    'Small (<1B)': COLOR_PALETTE['Small (<1B)'],
    'Medium (1B-10B)': COLOR_PALETTE['Medium (1B-10B)'],
    'Large (10B-100B)': COLOR_PALETTE['Large (10B-100B)'],
    'Very Large (100B-1T)': COLOR_PALETTE['Very Large (100B-1T)'],
    'Massive (≥1T)': COLOR_PALETTE['Massive (≥1T)'],
    'Unknown': COLOR_PALETTE['Unknown']
}

# Notability criteria colors
NOTABILITY_COLORS = [COLOR_PALETTE['FEYE'], COLOR_PALETTE['NATO'], COLOR_PALETTE['Adversaries'],
                     COLOR_PALETTE['Close Allies & Partners'], COLOR_PALETTE['Other']]

# Visual hierarchy settings (Law of Figure/Ground)
STYLE_CONFIG = {
    'figure_bg': '#FFFFFF',           # White background
    'axes_bg': '#FAFAFA',             # Very light gray for axes
    'grid_color': '#E0E0E0',          # Light gray grid
    'grid_alpha': 0.5,
    'text_color': '#2C3E50',          # Dark gray text
    'title_size': 16,
    'subtitle_size': 12,
    'label_size': 11,
    'tick_size': 9,
    'legend_size': 10,
    'spacing': 0.3,                   # Spacing between groups (Law of Proximity)
    'bar_width': 0.7,                 # Bar width for grouping
    'line_width': 2.5,                # Line width for continuity
    'marker_size': 6,                  # Marker size
    'alpha_fill': 0.25,               # Transparency for filled areas
    'alpha_line': 0.8,                # Transparency for lines
    'border_width': 1.5,              # Border width for separation
    'shadow': True,                   # Shadows for depth (Law of Figure/Ground)
    'padding': 0.15                    # Padding around elements
}

# Set style following Gestalt principles
sns.set_style("whitegrid", {
    'axes.facecolor': STYLE_CONFIG['axes_bg'],
    'figure.facecolor': STYLE_CONFIG['figure_bg'],
    'grid.color': STYLE_CONFIG['grid_color'],
    'grid.alpha': STYLE_CONFIG['grid_alpha'],
    'text.color': STYLE_CONFIG['text_color'],
    'axes.labelcolor': STYLE_CONFIG['text_color'],
    'xtick.color': STYLE_CONFIG['text_color'],
    'ytick.color': STYLE_CONFIG['text_color']
})

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.facecolor'] = STYLE_CONFIG['figure_bg']
plt.rcParams['axes.facecolor'] = STYLE_CONFIG['axes_bg']
plt.rcParams['font.size'] = STYLE_CONFIG['label_size']
plt.rcParams['axes.labelsize'] = STYLE_CONFIG['label_size']
plt.rcParams['axes.titlesize'] = STYLE_CONFIG['title_size']
plt.rcParams['xtick.labelsize'] = STYLE_CONFIG['tick_size']
plt.rcParams['ytick.labelsize'] = STYLE_CONFIG['tick_size']
plt.rcParams['legend.fontsize'] = STYLE_CONFIG['legend_size']
plt.rcParams['text.color'] = STYLE_CONFIG['text_color']
plt.rcParams['axes.labelcolor'] = STYLE_CONFIG['text_color']
plt.rcParams['xtick.color'] = STYLE_CONFIG['text_color']
plt.rcParams['ytick.color'] = STYLE_CONFIG['text_color']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = STYLE_CONFIG['grid_alpha']
plt.rcParams['grid.color'] = STYLE_CONFIG['grid_color']

# Output directory for visualizations
OUTPUT_DIR = "visualizations/shotgun examples"
# Output directory for JSON data files (processed data)
# Get the script directory and construct path relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
JSON_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATE_FORMAT = '%Y-%m-%d'

def load_data(json_file: Optional[str] = None) -> List[Dict]:
    """Load AI models data from JSON file.
    
    If json_file is not provided, loads the most recent notable JSON file from the processed data directory.
    """
    if json_file is None:
        if not os.path.exists(JSON_OUTPUT_DIR):
            raise FileNotFoundError(f"Directory {JSON_OUTPUT_DIR} does not exist")
        
        # Find all notable JSON files in the processed directory
        json_files = [f for f in os.listdir(JSON_OUTPUT_DIR) if f.endswith('.json') and 'notable' in f.lower()]
        
        if not json_files:
            raise FileNotFoundError(f"No notable JSON files found in {JSON_OUTPUT_DIR}")
        
        # Sort by modification time and use the most recent
        json_files.sort(key=lambda f: os.path.getmtime(os.path.join(JSON_OUTPUT_DIR, f)), reverse=True)
        json_file = os.path.join(JSON_OUTPUT_DIR, json_files[0])
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def parse_date(date_str: str):
    """Parse date string to datetime object."""
    if not date_str:
        return None
    return datetime.strptime(date_str, DATE_FORMAT)

def prepare_dataframe(data: List[Dict]) -> pd.DataFrame:
    """Convert data to pandas DataFrame with processed fields."""
    records = []
    
    for record in data:
        # Parse date
        pub_date_str = record.get('publicationDate')
        if not pub_date_str:
            continue
        pub_date = parse_date(pub_date_str)
        if not pub_date:
            continue
        
        # Extract sector (Private/Public)
        org_cat = record.get('organizationCategorization', 'Unknown')
        sector = "Private" if org_cat == "Industry" else "Public" if org_cat == "Academia" else "Other"
        
        # Get categorized accessibility (already done in modelsfetch.py)
        accessibility = record.get('accessibilityCategory', 'Unknown')
        if not accessibility or accessibility == 'Unknown':
            # Fallback: categorize from modelAccessibility if category not available
            acc_raw = record.get('modelAccessibility', '')
            if acc_raw:
                accessibility_lower = str(acc_raw).lower()
                if "open" in accessibility_lower:
                    accessibility = "Open Source"
                elif "api" in accessibility_lower or "closed" in accessibility_lower or "unreleased" in accessibility_lower:
                    accessibility = "Closed Source"
                else:
                    accessibility = "Other"
        
        # Extract country relationship
        country_rel = record.get('countryRelationship', 'Unknown')
        if isinstance(country_rel, list):
            country_rel = country_rel[0] if country_rel else 'Unknown'
        
        # Extract country name
        country = record.get('countryOfOrganization', 'Unknown')
        if isinstance(country, list):
            country = country[0] if country else 'Unknown'
        
        # Extract company
        company = record.get('company', 'Unknown')
        if isinstance(company, list):
            company = company[0] if company else 'Unknown'
        
        # Extract parameters
        parameters = record.get('parameters')
        if parameters and isinstance(parameters, (int, float)):
            parameters = float(parameters)
        else:
            parameters = None
        
        # Extract domain
        domain = record.get('domain', None)
        if isinstance(domain, list):
            domain = domain[0] if domain else None
        elif not domain:
            domain = None
        
        # Extract task
        task = record.get('task', None)
        if isinstance(task, list):
            task = task[0] if task else None
        elif not task:
            task = None
        
        # Extract training compute (FLOP)
        training_compute = record.get('trainingCompute')
        if training_compute and isinstance(training_compute, (int, float)):
            training_compute = float(training_compute)
        elif training_compute and isinstance(training_compute, str):
            try:
                training_compute = float(training_compute)
            except (ValueError, TypeError):
                training_compute = None
        else:
            training_compute = None
        
        # Extract notability criteria
        notability_criteria = record.get('notabilityCriteria', None)
        if isinstance(notability_criteria, list):
            notability_criteria = notability_criteria[0] if notability_criteria else None
        elif not notability_criteria:
            notability_criteria = None
        
        # Extract training hardware
        training_hardware = record.get('trainingHardware', None)
        if isinstance(training_hardware, list):
            training_hardware = training_hardware[0] if training_hardware else None
        elif not training_hardware:
            training_hardware = None
        
        # Get model capacity (already categorized in modelsfetch.py)
        model_capacity = record.get('modelCapacity')
        if not model_capacity:
            # Fallback: categorize from parameters if capacity not available
            model_capacity = categorize_model_capacity(parameters)
        
        records.append({
            'date': pub_date,
            'year': pub_date.year,
            'year_month': pub_date.strftime('%Y-%m'),
            'sector': sector,
            'organization_cat': org_cat,
            'accessibility': accessibility,
            'country_relationship': country_rel,
            'country': country,
            'company': company,
            'parameters': parameters,
            'model_capacity': model_capacity,
            'training_compute': training_compute,
            'domain': domain,
            'task': task,
            'notability_criteria': notability_criteria,
            'training_hardware': training_hardware,
            'model': record.get('model', 'Unknown')
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values('date')
    return df

def categorize_model_capacity(parameters):
    """Categorize model capacity based on parameters."""
    if not parameters or parameters is None:
        return "Unknown"
    
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

def apply_gestalt_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply Gestalt-compliant styling to axes (Law of Similarity - consistent styling)."""
    ax.set_facecolor(STYLE_CONFIG['axes_bg'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(STYLE_CONFIG['grid_color'])
    ax.spines['bottom'].set_color(STYLE_CONFIG['grid_color'])
    ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'], color=STYLE_CONFIG['grid_color'])
    
    if title:
        ax.set_title(title, fontsize=STYLE_CONFIG['title_size'], fontweight='bold',
                    pad=20, color=STYLE_CONFIG['text_color'])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=STYLE_CONFIG['label_size'], color=STYLE_CONFIG['text_color'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=STYLE_CONFIG['label_size'], color=STYLE_CONFIG['text_color'])
    
    for label in ax.get_xticklabels():
        label.set_color(STYLE_CONFIG['text_color'])
        label.set_fontsize(STYLE_CONFIG['tick_size'])
    for label in ax.get_yticklabels():
        label.set_color(STYLE_CONFIG['text_color'])
        label.set_fontsize(STYLE_CONFIG['tick_size'])

def create_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def plot_sector_distribution(df: pd.DataFrame):
    """Create visualizations for Private vs Public sector - focused on alliances."""
    # Removed pie chart - less actionable than bar/time series
    # Focus on alliance-specific sector analysis
    
    # 1. Sector by Alliance - Stacked bar chart (WHO LEADS IN PRIVATE VS PUBLIC?)
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    sector_alliance = pd.crosstab(df['country_relationship'], df['sector'])
    sector_alliance.plot(kind='bar', stacked=True, ax=ax,
                        color=[COLOR_PALETTE.get(s, COLOR_PALETTE['Other']) 
                              for s in sector_alliance.columns],
                        width=STYLE_CONFIG['bar_width'], alpha=STYLE_CONFIG['alpha_line'],
                        edgecolor='white', linewidth=STYLE_CONFIG['border_width'])
    ax.set_title('Sector Distribution by Alliance\n(Who Leads in Private vs Public Sector?)', 
                fontsize=STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel('Country Relationship (Alliance)', fontsize=STYLE_CONFIG['label_size'])
    ax.set_ylabel('Number of Models', fontsize=STYLE_CONFIG['label_size'])
    ax.legend(title='Sector', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
    ax.grid(axis='y', alpha=STYLE_CONFIG['grid_alpha'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/01_sector_by_alliance.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 2. Sector over time by Alliance (WHO IS ACCELERATING IN PRIVATE VS PUBLIC?)
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=STYLE_CONFIG['figure_bg'])
    alliances = ['FEYE', 'NATO', 'Adversaries', 'Close Allies & Partners']
    sectors = ['Private', 'Public']
    
    for alliance in alliances:
        df_alliance = df[df['country_relationship'] == alliance]
        if len(df_alliance) > 0:
            for sector in sectors:
                df_sector = df_alliance[df_alliance['sector'] == sector]
                if len(df_sector) > 0:
                    monthly = df_sector.groupby('year_month').size()
                    monthly.index = pd.to_datetime(monthly.index)
                    monthly = monthly.sort_index()
                    if len(monthly) > 0:
                        ax.plot(monthly.index, monthly.values,
                               marker='o', label=f'{alliance} - {sector}',
                               linewidth=STYLE_CONFIG['line_width'],
                               markersize=STYLE_CONFIG['marker_size'],
                               alpha=STYLE_CONFIG['alpha_line'])
    
    ax.set_title('Sector Over Time by Alliance\n(Who is Accelerating in Private vs Public?)', 
                fontsize=STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=STYLE_CONFIG['label_size'])
    ax.set_ylabel('Number of Models', fontsize=STYLE_CONFIG['label_size'])
    ax.legend(loc='best', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
             ncol=2, fontsize=9)
    ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/02_sector_time_by_alliance.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def plot_accessibility_distribution(df: pd.DataFrame):
    """Create visualizations for Open vs Closed source - focused on alliances."""
    # Simplify accessibility categories for main analysis
    df['accessibility_simple'] = df['accessibility'].apply(
        lambda x: 'Open Source' if 'Open' in str(x) 
        else 'Closed Source' if 'Closed' in str(x) 
        else 'Other'
    )
    
    # Removed pie charts and detailed bars - less actionable
    # Focus on alliance-specific analysis
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    acc_alliance = pd.crosstab(df['country_relationship'], df['accessibility_simple'])
    acc_alliance.plot(kind='bar', stacked=True, ax=ax,
                     color=[COLOR_PALETTE.get(a, COLOR_PALETTE['Other Source']) 
                           for a in acc_alliance.columns],
                     width=STYLE_CONFIG['bar_width'], alpha=STYLE_CONFIG['alpha_line'],
                     edgecolor='white', linewidth=STYLE_CONFIG['border_width'])
    ax.set_title('Source Type Distribution by Alliance\n(Who Shares vs Hides Technology?)', 
                fontsize=STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel('Country Relationship (Alliance)', fontsize=STYLE_CONFIG['label_size'])
    ax.set_ylabel('Number of Models', fontsize=STYLE_CONFIG['label_size'])
    ax.legend(title='Source Type', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
    ax.grid(axis='y', alpha=STYLE_CONFIG['grid_alpha'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/03_accessibility_by_alliance.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 2. Accessibility over time by Alliance (WHO IS CHANGING STRATEGY?)
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=STYLE_CONFIG['figure_bg'])
    alliances = ['FEYE', 'NATO', 'Adversaries', 'Close Allies & Partners']
    acc_types = ['Open Source', 'Closed Source']
    
    for alliance in alliances:
        df_alliance = df[df['country_relationship'] == alliance]
        if len(df_alliance) > 0:
            for acc_type in acc_types:
                df_acc = df_alliance[df_alliance['accessibility_simple'] == acc_type]
                if len(df_acc) > 0:
                    monthly = df_acc.groupby('year_month').size()
                    monthly.index = pd.to_datetime(monthly.index)
                    monthly = monthly.sort_index()
                    if len(monthly) > 0:
                        color = COLOR_PALETTE.get(acc_type, COLOR_PALETTE['Other Source'])
                        ax.plot(monthly.index, monthly.values,
                               marker='o', label=f'{alliance} - {acc_type}',
                               linewidth=STYLE_CONFIG['line_width'],
                               markersize=STYLE_CONFIG['marker_size'],
                               color=color, alpha=STYLE_CONFIG['alpha_line'],
                               markerfacecolor=color, markeredgecolor='white')
    
    ax.set_title('Source Type Over Time by Alliance\n(Who is Changing Strategy: Open vs Closed?)', 
                fontsize=STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=STYLE_CONFIG['label_size'])
    ax.set_ylabel('Number of Models', fontsize=STYLE_CONFIG['label_size'])
    ax.legend(loc='best', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
             ncol=2, fontsize=9)
    ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/04_accessibility_time_by_alliance.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def plot_country_distribution(df: pd.DataFrame):
    """Create visualizations for country/allies breakdown - focused on arms race."""
    # Removed pie chart - less actionable than bar/time series
    # Focus on actionable comparisons
    
    # 1. Bar chart - Top countries (WHO ARE THE LEADING COUNTRIES?)
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    country_counts = df['country'].value_counts().head(15)
    bars = ax.barh(y=range(len(country_counts)), width=country_counts.values, 
                   color=COLOR_PALETTE['FEYE'], height=STYLE_CONFIG['bar_width'],
                   edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                   alpha=STYLE_CONFIG['alpha_line'])
    ax.set_yticks(range(len(country_counts)))
    ax.set_yticklabels(country_counts.index)
    apply_gestalt_style(ax, 
                        title='Top 15 Countries by Number of AI Models\n(Who are the Leading Countries?)',
                        xlabel='Number of Models',
                        ylabel='Country')
    
    # Add value labels (Law of Connectedness - labels connected to bars)
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width)}',
                ha='left', va='center', fontsize=STYLE_CONFIG['tick_size'], 
                fontweight='bold', color=STYLE_CONFIG['text_color'])
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/05_top_countries_bar.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 2. Bar chart - Country relationships (WHO HAS MORE MODELS?)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=STYLE_CONFIG['figure_bg'])
    country_rel_counts = df['country_relationship'].value_counts()
    colors = [COLOR_PALETTE.get(rel, COLOR_PALETTE['Other']) for rel in country_rel_counts.index]
    bars = ax.bar(country_rel_counts.index, country_rel_counts.values,
                  color=colors, width=STYLE_CONFIG['bar_width'],
                  edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                  alpha=STYLE_CONFIG['alpha_line'])
    ax.set_title('AI Models by Alliance\n(Who Has More Models?)', 
                fontsize=STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel('Country Relationship (Alliance)', fontsize=STYLE_CONFIG['label_size'])
    ax.set_ylabel('Number of Models', fontsize=STYLE_CONFIG['label_size'])
    ax.grid(axis='y', alpha=STYLE_CONFIG['grid_alpha'])
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=STYLE_CONFIG['tick_size'], 
                fontweight='bold', color=STYLE_CONFIG['text_color'])
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/06_alliance_model_count.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 3. Time series - Country relationships over time (WHO IS ACCELERATING?)
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    
    monthly_data = df.groupby(['year_month', 'country_relationship']).size().unstack(fill_value=0)
    monthly_data.index = pd.to_datetime(monthly_data.index)
    monthly_data = monthly_data.sort_index()
    
    # Use consistent COLOR_PALETTE (Law of Similarity)
    colors_ts = {
        'FEYE': COLOR_PALETTE['FEYE'],
        'NATO': COLOR_PALETTE['NATO'],
        'Adversaries': COLOR_PALETTE['Adversaries'],
        'Close Allies & Partners': COLOR_PALETTE['Close Allies & Partners'],
        'Other': COLOR_PALETTE['Other']
    }
    
    for rel in monthly_data.columns:
        if rel in colors_ts:
            ax.plot(monthly_data.index, monthly_data[rel],
                   marker='o', label=rel, 
                   linewidth=STYLE_CONFIG['line_width'], 
                   markersize=STYLE_CONFIG['marker_size'],
                   color=colors_ts[rel], alpha=STYLE_CONFIG['alpha_line'],
                   markerfacecolor=colors_ts[rel], markeredgecolor='white',
                   markeredgewidth=1)
    
    apply_gestalt_style(ax,
                       title='AI Models by Alliance Over Time\n(Who is Accelerating in the Arms Race?)',
                       xlabel='Date',
                       ylabel='Number of Models Published')
    ax.legend(loc='best', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
             framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/07_alliance_time_series.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 4. Cumulative models by alliance (WHO IS AHEAD OVERALL?)
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    
    monthly_data = df.groupby(['year_month', 'country_relationship']).size().unstack(fill_value=0)
    monthly_data.index = pd.to_datetime(monthly_data.index)
    monthly_data = monthly_data.sort_index()
    cumulative_data = monthly_data.cumsum()
    
    for rel in cumulative_data.columns:
        if rel in colors_ts:
            ax.plot(cumulative_data.index, cumulative_data[rel],
                   marker='o', label=rel, linewidth=STYLE_CONFIG['line_width'], 
                   markersize=STYLE_CONFIG['marker_size'],
                   color=colors_ts[rel], alpha=STYLE_CONFIG['alpha_line'],
                   markerfacecolor=colors_ts[rel], markeredgecolor='white', markeredgewidth=1)
    
    apply_gestalt_style(ax,
                       title='Cumulative AI Models by Alliance Over Time\n(Who is Ahead in the Arms Race?)',
                       xlabel='Date',
                       ylabel='Cumulative Number of Models')
    ax.legend(loc='best', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
             framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/08_cumulative_alliance.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def plot_combined_analysis(df: pd.DataFrame):
    """Create combined analysis visualizations."""
    # 1. Sector vs Accessibility heatmap
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=STYLE_CONFIG['figure_bg'])
    df['accessibility_simple'] = df['accessibility'].apply(
        lambda x: 'Open Source' if 'Open' in str(x) 
        else 'Closed Source' if 'Closed' in str(x) 
        else 'Other'
    )
    crosstab = pd.crosstab(df['sector'], df['accessibility_simple'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Number of Models'}, linewidths=0.5,
                annot_kws={'fontsize': STYLE_CONFIG['tick_size'], 
                          'color': STYLE_CONFIG['text_color']})
    apply_gestalt_style(ax,
                       title='Sector vs Source Type\n(Private/Public vs Open/Closed)',
                       xlabel='Source Type',
                       ylabel='Sector')
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/14_sector_vs_accessibility_heatmap.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 2. Country Relationship vs Accessibility heatmap
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    crosstab = pd.crosstab(df['country_relationship'], df['accessibility_simple'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Number of Models'}, linewidths=0.5,
                annot_kws={'fontsize': STYLE_CONFIG['tick_size'], 
                          'color': STYLE_CONFIG['text_color']})
    apply_gestalt_style(ax,
                       title='Country Relationship vs Source Type',
                       xlabel='Source Type',
                       ylabel='Country Relationship')
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/15_country_vs_accessibility_heatmap.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 3. Sector vs Country Relationship heatmap
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    crosstab = pd.crosstab(df['sector'], df['country_relationship'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Number of Models'}, linewidths=0.5,
                annot_kws={'fontsize': STYLE_CONFIG['tick_size'], 
                          'color': STYLE_CONFIG['text_color']})
    apply_gestalt_style(ax,
                       title='Sector vs Country Relationship',
                       xlabel='Country Relationship',
                       ylabel='Sector')
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/16_sector_vs_country_heatmap.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 4. Top companies by model count
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    company_counts = df[df['company'] != 'Unknown']['company'].value_counts().head(20)
    bars = ax.barh(range(len(company_counts)), company_counts.values, 
                   color=COLOR_PALETTE['FEYE'], height=STYLE_CONFIG['bar_width'],
                   edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                   alpha=STYLE_CONFIG['alpha_line'])
    ax.set_yticks(range(len(company_counts)))
    ax.set_yticklabels(company_counts.index)
    apply_gestalt_style(ax,
                       title='Top 20 Companies/Organizations by Number of AI Models',
                       xlabel='Number of Models',
                       ylabel='Company/Organization')
    
    # Add value labels (Law of Connectedness)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width)}',
                ha='left', va='center', fontsize=STYLE_CONFIG['tick_size'], 
                fontweight='bold', color=STYLE_CONFIG['text_color'])
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/17_top_companies_bar.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 5. Yearly trends - All metrics combined
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=STYLE_CONFIG['figure_bg'])
    
    # Yearly sector distribution (Law of Similarity - consistent colors)
    yearly_sector = df.groupby(['year', 'sector']).size().unstack(fill_value=0)
    sector_colors = [COLOR_PALETTE.get(s, COLOR_PALETTE['Other']) for s in yearly_sector.columns]
    yearly_sector.plot(kind='bar', ax=axes[0, 0], color=sector_colors, 
                      width=STYLE_CONFIG['bar_width'], 
                      edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                      alpha=STYLE_CONFIG['alpha_line'])
    apply_gestalt_style(axes[0, 0],
                       title='Sector Distribution by Year',
                       xlabel='Year',
                       ylabel='Number of Models')
    axes[0, 0].legend(title='Sector', frameon=True, shadow=STYLE_CONFIG['shadow'],
                      facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Yearly accessibility distribution
    yearly_acc = df.groupby(['year', 'accessibility_simple']).size().unstack(fill_value=0)
    acc_colors = [COLOR_PALETTE.get(a, COLOR_PALETTE['Other Source']) for a in yearly_acc.columns]
    yearly_acc.plot(kind='bar', ax=axes[0, 1], color=acc_colors, 
                   width=STYLE_CONFIG['bar_width'],
                   edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                   alpha=STYLE_CONFIG['alpha_line'])
    apply_gestalt_style(axes[0, 1],
                       title='Source Type Distribution by Year',
                       xlabel='Year',
                       ylabel='Number of Models')
    axes[0, 1].legend(title='Source Type', frameon=True, shadow=STYLE_CONFIG['shadow'],
                     facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Yearly country relationship distribution
    yearly_country = df.groupby(['year', 'country_relationship']).size().unstack(fill_value=0)
    colors_country = [COLOR_PALETTE.get(c, COLOR_PALETTE['Other']) for c in yearly_country.columns]
    yearly_country.plot(kind='bar', ax=axes[1, 0], 
                       color=colors_country, width=STYLE_CONFIG['bar_width'],
                       edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                       alpha=STYLE_CONFIG['alpha_line'])
    apply_gestalt_style(axes[1, 0],
                       title='Country Relationship Distribution by Year',
                       xlabel='Year',
                       ylabel='Number of Models')
    axes[1, 0].legend(title='Country Relationship', bbox_to_anchor=(1.05, 1), loc='upper left',
                     frameon=True, shadow=STYLE_CONFIG['shadow'],
                     facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Total models per year
    yearly_total = df.groupby('year').size()
    axes[1, 1].plot(yearly_total.index, yearly_total.values, 
                   marker='o', linewidth=STYLE_CONFIG['line_width'], 
                   markersize=STYLE_CONFIG['marker_size'], 
                   color=COLOR_PALETTE['FEYE'], alpha=STYLE_CONFIG['alpha_line'],
                   markerfacecolor=COLOR_PALETTE['FEYE'], markeredgecolor='white',
                   markeredgewidth=1)
    apply_gestalt_style(axes[1, 1],
                       title='Total AI Models Published per Year',
                       xlabel='Year',
                       ylabel='Number of Models')
    
    # Add value labels (Law of Connectedness)
    for year, count in yearly_total.items():
        axes[1, 1].text(year, count, f'{int(count)}', ha='center', va='bottom', 
                      fontweight='bold', fontsize=STYLE_CONFIG['tick_size'],
                      color=STYLE_CONFIG['text_color'])
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/18_yearly_trends_combined.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 6. Yearly trends by Alliance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=STYLE_CONFIG['figure_bg'])
    
    alliances = ['FEYE', 'NATO', 'Adversaries', 'Close Allies & Partners']
    colors_alliance = [COLOR_PALETTE.get(a, COLOR_PALETTE['Other']) for a in alliances]
    
    # Yearly model count by alliance
    yearly_alliance = df.groupby(['year', 'country_relationship']).size().unstack(fill_value=0)
    yearly_alliance = yearly_alliance[[a for a in alliances if a in yearly_alliance.columns]]
    yearly_alliance.plot(kind='bar', ax=axes[0, 0], 
                        color=colors_alliance[:len(yearly_alliance.columns)], 
                        width=STYLE_CONFIG['bar_width'],
                        edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                        alpha=STYLE_CONFIG['alpha_line'])
    apply_gestalt_style(axes[0, 0],
                       title='Model Count by Year and Alliance',
                       xlabel='Year',
                       ylabel='Number of Models')
    axes[0, 0].legend(title='Alliance', bbox_to_anchor=(1.05, 1), loc='upper left',
                     frameon=True, shadow=STYLE_CONFIG['shadow'],
                     facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Yearly sector by alliance
    yearly_sector_alliance = df.groupby(['year', 'sector']).size().unstack(fill_value=0)
    if 'Private' in yearly_sector_alliance.columns and 'Public' in yearly_sector_alliance.columns:
        x_pos = np.arange(len(yearly_sector_alliance.index))
        width = STYLE_CONFIG['spacing']
        bars1 = axes[0, 1].bar(x_pos - width/2, yearly_sector_alliance['Private'].values, 
                      width, label='Private', color=COLOR_PALETTE['Private'], 
                      alpha=STYLE_CONFIG['alpha_line'],
                      edgecolor='white', linewidth=STYLE_CONFIG['border_width'])
        bars2 = axes[0, 1].bar(x_pos + width/2, yearly_sector_alliance['Public'].values, 
                      width, label='Public', color=COLOR_PALETTE['Public'], 
                      alpha=STYLE_CONFIG['alpha_line'],
                      edgecolor='white', linewidth=STYLE_CONFIG['border_width'])
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(yearly_sector_alliance.index)
        apply_gestalt_style(axes[0, 1],
                           title='Sector Distribution by Year (All Alliances)',
                           xlabel='Year',
                           ylabel='Number of Models')
        axes[0, 1].legend(frameon=True, shadow=STYLE_CONFIG['shadow'],
                        facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Top countries by alliance
    top_countries_alliance = {}
    for alliance in alliances:
        df_alliance = df[df['country_relationship'] == alliance]
        if len(df_alliance) > 0:
            top_countries_alliance[alliance] = df_alliance['country'].value_counts().head(5)
    
    if top_countries_alliance:
        y_pos = np.arange(len(alliances))
        max_countries = max([len(v) for v in top_countries_alliance.values()])
        bars = []
        for i, alliance in enumerate(alliances):
            if alliance in top_countries_alliance:
                countries = top_countries_alliance[alliance]
                bar = axes[1, 0].barh(y_pos[i], countries.sum(), 
                               color=COLOR_PALETTE.get(alliance, COLOR_PALETTE['Other']),
                               height=STYLE_CONFIG['bar_width'],
                               alpha=STYLE_CONFIG['alpha_line'],
                               edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                               label=alliance if i == 0 else "")
                bars.append(bar)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(alliances)
        apply_gestalt_style(axes[1, 0],
                           title='Total Models by Alliance',
                           xlabel='Number of Models',
                           ylabel='Alliance')
    
    # Parameters by alliance over time (if available)
    df_params = df[df['parameters'].notna()]
    if len(df_params) > 0:
        yearly_params_alliance = df_params.groupby(['year', 'country_relationship'])['parameters'].mean().unstack(fill_value=0)
        yearly_params_alliance = yearly_params_alliance[[a for a in alliances if a in yearly_params_alliance.columns]]
        if len(yearly_params_alliance.columns) > 0:
            for col in yearly_params_alliance.columns:
                color = COLOR_PALETTE.get(col, COLOR_PALETTE['Other'])
                axes[1, 1].plot(yearly_params_alliance.index, yearly_params_alliance[col],
                               marker='o', label=col,
                               linewidth=STYLE_CONFIG['line_width'],
                               markersize=STYLE_CONFIG['marker_size'],
                               color=color, alpha=STYLE_CONFIG['alpha_line'],
                               markerfacecolor=color, markeredgecolor='white',
                               markeredgewidth=1)
            apply_gestalt_style(axes[1, 1],
                               title='Average Parameters by Year and Alliance',
                               xlabel='Year',
                               ylabel='Average Parameters')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend(title='Alliance', bbox_to_anchor=(1.05, 1), loc='upper left',
                           frameon=True, shadow=STYLE_CONFIG['shadow'],
                           facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/18b_yearly_trends_by_alliance.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def plot_parameters_over_time(df: pd.DataFrame):
    """Analyze parameters over time, including unknown/undisclosed models."""
    # Use model capacity from data (already categorized in modelsfetch.py)
    df['param_category'] = df.get('model_capacity', df['parameters'].apply(categorize_model_capacity))
    
    # 1. Parameters disclosure over time
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    
    monthly_data = df.groupby(['year_month', 'param_category']).size().unstack(fill_value=0)
    monthly_data.index = pd.to_datetime(monthly_data.index)
    monthly_data = monthly_data.sort_index()
    
    for category in monthly_data.columns:
        if category in PARAM_COLORS:
            ax.plot(monthly_data.index, monthly_data[category],
                   marker='o', label=category, 
                   linewidth=STYLE_CONFIG['line_width'],
                   markersize=STYLE_CONFIG['marker_size'],
                   color=PARAM_COLORS[category], alpha=STYLE_CONFIG['alpha_line'],
                   markerfacecolor=PARAM_COLORS[category], markeredgecolor='white',
                   markeredgewidth=1)
    
    apply_gestalt_style(ax,
                       title='Model Parameters Over Time\n(Including Unknown/Undisclosed)',
                       xlabel='Date',
                       ylabel='Number of Models')
    ax.legend(loc='best', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
             framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/25_parameters_over_time.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 2. Stacked area showing disclosure vs unknown
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    
    ax.stackplot(monthly_data.index,
                 *[monthly_data[col] for col in monthly_data.columns if col in PARAM_COLORS],
                 labels=[col for col in monthly_data.columns if col in PARAM_COLORS],
                 colors=[PARAM_COLORS[col] for col in monthly_data.columns if col in PARAM_COLORS],
                 alpha=STYLE_CONFIG['alpha_fill'],
                 edgecolor='white', linewidth=0.5)
    
    apply_gestalt_style(ax,
                       title='Model Parameters Over Time (Stacked)\n(Showing Disclosure Patterns)',
                       xlabel='Date',
                       ylabel='Cumulative Number of Models')
    ax.legend(loc='upper left', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
             framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/26_parameters_stacked_over_time.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 3. Percentage of unknown parameters over time
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    
    monthly_total = df.groupby('year_month').size()
    monthly_unknown = df[df['param_category'] == 'Unknown'].groupby('year_month').size()
    monthly_known = monthly_total - monthly_unknown.reindex(monthly_total.index, fill_value=0)
    
    monthly_total_index = pd.to_datetime(monthly_total.index)
    pct_unknown = (monthly_unknown.reindex(monthly_total.index, fill_value=0) / monthly_total * 100).values
    pct_known = (monthly_known / monthly_total * 100).values
    
    ax.fill_between(monthly_total_index, 0, pct_unknown, 
                    alpha=STYLE_CONFIG['alpha_fill'], color=COLOR_PALETTE['Unknown'],
                    label='Unknown/Undisclosed', edgecolor='white', linewidth=0.5)
    ax.fill_between(monthly_total_index, pct_unknown, 100,
                    alpha=STYLE_CONFIG['alpha_fill'], color=COLOR_PALETTE['FEYE'],
                    label='Disclosed', edgecolor='white', linewidth=0.5)
    
    apply_gestalt_style(ax,
                       title='Parameter Disclosure Rate Over Time\n(% Unknown vs Disclosed)',
                       xlabel='Date',
                       ylabel='Percentage of Models')
    ax.set_ylim(0, 100)
    ax.legend(loc='best', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
             framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/27_parameter_disclosure_rate.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 4. Parameters over time by Alliance
    df_params = df[df['parameters'].notna()].copy()
    if len(df_params) > 0:
        fig, ax = plt.subplots(figsize=(16, 10), facecolor=STYLE_CONFIG['figure_bg'])
        
        alliances = ['FEYE', 'NATO', 'Adversaries', 'Close Allies & Partners']
        colors_alliance = [COLOR_PALETTE.get(a, COLOR_PALETTE['Other']) for a in alliances]
        
        for idx, alliance in enumerate(alliances):
            df_alliance = df_params[df_params['country_relationship'] == alliance]
            if len(df_alliance) > 0:
                monthly = df_alliance.groupby('year_month')['parameters'].mean()
                monthly.index = pd.to_datetime(monthly.index)
                monthly = monthly.sort_index()
                if len(monthly) > 0:
                    ax.plot(monthly.index, monthly.values,
                           marker='o', label=alliance,
                           linewidth=STYLE_CONFIG['line_width'],
                           markersize=STYLE_CONFIG['marker_size'],
                           color=colors_alliance[idx],
                           alpha=STYLE_CONFIG['alpha_line'],
                           markerfacecolor=colors_alliance[idx],
                           markeredgecolor='white', markeredgewidth=1)
        
        apply_gestalt_style(ax,
                           title='Average Parameters Over Time by Alliance\n(Who is Building Bigger Models?)',
                           xlabel='Date',
                           ylabel='Average Parameters (log scale)')
        ax.set_yscale('log')
        ax.legend(frameon=True, shadow=STYLE_CONFIG['shadow'],
                 facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
                 framealpha=0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        plt.tight_layout(pad=STYLE_CONFIG['padding'])
        plt.savefig(f"{OUTPUT_DIR}/27b_parameters_time_by_alliance.png", dpi=300, bbox_inches='tight',
                    facecolor=STYLE_CONFIG['figure_bg'])
        plt.close()
    
    # 5. Parameter capacity distribution by Alliance
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    
    alliances = ['FEYE', 'NATO', 'Adversaries', 'Close Allies & Partners']
    param_alliance = pd.crosstab(df['country_relationship'], df['param_category'])
    # Filter rows (alliances) and columns (param categories)
    param_alliance = param_alliance.loc[[a for a in alliances if a in param_alliance.index]]
    param_alliance = param_alliance[[c for c in param_alliance.columns if c in PARAM_COLORS]]
    param_alliance.plot(kind='barh', stacked=True, ax=ax,
                       color=[PARAM_COLORS.get(c, COLOR_PALETTE['Other']) 
                             for c in param_alliance.columns],
                       width=STYLE_CONFIG['bar_width'], alpha=STYLE_CONFIG['alpha_line'],
                       edgecolor='white', linewidth=STYLE_CONFIG['border_width'])
    ax.set_title('Model Parameter Capacity by Alliance\n(Who Has More Capable Models?)', 
                fontsize=STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel('Number of Models', fontsize=STYLE_CONFIG['label_size'])
    ax.set_ylabel('Alliance', fontsize=STYLE_CONFIG['label_size'])
    ax.legend(title='Parameter Category', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
             bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='x', alpha=STYLE_CONFIG['grid_alpha'])
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/27c_parameters_capacity_by_alliance.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def plot_matrix_visualizations(df: pd.DataFrame):
    """Create matrix/heatmap visualizations showing arms race metrics."""
    # 1. Correlation matrix: Parameters, Training Compute, Hardware by Country Relationship
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor=STYLE_CONFIG['figure_bg'])
    
    # Prepare data with numeric values
    df_matrix = df.copy()
    # Handle log calculations safely
    param_series = pd.to_numeric(df_matrix['parameters'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df_matrix['param_log'] = np.log10(param_series.fillna(0) + 1)
    compute_series = pd.to_numeric(df_matrix['training_compute'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df_matrix['compute_log'] = np.log10(compute_series.fillna(0) + 1)
    
    # Matrix 1: Country Relationship vs Sector - Model Count
    crosstab1 = pd.crosstab(df_matrix['country_relationship'], df_matrix['sector'])
    sns.heatmap(crosstab1, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0, 0],
                cbar_kws={'label': 'Number of Models'}, linewidths=0.5,
                annot_kws={'fontsize': STYLE_CONFIG['tick_size'], 
                          'color': STYLE_CONFIG['text_color']})
    apply_gestalt_style(axes[0, 0],
                       title='Arms Race Matrix: Country Alliance vs Sector\n(Model Count)',
                       xlabel='Sector',
                       ylabel='Country Relationship')
    
    # Matrix 2: Country Relationship vs Sector - Average Parameters
    df_with_params = df_matrix[df_matrix['parameters'].notna()]
    if len(df_with_params) > 0:
        param_matrix = df_with_params.groupby(['country_relationship', 'sector'])['parameters'].mean().unstack(fill_value=0)
        sns.heatmap(param_matrix, annot=True, fmt='.2e', cmap='viridis', ax=axes[0, 1],
                    cbar_kws={'label': 'Avg Parameters'}, linewidths=0.5,
                    annot_kws={'fontsize': STYLE_CONFIG['tick_size'], 
                              'color': STYLE_CONFIG['text_color']})
        apply_gestalt_style(axes[0, 1],
                           title='Arms Race Matrix: Country Alliance vs Sector\n(Average Parameters)',
                           xlabel='Sector',
                           ylabel='Country Relationship')
    
    # Matrix 3: Country Relationship vs Sector - Average Training Compute
    df_with_compute = df_matrix[df_matrix['training_compute'].notna()]
    if len(df_with_compute) > 0:
        compute_matrix = df_with_compute.groupby(['country_relationship', 'sector'])['training_compute'].mean().unstack(fill_value=0)
        sns.heatmap(compute_matrix, annot=True, fmt='.2e', cmap='plasma', ax=axes[1, 0],
                    cbar_kws={'label': 'Avg Training Compute (FLOP)'}, linewidths=0.5,
                    annot_kws={'fontsize': STYLE_CONFIG['tick_size'], 
                              'color': STYLE_CONFIG['text_color']})
        apply_gestalt_style(axes[1, 0],
                           title='Arms Race Matrix: Country Alliance vs Sector\n(Average Training Compute)',
                           xlabel='Sector',
                           ylabel='Country Relationship')
    
    # Matrix 4: Top Countries vs Top Hardware
    top_countries = df_matrix['country'].value_counts().head(8).index.tolist()
    top_hardware = df_matrix[df_matrix['training_hardware'].notna()]['training_hardware'].value_counts().head(8).index.tolist()
    if top_hardware:
        df_hw_matrix = df_matrix[(df_matrix['country'].isin(top_countries)) & 
                                 (df_matrix['training_hardware'].isin(top_hardware))]
        hw_matrix = pd.crosstab(df_hw_matrix['country'], df_hw_matrix['training_hardware'])
        sns.heatmap(hw_matrix, annot=True, fmt='d', cmap='coolwarm', ax=axes[1, 1],
                    cbar_kws={'label': 'Number of Models'}, linewidths=0.5,
                    annot_kws={'fontsize': STYLE_CONFIG['tick_size'], 
                              'color': STYLE_CONFIG['text_color']})
        apply_gestalt_style(axes[1, 1],
                           title='Arms Race Matrix: Top Countries vs Top Hardware',
                           xlabel='Hardware',
                           ylabel='Country')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/35_matrix_arms_race_overview.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 2. FEYE vs NATO vs Adversaries comparison matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=STYLE_CONFIG['figure_bg'])
    
    alliance_groups = ['FEYE', 'NATO', 'Adversaries']
    df_alliances = df_matrix[df_matrix['country_relationship'].isin(alliance_groups)]
    
    if len(df_alliances) > 0:
        # Model count by alliance
        alliance_counts = df_alliances['country_relationship'].value_counts()
        bars0 = axes[0].bar(alliance_counts.index, alliance_counts.values,
                   color=[COLOR_PALETTE.get(a, COLOR_PALETTE['Other']) for a in alliance_counts.index],
                   width=STYLE_CONFIG['bar_width'],
                   edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                   alpha=STYLE_CONFIG['alpha_line'])
        apply_gestalt_style(axes[0],
                           title='Model Count by Alliance',
                           xlabel='Alliance',
                           ylabel='Number of Models')
        for i, (alliance, count) in enumerate(alliance_counts.items()):
            axes[0].text(i, count, f'{int(count)}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=STYLE_CONFIG['tick_size'],
                        color=STYLE_CONFIG['text_color'])
        
        # Average parameters by alliance
        df_alliance_params = df_alliances[df_alliances['parameters'].notna()]
        if len(df_alliance_params) > 0:
            alliance_params = df_alliance_params.groupby('country_relationship')['parameters'].mean()
            bars1 = axes[1].bar(alliance_params.index, alliance_params.values,
                       color=[COLOR_PALETTE.get(a, COLOR_PALETTE['Other']) for a in alliance_params.index],
                       width=STYLE_CONFIG['bar_width'],
                       edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                       alpha=STYLE_CONFIG['alpha_line'])
            apply_gestalt_style(axes[1],
                               title='Average Parameters by Alliance',
                               xlabel='Alliance',
                               ylabel='Average Parameters')
            axes[1].set_yscale('log')
        
        # Average training compute by alliance
        df_alliance_compute = df_alliances[df_alliances['training_compute'].notna()]
        if len(df_alliance_compute) > 0:
            alliance_compute = df_alliance_compute.groupby('country_relationship')['training_compute'].mean()
            bars2 = axes[2].bar(alliance_compute.index, alliance_compute.values,
                       color=[COLOR_PALETTE.get(a, COLOR_PALETTE['Other']) for a in alliance_compute.index],
                       width=STYLE_CONFIG['bar_width'],
                       edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                       alpha=STYLE_CONFIG['alpha_line'])
            apply_gestalt_style(axes[2],
                               title='Average Training Compute by Alliance',
                               xlabel='Alliance',
                               ylabel='Average Training Compute (FLOP)')
            axes[2].set_yscale('log')
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/36_matrix_alliance_comparison.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def plot_bubble_visualizations(df: pd.DataFrame):
    """Create bubble chart visualizations for arms race metrics."""
    # 1. Bubble chart: Country Relationship vs Parameters vs Training Compute
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=STYLE_CONFIG['figure_bg'])
    
    df_bubble = df[(df['parameters'].notna()) & (df['training_compute'].notna())].copy()
    
    if len(df_bubble) > 0:
        alliances = ['FEYE', 'NATO', 'Adversaries', 'Close Allies & Partners']
        colors_alliance = [COLOR_PALETTE.get(a, COLOR_PALETTE['Other']) for a in alliances]
        labeled_alliances = set()
        
        for idx, alliance in enumerate(alliances):
            df_alliance = df_bubble[df_bubble['country_relationship'] == alliance]
            if len(df_alliance) > 0:
                # Bubble size based on model count per country
                country_counts = df_alliance.groupby('country').size()
                for country, count in country_counts.items():
                    df_country = df_alliance[df_alliance['country'] == country]
                    if len(df_country) > 0:
                        avg_params = df_country['parameters'].mean()
                        avg_compute = df_country['training_compute'].mean()
                        bubble_size = count * 50  # Scale bubble size
                        
                        # Only label first country of each alliance to avoid duplicate labels
                        label_text = alliance if alliance not in labeled_alliances else ""
                        if alliance not in labeled_alliances:
                            labeled_alliances.add(alliance)
                        ax.scatter(avg_params, avg_compute, s=bubble_size,
                                 c=[colors_alliance[idx]], alpha=0.6,
                                 edgecolors='white', linewidths=2,
                                 label=label_text)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        apply_gestalt_style(ax,
                           title='Arms Race Bubble Chart: Parameters vs Training Compute\n(Bubble size = Model count per country)',
                           xlabel='Average Parameters (log scale)',
                           ylabel='Average Training Compute (FLOP, log scale)')
        ax.legend(frameon=True, shadow=STYLE_CONFIG['shadow'],
                 facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
                 framealpha=0.9)
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/43_bubble_params_vs_compute.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()
    
    # 2. Bubble chart: FEYE vs NATO vs Adversaries - Parameters and Compute
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=STYLE_CONFIG['figure_bg'])
    
    if len(df_bubble) > 0:
        for alliance in ['FEYE', 'NATO', 'Adversaries']:
            df_alliance = df_bubble[df_bubble['country_relationship'] == alliance]
            if len(df_alliance) > 0:
                # Aggregate by alliance
                avg_params = df_alliance['parameters'].mean()
                avg_compute = df_alliance['training_compute'].mean()
                count = len(df_alliance)
                bubble_size = count * 30
                
                ax.scatter(avg_params, avg_compute, s=bubble_size,
                         c=[COLOR_PALETTE.get(alliance, COLOR_PALETTE['Other'])],
                         alpha=0.7, edgecolors='white', linewidths=3,
                         label=f'{alliance} (n={count})')
                
                # Add text label
                ax.text(float(avg_params), float(avg_compute), alliance, ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        apply_gestalt_style(ax,
                           title='Arms Race: FEYE vs NATO vs Adversaries\n(Who Has More Capability?)',
                           xlabel='Average Parameters (log scale)',
                           ylabel='Average Training Compute (FLOP, log scale)')
        ax.legend(frameon=True, shadow=STYLE_CONFIG['shadow'],
                 facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'],
                 framealpha=0.9)
    
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/44_bubble_alliance_comparison.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def generate_summary_statistics(df: pd.DataFrame):
    """Generate and save summary statistics."""
    stats = []
    stats.append("=" * 80)
    stats.append("AI ARMS RACE - SUMMARY STATISTICS")
    stats.append("=" * 80)
    stats.append("")
    
    stats.append(f"Total Models Analyzed: {len(df)}")
    stats.append(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    stats.append("")
    
    stats.append("SECTOR DISTRIBUTION:")
    stats.append("-" * 40)
    sector_counts = df['sector'].value_counts()
    for sector, count in sector_counts.items():
        pct = (count / len(df)) * 100
        stats.append(f"  {sector}: {count} ({pct:.1f}%)")
    stats.append("")
    
    stats.append("SOURCE TYPE DISTRIBUTION:")
    stats.append("-" * 40)
    df['accessibility_simple'] = df['accessibility'].apply(
        lambda x: 'Open Source' if 'Open' in str(x) 
        else 'Closed Source' if 'Closed' in str(x) 
        else 'Other'
    )
    acc_counts = df['accessibility_simple'].value_counts()
    for acc, count in acc_counts.items():
        pct = (count / len(df)) * 100
        stats.append(f"  {acc}: {count} ({pct:.1f}%)")
    stats.append("")
    
    stats.append("COUNTRY RELATIONSHIP DISTRIBUTION:")
    stats.append("-" * 40)
    country_rel_counts = df['country_relationship'].value_counts()
    for rel, count in country_rel_counts.items():
        pct = (count / len(df)) * 100
        stats.append(f"  {rel}: {count} ({pct:.1f}%)")
    stats.append("")
    
    stats.append("TOP 10 COUNTRIES:")
    stats.append("-" * 40)
    country_counts = df['country'].value_counts().head(10)
    for country, count in country_counts.items():
        pct = (count / len(df)) * 100
        stats.append(f"  {country}: {count} ({pct:.1f}%)")
    stats.append("")
    
    stats.append("TOP 10 COMPANIES/ORGANIZATIONS:")
    stats.append("-" * 40)
    company_counts = df[df['company'] != 'Unknown']['company'].value_counts().head(10)
    for company, count in company_counts.items():
        pct = (count / len(df)) * 100
        stats.append(f"  {company}: {count} ({pct:.1f}%)")
    stats.append("")
    
    stats.append("YEARLY BREAKDOWN:")
    stats.append("-" * 40)
    yearly_counts = df.groupby('year').size().sort_index()
    for year, count in yearly_counts.items():
        stats.append(f"  {year}: {count} models")
    stats.append("")
    
    stats_text = "\n".join(stats)
    
    # Save to file
    with open(f"{OUTPUT_DIR}/summary_statistics.txt", 'w', encoding='utf-8') as f:
        f.write(stats_text)

def plot_growth_rate_analysis(df: pd.DataFrame):
    """Analyze growth rates by alliance - who is accelerating fastest?"""
    # Calculate year-over-year growth rates
    alliances = ['FEYE', 'NATO', 'Adversaries', 'Close Allies & Partners']
    colors_alliance = [COLOR_PALETTE.get(a, COLOR_PALETTE['Other']) for a in alliances]
    
    # Yearly counts by alliance
    yearly_alliance = df.groupby(['year', 'country_relationship']).size().unstack(fill_value=0)
    yearly_alliance = yearly_alliance[[a for a in alliances if a in yearly_alliance.columns]]
    
    # Calculate growth rates
    growth_rates = yearly_alliance.pct_change(axis=0) * 100  # Percentage change
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    
    for idx, alliance in enumerate(alliances):
        if alliance in growth_rates.columns:
            ax.plot(growth_rates.index, growth_rates[alliance],
                   marker='o', label=alliance,
                   linewidth=STYLE_CONFIG['line_width'],
                   markersize=STYLE_CONFIG['marker_size'],
                   color=colors_alliance[idx],
                   alpha=STYLE_CONFIG['alpha_line'],
                   markerfacecolor=colors_alliance[idx],
                   markeredgecolor='white', markeredgewidth=1)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('Year-over-Year Growth Rate by Alliance\n(Who is Accelerating Fastest?)', 
                fontsize=STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=STYLE_CONFIG['label_size'])
    ax.set_ylabel('Growth Rate (%)', fontsize=STYLE_CONFIG['label_size'])
    ax.legend(loc='best', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
    ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'])
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/09_growth_rate_by_alliance.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def plot_market_share_analysis(df: pd.DataFrame):
    """Analyze market share over time by alliance."""
    alliances = ['FEYE', 'NATO', 'Adversaries', 'Close Allies & Partners']
    colors_alliance = [COLOR_PALETTE.get(a, COLOR_PALETTE['Other']) for a in alliances]
    
    # Monthly counts by alliance
    monthly_alliance = df.groupby(['year_month', 'country_relationship']).size().unstack(fill_value=0)
    monthly_alliance.index = pd.to_datetime(monthly_alliance.index)
    monthly_alliance = monthly_alliance.sort_index()
    monthly_alliance = monthly_alliance[[a for a in alliances if a in monthly_alliance.columns]]
    
    # Calculate market share (percentage of total models each month)
    monthly_total = monthly_alliance.sum(axis=1)
    market_share = monthly_alliance.div(monthly_total, axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=STYLE_CONFIG['figure_bg'])
    
    ax.stackplot(market_share.index,
                 *[market_share[col] for col in market_share.columns if col in alliances],
                 labels=[col for col in market_share.columns if col in alliances],
                 colors=[colors_alliance[alliances.index(col)] for col in market_share.columns if col in alliances],
                 alpha=STYLE_CONFIG['alpha_fill'],
                 edgecolor='white', linewidth=0.5)
    
    ax.set_title('Market Share Over Time by Alliance\n(Who is Gaining/Losing Ground?)', 
                fontsize=STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=STYLE_CONFIG['label_size'])
    ax.set_ylabel('Market Share (%)', fontsize=STYLE_CONFIG['label_size'])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', frameon=True, shadow=STYLE_CONFIG['shadow'],
             facecolor=STYLE_CONFIG['axes_bg'], edgecolor=STYLE_CONFIG['grid_color'])
    ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/10_market_share_by_alliance.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def plot_lead_indicators_dashboard(df: pd.DataFrame):
    """Create a comprehensive dashboard showing who's winning on multiple metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor=STYLE_CONFIG['figure_bg'])
    alliances = ['FEYE', 'NATO', 'Adversaries', 'Close Allies & Partners']
    colors_alliance = [COLOR_PALETTE.get(a, COLOR_PALETTE['Other']) for a in alliances]
    
    # 1. Total model count
    ax = axes[0, 0]
    alliance_counts = df['country_relationship'].value_counts()
    alliance_counts = alliance_counts[[a for a in alliances if a in alliance_counts.index]]
    bars = ax.bar(alliance_counts.index, alliance_counts.values,
                  color=[colors_alliance[alliances.index(a)] for a in alliance_counts.index],
                  width=STYLE_CONFIG['bar_width'],
                  edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                  alpha=STYLE_CONFIG['alpha_line'])
    ax.set_title('Total Model Count by Alliance\n(Who Has More Models?)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Models', fontsize=11)
    ax.grid(axis='y', alpha=STYLE_CONFIG['grid_alpha'])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Average parameters (model size)
    ax = axes[0, 1]
    df_params = df[df['parameters'].notna()]
    if len(df_params) > 0:
        param_avg = df_params.groupby('country_relationship')['parameters'].mean()
        param_avg = param_avg[[a for a in alliances if a in param_avg.index]]
        bars = ax.bar(param_avg.index, param_avg.values,
                     color=[colors_alliance[alliances.index(a)] for a in param_avg.index],
                     width=STYLE_CONFIG['bar_width'],
                     edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                     alpha=STYLE_CONFIG['alpha_line'])
        ax.set_title('Average Model Size (Parameters)\n(Who Builds Bigger Models?)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Average Parameters', fontsize=11)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=STYLE_CONFIG['grid_alpha'])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Recent activity (last 2 years)
    ax = axes[1, 0]
    recent_date = df['date'].max() - pd.Timedelta(days=730)  # 2 years
    df_recent = df[df['date'] >= recent_date]
    recent_counts = df_recent['country_relationship'].value_counts()
    recent_counts = recent_counts[[a for a in alliances if a in recent_counts.index]]
    bars = ax.bar(recent_counts.index, recent_counts.values,
                 color=[colors_alliance[alliances.index(a)] for a in recent_counts.index],
                 width=STYLE_CONFIG['bar_width'],
                 edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                 alpha=STYLE_CONFIG['alpha_line'])
    ax.set_title('Models Published in Last 2 Years\n(Who is Most Active Recently?)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Models', fontsize=11)
    ax.grid(axis='y', alpha=STYLE_CONFIG['grid_alpha'])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Open source percentage
    ax = axes[1, 1]
    df['accessibility_simple'] = df['accessibility'].apply(
        lambda x: 'Open Source' if 'Open' in str(x) else 'Closed Source' if 'Closed' in str(x) else 'Other'
    )
    open_pct = {}
    for alliance in alliances:
        df_alliance = df[df['country_relationship'] == alliance]
        if len(df_alliance) > 0:
            open_count = len(df_alliance[df_alliance['accessibility_simple'] == 'Open Source'])
            open_pct[alliance] = (open_count / len(df_alliance)) * 100
    
    if open_pct:
        open_pct_series = pd.Series(open_pct)
        open_pct_series = open_pct_series[[a for a in alliances if a in open_pct_series.index]]
        bars = ax.bar(open_pct_series.index, open_pct_series.values,
                     color=[colors_alliance[alliances.index(a)] for a in open_pct_series.index],
                     width=STYLE_CONFIG['bar_width'],
                     edgecolor='white', linewidth=STYLE_CONFIG['border_width'],
                     alpha=STYLE_CONFIG['alpha_line'])
        ax.set_title('Open Source Percentage by Alliance\n(Who Shares More Technology?)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Open Source %', fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=STYLE_CONFIG['grid_alpha'])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle('AI Arms Race Lead Indicators Dashboard\n(Who is Winning?)', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(pad=STYLE_CONFIG['padding'])
    plt.savefig(f"{OUTPUT_DIR}/11_lead_indicators_dashboard.png", dpi=300, bbox_inches='tight',
                facecolor=STYLE_CONFIG['figure_bg'])
    plt.close()

def main():
    """Main function to generate all visualizations."""
    # Create output directory
    create_output_dir()
    
    # Load data
    data = load_data()
    
    # Prepare DataFrame
    df = prepare_dataframe(data)
    
    # Generate core visualizations focused on arms race
    plot_sector_distribution(df)
    plot_accessibility_distribution(df)
    plot_country_distribution(df)
    
    # Generate new high-value arms race visualizations
    plot_growth_rate_analysis(df)
    plot_market_share_analysis(df)
    plot_lead_indicators_dashboard(df)
    
    # Generate parameters analysis (who builds bigger models)
    plot_parameters_over_time(df)
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    # Keep some additional visualizations that are still useful
    plot_combined_analysis(df)
    plot_matrix_visualizations(df)
    plot_bubble_visualizations(df)

if __name__ == "__main__":
    main()

