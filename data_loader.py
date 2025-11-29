import pandas as pd
import numpy as np
import streamlit as st
import os

def load_all_sheets(excel_path="marks_dataset.xlsx"):
    """Load all Excel sheets into a dictionary"""
    try:
        if not os.path.exists(excel_path):
            st.error(f"File {excel_path} not found. Please upload a file.")
            return {}
            
        sheets_dict = pd.read_excel(excel_path, sheet_name=None)
        st.success(f"✅ Successfully loaded {len(sheets_dict)} sheets")
        return sheets_dict
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {e}")
        return {}

def establish_assessment_timeline():
    """Define the temporal sequence of assessments to prevent data leakage"""
    timeline = {
        'early_assignments': ['As:1', 'As:2', 'As:3'],
        'pre_midterm1_quizzes': ['Qz:1', 'Qz:2', 'Qz:3', 'Qz:4'],
        'midterm1': ['S-I:1', 'S-I'],
        'between_midterms_assignments': ['As:4', 'As:5'],
        'between_midterms_quizzes': ['Qz:5', 'Qz:6', 'Qz:7'],
        'midterm2': ['S-II:1', 'S-II'],
        'late_assignments': ['As:6'],
        'late_quizzes': ['Qz:8'],
        'project': ['Proj:1', 'Proj'],
        'final_exam': ['Final:1', 'Final:2', 'Final:3', 'Final:4', 'Final:5', 'Final']
    }
    return timeline

def consolidate_data(sheets_dict):
    """Consolidate all sheets with proper indexing"""
    if not sheets_dict:
        return pd.DataFrame()
        
    consolidated_data = pd.DataFrame()
    
    for sheet_name, df in sheets_dict.items():
        df['sheet_source'] = sheet_name
        df['student_id'] = range(len(df))
        df.columns = [col.strip() if isinstance(col, str) else str(col) for col in df.columns]
        consolidated_data = pd.concat([consolidated_data, df], ignore_index=True)
    
    return consolidated_data

def handle_missing_data(df, timeline):
    """Handle missing values with domain-aware strategies"""
    if df.empty:
        return df
        
    df_clean = df.copy()
    
    assessment_columns = []
    for stage in timeline.values():
        assessment_columns.extend(stage)
    
    assessment_columns = [col for col in assessment_columns if col in df.columns]
    
    # Group by sheet source and impute missing values with median
    for column in assessment_columns:
        if df_clean[column].isnull().any():
            df_clean[column] = df_clean.groupby('sheet_source')[column].transform(
                lambda x: x.fillna(x.median())
            )
    
    # For remaining missing values, use overall median
    for column in assessment_columns:
        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
    
    return df_clean

def create_features(df, timeline):
    """Create engineered features for modeling"""
    if df.empty:
        return df
        
    df_engineered = df.copy()
    
    try:
        # Create cumulative scores for available columns only
        available_early_assignments = [col for col in timeline['early_assignments'] if col in df.columns]
        available_pre_midterm_quizzes = [col for col in timeline['pre_midterm1_quizzes'] if col in df.columns]
        
        if available_early_assignments:
            df_engineered['pre_midterm1_total'] = df_engineered[available_early_assignments].sum(axis=1)
        if available_pre_midterm_quizzes:
            df_engineered['pre_midterm1_quiz_total'] = df_engineered[available_pre_midterm_quizzes].sum(axis=1)
        
        # Create performance indicators
        if available_early_assignments:
            df_engineered['assignment_consistency'] = df_engineered[available_early_assignments].std(axis=1)
        if available_pre_midterm_quizzes:
            df_engineered['quiz_consistency'] = df_engineered[available_pre_midterm_quizzes].std(axis=1)
    except Exception as e:
        st.warning(f"Feature engineering encountered issues: {e}")
    
    return df_engineered

def prepare_modeling_datasets(preprocessed_data, timeline):
    """Prepare datasets for each research question"""
    modeling_datasets = {}
    
    if preprocessed_data.empty:
        return modeling_datasets
    
    # RQ1: Predict Midterm I
    rq1_features = timeline['early_assignments'] + timeline['pre_midterm1_quizzes'] + \
                   ['pre_midterm1_total', 'pre_midterm1_quiz_total', 
                    'assignment_consistency', 'quiz_consistency']
    rq1_features = [f for f in rq1_features if f in preprocessed_data.columns]
    
    if 'S-I:1' in preprocessed_data.columns and rq1_features:
        rq1_target = 'S-I:1'
        rq1_data = preprocessed_data[rq1_features + [rq1_target]].dropna()
        
        if not rq1_data.empty:
            modeling_datasets['RQ1'] = {
                'features': rq1_features,
                'target': rq1_target,
                'data': rq1_data
            }
    
    # RQ2: Predict Midterm II
    if 'RQ1' in modeling_datasets:
        rq2_features = rq1_features + timeline['midterm1'] + \
                       timeline['between_midterms_assignments'] + \
                       timeline['between_midterms_quizzes']
        rq2_features = [f for f in rq2_features if f in preprocessed_data.columns]
        
        if 'S-II:1' in preprocessed_data.columns and rq2_features:
            rq2_target = 'S-II:1'
            rq2_data = preprocessed_data[rq2_features + [rq2_target]].dropna()
            
            if not rq2_data.empty:
                modeling_datasets['RQ2'] = {
                    'features': rq2_features,
                    'target': rq2_target,
                    'data': rq2_data
                }
    
    # RQ3: Predict Final Exam
    if 'RQ2' in modeling_datasets:
        rq3_features = rq2_features + timeline['midterm2'] + \
                       timeline['late_assignments'] + timeline['late_quizzes'] + \
                       timeline['project']
        rq3_features = [f for f in rq3_features if f in preprocessed_data.columns]
        
        if 'Final:1' in preprocessed_data.columns and rq3_features:
            rq3_target = 'Final:1'
            rq3_data = preprocessed_data[rq3_features + [rq3_target]].dropna()
            
            if not rq3_data.empty:
                modeling_datasets['RQ3'] = {
                    'features': rq3_features,
                    'target': rq3_target,
                    'data': rq3_data
                }
    
    return modeling_datasets

def load_and_preprocess_data(excel_path="marks_dataset.xlsx"):
    """Main function to load and preprocess all data"""
    sheets_dict = load_all_sheets(excel_path)
    if not sheets_dict:
        return pd.DataFrame(), {}, {}
    
    timeline = establish_assessment_timeline()
    consolidated_data = consolidate_data(sheets_dict)
    
    if consolidated_data.empty:
        return pd.DataFrame(), {}, {}
        
    cleaned_data = handle_missing_data(consolidated_data, timeline)
    featured_data = create_features(cleaned_data, timeline)
    modeling_datasets = prepare_modeling_datasets(featured_data, timeline)
    
    return featured_data, modeling_datasets, timeline
