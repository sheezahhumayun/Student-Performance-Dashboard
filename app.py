import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(__file__))

try:
    from data_loader import load_and_preprocess_data
    from model_trainer import train_all_models
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Please make sure data_loader.py and model_trainer.py are in the same directory.")

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .flowchart-step {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .phase-header {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Student Performance Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Section",
        ["📊 Data Overview", "🤖 Model Training", "📈 Results & Analysis", "📋 Final Report", "📋 Workflow Flowchart"]
    )
    
    # File uploader in sidebar
    st.sidebar.title("Data Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    # Load data with caching and progress
    @st.cache_data(show_spinner=False)
    def load_data(uploaded_file):
        if uploaded_file is not None:
            try:
                with st.spinner("Loading and preprocessing data..."):
                    # Save uploaded file temporarily
                    with open("temp_marks_data.xlsx", "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    data = load_and_preprocess_data("temp_marks_data.xlsx")
                    
                    # Clean up temporary file
                    if os.path.exists("temp_marks_data.xlsx"):
                        os.remove("temp_marks_data.xlsx")
                    
                    return data
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return None, None, None
        else:
            # Try to load default file
            try:
                with st.spinner("Loading default dataset..."):
                    return load_and_preprocess_data("marks_dataset.xlsx")
            except Exception as e:
                st.warning("Please upload an Excel file or ensure 'marks_dataset.xlsx' exists.")
                return None, None, None
    
    if app_mode == "📋 Workflow Flowchart":
        show_workflow_flowchart()
        return
    
    data = load_data(uploaded_file)
    
    if data[0] is None:
        st.info("👆 Please upload an Excel file with student assessment data to get started.")
        return
    
    preprocessed_data, modeling_datasets, timeline = data
    
    if app_mode == "📊 Data Overview":
        show_data_overview(preprocessed_data, modeling_datasets, timeline)
    elif app_mode == "🤖 Model Training":
        show_model_training(preprocessed_data, modeling_datasets, timeline)
    elif app_mode == "📈 Results & Analysis":
        show_results_analysis()
    elif app_mode == "📋 Final Report":
        show_final_report()

def show_workflow_flowchart():
    st.header("📋 Complete Workflow Pipeline")
    st.info("This diagram shows the complete workflow from data loading to final reporting, including all preprocessing, modeling, and evaluation steps.")
    
    # Create a visual flowchart using columns and markdown
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Phase 1: Data Understanding & Exploration
        st.markdown('<div class="phase-header">📊 Phase 1: Data Understanding & Exploration</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 1: Comprehensive Data Inventory</strong><br>
        • Load all 6 Excel sheets<br>
        • Document assessment components<br>
        • Create temporal sequence<br>
        • Examine structure and weightages
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 2: Temporal Analysis & Domain Logic</strong><br>
        • Midterm I: Pre-midterm assignments/quizzes only<br>
        • Midterm II: Midterm I + intermediate assessments<br>
        • Final: All components except final exam<br>
        • Prevent data leakage
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 3: Initial EDA & Visualization</strong><br>
        • Distribution plots<br>
        • Missing value analysis<br>
        • Correlation matrices<br>
        • Outlier detection
        </div>
        """, unsafe_allow_html=True)
        
        # Phase 2: Data Preprocessing
        st.markdown('<div class="phase-header">🔧 Phase 2: Data Preprocessing</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 4: Data Consolidation</strong><br>
        • Combine all 6 sheets<br>
        • Consistent column naming<br>
        • Student identifiers<br>
        • Handle duplicates
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 5: Missing Data Handling</strong><br>
        • Analyze missing patterns<br>
        • Median imputation within sheets<br>
        • Domain-aware strategies<br>
        • Document decisions
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 6: Feature Engineering</strong><br>
        • Normalized scores<br>
        • Cumulative scores<br>
        • Performance indicators<br>
        • Temporal features
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 7: Train-Test Split</strong><br>
        • 80-20 split<br>
        • Temporal consistency<br>
        • No data leakage<br>
        • Random state 42
        </div>
        """, unsafe_allow_html=True)
        
        # Phase 3: Model Development
        st.markdown('<div class="phase-header">🤖 Phase 3: Model Development</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 8: Baseline Models</strong><br>
        • DummyRegressor (mean)<br>
        • DummyRegressor (median)<br>
        • Performance benchmarks<br>
        • Comparison baseline
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 9: Model Selection & Training</strong><br>
        • Simple Linear Regression<br>
        • Multiple Linear Regression<br>
        • Polynomial Regression (deg 2, 3)<br>
        • Temporal feature selection
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 10: Feature Selection</strong><br>
        • RQ1: Pre-midterm assessments<br>
        • RQ2: Midterm I + intermediate<br>
        • RQ3: All except final exam<br>
        • Domain-aware selection
        </div>
        """, unsafe_allow_html=True)
    
    # Second row of columns for remaining phases
    col3, col4 = st.columns([1, 1])
    
    with col3:
        # Phase 4: Model Evaluation
        st.markdown('<div class="phase-header">📈 Phase 4: Model Evaluation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 11: Bootstrapping</strong><br>
        • 500 bootstrap samples<br>
        • Training data only<br>
        • 95% confidence intervals<br>
        • MAE stability analysis
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 12: Comprehensive Evaluation</strong><br>
        • MAE (primary metric)<br>
        • RMSE (penalizes large errors)<br>
        • R² (variance explained)<br>
        • Multiple metrics comparison
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 13: Model Interpretation</strong><br>
        • Coefficient analysis<br>
        • Feature importance<br>
        • Residual patterns<br>
        • Domain consistency
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Phase 5: Results Comparison & Reporting
        st.markdown('<div class="phase-header">📋 Phase 5: Results & Reporting</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 14: Comparative Analysis</strong><br>
        • Model comparison tables<br>
        • Best model identification<br>
        • Baseline comparison<br>
        • Overfitting analysis
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="flowchart-step">
        <strong>Step 15: Final Reporting</strong><br>
        • Preprocessing documentation<br>
        • Confidence intervals<br>
        • Business insights<br>
        • Deployment recommendations
        </div>
        """, unsafe_allow_html=True)
    
    # Visual flowchart diagram
    st.markdown("---")
    st.subheader("🎯 Visual Workflow Diagram")
    
    # Create a simplified flowchart using Plotly
    fig = create_flowchart_diagram()
    st.plotly_chart(fig, use_container_width=True)
    
    # Key considerations
    st.markdown("---")
    st.subheader("🔑 Key Workflow Considerations")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("""
        **Data Integrity:**
        - Temporal separation to prevent leakage
        - Consistent preprocessing across all sheets
        - Domain-aware feature selection
        - Proper train-test splitting
        """)
        
        st.markdown("""
        **Model Robustness:**
        - Multiple model types for comparison
        - Bootstrap confidence intervals
        - Comprehensive evaluation metrics
        - Overfitting detection
        """)
    
    with col6:
        st.markdown("""
        **Reproducibility:**
        - Fixed random seeds (42)
        - Cached data processing
        - Documented decisions
        - Version control
        """)
        
        st.markdown("""
        **Practical Application:**
        - Educational insights
        - Actionable recommendations
        - Confidence intervals for decisions
        - Model interpretability
        """)

def create_flowchart_diagram():
    """Create a visual flowchart diagram using Plotly"""
    
    # Define nodes for the flowchart
    nodes = [
        # Phase 1
        {"label": "Data Loading", "phase": 1, "x": 0, "y": 0},
        {"label": "Data Inventory", "phase": 1, "x": 0, "y": -1},
        {"label": "Temporal Analysis", "phase": 1, "x": 0, "y": -2},
        {"label": "EDA & Visualization", "phase": 1, "x": 0, "y": -3},
        
        # Phase 2
        {"label": "Data Consolidation", "phase": 2, "x": 2, "y": 0},
        {"label": "Missing Data Handling", "phase": 2, "x": 2, "y": -1},
        {"label": "Feature Engineering", "phase": 2, "x": 2, "y": -2},
        {"label": "Train-Test Split", "phase": 2, "x": 2, "y": -3},
        
        # Phase 3
        {"label": "Baseline Models", "phase": 3, "x": 4, "y": 0},
        {"label": "Model Training", "phase": 3, "x": 4, "y": -1},
        {"label": "Feature Selection", "phase": 3, "x": 4, "y": -2},
        
        # Phase 4
        {"label": "Bootstrapping", "phase": 4, "x": 6, "y": 0},
        {"label": "Model Evaluation", "phase": 4, "x": 6, "y": -1},
        {"label": "Interpretation", "phase": 4, "x": 6, "y": -2},
        
        # Phase 5
        {"label": "Comparative Analysis", "phase": 5, "x": 8, "y": -1},
        {"label": "Final Reporting", "phase": 5, "x": 8, "y": -2},
    ]
    
    # Create edges (connections between nodes)
    edges = [
        (0, 1), (1, 2), (2, 3),  # Phase 1
        (3, 4), (4, 5), (5, 6), (6, 7),  # Phase 1->2
        (7, 8), (8, 9), (9, 10),  # Phase 2->3
        (10, 11), (11, 12), (12, 13),  # Phase 3->4
        (13, 14), (14, 15)  # Phase 4->5
    ]
    
    # Create the figure
    fig = go.Figure()
    
    # Phase colors
    phase_colors = {
        1: '#FF6B6B',  # Red
        2: '#4ECDC4',  # Teal
        3: '#45B7D1',  # Blue
        4: '#96CEB4',  # Green
        5: '#FFEAA7'   # Yellow
    }
    
    phase_names = {
        1: "Data Understanding",
        2: "Preprocessing",
        3: "Model Development", 
        4: "Model Evaluation",
        5: "Results & Reporting"
    }
    
    # Add edges (lines between nodes)
    for start_idx, end_idx in edges:
        start_node = nodes[start_idx]
        end_node = nodes[end_idx]
        
        fig.add_trace(go.Scatter(
            x=[start_node['x'], end_node['x']],
            y=[start_node['y'], end_node['y']],
            mode='lines',
            line=dict(color='gray', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Add nodes
    for i, node in enumerate(nodes):
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            mode='markers+text',
            marker=dict(
                size=25,
                color=phase_colors[node['phase']],
                line=dict(width=2, color='white')
            ),
            text=[node['label']],
            textposition="middle center",
            textfont=dict(size=9, color='white'),
            name=phase_names[node['phase']],
            hovertemplate=f"<b>{node['label']}</b><br>Phase: {phase_names[node['phase']]}<extra></extra>"
        ))
    
    # Add phase labels
    phase_y_positions = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5}
    for phase_num, phase_name in phase_names.items():
        fig.add_annotation(
            x=(phase_num-1)*2,
            y=phase_y_positions[phase_num],
            text=phase_name,
            showarrow=False,
            font=dict(size=12, color=phase_colors[phase_num], weight='bold'),
            bgcolor='white',
            bordercolor=phase_colors[phase_num],
            borderwidth=2,
            borderpad=4
        )
    
    # Update layout
    fig.update_layout(
        title="Complete Workflow Pipeline - Student Performance Prediction",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, 1]),
        width=1000,
        height=500,
        plot_bgcolor='white'
    )
    
    return fig


def show_data_overview(preprocessed_data, modeling_datasets, timeline):
    st.header("📊 Data Overview & Exploratory Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Dataset Info", "Missing Data Analysis", "Feature Distributions"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Summary")
            st.metric("Total Students", len(preprocessed_data))
            st.metric("Total Features", len(preprocessed_data.columns))
            st.metric("Research Questions", len(modeling_datasets))
        
        with col2:
            st.subheader("Research Questions Summary")
            for rq, dataset in modeling_datasets.items():
                st.write(f"**{rq}**: {dataset['data'].shape[0]} samples, {len(dataset['features'])} features")
        
        st.subheader("Sample Data (First 10 rows)")
        st.dataframe(preprocessed_data.head(10), use_container_width=True)
        
        # Show basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(preprocessed_data.describe(), use_container_width=True)
    
    with tab2:
        st.subheader("Missing Data Analysis")
        
        missing_data = preprocessed_data.isnull().sum()
        missing_percent = (missing_data / len(preprocessed_data)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percent': missing_percent.values
        }).sort_values('Missing_Percent', ascending=False)
        
        # Filter only columns with missing values
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df.head(20), x='Missing_Percent', y='Column', 
                        title='Top 20 Columns with Missing Data',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Missing Data Details")
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("🎉 No missing data found in the dataset!")
    
    with tab3:
        st.subheader("Feature Distributions")
        
        # Select features to visualize
        numeric_cols = preprocessed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_features = st.multiselect(
                "Select features to visualize:",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            
            if selected_features:
                # Create subplots
                n_cols = 2
                n_rows = (len(selected_features) + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=selected_features
                )
                
                for i, feature in enumerate(selected_features):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    
                    fig.add_trace(
                        go.Histogram(x=preprocessed_data[feature], name=feature, nbinsx=20),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    height=300 * n_rows, 
                    showlegend=False, 
                    title_text="Feature Distributions",
                    margin=dict(t=100)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found for visualization.")

def show_model_training(preprocessed_data, modeling_datasets, timeline):
    st.header("🤖 Model Training & Evaluation")
    
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    
    st.info("""
    This section trains multiple regression models for each research question:
    - **RQ1**: Predict Midterm I scores using early assignments and quizzes
    - **RQ2**: Predict Midterm II scores using Midterm I + intermediate assessments  
    - **RQ3**: Predict Final Exam scores using all previous assessments
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🚀 Train All Models", type="primary", use_container_width=True):
            with st.spinner("Training models... This may take a few moments."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Train models
                    results = train_all_models(modeling_datasets)
                    st.session_state.training_results = results
                    
                    # Simulate progress for better UX
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                    
                    progress_bar.empty()
                    status_text.success("✅ All models trained successfully!")
                    
                except Exception as e:
                    status_text.error(f"❌ Error training models: {e}")
    
    with col2:
        if st.session_state.training_results:
            st.success("Models are trained and ready for analysis!")
        else:
            st.info("Click the button to train models")
    
    if st.session_state.training_results:
        results = st.session_state.training_results
        all_results, baseline_results, bootstrap_results = results
        
        # Model comparison for each RQ
        st.subheader("Model Performance Comparison")
        
        rq_selection = st.selectbox(
            "Select Research Question:",
            list(all_results.keys())
        )
        
        if rq_selection:
            display_model_comparison(all_results[rq_selection], rq_selection)
            
            # Feature importance
            st.subheader("🔍 Feature Importance")
            display_feature_importance(all_results[rq_selection], modeling_datasets[rq_selection])

def display_model_comparison(results, rq_name):
    # Create comparison table
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'MAE': f"{metrics['MAE']:.3f}",
            'RMSE': f"{metrics['RMSE']:.3f}",
            'R²': f"{metrics['R2']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display as metrics
    st.write(f"**Performance Metrics for {rq_name}**")
    
    # Find best model (excluding dummies)
    non_dummy_models = {k: v for k, v in results.items() if not k.startswith('dummy')}
    if non_dummy_models:
        best_model_name = min(non_dummy_models.keys(), key=lambda x: non_dummy_models[x]['MAE'])
        best_metrics = results[best_model_name]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", best_model_name)
        with col2:
            st.metric("MAE", f"{best_metrics['MAE']:.3f}")
        with col3:
            st.metric("RMSE", f"{best_metrics['RMSE']:.3f}")
        with col4:
            st.metric("R²", f"{best_metrics['R2']:.3f}")
    
    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualization
    if non_dummy_models:
        models = [item['Model'] for item in comparison_data if not item['Model'].startswith('dummy')]
        maes = [float(item['MAE']) for item in comparison_data if not item['Model'].startswith('dummy')]
        r2s = [float(item['R²']) for item in comparison_data if not item['Model'].startswith('dummy')]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='MAE', x=models, y=maes, marker_color='lightcoral'))
        fig.add_trace(go.Bar(name='R²', x=models, y=r2s, marker_color='lightgreen', yaxis='y2'))
        
        fig.update_layout(
            title=f"Model Comparison for {rq_name}",
            xaxis_title="Models",
            yaxis_title="MAE",
            yaxis2=dict(title="R²", overlaying='y', side='right'),
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_feature_importance(results, dataset_info):
    # Find the best linear model for interpretation
    linear_models = {k: v for k, v in results.items() if 'linear' in k and not k.startswith('dummy')}
    
    if linear_models and 'multiple_linear' in linear_models:
        model_info = results['multiple_linear']
        model = model_info['model']
        features = dataset_info['features']
        coefficients = model.coef_
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients,
            'Abs_Importance': np.abs(coefficients)
        }).sort_values('Abs_Importance', ascending=False).head(10)
        
        fig = px.bar(importance_df, x='Abs_Importance', y='Feature', 
                    orientation='h', title='Top 10 Feature Importance (Multiple Linear Regression)')
        st.plotly_chart(fig, use_container_width=True)

def show_results_analysis():
    st.header("📈 Detailed Results & Analysis")
    
    if 'training_results' not in st.session_state:
        st.warning("Please train models first in the 'Model Training' section.")
        return
    
    results = st.session_state.training_results
    all_results, baseline_results, bootstrap_results = results
    
    tab1, tab2 = st.tabs(["Bootstrap Analysis", "Model Interpretation"])
    
    with tab1:
        st.subheader("📊 Bootstrap Confidence Intervals")
        
        if not bootstrap_results:
            st.info("No bootstrap results available. Please train models first.")
            return
            
        for rq_name in bootstrap_results.keys():
            st.write(f"**{rq_name}**")
            
            for model_name, bs_data in bootstrap_results[rq_name].items():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean MAE", f"{bs_data['mean_MAE']:.3f}")
                with col2:
                    st.metric("95% CI Lower", f"{bs_data['CI_lower']:.3f}")
                with col3:
                    st.metric("95% CI Upper", f"{bs_data['CI_upper']:.3f}")
                
                # Bootstrap distribution plot
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=bs_data['all_MAE_scores'], nbinsx=20, 
                                         name='Bootstrap MAE Distribution',
                                         marker_color='lightblue'))
                fig.add_vline(x=bs_data['CI_lower'], line_dash="dash", line_color="red", 
                            annotation_text=f"Lower CI: {bs_data['CI_lower']:.3f}")
                fig.add_vline(x=bs_data['CI_upper'], line_dash="dash", line_color="red",
                            annotation_text=f"Upper CI: {bs_data['CI_upper']:.3f}")
                fig.add_vline(x=bs_data['mean_MAE'], line_color="green",
                            annotation_text=f"Mean: {bs_data['mean_MAE']:.3f}")
                
                fig.update_layout(
                    title=f"Bootstrap MAE Distribution - {rq_name} ({model_name})",
                    xaxis_title="MAE", 
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🤔 Model Interpretation")
        st.info("Feature importance and model coefficients are shown in the Model Training section for each research question.")

def show_final_report():
    st.header("📋 Final Report & Recommendations")
    
    if 'training_results' not in st.session_state:
        st.warning("Please train models first to generate the final report.")
        return
    
    results = st.session_state.training_results
    all_results, baseline_results, bootstrap_results = results
    
    # Executive Summary
    st.subheader("📊 Executive Summary")
    
    summary_data = []
    for rq_name in all_results.keys():
        # Find best model (excluding dummies)
        non_dummy_models = {k: v for k, v in all_results[rq_name].items() if not k.startswith('dummy')}
        if non_dummy_models:
            best_model = min(non_dummy_models.keys(), key=lambda x: non_dummy_models[x]['MAE'])
            best_metrics = all_results[rq_name][best_model]
            
            bootstrap_ci = "N/A"
            if rq_name in bootstrap_results and best_model in bootstrap_results[rq_name]:
                bs_data = bootstrap_results[rq_name][best_model]
                bootstrap_ci = f"[{bs_data['CI_lower']:.3f}, {bs_data['CI_upper']:.3f}]"
            
            summary_data.append({
                'Research Question': rq_name,
                'Best Model': best_model,
                'MAE': f"{best_metrics['MAE']:.3f}",
                'R²': f"{best_metrics['R2']:.3f}",
                '95% CI MAE': bootstrap_ci
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Key Insights
        st.subheader("🔑 Key Insights")
        
        best_rq = min(summary_data, key=lambda x: float(x['MAE']))
        worst_rq = max(summary_data, key=lambda x: float(x['MAE']))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Most Predictable", best_rq['Research Question'], 
                     f"MAE: {best_rq['MAE']}")
        with col2:
            st.metric("Least Predictable", worst_rq['Research Question'],
                     f"MAE: {worst_rq['MAE']}")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = """
    1. **Use Best Performing Models**: Implement the identified best models for each prediction task
    2. **Consider Confidence Intervals**: Use bootstrap confidence intervals for uncertainty quantification
    3. **Monitor Performance**: Regularly validate models on new data
    4. **Feature Engineering**: Continue to develop meaningful features based on domain knowledge
    5. **Model Updates**: Retrain models periodically with new student data
    """
    st.markdown(recommendations)
    
    # Limitations
    st.subheader("⚠️ Limitations & Caveats")
    
    limitations = """
    - **Historical Data**: Models are trained on past data and should be validated on new semesters
    - **Data Imputation**: Missing values were handled via median imputation which may introduce bias
    - **Temporal Assumptions**: Assumes consistent assessment patterns across different time periods
    - **Feature Availability**: Model performance depends on available assessment data
    - **Domain Knowledge**: Interpretation requires understanding of educational assessment practices
    """
    st.markdown(limitations)

if __name__ == "__main__":
    main()
