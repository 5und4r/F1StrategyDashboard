import streamlit as st
from PIL import Image
import os

def render_tab1(df):
    st.header("Dataset Overview: The Anatomy of F1 Strategy")
    
    # Attempt to load a cool F1 image if you placed one in the assets folder
    img_path = os.path.join('assets', 'f1_car.jpg')
    if os.path.exists(img_path):
        image = Image.open(img_path)
        st.image(image, caption="The Intersection of Engineering and Data", use_container_width=True)
    else:
        st.info("💡 Tip: Add an image named `f1_car.jpg` to your `assets/` folder to display a cool banner here!")
    
    # --- UPDATED KAGGLE DATASET OVERVIEW ---
    st.markdown("""
    ### F1 Strategy Dataset: Pit Stop Prediction
    **Overview:**
    The F1 Strategy Dataset: Pit Stop Prediction is a lap-by-lap Formula 1 race dataset designed for machine learning, predictive analytics, and motorsport strategy research. It captures driver performance, tyre behavior, race progression, and engineered features that influence pit stop decisions.
    
    The dataset is intended for building classification models that predict whether a driver will make a pit stop on the next lap, while also supporting exploratory data analysis, feature engineering, and race strategy optimization.
    """)
    
    st.divider()
    
    # Dataset Statistics using columns for visual appeal
    st.subheader("Dataset Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", "101,371")
    col2.metric("Columns", "16")
    col3.metric("Drivers", "31")
    col4.metric("Races", "28")
    col5.metric("Seasons", "2025")
    
    st.markdown("**Target Variable:** `PitNextLap`")
    st.markdown("The dataset contains lap-level observations, where each row represents a driver's state during a specific lap.")
    
    st.divider()
    
    # Features Table
    st.subheader("Features Description")
    st.markdown("""
    | Feature | Description |
    | :--- | :--- |
    | **Driver** | Driver identifier |
    | **LapNumber** | Current lap number |
    | **Compound** | Tyre compound (Soft, Medium, Hard, etc.) |
    | **Stint** | Current tyre stint |
    | **TyreLife** | Number of laps completed on the current tyre set |
    | **Position** | Driver's race position |
    | **LapTime (s)** | Lap completion time in seconds |
    | **Race** | Grand Prix name |
    | **Year** | Race season |
    | **LapTime_Delta** | Difference in lap time compared to the previous lap |
    | **Cumulative_Degradation** | Estimated cumulative tyre performance degradation |
    | **PitStop** | Indicates whether a pit stop occurred on the current lap |
    | **PitNextLap** | Target variable indicating whether the driver pits on the next lap |
    | **RaceProgress** | Percentage of race completed |
    | **Normalized_TyreLife** | Tyre life normalized between 0 and 1 |
    | **Position_Change** | Position gained or lost compared to the previous lap |
    """)
    
    st.divider()
    
    col_ml, col_data = st.columns(2)
    
    with col_ml:
        st.subheader("Machine Learning Applications")
        st.markdown("""
        This dataset can be used for:
        * Binary Classification
        * Pit Stop Prediction
        * Race Strategy Optimization
        * Driver Performance Analysis
        * Feature Engineering
        * Time-Series Feature Creation
        * Explainable AI (SHAP/LIME)
        
        **Suitable algorithms include:** Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, and Neural Networks.
        """)
        
    with col_data:
        st.subheader("Data Quality & Engineering")
        st.markdown("""
        **Feature Engineering Included:**
        The dataset contains engineered features that capture race dynamics highly relevant for pit stop prediction: Lap Time Delta, Race Progress, Normalized Tyre Life, and Cumulative Tyre Degradation.
        
        **Data Quality:**
        * Total Records: 101,371
        * Missing Values: Only 66 missing values in the `Compound` column (Handled via Time-Series Forward-Fill).
        * All numerical features are complete.
        """)
        
        st.info("**Target Variable (`PitNextLap`):** \n\n 0 → Driver does not pit on the next lap. \n\n 1 → Driver will pit on the next lap.")

    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Telemetry Data (Head)")
        st.dataframe(df.head(15), use_container_width=True)
        
    with col2:
        st.subheader("Statistical Summary")
        # Displaying describe() helps instantly spot scale differences (perfect for normalization references)
        st.dataframe(df.describe(), use_container_width=True)