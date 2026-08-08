import streamlit as st
import pandas as pd
import os
from components.tab1_intro import render_tab1
from components.tab2_eda import render_tab2
from components.tab3_ml import render_tab3

st.set_page_config(
    page_title="F1 Strategy & Telemetry Analytics",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """
    Loads the F1 dataset. 
    @st.cache_data ensures the dataset is loaded only once into memory,
    making the interactive web app incredibly fast.
    """
    file_path = os.path.join('data', 'f1_strategy_dataset_v6.csv')
    
    # Handle the case where the file might not be exactly where expected yet
    if not os.path.exists(file_path):
        st.error(f"⚠️ Dataset not found at: `{file_path}`. Please ensure your CSV is named 'f1_dataset.csv' and placed inside the 'data' folder.")
        # Return a tiny dummy dataframe so the app doesn't crash completely while you fix the path
        return pd.DataFrame({'LapNumber': [1,2,3], 'TyreLife': [1,2,3], 'LapTime (s)': [90,91,92], 'Compound': ['Soft','Medium','Hard'], 'Position': [1,2,3], 'Cumulative_Degradation': [0.1,0.2,0.3], 'Driver': ['VER','HAM','LEC']})
        
    df = pd.read_csv(file_path)
    
    # ---------------------------------------------------------
    # DATA IMPUTATION: Handling the 66 missing 'Compound' values
    # We use forward-fill (ffill) because F1 data is sequential.
    # If a driver was on Softs on Lap 10 and Lap 12, they were on Softs on Lap 11.
    # ---------------------------------------------------------
    df['Compound'] = df['Compound'].ffill()
    
    # Quick standard cleaning just in case there are other random NaNs
    df.dropna(inplace=True)
    return df

def main():
    # App Header
    st.title("Formula 1 EDA & ML Operations Dashboard")
    st.markdown("Explore race strategies, visualize tyre degradation, and predict tactical pit windows using Machine Learning.")
    
    # Load data
    df = load_data()
    
    # Create the Tabs
    tab1, tab2, tab3 = st.tabs([
        " 1. Dataset & Overview", 
        " 2. Interactive EDA", 
        " 3. Machine Learning Models"
    ])
    
    # Render the contents of each tab from our modular files
    with tab1:
        render_tab1(df)
        
    with tab2:
        render_tab2(df)
        
    with tab3:
        render_tab3(df)

if __name__ == "__main__":
    main()