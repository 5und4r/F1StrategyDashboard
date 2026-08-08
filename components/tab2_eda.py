import streamlit as st
import plotly.express as px
import pandas as pd

def render_tab2(df):
    st.header("Exploratory Data Analysis: Deep Dive")
    st.markdown("Uncovering the hidden relationships between strategy, speed, and track position.")
    
    st.subheader("1. Multidimensional Correlation Heatmap")
    st.markdown("Identify which numerical variables influence each other the most (e.g., TyreLife vs LapTime).")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                         color_continuous_scale='RdBu_r', 
                         title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("📊 Key Strategic Takeaway: Correlation Heatmap", expanded=False):
        st.markdown("""
        *   **Strong Positive Correlations (Red):** Variables like `TyreLife` and `Cumulative_Degradation` have a high positive correlation, meaning as the tyre gets older, degradation directly increases.
        *   **Negative Correlations (Blue):** Look for variables that inversely affect each other. For example, higher `RaceProgress` usually correlates with lower `LapTime (s)` as the car burns fuel and becomes lighter.
        """)
    
    st.divider()
    
    st.subheader("2. Dynamic Race Pace Analysis")
    st.markdown("Analyze how lap times fluctuate as the race progresses.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Let user choose what to color the scatter plot by
        color_option = st.selectbox("Group Pace By:", ['Compound', 'Driver'])
        
    with col2:
        # Plotly scatter plot with trendlines
        fig_pace = px.scatter(df, x="LapNumber", y="LapTime (s)", color=color_option,
                              opacity=0.7, hover_data=["Position", "TyreLife"],
                              title=f"Lap Time Evolution Grouped by {color_option}")
        
        # Reverse Y axis if needed (lower lap time is better, though standard plot is fine too)
        st.plotly_chart(fig_pace, use_container_width=True)

    with st.expander("⏱️ Key Strategic Takeaway: Race Pace Evolution"):
        st.markdown("""
        *   **Tyre Degradation Slopes:** Notice how the lap times gradually curve upwards as the stint goes on. The steeper the slope, the worse the tyre degradation.
        *   **Compound Differences:** If you color by 'Compound', you'll likely observe that softer tyres start fast (low on the Y-axis) but climb quickly, while harder tyres start slower but have a much flatter, consistent line.
        """)
        
    st.divider()
    
    st.subheader("3. Positional Battles (Bump Chart)")
    
    # --- ADDED EXPLANATION FOR THE BUMP CHART ---
    with st.expander("📖 How to read this chart", expanded=True):
        st.markdown("""
        **Understanding the Spaghetti:**
        *   The **Y-Axis (Vertical)** represents the driver's current race position (1st place is at the top).
        *   The **X-Axis (Horizontal)** is the progression of the race via Lap Number.
        *   **Crossing Lines:** Every time two lines intersect, a position change has occurred. This could be due to a spectacular on-track overtake, or a strategic pit stop where a driver temporarily drops down the order to get fresh tyres.
        """)
    
    # Check if Driver column exists for this plot
    if 'Driver' in df.columns and 'Position' in df.columns:
        fig_pos = px.line(df, x="LapNumber", y="Position", color="Driver", markers=True,
                          title="Driver Track Position over Time")
        
        # In F1, Position 1 is at the top, so we reverse the Y-axis!
        fig_pos.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_pos, use_container_width=True)

        with st.expander("🏎️ Key Strategic Takeaway: Reading the Overcuts and Undercuts"):
            st.markdown("""
            *   **The Undercut:** Look for instances where a line dips sharply (a pit stop dropping them down the order), but then rapidly climbs past intersecting lines within 2-3 laps. That is a successful undercut strategy!
            *   **Traffic Management:** Sometimes drivers get stuck behind a slower car. You'll see their lines perfectly parallel, highlighting how track position can completely dictate race pace.
            """)


    else:
        st.warning("Position or Driver data missing from dataset to generate Bump Chart.")