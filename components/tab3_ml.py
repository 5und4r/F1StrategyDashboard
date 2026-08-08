import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

@st.cache_resource
def train_poly_model(X_train, y_train, degree):
    """Caches the polynomial transformation and linear regression fit."""
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    return poly, model

@st.cache_resource
def train_tree_model(X_train, y_train, max_depth):
    """Caches the decision tree training process."""
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
    tree_clf.fit(X_train, y_train)
    return tree_clf

def render_tab3(df):
    st.header("Machine Learning Models")
    st.markdown("Applying predictive algorithms to reverse-engineer racing strategies.")
    
    # Check for required columns
    required_cols = ['TyreLife', 'LapTime (s)', 'Driver', 'Cumulative_Degradation']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns for ML. Need: {required_cols}")
        return

    # ==========================================
    # TASK 1: MULTIPLE POLYNOMIAL REGRESSION
    # ==========================================
    st.subheader("Model 1: Predicting the 'Tyre Cliff' (Polynomial Regression)")
    st.markdown("Tyres don't degrade linearly. We use Polynomial Regression to capture the sudden drop-off in performance as tyres age.")
    
    # --- ADDED EXPLANATION FOR POLYNOMIAL FLEXIBILITY ---
    with st.expander("📖 What does 'Flexibility of the Curve' mean?"):
        st.markdown("""
        Formula 1 tyres do not lose performance in a straight line. They often hold steady grip for several laps and then suddenly experience a severe drop in performance, commonly referred to as **"falling off a cliff."**
        
        *   **Standard Linear Regression (Degree 1):** Draws a rigid, straight line. It completely misses this sudden drop-off.
        *   **Polynomial Regression (Degree 2, 3, or 4):** Allows the mathematical line to "bend" and curve. By increasing the degree (flexibility), the model can accurately wrap around the data points and predict exactly when the lap times will suddenly spike due to tyre wear.
        """)

    # Let user select polynomial degree
    poly_degree = st.slider("Select Polynomial Degree (Flexibility of the curve)", min_value=1, max_value=4, value=2)
    
    # We sample the data for real-time slider responsiveness
    sample_df = df.sample(n=min(20000, len(df)), random_state=42)
    
    # Prepare Data
    X_reg = sample_df[['TyreLife']]
    y_reg = sample_df['LapTime (s)']
    
    # Fetch the cached model
    poly, poly_model = train_poly_model(X_reg, y_reg, poly_degree)
    
    # Generate smooth predictions for plotting
    x_range = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    y_pred_smooth = poly_model.predict(x_range_poly)
    
    # Plotting actual data vs Predicted Curve using Plotly Graph Objects
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=sample_df['TyreLife'], y=sample_df['LapTime (s)'], mode='markers', name='Actual Laps', marker=dict(color='gray', opacity=0.5)))
    fig_reg.add_trace(go.Scatter(x=x_range.flatten(), y=y_pred_smooth, mode='lines', name=f'Degree {poly_degree} Prediction', line=dict(color='red', width=3)))
    fig_reg.update_layout(title="Tyre Life vs. Lap Time Degradation", xaxis_title="Tyre Life (Laps)", yaxis_title="Lap Time (s)")
    st.plotly_chart(fig_reg, use_container_width=True)

    with st.expander("📈 Key Strategic Takeaway: Predicting the Tyre Cliff"):
        st.markdown(f"""
        *   **The Model in Action:** With a degree of **{poly_degree}**, this model maps the trajectory of lap times as the tyre gets older.
        *   **The Cliff:** Notice how the red prediction line behaves towards the right side of the graph. If it suddenly curves upwards sharply, that is the exact mathematical representation of the "tyre cliff"—the point where performance drops exponentially and a pit stop is absolutely mandatory to prevent losing track position.
        """)

    st.divider()

    # ==========================================
    # TASK 2: K-MEANS CLUSTERING
    # ==========================================
    st.subheader("Model 2: Driver Pace vs. Consistency (K-Means Clustering)")
    st.markdown("Unsupervised learning groups drivers by their behavioral profiles: Are they fast but erratic, or slower but highly consistent?")
    
    # Data Preparation: Aggregate by Driver
    driver_stats = df.groupby('Driver').agg(
        Mean_Pace=('LapTime (s)', 'mean'),
        Pace_Variance=('LapTime (s)', 'std')  # High standard deviation = inconsistent
    ).dropna().reset_index()
    
    num_clusters = st.slider("Select Number of Clusters (K)", min_value=2, max_value=5, value=3)
    
    # Train K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    driver_stats['Cluster'] = kmeans.fit_predict(driver_stats[['Mean_Pace', 'Pace_Variance']])
    driver_stats['Cluster'] = driver_stats['Cluster'].astype(str) # For categorical coloring
    
    # Plot Clusters
    fig_cluster = px.scatter(driver_stats, x='Mean_Pace', y='Pace_Variance', color='Cluster', 
                             text='Driver', title="Driver Profiling Clusters",
                             labels={'Mean_Pace': 'Average Lap Time (Lower is Faster)', 'Pace_Variance': 'Lap Time Variance (Lower is more Consistent)'})
    fig_cluster.update_traces(textposition='top center', marker=dict(size=12))
    st.plotly_chart(fig_cluster, use_container_width=True)

    with st.expander("🎯 Key Strategic Takeaway: Driver Profiling"):
        st.markdown("""
        *   **Bottom Left (The Elite):** Drivers with low average lap times and low variance. These are the front-runners who set fast laps consistently without making mistakes.
        *   **Bottom Right (The Midfield Rocks):** Drivers who might not have the outright pace (higher average lap time) but boast very low variance, executing their race strategies perfectly.
        *   **Top Half (The Erratic):** Drivers with high variance. This could indicate aggressive driving styles, inconsistent machinery, or having been involved in multiple track incidents that disrupted their pace.
        """)

    st.divider()

    # ==========================================
    # TASK 3: DECISION TREE CLASSIFIER
    # ==========================================
    st.subheader("Model 3: Pit Stop Window Prediction (Decision Tree Classifier)")
    st.markdown("We simulate a 'Pit Window' flag when Cumulative Degradation exceeds a critical threshold, and train a Decision Tree to learn these rules.")
    
    # --- REBUILT DECISION TREE TO USE THE REAL PITNEXTLAP TARGET ---
    if 'PitNextLap' in df.columns:
        st.markdown("We are training this model to look into the future: predicting `PitNextLap` using leading indicators rather than just reacting to a slow current lap time.")
        
        # Features and Target - Using predictive indicators
        features_to_use = ['Normalized_TyreLife', 'Cumulative_Degradation', 'LapTime_Delta', 'RaceProgress', 'Position']
        
        # Ensure all requested features exist in the dataframe before training
        available_features = [f for f in features_to_use if f in df.columns]
        
        # Sample for speed optimization
        sample_df_clf = df.dropna(subset=available_features + ['PitNextLap']).sample(n=min(20000, len(df)), random_state=42)
        
        X_clf = sample_df_clf[available_features]
        y_clf = sample_df_clf['PitNextLap']
        
        # Interactive Depth Slider
        tree_depth = st.slider("Select Maximum Tree Depth (Complexity)", min_value=2, max_value=8, value=4)
        
        # Train Decision Tree using cached function
        tree_clf = train_tree_model(X_clf, y_clf, tree_depth)
        
        # Extract Feature Importances to explain the model's logic
        importances = pd.DataFrame({
            'Feature': X_clf.columns,
            'Importance': tree_clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Plot Feature Importances
        fig_tree = px.bar(importances, x='Importance', y='Feature', orientation='h',
                          title="Decision Tree Feature Importances (What predicts a pit stop on the NEXT lap?)",
                          color='Importance', color_continuous_scale='viridis')
        st.plotly_chart(fig_tree, use_container_width=True)
        
        with st.expander("🌳 Key Strategic Takeaway: Inside the Decision Engine", expanded=True):
            st.markdown(f"""
            *   **Top Predictive Feature:** The bar chart above shows the most important variables the model uses to forecast a pit stop. If `{importances.iloc[0]['Feature']}` is at the top, it means this is the strongest leading indicator that a driver is about to pit.
            *   **Model Complexity (Depth {tree_depth}):** By adjusting the tree depth, you allow the model to find more intricate combinations of rules. However, keeping it relatively shallow prevents overfitting and allows humans to easily read the strategic triggers.
            *   **Forecasting vs Reacting:** Because this model predicts `PitNextLap` instead of the current lap, it is learning the *setup* to a pit stop (e.g., increasing degradation and high normalized tyre life) rather than just reacting to the 25-second penalty of the pit stop itself.
            """)
        
        st.success("By feeding the tree leading indicators like 'Normalized Tyre Life' and 'Cumulative Degradation', the model acts as a true forecasting tool rather than just identifying laps that are already slow.")
    else:
        st.error("The 'PitNextLap' target variable was not found in the dataset. Please ensure you are using the updated F1 Strategy dataset.")