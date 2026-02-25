import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Tourism Analytics", layout="wide")

# 2. Load Data and ML Assets
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_tourism_data.csv')

@st.cache_resource
def load_models():
    with open('visit_mode_model.pkl', 'rb') as f:
        m_cls = pickle.load(f)
    with open('rating_model.pkl', 'rb') as f:
        m_reg = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        enc = pickle.load(f)
    with open('eng_features.pkl', 'rb') as f:
        eng = pickle.load(f)
    return m_cls, m_reg, enc, eng

df = load_data()
model_cls, model_reg, encoders, engineering = load_models()

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predictions & Recommendations"])

if page == "Dashboard":
    st.title("Tourism Experience Analytics Dashboard")
    st.markdown("### Deep-Dive into Trends, Behaviors, and Hotspots")

    # --- ROW 1: Geographic Distribution ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tourist Origin (Continents)")
        # Visualize user distribution across continents [cite: 69]
        fig_pie = px.pie(df, names='Continent', hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Top Regions by Traffic")
        # Identify top regions based on transaction volume [cite: 89, 92]
        region_counts = df['Region'].value_counts().nlargest(10).reset_index()
        fig_region = px.bar(region_counts, x='count', y='Region', orientation='h',
                            color='count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_region, use_container_width=True)

    st.divider()

    # --- ROW 2: Attraction Analysis ---
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Attraction Popularity & Ratings")
        # Explore attraction types and their popularity [cite: 70]
        attr_stats = df.groupby('AttractionType').agg({'Rating': 'mean', 'UserId': 'count'}).reset_index()
        attr_stats.columns = ['AttractionType', 'Average Rating', 'Visit Count']
        fig_scatter = px.scatter(attr_stats, x='Visit Count', y='Average Rating', 
                                 size='Visit Count', color='AttractionType', 
                                 hover_name='AttractionType', title="Popularity vs. Satisfaction")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col4:
        st.subheader("Monthly Travel Trends")
        # Standardize and analyze date/time formats for seasonal patterns [cite: 60, 92]
        month_trends = df.groupby('VisitMonth').size().reset_index(name='Visits')
        fig_line = px.line(month_trends, x='VisitMonth', y='Visits', markers=True,
                           title="Seasonality: Total Visits per Month")
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # --- ROW 3: Behavioral Insights ---
    st.subheader("Correlation: Continent vs. Visit Mode")
    # Investigate correlation between VisitMode and demographics [cite: 71]
    behavior_matrix = df.groupby(['Continent', 'VisitMode']).size().reset_index(name='Counts')
    fig_heatmap = px.density_heatmap(behavior_matrix, x='Continent', y='VisitMode', z='Counts',
                                     color_continuous_scale='Blues', text_auto=True,
                                     title="Heatmap of Travel Behaviors per Continent")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- ROW 4: Rating Distribution ---
    st.subheader("Rating Distribution across All Transactions")
    # Analyze distribution of ratings to identify quality benchmarks [cite: 72]
    fig_hist = px.histogram(df, x='Rating', nbins=5, color='VisitMode',
                            title="How Different Groups Rate Their Experience")
    st.plotly_chart(fig_hist, use_container_width=True)

elif page == "Predictions & Recommendations":
    st.title("Personalized Tourism Insights")
    
    tab1, tab2 = st.tabs(["Visit Mode & Rating Prediction", "Attraction Recommendations"])
    
    with tab1:
        st.subheader("Predict Your Experience")
        # User Inputs [cite: 19, 20, 32]
        c1, c2 = st.columns(2)
        with c1:
            u_cont = st.selectbox("Continent", encoders['Continent'].classes_)
            u_country = st.selectbox("Country", encoders['Country'].classes_)
            u_region = st.selectbox("Region", encoders['Region'].classes_)
        with c2:
            u_type = st.selectbox("Attraction Type", encoders['AttractionType'].classes_)
            u_month = st.slider("Month of Visit", 1, 12, 6)

        if st.button("Generate Predictions"):
            # Prepare inputs for Classification [cite: 77, 93]
            # We use 1 as default for 'User_Travel_Freq' for new users
            input_cls = [[
                encoders['Continent'].transform([u_cont])[0],
                encoders['Country'].transform([u_country])[0],
                encoders['Region'].transform([u_region])[0],
                encoders['AttractionType'].transform([u_type])[0],
                u_month,
                1 
            ]]
            res_cls = model_cls.predict(input_cls)
            mode_name = encoders['VisitMode'].inverse_transform(res_cls)[0]
            
            # Prepare inputs for Regression [cite: 75, 87]
            # Use the engineering dict to get the historical average for that type
            avg_type_score = engineering['type_means'].get(u_type, df['Rating'].mean())
            input_reg = [[
                encoders['Continent'].transform([u_cont])[0],
                encoders['Country'].transform([u_country])[0],
                res_cls[0], # Predicted VisitMode
                encoders['AttractionType'].transform([u_type])[0],
                u_month,
                avg_type_score
            ]]
            res_reg = model_reg.predict(input_reg)[0]
            
            st.success(f"Predicted Visit Mode: **{mode_name}**")
            st.info(f"Estimated Rating: **{res_reg:.2f} / 5.0**")

    with tab2:
        st.subheader("Recommended for You")
        # Content-Based Filtering [cite: 46, 79]
        rec_type = st.selectbox("What do you want to see?", df['AttractionType'].unique())
        if st.button("Get Recommendations"):
            # Filter and rank by rating [cite: 54, 55, 88]
            recs = df[df['AttractionType'] == rec_type].groupby('Attraction').agg({
                'Rating': 'mean',
                'AttractionAddress': 'first'
            }).sort_values(by='Rating', ascending=False).head(5)
            
            for i, (name, row) in enumerate(recs.iterrows()):
                st.write(f"**{i+1}. {name}** (Avg Rating: {row['Rating']:.1f})")
                st.caption(f"📍 {row['AttractionAddress']}")