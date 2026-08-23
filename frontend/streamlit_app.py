# app.py
# =========================================================
# BMW Price Prediction Dashboard (Professional UI Template)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pickle
from pathlib import Path
import requests


API_URL = "http://backend:8000/predict"
#load bmw.csv data 
BASE_DIR = Path(__file__).resolve().parent.parent

df = BASE_DIR / "data" / "bmw.csv"

df = pd.read_csv(df)



# Load Artifact
BASE_DIR = Path(__file__).resolve().parent.parent


RESULT_PATH = BASE_DIR / "artifacts" / "model_results.csv"

#load model performances table 
results_df = pd.read_csv(
    RESULT_PATH,
    index_col="Rank"
)



with open(BASE_DIR / "model" / "bmw_model.pkl", "rb") as file:
    artifact = pickle.load(file)

model = artifact["model"]
columns = artifact["columns"]
encoder = artifact["encoder"]
scaler = artifact["scaler"]


# =========================================================
# PAGE CONFIG
# ========================================================

st.set_page_config(
    page_title="BMW Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg,#1f2937,#111827);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #374151;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

/* Main Title */
.main-title {
    font-size: 45px;
    font-weight: 700;
    color: white;
}

/* Subtitle */
.subtitle {
    color: #9ca3af;
    font-size: 18px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1f2937;
    border-radius: 10px;

    padding: 12px 22px;
    color: white;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
}

/* Prediction Box */
.prediction-box {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    color: white;
    text-align: center;
}

/* Footer */
.footer {
    text-align:center;
    color:gray;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚘 BMW Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Business Insights",
        "📊 Plots",
        "📈 Model Performance",
        "🧠 About Model",
        "👨‍💻 Developer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
Project Features:
- BMW Price Prediction
- Business Insights
- Model Evaluation
- Interactive Dashboard
""")

# =========================================================
# SAMPLE DATA
# =========================================================

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    col1, col2 = st.columns([3,1])

    with col1:
        st.markdown("""
        <div class="main-title">
        BMW Price Prediction Tool
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="subtitle">
        AI-powered machine learning application for predicting BMW resale prices.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg",
            width=130
        )

    st.markdown("---")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Dataset Size", "10K+")

    with c2:
        st.metric("Best Model R²", "0.94")

    with c3:
        st.metric("Features", "15")

    with c4:
        st.metric("Models Tested", "95%")

    st.markdown("---")

    # Charts
    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            df,
            x="price",
            nbins=30,
            title="BMW Price Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig2 = px.scatter(
            df,
            x="mileage",
            y="price",
            color="year",
            title="Mileage vs Price"
        )

        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    <div class="insight-box">

    ## 📌 Suggested Business Insights Section

    ### 1. Customer Behavior
    - Which BMW models attract buyers?
    - Mileage tolerance among customers.
    - Fuel efficiency demand trend.

    ### 2. Resale Market Insights
    - Best resale years.
    - Depreciation analysis.
    - Premium feature effect on price.

    ### 3. Market Opportunities
    - Affordable luxury segment.
    - High demand engine categories.
    - Emerging efficient vehicle segment.

    ### 4. Data Science Insights
    - Most influential features.
    - Correlation-based findings.
    - Predictive model implications.

    ### 5. Recommendations
    - Inventory optimization.
    - Price optimization strategy.
    - Customer targeting strategy.

    </div>
    """, unsafe_allow_html=True)
# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Prediction":

    st.title("🔮 BMW Price Prediction")

    st.markdown("Enter car details below:")
    car_model = st.selectbox(
    "Car Model",
    [' 5 Series', ' 6 Series', ' 1 Series', ' 7 Series',
     ' 2 Series', ' 4 Series', ' X3', ' 3 Series',
     ' X5', ' X4', ' i3', ' X1', ' M4',
     ' X2', ' X6', ' 8 Series', ' Z4',
     ' X7', ' M5', ' i8', ' M2',
     ' M3', ' M6', ' Z3']
    )

    transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic", "Semi-Auto"]
        )
    

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.slider("Year", 1990, 2025, 2024)
        mileage = st.number_input("Mileage", 1,214000,2000)

    with col2:
        mpg = st.number_input(
            'MPG',
            min_value=5.5,
            max_value=470.8
        )
        engineSize = st.number_input(
            "Engine Size",
            min_value=0.0,
            max_value=6.6
        )

    with col3:
        tax = st.slider("Tax", 0,580,60)

        fuel = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "Hybrid","Other"]
        )

    st.markdown("")

    if st.button("🚀 Predict Price", use_container_width=True):


        input_df = {
            "fuelType": fuel,
            "year": year,
            "model": car_model,
            "transmission": transmission,
            "mileage": mileage,
            "tax": tax,
            "engineSize":engineSize,
            "mpg": mpg
        }
        try:
            response = requests.post(API_URL,json=input_df)
            if response.status_code ==200:
                result = response.json()
                price = result["prediction"]
                st.markdown(f"""
                <div class="prediction-box">
                    <h1>Estimated Price</h1>
                    <h1>£ {price:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Error:{response.status_code} - {response.text}")
        except Exception as e:
            st.error(e)

# =========================================================
# BUSINESS INSIGHTS
# =========================================================

elif page == "📊 Business Insights":

    st.subheader("1. Fuel Type vs Transmission Analysis")

    st.markdown("""
    ### Key Insights
    For Dealerships\n
    Keep higher inventory of:\n
    Diesel Semi-Automatic\n
    Petrol Semi-Automatic\n

    These are clearly the highest demand segments.\n

    For Marketing Teams\n
    Promote:\n
    smooth driving experience\n
    luxury automatic transmission\n
    premium comfort\n

    BMW customers prefer convenience over manual driving.\n

    For EV Strategy
    Electric market is currently very small.
    Avoid overstocking EV inventory unless targeting urban premium buyers.
                    
                    
        
        """)
    st.subheader("Transmission Market Share Pie Chart")

    st.markdown("""
    Nearly 76% of customers prefer automatic driving experience.
    Manual transmission is becoming a niche segment.
    Strategic Decision
    Sales Strategy

    Focus advertisements on:

    luxury driving
    traffic convenience
    premium automatic experience
    Inventory Optimization

    Reduce manual inventory gradually.

        """)
    st.subheader("3. Tax vs Price Scatter Plot")

    st.markdown("""
    Customer Psychology

    Customers buying luxury BMWs are:

    less sensitive to tax
    more focused on performance, brand, and features.
    Pricing Strategy

    Tax alone should NOT determine resale pricing.

    Use:

    model
    engine size
    mileage
    year

    as primary pricing factors.

    Market Segmentation

    High tax luxury vehicles still maintain premium pricing.

    This indicates:

    strong luxury buyer segment
    premium brand loyalty.
    """)
    st.subheader("4. Average Price by BMW Model")

    st.markdown("""
    Premium Segment

    Models like:

    X7
    M Series
    i8

    should be marketed as:

    luxury performance vehicles
    premium executive class.
    High Profit Opportunity

    SUV and M-series likely provide:

    higher margins
    higher resale value
    premium customer segment.
    Entry-Level Strategy

    1 Series / 2 Series can be:

    gateway products
    used for first-time luxury buyers.
    """)

    st.subheader("Price Distribution by Model")
    st.markdown("""
    Resale Stability

    Models with tight box ranges:

    more stable resale market
    easier pricing strategy.
    Variant-Based Pricing

    Large spread models require:

    trim-level pricing
    feature-based valuation.
    Investment Vehicles

    M models retain premium resale value.
    Good for:

    premium inventory investment
    luxury targeting
    """)
elif page == "📊 Plots":

    st.title("📊 Plots")

    tab1, tab2, tab3,tab4 = st.tabs([
        "Market Trends",
        "Price Analysis",
        "Feature Impact",
        "correlation_analysis"
    ])

    # -----------------------------------------------------

    with tab1:
        
        fig1 = px.histogram(
            df,
            x="price",
            nbins=40,
            title="BMW Price Distribution",
            template="plotly_dark"
        )

        fig1.update_layout(height=600)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(
            df,
            y="price",
            title="Price Outlier Analysis",
            template="plotly_dark"
        )

        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

        year_price = df.groupby("year")["price"].mean().reset_index()

        fig3 = px.line(
            year_price,
            x="year",
            y="price",
            markers=True,
            title="Average BMW Price by Year",
            template="plotly_dark"
        )
        st.plotly_chart(fig3, use_container_width=True)

        model_avg = df.groupby("model")["price"].mean().sort_values()

        fig4 = px.bar(
            model_avg,
            title="Average Price by Model",
            template="plotly_dark"
        )
        st.plotly_chart(fig4, use_container_width=True)
    # -----------------------------------------------------

    with tab2:

        fig5 = px.box(
            df,
            x="year",
            y="price",
            title="Price Spread by Manufacturing Year"
        )

        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("")

        fig6 = px.histogram(
            df,
            x="fuelType",
            color="transmission",
            barmode="group",
            title="Fuel Type vs Transmission",
            template="plotly_dark"
        )

        fig6.update_layout(height=650)

        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("" \
        "### Key Insight: \n" \
        """Semi-Automatic transmission dominates the BMW market.\n
        Diesel + Semi-Auto has the highest count (~2908 vehicles).\n
        Petrol + Semi-Auto is also very strong (~1652 vehicles).\n
        Electric BMW presence is extremely low\n
        Only a few electric vehicles exist in the dataset.\n
        Indicates either:\n
        old market data\n
        or low EV adoption in premium segment.\n
        Hybrid market is small but emerging
        Mostly Automatic transmission.\n
        Manual transmission demand is shrinking
        Manual counts are much lower than Semi-Auto.""")

        trans = df["transmission"].value_counts()

        fig7 = px.pie(
            values=trans.values,
            names=trans.index,
            title="Transmission Market Share",
            template="plotly_dark"
        )
        st.plotly_chart(fig7, use_container_width=True)

        fig8 = px.bar(
            df["fuelType"].value_counts(),
            title="Fuel Type Market Distribution",
            template="plotly_dark"
        )
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("")

        fig9 = px.histogram(
            df,
            x="model",
            title="BMW Model Popularity",
            template="plotly_dark"
        )

        fig9.update_layout(height=650)
        st.plotly_chart(fig9, use_container_width=True)


        st.markdown("""
        ## 📌 Key Insights

        - SUVs dominate premium market
        - Semi-auto most preferred
        - Diesel has highest resale demand

        ## 💼 Business Recommendation

        ✅ Increase SUV inventory

        ✅ Focus premium automatic cars

        ✅ Reduce manual inventory

        """)
    # -----------------------------------------------------

    with tab3:
        

        fig11 = px.scatter(
            df,
            x="mileage",
            y="price",
            color="year",
            trendline="ols",
            title="Mileage Impact on Price",
            template="plotly_dark"
        )

        fig11.update_layout(height=650)
        st.plotly_chart(fig11, use_container_width=True)

        fig12 = px.scatter(
            df,
            x="engineSize",
            y="price",
            color="fuelType",
            size="tax",
            title="Engine Size Impact on Price",
            template="plotly_dark"
        )
        fig12.update_layout(height=650)
        st.plotly_chart(fig12, use_container_width=True)

        fig13 = px.scatter(
            df,
            x="mpg",
            y="price",
            color="fuelType",
            title="MPG vs Price",
            template="plotly_dark"
        )
        fig13.update_layout(height=650)
        st.plotly_chart(fig13, use_container_width=True)


        fig14 = px.scatter(
            df,
            x="tax",
            y="price",
            color="transmission",
            title="Tax Impact on Price",
            template="plotly_dark"
        )
        fig14.update_layout(height=650)
        st.plotly_chart(fig14, use_container_width=True)

        st.markdown("" \
        "### Key Insight: \n" \
        """No strong linear relationship between tax and price.\n
        No strong linear relationship between tax and price.\n
        High-priced BMWs exist across multiple tax ranges.\n
        Some luxury outliers:
          - ₹100k+ cars around tax 300.""")


        fig15 = px.box(
            df,
            x="model",
            y="price",
            title="Price Spread Across Models",
            template="plotly_dark"
        )

        fig15.update_layout(height=700)

        st.plotly_chart(
            fig15,
            use_container_width=True,
            key="fig15_box_model_price"
        )
        st.markdown("" \
        "### Key Insight: \n" \
        """M Series vehicles show:\n
        high median price
        high variance
        premium positioning.
        X Series SUVs maintain strong resale value.\n
        Some models have:\n
        huge price spread 
        indicating multiple trims/variants.""")

    with tab4:

        correlation = df.corr(numeric_only=True)

        fig16 = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Feature Correlation Heatmap"
        )

        fig16.update_layout(
            width=500,
            height=400, 
            title_font=dict(size=24),
            font=dict(size=16)
        )

        fig16.update_traces(
            textfont=dict(size=14)
        )

        st.plotly_chart(fig16, use_container_width=True)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")


    st.dataframe(results_df, use_container_width=True)

    fig = px.bar(
        results_df,
        x="Model",
        y="Test R",
        title="Model Comparison",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Best Model")

    st.success("""
    Random Forest achieved the best balance between:
    - High accuracy
    - Low RMSE
    - Good generalization
    """)

# =========================================================
# ABOUT MODEL
# =========================================================

elif page == "🧠 About Model":

    st.title("🧠 About The Model")

    st.markdown("""
    ### Workflow

    1. Data Collection  
    2. Data Cleaning  
    3. Data preprocessing           
    4. Feature Engineering  
    5. Feature extraction  
    6. Model Training 
    7. Hyperpeimeter Tunning


    ---

    ### Algorithms Used

    - Linear Regression
    - Random Forest
    - KNN
    - Descision Tree

    ---

    ### Evaluation Metrics

    - R² Score
    - RMSE
    - MAE
    - MSE
    - Cross Validation

    """)

# =========================================================
# DEVELOPER
# =========================================================

elif page == "👨‍💻 Developer":

    st.title("👨‍💻 Developer")

    st.markdown("""
    ### Sachin Koranga

    Machine Learning Enthusiast focused on:

    - Machine Learning
    - Data Science
    - MLOps
    - Deployment
    - Dashboard Development

    ---

    ### Tech Stack

    - Python
    - Scikit-learn
    - Streamlit
    - Pandas
    - Plotly
    - Numpy
    - seaborn
    - Matplotlib

    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)