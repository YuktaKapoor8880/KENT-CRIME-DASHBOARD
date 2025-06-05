import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import gdown


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(page_title="Kent Police Crime Dashboard", layout="wide")
st.title("📊 Kent Police Predictive Crime Dashboard")

# -----------------------------------
# LOAD AND CLEAN DATA
# -----------------------------------
@st.cache_data
def load_data():
    file_id = "1nYy7ofK-_cYJhMRGjIXoVA_na2iTsOYK"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "kent_police.csv", quiet=False)
df = pd.read_csv("kent_police.csv")
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
return df

df = load_data()

# -----------------------------------
# DATA OVERVIEW
# -----------------------------------
st.markdown("#### 🎯 Business Goal: Help Kent Police anticipate crime patterns and investigation outcomes.")
with st.expander("📋 Columns in Dataset"):
    st.write(df.columns.tolist())

# -----------------------------------
# FILTERS
# -----------------------------------
st.sidebar.header("🔍 Filter Crime Data")
crime_types = st.sidebar.multiselect("Select Crime Types", df['Crime_Category'].dropna().unique())
seasons = st.sidebar.multiselect("Select Seasons", df['Season'].dropna().unique())
months = st.sidebar.multiselect("Select Months", df['Month'].dropna().dt.month_name().unique())

# Date range slider
min_date = df['Month'].min().date()
max_date = df['Month'].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

filtered_df = df.copy()
if crime_types:
    filtered_df = filtered_df[filtered_df['Crime_Category'].isin(crime_types)]
if seasons:
    filtered_df = filtered_df[filtered_df['Season'].isin(seasons)]
if months:
    filtered_df = filtered_df[filtered_df['Month'].dt.month_name().isin(months)]
filtered_df = filtered_df[(filtered_df['Month'].dt.date >= start_date) & (filtered_df['Month'].dt.date <= end_date)]

# -----------------------------------
# KPIs
# -----------------------------------
st.subheader("📌 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(filtered_df):,}")
col2.metric("Unique Crime Types", filtered_df['Crime_Category'].nunique())
col3.metric("Distinct Outcomes", filtered_df['Last outcome category'].nunique())

# -----------------------------------
# VISUALS
# -----------------------------------
st.markdown("### 📅 1. When does crime happen most?")
chart_type = st.selectbox("Choose Chart Type for Monthly Trend", ['Line', 'Bar'])
monthly_crimes = filtered_df['Month'].dt.to_period('M').value_counts().sort_index()
if chart_type == 'Line':
    fig1 = px.line(x=monthly_crimes.index.astype(str), y=monthly_crimes.values,
                   labels={"x": "Month", "y": "Number of Crimes"},
                   title="Monthly Crime Trends")
else:
    fig1 = px.bar(x=monthly_crimes.index.astype(str), y=monthly_crimes.values,
                  labels={"x": "Month", "y": "Number of Crimes"},
                  title="Monthly Crime Trends")
st.plotly_chart(fig1, use_container_width=True)

# 2
st.markdown("### 🔍 2. What types of crime are most common?")
crime_counts = filtered_df['Crime_Category'].value_counts()
fig2 = px.pie(values=crime_counts.values, names=crime_counts.index, title="Crime Types Distribution", hole=0.4)
st.plotly_chart(fig2, use_container_width=True)

# 3
st.markdown("### 🌦️ 3. How does crime vary by season?")
season_chart = filtered_df.groupby(['Season', 'Crime_Category']).size().unstack(fill_value=0)
fig3 = px.bar(season_chart, barmode='group', title="Crime Types by Season")
st.plotly_chart(fig3, use_container_width=True)


#4
# -----------------------------------
# MAP VISUALISATION
# -----------------------------------
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    st.markdown("### 🗺️ 4. Where are crimes located?")
    st.map(filtered_df[['Latitude', 'Longitude']].dropna().rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}))


# 5. Day of Week Patterns
st.markdown("### 📆 5. On which days do crimes happen most?")
filtered_df['DayOfWeek'] = filtered_df['Month'].dt.day_name()
dow_counts = filtered_df['DayOfWeek'].value_counts().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
fig6 = px.bar(x=dow_counts.index, y=dow_counts.values, labels={"x": "Day of Week", "y": "Crime Count"},
              title="Crime Frequency by Day of Week")
st.plotly_chart(fig6, use_container_width=True)

# 6. Crime Type vs Outcome
st.markdown("### 📊 6. Which crimes lead to successful outcomes?")
outcome_success = filtered_df.groupby(['Crime_Category', 'Last outcome category']).size().unstack(fill_value=0)
fig7 = px.bar(outcome_success, barmode='stack', title="Outcome Distribution by Crime Type")
st.plotly_chart(fig7, use_container_width=True)

# 7. Location-based Trends
st.markdown("### 📍 7. Which LSOA areas have the most crime?")
lsoa_counts = filtered_df['LSOA name'].value_counts().nlargest(10)
fig8 = px.bar(x=lsoa_counts.values, y=lsoa_counts.index, orientation='h',
              labels={"x": "Crime Count", "y": "LSOA Name"}, title="Top 10 Crime-Prone Areas")
st.plotly_chart(fig8, use_container_width=True)

# 8. Crime vs Outcome Treemap (Fixed)
st.markdown("### 🌲 8. Crime Categories vs Outcomes (Treemap)")

# Count each (Crime_Category, Last outcome category) pair
treemap_df = (
    filtered_df.groupby(['Crime_Category', 'Last outcome category'])
    .size()
    .reset_index(name='Count')
)

fig9 = px.treemap(
    treemap_df,
    path=['Crime_Category', 'Last outcome category'],
    values='Count',
    title="Crime Category and Outcomes Treemap"
)

st.plotly_chart(fig9, use_container_width=True)

# 9. Outcome Trends Over Time (Fixed)
st.markdown("### 📈 9. How do investigation outcomes change over time?")

# Group and convert Period to string or datetime
outcome_trends = (
    filtered_df
    .groupby([filtered_df['Month'].dt.to_period('M'), 'Last outcome category'])
    .size()
    .unstack(fill_value=0)
)
outcome_trends.index = outcome_trends.index.to_timestamp()  # Convert PeriodIndex to datetime

fig10 = px.line(
    outcome_trends,
    x=outcome_trends.index,
    y=outcome_trends.columns,
    labels={"value": "Crime Count", "index": "Month"},
    title="Monthly Outcome Trends"
)
st.plotly_chart(fig10, use_container_width=True)


# 10. Crime Type vs LSOA
st.markdown("### 🧭 10. Which areas are most affected by specific crime types?")
top_crimes = filtered_df.groupby(['LSOA name', 'Crime_Category']).size().reset_index(name='Counts')
fig11 = px.bar(top_crimes, x='Counts', y='LSOA name', color='Crime_Category',
               title="Crime Types per LSOA", orientation='h')
st.plotly_chart(fig11, use_container_width=True)

# 11. Seasonal Resolution Rates
st.markdown("### ☀️ 11. Which seasons see the highest resolution rates?")
resolved = filtered_df['Last outcome category'].str.contains("charged|summonsed|identified", case=False, na=False)
resolution_rates = filtered_df[resolved].groupby('Season').size() / filtered_df.groupby('Season').size()
fig12 = px.bar(resolution_rates.fillna(0), labels={'value': 'Resolution Rate'}, title="Resolution Rate by Season")
st.plotly_chart(fig12, use_container_width=True)

# 12. Outliers in Monthly Crime Counts
st.markdown("### 🚨 12. Are there anomalies in monthly crime reporting?")
monthly_totals = filtered_df['Month'].dt.to_period('M').value_counts()
z_scores = ((monthly_totals - monthly_totals.mean()) / monthly_totals.std()).abs()
outliers = z_scores[z_scores > 2]
fig13 = px.scatter(x=monthly_totals.index.astype(str), y=monthly_totals.values,
                   color=z_scores > 2, title="Monthly Crime Volume Anomalies",
                   labels={"x": "Month", "y": "Crime Count"})
st.plotly_chart(fig13, use_container_width=True)

# -----------------------------------
# PREDICTIVE MODEL
# -----------------------------------
st.markdown("### 🤖 13. Predict Crime Outcome")
model_df = df[['Crime_Category', 'Season', 'Last outcome category']].dropna()
le_crime = LabelEncoder()
le_season = LabelEncoder()
le_outcome = LabelEncoder()

model_df['Crime_enc'] = le_crime.fit_transform(model_df['Crime_Category'])
model_df['Season_enc'] = le_season.fit_transform(model_df['Season'])
model_df['Outcome_enc'] = le_outcome.fit_transform(model_df['Last outcome category'])

X = model_df[['Crime_enc', 'Season_enc']]
y = model_df['Outcome_enc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

st.success(f"✅ Model Accuracy on Test Data: {accuracy:.2%}")
st.caption("ℹ️ Model: Decision Tree Classifier. Could be improved with RandomForest or more features like Location.")

st.markdown("#### 🎯 Try predicting an outcome")
user_crime = st.selectbox("Crime Type", le_crime.classes_)
user_season = st.selectbox("Season", le_season.classes_)

try:
    input_df = pd.DataFrame({
        'Crime_enc': le_crime.transform([user_crime]),
        'Season_enc': le_season.transform([user_season])
    })
    pred_enc = model.predict(input_df)[0]
    pred_label = le_outcome.inverse_transform([pred_enc])[0]
    st.info(f"🔮 **Predicted Outcome:** _{pred_label}_")

    probs = model.predict_proba(input_df)[0]
    trained_classes = model.classes_
    trained_labels = le_outcome.inverse_transform(trained_classes)
    confidence_df = pd.DataFrame({
        'Outcome': trained_labels,
        'Probability': probs
    }).sort_values(by='Probability', ascending=False).head()
    st.write("🔢 Prediction Confidence:", confidence_df)
except ValueError as e:
    st.error(f"Prediction failed. Try selecting a different crime or season. Error: {e}")

# -----------------------------------
# FOOTER
# -----------------------------------
st.caption("📘 Dashboard built for MSc Advanced Topics in Data Analytics — Predictive Analytics Assessment")


