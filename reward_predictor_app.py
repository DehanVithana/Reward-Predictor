
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

st.set_page_config(page_title='Employee Reward Recommender', page_icon='🎁', layout='wide', initial_sidebar_state='expanded')
st.title('🎁 Employee Preferred Reward — AI Predictor')
st.write('Enter employee attributes to predict the most likely **Preferred Reward**. Upload your dataset or use the bundled sample.')

# Data loading
DEFAULT_FILE = 'employee_reward_data.csv'
file = None
if Path(DEFAULT_FILE).exists():
    file = DEFAULT_FILE
st.sidebar.header('Data')
upload = st.sidebar.file_uploader('Upload a CSV (same schema as sample)', type=['csv'])
if upload is not None:
    file = upload

if file is None:
    st.warning('No data found. Please upload `employee_reward_data.csv`.')
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data(file)

# Basic stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Records', len(df))
with col2:
    st.metric('Departments', df['Department'].nunique())
with col3:
    st.metric('Reward Classes', df['PreferredReward'].nunique())

# Build model pipeline
cat_cols = ['Department','PerformanceScore','WorkStyle']
num_cols = ['TenureYears']

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols),
])

model = LogisticRegression(multi_class='multinomial', max_iter=500)
pipe = Pipeline([
    ('prep', preprocess),
    ('clf', model)
])

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['PreferredReward'])
X_train = train_df[cat_cols + num_cols]
y_train = train_df['PreferredReward']
X_test = test_df[cat_cols + num_cols]
y_test = test_df['PreferredReward']

pipe.fit(X_train, y_train)

# Evaluate
pred_test = pipe.predict(X_test)
proba_test = pipe.predict_proba(X_test)
acc = accuracy_score(y_test, pred_test)
cm = confusion_matrix(y_test, pred_test, labels=pipe.classes_)

with st.expander('Model performance on holdout (20%)'):
    st.write(f'**Accuracy:** {acc:.3f}')
    cm_df = pd.DataFrame(cm, index=pipe.classes_, columns=pipe.classes_)
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues', aspect='auto', title='Confusion Matrix')
    st.plotly_chart(fig, use_container_width=True)

# Sidebar: user input
st.sidebar.header('Predict for a new employee')
department = st.sidebar.selectbox('Department', sorted(df['Department'].unique()))
perf = st.sidebar.selectbox('Performance Score', sorted(df['PerformanceScore'].unique()))
work = st.sidebar.selectbox('Work Style', sorted(df['WorkStyle'].unique()))
tenure = st.sidebar.slider('Tenure (years)', float(df['TenureYears'].min()), float(df['TenureYears'].max()), float(df['TenureYears'].median()), 0.1)

new_X = pd.DataFrame([{ 'Department': department, 'PerformanceScore': perf, 'WorkStyle': work, 'TenureYears': tenure }])

if st.sidebar.button('🔮 Predict Preferred Reward'):
    pred = pipe.predict(new_X)[0]
    probs = pipe.predict_proba(new_X)[0]
    prob_df = pd.DataFrame({ 'Reward': pipe.classes_, 'Probability': probs }).sort_values('Probability', ascending=False)
    st.subheader('Prediction')
    st.success(f'**Recommended Reward:** {pred}')
    st.subheader('Class probabilities')
    st.bar_chart(prob_df.set_index('Reward'))

st.markdown('---')
st.caption('Built with scikit-learn & Streamlit.')
