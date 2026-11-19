import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title='Smart Spend | Expense Predictor', page_icon='💸', layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1250px}
h1{color:#126e65}
div[data-testid="stMetric"]{background:#f2f8f6;padding:18px;border-radius:14px;border:1px solid #d9eae6}
</style>''', unsafe_allow_html=True)

CATEGORIES = ['housing', 'groceries', 'transport', 'utilities', 'healthcare', 'entertainment', 'shopping', 'dining_out', 'education']
FEATURE_COLS = ['monthly_income', 'dependents', 'city_tier', 'employment_type']

@st.cache_data
def load_data():
    return pd.read_csv('smart_spend_dataset.csv')

@st.cache_resource
def train_models():
    df = load_data().dropna(subset=['monthly_expense'] + CATEGORIES)
    X = df[FEATURE_COLS]
    y_total = df['monthly_expense']
    y_cats = df[CATEGORIES]
    cat_features = ['city_tier', 'employment_type']
    
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)], remainder='passthrough')
    
    total_model = Pipeline([('pre', pre), ('model', RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2, n_jobs=-1))])
    total_model.fit(X, y_total)
    
    cat_model = Pipeline([('pre', pre), ('model', RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2, n_jobs=-1))])
    cat_model.fit(X, y_cats)
    
    return total_model, cat_model

def money(v):
    return f'₹{v:,.0f}'

df = load_data()
total_model, cat_model = train_models()

st.title('💸 Smart Spend')
st.caption('AI-powered monthly expense estimation and budget insights based on your financial profile')

with st.sidebar:
    st.header('Your Financial Profile')
    income = st.number_input('Monthly income (₹)', min_value=0, max_value=10000000, value=60000, step=1000)
    dependents = st.number_input('Number of dependents', min_value=0, max_value=20, value=1)
    city = st.selectbox('City tier', sorted(df.city_tier.dropna().unique().tolist()))
    employment = st.selectbox('Employment type', sorted(df.employment_type.dropna().unique().tolist()))

profile_df = pd.DataFrame([{
    'monthly_income': income,
    'dependents': dependents,
    'city_tier': city,
    'employment_type': employment
}])

# AI Predictions
predicted_total = max(0.0, float(total_model.predict(profile_df)[0]))
predicted_cats_raw = cat_model.predict(profile_df)[0]
predicted_cats = {c: max(0.0, float(val)) for c, val in zip(CATEGORIES, predicted_cats_raw)}

savings = income - predicted_total
savings_rate = (savings / income * 100) if income > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric('Monthly Income', money(income))
c2.metric('Predicted Total Expense', money(predicted_total))
c3.metric('Estimated Savings', money(savings), f'{savings_rate:.1f}% savings rate' if income > 0 else None)

st.info('💡 Expense prediction is calculated by machine learning trained on demographic and spending patterns.')

left, right = st.columns([1.1, 1])

with left:
    st.subheader('Predicted Spending by Category')
    chart_df = pd.DataFrame({
        'Category': [c.replace('_', ' ').title() for c in CATEGORIES],
        'Predicted Amount': list(predicted_cats.values())
    })
    fig = px.pie(
        chart_df,
        values='Predicted Amount',
        names='Category',
        hole=0.55,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation='h', y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader('Budget Overview')
    compare = pd.DataFrame({
        'Metric': ['Monthly Income', 'Predicted Expenses', 'Estimated Savings'],
        'Amount': [income, predicted_total, max(0.0, savings)]
    })
    fig = px.bar(
        compare,
        x='Metric',
        y='Amount',
        color='Metric',
        color_discrete_sequence=['#178d80', '#ef9a52', '#36a2eb']
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), yaxis_title='Amount (₹)', xaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    if savings >= 0:
        st.success(f'Based on your profile, you are estimated to save **{money(savings)}** ({savings_rate:.1f}%) each month.')
    else:
        st.warning(f'Predicted expenses exceed your income by **{money(abs(savings))}**. Consider reviewing budget allocations.')

st.subheader('Predicted Category Breakdown')
breakdown_table = pd.DataFrame({
    'Category': [c.replace('_', ' ').title() for c in CATEGORIES],
    'Predicted Monthly Spending (₹)': [f'{v:,.2f}' for v in predicted_cats.values()],
    'Share of Budget': [f'{(v / predicted_total * 100):.1f}%' if predicted_total > 0 else '0.0%' for v in predicted_cats.values()]
})
st.dataframe(breakdown_table, hide_index=True, use_container_width=True)

# Export / download
result_dict = {
    'monthly_income': income,
    'dependents': dependents,
    'city_tier': city,
    'employment_type': employment,
    'predicted_total_expense': round(predicted_total, 2),
    'estimated_savings': round(savings, 2),
    'savings_rate_pct': round(savings_rate, 2),
    **{f'pred_{c}': round(v, 2) for c, v in predicted_cats.items()}
}
result = pd.DataFrame([result_dict])
st.download_button('⬇️ Download prediction as CSV', data=result.to_csv(index=False), file_name='smart_spend_prediction.csv', mime='text/csv')

with st.expander('About the AI Model'):
    st.write('The model uses Random Forest Regression trained on demographic profile features (income, dependents, city tier, employment type) to predict both overall monthly expenses and category-level allocations. Trained locally and cached.')
