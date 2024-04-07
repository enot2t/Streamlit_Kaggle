import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Title
st.title("House Prices - Advanced Regression Techniques")

# Header
st.header("Predict sales prices and practice feature engineering, RFs, and gradient boosting", anchor=None)

# Download files
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

st.text("Train data")
st.dataframe(train.head())

st.text("Test data")
st.table(test.head())

st.text("Distribution of Sale Prices")
fig, ax = plt.subplots()
ax.hist(train['SalePrice'], bins=20, color='green')

st.pyplot(fig)

train_group = train.groupby(['SaleCondition'], as_index=False)['SalePrice'].count()
test_group = test.groupby(['SaleCondition'], as_index=False)['MSZoning'].count()


st.text("Relation SaleCondition in DataSets")
relation = st.radio('Choose dataset: ', ('train', 'test'))

if relation == 'train':
    fig = px.pie(train_group, values='SalePrice', names='SaleCondition')
    st.plotly_chart(fig, use_container_width=True)
elif relation == 'test':
    fig = px.pie(test_group, values='MSZoning', names='SaleCondition')
    st.plotly_chart(fig, use_container_width=True)

