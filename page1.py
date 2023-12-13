import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    # Load the CSV file
    data = pd.read_csv('modified_data.csv')

    # Group the data by 'id' and 'condition' and calculate the sum of 'value'
    flow = data.groupby(['id', 'condition'])['Sum'].mean().reset_index()

    # title
    st.title('Subjective Flow Score Distribution by Condition')

    # Dropdown to select condition
    condition = st.sidebar.selectbox('Select a Condition', flow['condition'].unique())

    # Filter data based on the selected condition
    filtered_data = flow[flow['condition'] == condition]

    # Show distribution plot
    st.write(f'Subjective Flow Score Distribution Plot for Condition: {condition}')
    fig, ax = plt.subplots()
    sns.histplot(filtered_data['Sum'], ax=ax, kde=True)
    st.pyplot(fig)