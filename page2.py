import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show():
    # Load the CSV file
    data = pd.read_csv('modified_data.csv')

    # Group the data by 'id' and 'condition' and calculate the sum of 'value'
    flow = data.groupby(['id', 'condition'])['Sum'].mean().reset_index()


    # Streamlit app starts here
    st.title('PSD Value Bar Chart by Segment and Frequency Band')

    # Segment selection
    segment = st.sidebar.selectbox('Select a Time Segment', sorted(data['segment'].unique()))

    # PSD Type selection
    freq_band = st.sidebar.selectbox('Select a Frequency Band', ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])

    # Filter data based on the selected segment
    filtered_data_2 = data[data['segment'] == segment]

    # Plotting
    fig, ax = plt.subplots()
    filtered_data_2.groupby('condition')[freq_band].mean().plot(kind='bar', ax=ax)
    ax.set_title(f'Average {freq_band} Value for Each Condition')
    ax.set_ylabel(f'PSD Value of {freq_band}')
    ax.set_xlabel('Condition')

    # Display the plot
    st.pyplot(fig)