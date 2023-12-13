import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show():
    data = pd.read_csv('modified_data.csv')

    # Group the data by 'id' and 'condition' and calculate the sum of 'value'
    flow = data.groupby(['id', 'condition'])['Sum'].mean().reset_index()


    # Streamlit app starts here
    st.title('Correlation Plot by Condition and Frequency Band')

    # Frequency band selection
    freq_band_3 = st.sidebar.selectbox('Please Select a Frequency Band', ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])

    # Condition selection
    condition_3 = st.sidebar.selectbox('Please Select a Condition', data['condition'].unique())

    segments = st.sidebar.multiselect('Select a Time Segment', sorted(data['segment'].unique()))

    # Check if any segments are selected
    if segments:
        # Filter data based on the selected frequency band, condition, and segments
        filtered_data = data[(data['condition'] == condition_3) & (data['segment'].isin(segments))][['Sum', freq_band_3]]

        # Calculate correlation coefficient
        correlation_coefficient = filtered_data.corr().iloc[0, 1]

        # Plotting
        st.write(
            f'Correlation Plot for Condition: {condition_3}, Frequency Band: {freq_band_3}, and Segments: {", ".join(map(str, segments))}')
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_data, x='Sum', y=freq_band_3, ax=ax)
        sns.regplot(data=filtered_data, x='Sum', y=freq_band_3, ax=ax, scatter=False, color='red')

        # Annotate correlation coefficient on the plot
        ax.annotate(f'Correlation: {correlation_coefficient:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2))

        ax.set_title(f'Correlation between Subjective Flow Score and {freq_band_3}')
        ax.set_xlabel('Subjective Flow Score')
        ax.set_ylabel(freq_band_3)

        # Display the plot
        st.pyplot(fig)

    else:
        st.write("Please select at least one segment.")