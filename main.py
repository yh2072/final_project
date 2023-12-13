import streamlit as st
import importlib

# Dictionary of pages
pages = {
    "Q1 What is subjective flow score for different conditions?": "page1",
    "Q2 What is EEG signals(PSD value) like among different conditions for a specific time segment and frequency band?": "page2",
    "Q3 What is correlation between EEG signals and subjective flow score for a specific frequency band and condition within certain time segments?": "page3",
    "Q4 How can we predict subjective flow score using EEG signals through a deep learning model ?": "page4"
    # Add more pages as needed
}

# Sidebar selection
page = st.sidebar.selectbox("Select a page:", options=list(pages.keys()))

# Import the selected page module and call its show function
if page:
    page_module = importlib.import_module(pages[page])
    page_module.show()

