import os
import io
import time

import streamlit as st
st. set_page_config(layout="wide")
from openai import OpenAI
import asyncio
from src.control_classification import *
from src.control_summary import *
from src.control_risks import *
from src.control_dependencies import *
from src.control_gaps import *
from src.control_industry_practices import *
from src.control_score import *
from src.control_score_reasoning import *
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# INITIAL SETUP: Initialize the OpenAI client and conversation history in session state
# -----------------------------------------------------------------------------
if 'model' not in st.session_state:
    # Instantiate the OpenAI client with your API key.
    st.session_state['model'] = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if 'conversation' not in st.session_state:
    # This list will hold the conversation messages.
    st.session_state['conversation'] = []

if 'new_conversation_flag' not in st.session_state:
    st.session_state['new_conversation_flag'] = 0

# Initialize download file storage in session state
if 'download_buffer' not in st.session_state:
    st.session_state['download_buffer'] = None
if 'download_available' not in st.session_state:
    st.session_state['download_available'] = False

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.title('Control Assessment Options')

# Button to start a new conversation
if st.sidebar.button("New Conversation"):
    st.session_state['conversation'] = []         # Clear conversation history
    st.session_state['new_conversation_flag'] = 0    # Reset flag
    st.session_state['download_buffer'] = None       # Clear stored download data
    st.session_state['download_available'] = False   # Reset download availability

# -----------------------------------------------------------------------------
# PAGE TITLE
# -----------------------------------------------------------------------------
st.title("Control Assessment")

# Display all previous conversation messages in order (rendered as Markdown)
for msg in st.session_state['conversation']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# -----------------------------------------------------------------------------
# USER INPUT AND RESPONSE HANDLING
# -----------------------------------------------------------------------------
# Get the user's legal query.
user_input = st.chat_input("Hello! Please provide the Control Description:")

if user_input:
    # Append the user's message to the conversation history.
    st.session_state['conversation'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**User:** {user_input}")

    # Retrieve and display the legal response.
    with st.chat_message("assistant"):
        # Create an in-memory binary buffer for Excel File
        buffer = io.BytesIO()
        # Create the response
        state = {"openai_api_key": st.secrets["OPENAI_API_KEY"], "original_input": user_input}
        with st.status("Running analysis..."):
            ### Validating Documents
            st.write("***Classifying Control...*** \n")
            state = classify(state)
            st.markdown(f"{state['control_classification']}", unsafe_allow_html=True)
            st.markdown("\n")

            ### Creating Control 1 Line Summary
            st.write("***Creating Control Summary...*** \n")
            state = summary(state)
            st.markdown(f"{state['control_summary']}", unsafe_allow_html=True)
            st.markdown("\n")

            ### Control Risks
            st.write("***Control Risks...*** \n")
            state = risks(state)
            st.markdown(f"{state['control_risk']}", unsafe_allow_html=True)
            st.markdown("\n")

            ## Control Dependencies
            st.write("***Control Dependencies...*** \n")
            state = dependencies(state)
            st.markdown(f"{state['control_dependencies']}", unsafe_allow_html=True)
            st.markdown("\n")

            # Control Gaps
            st.write("***Control Gaps...*** \n")
            state = gaps(state)
            st.markdown(f"{state['control_gaps']}", unsafe_allow_html=True)
            st.markdown("\n")

            # Control Industry Best Practices
            st.write("***Control Industry Best Practices...*** \n")
            state = industry_practices(state)
            st.markdown(f"{state['control_industry_practices']}", unsafe_allow_html=True)
            st.markdown("\n")

            # Control Scores
            st.write("***Control Score...*** \n")
            state = score(state)
            st.markdown(f"{state['control_score']}", unsafe_allow_html=True)
            st.markdown("\n")

            # Control Score Reasoning
            st.write("***Control Score Reasoning...*** \n")
            state = score_reasoning(state)
            st.markdown(f"{state['control_score_reasoning']}", unsafe_allow_html=True)
            st.markdown("\n")




