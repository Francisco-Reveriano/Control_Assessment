import os
import io
import time
import streamlit as st
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# INITIAL SETUP: Initialize the OpenAI client and conversation history in session state
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Control Assessment Demo")
# Instantiate the OpenAI client with your API key on first run
if 'model' not in st.session_state:
    st.session_state['model'] = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Setup conversation history
if 'conversation' not in st.session_state:
    # Start with a system prompt for context
    st.session_state['conversation'] = [
        {"role": "system", "content": "You are a control assessment assistant. Provide detailed analyses and answer follow-up questions based on earlier parts of the conversation."}
    ]

# Initialize download file storage in session state
if 'download_buffer' not in st.session_state:
    st.session_state['download_buffer'] = None
if 'download_available' not in st.session_state:
    st.session_state['download_available'] = False

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.title('Control Assessment Options')
if st.sidebar.button("New Conversation"):
    # Clear conversation but retain system prompt
    st.session_state['conversation'] = [st.session_state['conversation'][0]]
    st.session_state['download_buffer'] = None
    st.session_state['download_available'] = False

# -----------------------------------------------------------------------------
# PAGE TITLE
# -----------------------------------------------------------------------------
st.title("Control Assessment Chat")

# Display previous conversation
for msg in st.session_state['conversation'][1:]:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# -----------------------------------------------------------------------------
# USER INPUT AND RESPONSE HANDLING
# -----------------------------------------------------------------------------
user_input = st.chat_input("Ask me about the control or provide a new description...")
if user_input:
    # Append user message
    st.session_state['conversation'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    if len(st.session_state['conversation']) == 2:
        # First query: run full control pipeline
        from src.control_classification import classify
        from src.control_summary import summary
        from src.control_risks import risks
        from src.control_dependencies import dependencies
        from src.control_gaps import gaps
        from src.control_industry_practices import industry_practices
        from src.control_score import score
        from src.control_score_reasoning import score_reasoning

        buffer = io.BytesIO()
        state = {"openai_api_key": st.secrets["OPENAI_API_KEY"], "original_input": user_input}
        with st.chat_message("assistant"):
            st.write("***Running full control assessment pipeline...***")

            state = classify(state)
            st.markdown(f"## **Classification:** \n {state['control_classification']}")

            state = summary(state)
            st.markdown(f"## **Summary:** \n {state['control_summary']}")

            state = risks(state)
            st.markdown(f"## **Risks:** \n {state['control_risk']}")

            state = dependencies(state)
            st.markdown(f"## **Dependencies:** \n {state['control_dependencies']}")

            state = gaps(state)
            st.markdown(f"## **Gaps:** \n {state['control_gaps']}")

            state = industry_practices(state)
            st.markdown(f"## **Industry Practices:** \n {state['control_industry_practices']}")

            state = score(state)
            st.markdown(f"## **Score:** \n {state['control_score']}")

            state = score_reasoning(state)
            st.markdown(f"## **Score Reasoning:** \n {state['control_score_reasoning']}")

        # Store combined message after pipeline
        combined = (
            f"## Classification: \n {state['control_classification']}\n"+
            f"## Summary: \n {state['control_summary']}\n"+
            f"## Risks: \n {state['control_risk']}\n"+
            f"## Dependencies: \n {state['control_dependencies']}\n"+
            f"## Gaps: \n {state['control_gaps']}\n"+
            f"## Industry Practices: \n {state['control_industry_practices']}\n"+
            f"## Score: \n {state['control_score']}\n"+
            f"## Score Reasoning: \n {state['control_score_reasoning']}"
        )
        st.session_state['conversation'].append({"role": "assistant", "content": combined})
    else:
        # Follow-up: stream responses using chat model
        with st.chat_message("assistant"):
            stream = st.session_state['model'].chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state['conversation'],
                stream=True
            )
            st.write_stream(stream)
            # Append the full streamed reply to history
            full_reply = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_reply += token

            print(full_reply)
            st.session_state['conversation'].append({
                "role": "assistant",
                "content": full_reply
            })

        # --------------------------------------------
        # DOWNLOAD OPTION
        # -----------------------------------------------------------------------------
    if st.session_state['download_buffer'] is not None and not st.session_state['download_available']:
        st.download_button(
            label="Download Excel Results",
            data=st.session_state['download_buffer'],
            file_name="control_assessment.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.session_state['download_available'] = True
