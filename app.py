import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# ---- Load environment variables ----
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")

# ---- Streamlit Page Setup ----
st.set_page_config(page_title="Mutual Fund Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Mutual Fund Advisor Chatbot")
st.caption("Ask me anything about mutual funds based on the uploaded dataset.")

# ---- Load Dataset ----
@st.cache_data
def load_data():
    return pd.read_csv("FIRE.mfdetails.csv")

# ---- Create Agent ----
def load_agent(dataframe):
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not found. Please set it in the `.env` file.")
        st.stop()

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    return create_pandas_dataframe_agent(
        llm,
        dataframe,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )

# ---- Main Logic ----
try:
    df = load_data()
    st.success("✅ Mutual fund dataset loaded.")

    agent = load_agent(df)

    # ---- Initialize Chat History ----
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! 👋 I'm your Mutual Fund advisor. Ask me anything about mutual funds!"}
        ]

    # ---- Display Chat Messages ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- Chat Input ----
    user_input = st.chat_input("Ask a question about mutual funds...")

    if user_input:
        # Show user message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get response from the agent
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    response = agent.run(user_input)
                except Exception as e:
                    response = f"❌ Sorry, an error occurred: {e}"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

except FileNotFoundError:
    st.error("❌ Could not find 'FIRE.mfdetails.csv'. Please ensure it's in the same folder as this script.")
