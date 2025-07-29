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

# ---- Page Config ----
st.set_page_config(page_title="Mutual Fund Chatbot", page_icon="💬", layout="centered")

# ---- Custom Glassmorphism CSS ----
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
    }

    .main {
        padding-top: 30px;
    }

    h1 {
        text-align: center;
        color: #1b1c1d;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #444;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .chat-card {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0,0,0,0.1);
        padding: 1rem;
        margin: 1rem 0;
        display: flex;
        gap: 10px;
        max-width: 85%;
    }

    .chat-card.user {
        margin-left: auto;
        flex-direction: row-reverse;
    }

    .chat-card.assistant {
        margin-right: auto;
    }

    .chat-avatar {
        height: 40px;
        width: 40px;
        border-radius: 50%;
        background-color: #ffffff66;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 20px;
        color: #333;
    }

    .chat-content {
        font-size: 1rem;
        color: #222;
        word-wrap: break-word;
        line-height: 1.5;
    }

    .stChatInput input {
        border-radius: 16px !important;
        padding: 0.6rem 1rem !important;
        font-size: 1rem !important;
        border: 1px solid #ccc !important;
    }

    </style>
""", unsafe_allow_html=True)

# ---- Title and subtitle ----
st.markdown("<h1>💬 Mutual Fund Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smart financial insights with a stylish twist ✨</p>", unsafe_allow_html=True)

# ---- Load dataset ----
@st.cache_data
def load_data():
    return pd.read_csv("FIRE.mfdetails.csv")

# ---- Load AI agent ----
def load_agent(dataframe):
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not found. Add it to .env")
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

# ---- App Logic ----
try:
    df = load_data()
    st.success("✅ Dataset loaded.")

    agent = load_agent(df)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi 👋 I’m your stylish Mutual Fund Advisor. Ask me anything financial!"}
        ]

    # ---- Display chat history ----
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "🧑" if role == "user" else "🤖"
        bubble_class = f"chat-card {role}"
        st.markdown(f"""
            <div class="{bubble_class}">
                <div class="chat-avatar">{avatar}</div>
                <div class="chat-content">{msg["content"]}</div>
            </div>
        """, unsafe_allow_html=True)

    # ---- Input ----
    user_input = st.chat_input("Ask your question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("🤖 Thinking..."):
            try:
                response = agent.run(user_input)
            except Exception as e:
                response = f"❌ Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

except FileNotFoundError:
    st.error("❌ FIRE.mfdetails.csv not found. Please upload it.")
