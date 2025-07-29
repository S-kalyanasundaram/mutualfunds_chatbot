from langchain_community.llms import Ollama
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

# Load your CSV
df = pd.read_csv("FIRE.mfdetails.csv")

# Load the Ollama model
llm = Ollama(model="llama3")  # You can change this to "mistral" or "phi3" if installed

# Create the agent
agent = create_pandas_dataframe_agent(llm, df, verbose=True)

def get_agent():
    return agent
