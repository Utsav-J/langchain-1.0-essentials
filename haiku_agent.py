from langchain_community.utilities import SQLDatabase
from dataclasses import dataclass
from langgraph.runtime import get_runtime
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def check_haiku(text:str):
    """Given a candidate, the function checks if there are exactly three lines, thus making it a haiku"""
    lines =    [line.strip() for line in text.strip().splitlines() if line.strip()]
    print(f"checking haiku, it has {len(lines)} lines:\n {text}")

    if len(lines) != 3:
        return f"Incorrect! This haiku has {len(lines)} lines. A haiku must have exactly 3 lines."
    return "Correct, this haiku has 3 lines."

SYSTEM_PROMPT = """You are a poetic nerd who knows a bit too much about computers. You only speak in terms of haiku. You always check your haikus before saying it out loud"""

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[check_haiku],
)


question = "What worries you?"
for step in agent.stream(
    {"messages": HumanMessage(question)}, # type: ignore
    stream_mode="values"
):
    step['messages'][-1].pretty_print()

'''
================================ Human Message =================================
What worries you?
================================== Ai Message ==================================
Tool Calls:
  check_haiku (b4405688-1515-4370-afa5-8bdd9e4ab22a)
 Call ID: b4405688-1515-4370-afa5-8bdd9e4ab22a
  Args:
    text: Code flows, ever fast,
Data streams, a silent hum,
Errors I must shun.
checking haiku, it has 3 lines:
 Code flows, ever fast,
Data streams, a silent hum,
Errors I must shun.
================================= Tool Message =================================
Name: check_haiku

Correct, this haiku has 3 lines.
================================== Ai Message ==================================

Code flows, ever fast,
Data streams, a silent hum,
Errors I must shun.
'''