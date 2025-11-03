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
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

@dataclass
class RuntimeContext:
    db: SQLDatabase

@tool
def execute_SQL(sql_query:str):
    """Executes an SQL query and returns the results"""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        db.run(sql_query)
    except Exception as e:
        return f"Error: {e}"

SYSTEM_PROMPT = """You are a careful SQLite analyst.
Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
"""

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[execute_SQL],
    context_schema=RuntimeContext
)


question = "List all the tables"
for step in agent.stream(
    {"messages": HumanMessage(question)}, # type: ignore
    context= RuntimeContext(db=db),
    stream_mode="values"
):
    step['messages'][-1].pretty_print()
