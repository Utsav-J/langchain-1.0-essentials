from langchain_community.utilities import SQLDatabase
from dataclasses import dataclass
from langgraph.runtime import get_runtime
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pyprojroot import here
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()
db_path = here("Chinook.db").resolve()
db_uri = f"sqlite:///{db_path.as_posix()}"

print("Using DB file:", db_path)
print("DB exists?", db_path.exists())
print("Using DB URI:", db_uri)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
try:
    db = SQLDatabase.from_uri(db_uri)
    print(db.get_usable_table_names())
except Exception as e:
    print("Failed to open DB at", db_path)
    print("Exception:", repr(e))
    raise

@dataclass
class RuntimeContext:
    db: SQLDatabase

@tool
def execute_SQL(sql_query:str):
    """Executes an SQL query and returns the results"""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db
    try:
        print(sql_query)
        return db.run(sql_query)
    except Exception as e:
        return f"Error: {e}"

SYSTEM_PROMPT = f"""You are a careful SQLite analyst.
Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
- Here are the usable tables in the sql db:
{db.get_usable_table_names()}
"""

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[execute_SQL],
    context_schema=RuntimeContext,
    checkpointer=InMemorySaver()
)



question = input()
while question != "exit":
    steps = []
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": "1"}},
        context=RuntimeContext(db=db),
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
        steps.append(step)
    question = input()

'''
$ uv run agents/agent_with_simple_memory.py 
Using DB file: E:\\Python Stuff\\langchain-1.0-samples\\Chinook.db
DB exists? True
Using DB URI: sqlite:///E:/Python Stuff/langchain-1.0-samples/Chinook.db
['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']
im frank harris what was my last total
================================ Human Message =================================

im frank harris what was my last total
================================== Ai Message ==================================
Tool Calls:
  execute_SQL (8c841ebe-9e02-45df-876b-92cf3f64eb6d)
 Call ID: 8c841ebe-9e02-45df-876b-92cf3f64eb6d
  Args:
    sql_query: SELECT CustomerId FROM Customer WHERE FirstName = 'Frank' AND LastName = 'Harris'
SELECT CustomerId FROM Customer WHERE FirstName = 'Frank' AND LastName = 'Harris'
================================= Tool Message =================================
Name: execute_SQL

[(16,)]
================================== Ai Message ==================================
Tool Calls:
  execute_SQL (e1af6ce0-33c5-4eab-be5e-9c80b710b983)
 Call ID: e1af6ce0-33c5-4eab-be5e-9c80b710b983
  Args:
    sql_query: SELECT Total FROM Invoice WHERE CustomerId = 16 ORDER BY InvoiceDate DESC LIMIT 1
SELECT Total FROM Invoice WHERE CustomerId = 16 ORDER BY InvoiceDate DESC LIMIT 1
================================= Tool Message =================================
Name: execute_SQL

[(5.94,)]
================================== Ai Message ==================================

Your last total was 5.94.
what were the titles on that
================================ Human Message =================================

what were the titles on that
================================== Ai Message ==================================
Tool Calls:
  execute_SQL (f8eac749-d801-4b0a-93ac-58023fd9c44c)
 Call ID: f8eac749-d801-4b0a-93ac-58023fd9c44c
  Args:
    sql_query: SELECT InvoiceId FROM Invoice WHERE CustomerId = 16 ORDER BY InvoiceDate DESC LIMIT 1
SELECT InvoiceId FROM Invoice WHERE CustomerId = 16 ORDER BY InvoiceDate DESC LIMIT 1
================================= Tool Message =================================
Name: execute_SQL

================================== Ai Message ==================================
Tool Calls:
  execute_SQL (59c52234-99bc-481c-aedd-40c5ad28fee7)
 Call ID: 59c52234-99bc-481c-aedd-40c5ad28fee7
  Args:
    sql_query: SELECT T.Name FROM Track AS T JOIN InvoiceLine AS IL ON T.TrackId = IL.TrackId WHERE IL.InvoiceId = 374
SELECT T.Name FROM Track AS T JOIN InvoiceLine AS IL ON T.TrackId = IL.TrackId WHERE IL.InvoiceId = 374
================================= Tool Message =================================
Name: execute_SQL

[('Holier Than Thou',), ('Through The Never',), ('My Friend Of Misery',), ('The Wait',), ('Blitzkrieg',), ('So What',)]
================================== Ai Message ==================================

The titles on that invoice were: Holier Than Thou, Through The Never, My Friend Of Misery, The Wait, Blitzkrieg, and So What.
great thanks
================================ Human Message =================================

great thanks
================================== Ai Message ==================================

You're welcome! Is there anything else I can help you with today?
'''