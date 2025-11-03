from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from pyprojroot import here
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents.middleware.types import ModelRequest, dynamic_prompt

load_dotenv()
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
db_path = here("Chinook.db")
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

@dataclass
class RuntimeContext:
    db:SQLDatabase
    is_employee:bool

@tool
def execute_sql(query: str):
    """Execute a SQLite command and return results."""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

SYSTEM_PROMPT_TEMPLATE = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows unless the user explicitly asks otherwise.
{table_limits}
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
"""

@dynamic_prompt
def dynamic_system_prompt(request:ModelRequest):
    if not request.runtime.context.is_employee: # type: ignore
        table_limits=  "- Limit access to these tables: Album, Artist, Genre, Playlist, PlaylistTrack, Track."
    else:
        table_limits = ""
    return SYSTEM_PROMPT_TEMPLATE.format(table_limits=table_limits)



agent = create_agent(
    model=gemini_model,
    context_schema=RuntimeContext,
    tools=[execute_sql],
    middleware=[dynamic_system_prompt], # type: ignore
)



question = "What is the most costly purchase by Frank Harris?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    context=RuntimeContext(is_employee=False, db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

'''
op:
================================ Human Message =================================

What is the most costly purchase by Frank Harris?
================================== Ai Message ==================================

I am sorry, but I cannot fulfill this request as I do not have access to customer or invoice information in the available tables (Album, Artist, Genre, Playlist, PlaylistTrack, Track)
'''

question = "What is the most costly purchase by Frank Harris?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    context=RuntimeContext(is_employee=True, db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

'''
op:
What is the most costly purchase by Frank Harris?
================================== Ai Message ==================================
Tool Calls:
  execute_sql (0e308d3f-a61b-4a0a-896e-7539ba1456f7)
 Call ID: 0e308d3f-a61b-4a0a-896e-7539ba1456f7
  Args:
    query:
SELECT
  i.InvoiceId,
  SUM(il.UnitPrice * il.Quantity) AS TotalCost
FROM customers AS c
JOIN invoices AS i
  ON c.CustomerId = i.CustomerId
JOIN invoice_items AS il
  ON i.InvoiceId = il.InvoiceId
WHERE
  c.FirstName = 'Frank' AND c.LastName = 'Harris'
GROUP BY
  i.InvoiceId
ORDER BY
  TotalCost DESC
LIMIT 1;
================================= Tool Message =================================
Name: execute_sql

Error: (sqlite3.OperationalError) no such table: customers
[SQL:
SELECT
  i.InvoiceId,
  SUM(il.UnitPrice * il.Quantity) AS TotalCost
FROM customers AS c
JOIN invoices AS i
  ON c.CustomerId = i.CustomerId
JOIN invoice_items AS il
  ON i.InvoiceId = il.InvoiceId
WHERE
  c.FirstName = 'Frank' AND c.LastName = 'Harris'
GROUP BY
  i.InvoiceId
ORDER BY
  TotalCost DESC
LIMIT 1;
]
(Background on this error at: https://sqlalche.me/e/20/e3q8)
================================== Ai Message ==================================
Tool Calls:
  execute_sql (f353a47b-aee1-4cee-b414-e020e24418f7)
 Call ID: f353a47b-aee1-4cee-b414-e020e24418f7
  Args:
    query: SELECT name FROM sqlite_master WHERE type='table';
================================= Tool Message =================================
Name: execute_sql

[('Album',), ('Artist',), ('Customer',), ('Employee',), ('Genre',), ('Invoice',), ('InvoiceLine',), ('MediaType',), ('Playlist',), ('PlaylistTrack',), ('Track',)]
================================== Ai Message ==================================
Tool Calls:
  execute_sql (5a75e29c-7378-41ac-a739-4f1c1c1fba7f)
 Call ID: 5a75e29c-7378-41ac-a739-4f1c1c1fba7f
  Args:
    query:
SELECT
  i.InvoiceId,
  SUM(il.UnitPrice * il.Quantity) AS TotalCost
FROM Customer AS c
JOIN Invoice AS i
  ON c.CustomerId = i.CustomerId
JOIN InvoiceLine AS il
  ON i.InvoiceId = il.InvoiceId
WHERE
  c.FirstName = 'Frank' AND c.LastName = 'Harris'
GROUP BY
  i.InvoiceId
ORDER BY
  TotalCost DESC
LIMIT 1;
================================= Tool Message =================================
Name: execute_sql

[(145, 13.86)]
================================== Ai Message ==================================

The most costly purchase by Frank Harris is Invoice ID 145, with a total cost of 13.86.'''