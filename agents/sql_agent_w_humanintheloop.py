from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from pyprojroot import here
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langgraph.types import Command

load_dotenv()
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
db_path = here("Chinook.db")
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

@dataclass
class RuntimeContext:
    db:SQLDatabase

@tool
def execute_sql(query: str):
    """Execute a SQLite command and return results."""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

SYSTEM_PROMPT = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
- If the database is offline, ask user to try again later without further comment.
"""

agent = create_agent(
    model=gemini_model,
    context_schema=RuntimeContext,
    system_prompt=SYSTEM_PROMPT,
    tools=[execute_sql],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                'execute_sql':{
                    "allowed_decisions":['approve','reject']
                }
            }
        ) # type: ignore
    ]
)

question = "What are the names of all the employees?"

config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config=config, # type: ignore
    context=RuntimeContext(db=db)
)

while "__interrupt__" in result:
    description = result['__interrupt__'][-1].value['action_requests'][-1]['description']
    print(f"\033[1;3;31m{80 * '-'}\033[0m")
    print(
        f"\033[1;3;31m Interrupt:{description}\033[0m"
    )

    result = agent.invoke(
        Command(
            # resume={
            #     "decisions": [{"type": "reject", "message": "the database is offline."}]
            # }
            resume={
                "decisions": [{"type": "approve"}]
            }
        ),
        config=config,  # Same thread ID to resume the paused conversation
        context=RuntimeContext(db=db),
    )
    print(f"\033[1;3;31m{80 * '-'}\033[0m")

print(result["messages"][-1].content)

for msg in result["messages"]:
    msg.pretty_print()