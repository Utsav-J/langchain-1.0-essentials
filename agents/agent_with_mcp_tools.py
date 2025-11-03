import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


mcp_client = MultiServerMCPClient(
    {
        "time":{
            "transport":"stdio",
            "command":"npx",
            "args":["-y", "@theo.foobar/mcp-time"]
        }
    }
)
mcp_tools = asyncio.run(mcp_client.get_tools())
print(f"Loaded {len(mcp_tools)} from MCP servers.")
print(f"Tools: {mcp_tools}")


agent_with_mcp = create_agent(
    model=model,
    tools=mcp_tools,
    system_prompt="You are a helpful assistant",
)

async def run():
    result = await agent_with_mcp.ainvoke(
        {"messages": [{"role": "user", "content": "What's the time in SF right now?"}]}
    )
    for msg in result["messages"]:
        msg.pretty_print()

if __name__ == "__main__":
    asyncio.run(run())