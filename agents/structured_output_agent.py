from typing_extensions import TypedDict, List
from langchain.agents import create_agent
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class ContactInfo(TypedDict):
    name:str
    phone:str
    email:str

class OutputSchema(TypedDict):
    contacts: List[ContactInfo]

agent = create_agent(
    model=gemini_model,
    system_prompt="You are a helpful AI assistant that specializes in extrating contact information from a given conversation recording.",
    response_format=OutputSchema
)

recorded_conversation = """We talked with John Doe. He works over at Example. His number is, let's see, 
five, five, five, one two three, four, five, six seven. Did you get that?
And, his email was john at example.com. He wanted to order 50 boxes of Captain Crunch."""

result = agent.invoke(
    {"messages": [{"role": "user", "content": recorded_conversation}]}
)

print(result)
print(f"\n\n{result['structured_response']}")

'''
eg.
{'contacts': [{'name': 'John Doe', 'phone': '555-123-4567', 'email': 'john@example.com'}]}
'''