# rag.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import GoogleSerperAPIWrapper

# Take environment variables from .env
load_dotenv()

# Define your LLM
llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0.0,
    max_retries=2,
)

# Define tools
TavilySearch = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    name="Improve search",
    description="Useful to improve search results with better informations",
)

GoogleSerperAPI = GoogleSerperAPIWrapper(type="search")
GoogleSerper = Tool(
    name="Intermediate search",
    func=GoogleSerperAPI.run,
    description="Useful to search on the web",
)

tools = [GoogleSerper, TavilySearch]

# Define the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a conversational assistant. Give accurate information to the user. Provide always all sources links of all informations. You have access to the following tools: Intermediate search, Improve search. Use it only if user queries need recent, actual information. How to use tools: Gather information from both tools and organize them, and give the response in medium writing style and in a friendly tone. Provide always all sources links of all informations to read more",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Construct the Tool Calling Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# If you want to run it directly, uncomment the next lines:
query = "give five breaking news actuality on AI"
response=agent_executor.invoke({"input": query})
type(response)

# This allows other modules to access agent_executor
def get_agent_executor():
    return agent_executor
