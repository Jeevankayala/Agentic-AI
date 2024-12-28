from phi.agent import Agent
from phi.tools.googlesearch import GoogleSearch
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

websearch_agent = Agent(
        name = "web search Agent",
        role = "search the web for the information",
        model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools = [GoogleSearch(), DuckDuckGo()],
        instructions=['Always include sources'],
        show_tool_calls=True,
        markdown=True
)


financial_agent = Agent(
        name = "Finance AI Agent",
        model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
        instructions=['Use tables to display the data'],
        show_tool_calls=True,
        markdown=True
)

combined_final_agent = Agent(
        team = [websearch_agent, financial_agent],
        model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        instructions = ["Always include sources", "Use tables to display the data"],
        show_tool_calls=True,
        markdown=True

)


combined_final_agent.print_response("Summarize analyst recommendation and share the latest news for Nvidia.",stream=True)
