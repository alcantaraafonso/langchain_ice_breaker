import os
from dotenv import load_dotenv
from my_tools.tools import get_profile_url_tavily


load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)

# to download pre made prompts
from langchain import hub

def lookup(name: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
        Your answer should contain only a URL"""
    
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["name_of_person"]
    )
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need to get the Linkedin Page URL"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    #é como a receita do que deve ser feito
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    #É a execução propriamente dita, inclusive com a execução da função em Python
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )
    
    
    return result["output"]

if __name__ == "__main__":
    linkedin_url = lookup(name="Afonso Alcantara")
    print(linkedin_url)