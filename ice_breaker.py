from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import summary_parser

from dotenv import load_dotenv
import os

def ice_breaker_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=True)

    summary_template = """
        Dada a informação do Linkedin {information} sobre uma pessoa de Eu quero que você crie:
        1. Um breve resumo
        2. Dois fatos interessantes sobre a pessoa

        \n{format_instructions}
        """

    # summary_prompt_template = PromptTemplate(input_variables=["information"], 
    #     template=summary_template,
    #     partial_variables={"format_instructions": "Por favor, siga o formato: \n\nResumo: \nFato 1: \nFato 2: \n\n"})
    summary_prompt_template = PromptTemplate(input_variables=["information"], 
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()})
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    res = chain.invoke(input={"information": linkedin_data})

if __name__ == '__main__':
    load_dotenv()

    print("Ice Breaker")

    ice_breaker_with("Eden Marco")
    # {information} is the parameter into the prompt


    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # llm = ChatOllama(model="llama3")

    # #calling the ChatGPT API
    # chain = summary_prompt_template | llm | StrOutputParser()
    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url="", mock=True)
    # res = chain.invoke(input={"information": linkedin_data})

    # print(res)