from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile

# from dotenv import load_dotenv
# import os

if __name__ == '__main__':

    # {information} is the parameter into the prompt
    summary_template = """
        Dada a informação do Linkedin {information} sobre uma pessoa de Eu quero que você crie:
        1. Um breve resumo
        2. Dois fatos interessantes sobre a pessoa
"""

summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm = ChatOllama(model="llama3")

#calling the ChatGPT API
chain = summary_prompt_template | llm | StrOutputParser()
linkedin_data = scrape_linkedin_profile(linkedin_profile_url="", mock=True)
res = chain.invoke(input={"information": linkedin_data})

print(res)