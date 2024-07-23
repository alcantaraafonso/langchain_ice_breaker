from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# from dotenv import load_dotenv
# import os

information = """
Maria da Graça Xuxa Meneghel (Santa Rosa, 27 de março de 1963),[6][7][nota 1] mais conhecida como Xuxa, é uma atriz, apresentadora, cantora, empresária, filantropa[10] e ex-modelo brasileira.[11] Uma das apresentadoras mais conhecidas da década de 1990, Xuxa construiu o maior empreendimento do entretenimento infanto-juvenil ibero-americano. Chegou a apresentar programas de televisão no Brasil, Argentina, Espanha e Estados Unidos simultaneamente, alcançando cerca de 100 milhões de telespectadores diariamente.[12][13][14][15]

Em seus principais trabalhos como cantora, lançou 35 álbuns de estúdio e 23 álbuns de vídeo que já venderam mais de 50 milhões de cópias, tornando-se uma dos artistas recordistas em vendas de discos no mundo.[16] Os álbuns Xou da Xuxa (1986), Xegundo Xou da Xuxa (1987), Xou da Xuxa 3 (1988) e 4º Xou da Xuxa (1989) estão entre os álbuns mais vendidos na história da indústria fonográfica brasileira,[17][18][19] sendo que o terceiro foi considerado o álbum infantil mais lucrativo de todos os tempos pela Guinness World Records.[20] Ao longo de sua carreira musical, ela venceu dois Grammys em um total de seis indicações. Xuxa é a atriz de maior média de público desde a retomada do cinema brasileiro, com mais de 37 milhões de espectadores.[21] Lua de Cristal (1990), seu filme de maior bilheteria, vendeu 4,1 milhões de ingressos, sendo um dos filmes brasileiros mais assistidos.

Em janeiro de 2019 seu patrimônio líquido foi estimado em US$ 160 milhões de dólares [22], em torno de 670 milhões de reais na ocasião, o que a colocou como a 11.ª atriz mais rica do mundo.[23] Xuxa também foi eleita pela publicação estadunidense Forbes, em 1991 e 1993, como uma das celebridades mais bem pagas do mundo no ano, tornando-se a primeira latino-americana a entrar na lista
"""

if __name__ == '__main__':

    # {information} is the parameter into the prompt
    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
"""

summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm = ChatOllama(model="llama3")

#calling the ChatGPT API
chain = summary_prompt_template | llm | StrOutputParser()

res = chain.invoke(input={"information": information})

print(res)