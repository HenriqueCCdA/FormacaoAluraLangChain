from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser

import os
from dotenv import load_dotenv

load_dotenv()
set_debug(True)

class Destino(BaseModel):
    cidade = Field("cidada a visitar")
    motivo = Field("motivo pelo qual é interessante visitar")

paseador = JsonOutputParser(pydantic_object=Destino)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv('OPENAI_API_KEY'),
)

modelo_cidade = PromptTemplate(
    template="""Sugira um cidade dado meu interesse por {interesses}
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={
        "formatacao_de_saida": paseador.get_format_instructions()
    },
)

modelo_restaurantes = PromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = PromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

cadeia_cidade = LLMChain(prompt=modelo_cidade, llm=llm)
cadeia_restaurantes = LLMChain(prompt=modelo_restaurantes, llm=llm)
cadeia_cultural = LLMChain(prompt=modelo_cultural, llm=llm)

cadeia = SimpleSequentialChain(
    chains=[
        cadeia_cidade, cadeia_restaurantes, cadeia_cultural
    ],
    verbose=True,
)

resultado = cadeia.invoke("praias")
print(resultado)
