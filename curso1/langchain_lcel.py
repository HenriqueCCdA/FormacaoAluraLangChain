from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

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

parte1 = modelo_cidade | llm | paseador
parte2 = modelo_restaurantes | llm | StrOutputParser()
parte3 = modelo_cultural | llm | StrOutputParser()

cadeia = parte1 | {
    "restaurantes": parte2,
    "locais_culturais": parte3
}

resultado = cadeia.invoke({"interesses": "praias"})
print(resultado)
