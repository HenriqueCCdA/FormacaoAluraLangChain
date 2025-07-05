from langchain.agents import AgentExecutor
from agente import AgenteOpenAIFunctions


pergunta = "Quais os dados da Ana e da Bianca?"
pergunta = "Crie um perfil acadêmico para a Bianca (Alemanha)!"
pergunta = "Compare o perfil acadêmico da Ana com o da Bianca!"
pergunta = "Tenho sentido Ana desanimada com cursos de matemática. Seria uma boa parear ela com a Bianca?"
pergunta = "Quais os dados da USP?"
pergunta = "Quais os dados da unNiCAmP?"
pergunta = "Dentre USP e UFRJ, qual você recomenda parrra a acadêmica Ana"
pergunta = "Dentre uni camp e USP, qual você recomenda para a Ana?"
pergunta = "Dentre todas as faculdades existentes, quais Ana possui mais chance de entrar?"

agente = AgenteOpenAIFunctions()
executor = AgentExecutor(agent=agente.agente, tools=agente.tools, verbose=True)

resposta = executor.invoke({'input': pergunta})

print(resposta)
