from langchain.agents import AgentExecutor
from agente import AgenteOpenAIFunctions


pergunta = "Quais os dados da Ana e da Bianca?"
pergunta = "Crie um perfil acadêmico para a Bianca (Alemanha)!"

agente = AgenteOpenAIFunctions()
executor = AgentExecutor(agent=agente.agente, tools=agente.tools, verbose=True)

resposta = executor.invoke({'input': pergunta})

print(resposta)
