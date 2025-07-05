from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryMemory
from langchain.chains.conversation.base import ConversationChain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()
set_debug(True)


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv('OPENAI_API_KEY'),
)

carregador = TextLoader("GTB_gold_Nov23.txt")
documentos = carregador.load()

quebrador = CharacterTextSplitter(chunk_size=1_000)
textos = quebrador.split_documents(documentos)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(textos, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

pergunta = "Como devo proceder caso tenha um item comprado roubado"
resultado = qa_chain.invoke({ "query" : pergunta})
print(resultado)
