from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

#1.โหลดเอกสาร
loader = TextLoader("data.txt",encoding="utf-8")
documents = loader.load()
# print(documents)

#2.แบ่งข้อมูลออกเป็นชิ้นส่วนย่อย
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

#3. ตัวแปลงข้อมูลเป็นเวกเตอร์
embedding = OpenAIEmbeddings()

#4.เก็บข้อมูลลงใน vector store
vectorstore = FAISS.from_documents(chunks,embedding)

#5. ตัวดึงข้อมูลจาก store ไปใช้งาน
retrievers = vectorstore.as_retriever()

#prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system","ใช้ข้อมูลจากเอกสารในการตอบคำถามให้สั้นกระชับด้วยความสุภาพเป็นกันเอง"),
    ("human","คำถาม : {question} , ข้อมูลที่เกี่ยวข้อง : {context}")
])

# model 
llm = ChatOpenAI(model="gpt-4o-mini")

#chain 
rag_chain = (
    {"context":retrievers,"question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)

result = rag_chain.invoke("มีสินค้าและบริการอะไรบ้าง")
print(result)