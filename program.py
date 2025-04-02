from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,CommaSeparatedListOutputParser
from dotenv import load_dotenv

load_dotenv()

#prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system","คุณเป็น {expertise}"),
    ("human","แนะนำเมนู {menu} จำนวน {amount} รายการ")
])

# สร้างโมเดล
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)

# สร้าง Chain 
chain = prompt | llm | CommaSeparatedListOutputParser()

# print(chain)

# #เรียกใช้งาน Chain
response = chain.invoke({"expertise":"เชฟอาหารไทย","menu":"อาหารเหนือ","amount":5})

print(response[1])
print(response[2])