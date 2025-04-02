from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# สร้างโมเดล
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
# tools
tools = {"type":"web_search_preview"}
#ติดเครื่องมือใน AI
llm_search = llm.bind_tools([tools])
#เรียกใช้งาน Model
response = llm_search.invoke("นายกรัฐมนตรีคนล่าสุดของประเทศไทยชื่อว่าอะไร")

print(response)