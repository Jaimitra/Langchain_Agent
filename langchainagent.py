from langchain_ollama import ChatOllama
import streamlit as st
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.globals import set_debug

llm = ChatOllama(model="llama3.2:3b")

set_debug(True)
prompt = '''You are an agent that answer the queries of user by using the tools provided to you.
                     Here based on user query choose the most suitable tool to get the answer.
                    '''
  
tools = load_tools(["wikipedia","ddg-search"])


agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt = prompt
)

st.title("LangChain Agents")
query = st.text_input("Enter query:")
if query:
    result = agent.invoke(
        {
            "messages":{
                "role":"user",
                "content":query
            }
        }
    )
   
    response = result["messages"][-1]
    st.write(response.content)
    # print(result)
    # print(response)