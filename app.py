import os
from apikey import apikey

import streamlit 
from langchain import OpenAI

os.environ['OPENAI_API_KEY']=apikey

streamlit.title('Youtube Video App')
prompt=streamlit.text_input('Your prompt')

llm=OpenAI(temperature=0.9)
if prompt:
    response=llm(prompt)
    streamlit.write(response)