import os
from langchain.document_loaders import YoutubeLoader


import streamlit 
from langchain import OpenAI
from apikey import apikey
# import openai


from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


os.environ['OPENAI_API_KEY']=apikey


loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=Cm_uIxcczWM&ab_channel=CaseyNeistat", add_video_info=True
)

docs=loader.load()
transcribedText=docs[0].page_content
text=transcribedText
print(transcribedText)

# Split them
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(text)

# Build an index
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_texts(splits, embeddings)

# Build a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

streamlit.title('Youtube Video App')
prompt=streamlit.text_input('Your Query')
# Ask a question!
query = "what are the tips mentioned?"
answer=qa_chain.run(query)
print(answer)

# def ask_question(prompt, context):
#     response = openai.Completion.create(
#         engine='text-davinci-003',
#         prompt=prompt,
#         max_tokens=100,
#         temperature=0.7,
#         n=1,
#         stop=None,
#         context=context
#     )
#     return response.choices[0].text.strip()



llm=OpenAI(temperature=0.9)
if prompt:
    response=ask_question(prompt,transcript[0])
    streamlit.write(response)