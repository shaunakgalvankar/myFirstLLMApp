import os
from apikey import apikey
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

import streamlit 
from langchain import OpenAI
# import openai

os.environ['OPENAI_API_KEY']=apikey

videoID='h9xaHwBsRUw'
transcript=YouTubeTranscriptApi.get_transcript(videoID)

formatter=JSONFormatter()
text_formatted=formatter.format_transcript(transcript)

print(text_formatted)

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

streamlit.title('Youtube Video App')
prompt=streamlit.text_input('Your prompt')

llm=OpenAI(temperature=0.9)
if prompt:
    response=ask_question(prompt,transcript[0])
    streamlit.write(response)