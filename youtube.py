from langchain.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=Cm_uIxcczWM&ab_channel=CaseyNeistat", add_video_info=True
)

docs=loader.load()
transcribedText=docs[0].page_content