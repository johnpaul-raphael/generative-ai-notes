import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

class ArticleSummarizer:
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.open_ai = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.3)
        
    def summarize(self, article: str):
        
        template = ChatPromptTemplate.from_messages([
            ("system", "Think as your a news article summarize expert. I want to summarize article with 1 to 2 lines."),
            ("human", "here is my article: {article}")]
        )
        
        output_format = StrOutputParser()
        
        llm = template | self.open_ai | output_format
        
        response = llm.invoke(article)
        return response