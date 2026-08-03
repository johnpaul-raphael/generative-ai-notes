import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

class ArticleSummarizer:
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.open_ai = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.3)
        
    def _summarize(self, article: str):
        
        template = ChatPromptTemplate.from_messages([
            ("system", "Think as your a news article summarize expert. I want to summarize article with 1 line."),
            ("human", "here is my article: {article}")]
        )
        
        output_format = StrOutputParser()
        
        llm = template | self.open_ai | output_format
        
        response = llm.invoke(article)
        return response
    
    def summarize_article_concurrent(self, articles: list[str]) -> str:
        """Process multiple items concurrently and maintain order"""
        print("calling concurrent summarize article")
        results = [None] * len(articles)  # Pre-allocate list to maintain order
        future_to_index = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all articles to workers
            for index, article in enumerate(articles):
                future = executor.submit(self._summarize, article)
                future_to_index[future] = index
            
            # Collect results as they complete
            for future in future_to_index:
                index = future_to_index[future]
                results[index] = future.result()
        
        print("concurrent summary completed final summary started")
        combined_summary = ""
        for result in results:
            combined_summary += result + ".\n"

        return self._summarize(combined_summary)
        
        
