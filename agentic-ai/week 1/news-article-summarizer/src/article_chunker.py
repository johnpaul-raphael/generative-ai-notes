from langchain_text_splitters import CharacterTextSplitter

class ArticleChunker:
    
    def __init__(self):
        pass
    
    def chunk(self, article: str) -> list[str]:
        
        if article:
            length = len(article)
            if length < 500:
                return [article]
            elif length > 500 and length < 1000:
                # A fixed-size chunking strategy is a text-splitting method 
                # that divides documents into uniform segments based on a preset number 
                # of characters, words, or tokens
                self._fixed_text_splitter(article=article)
                
            elif length > 1000 and length < 2000:
                # 
        
    def _fixed_text_splitter(self, article) -> list[str]:
        print("fixed size text splitter")
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        fixed_chunks = splitter.split_text(article)
        return fixed_chunks 