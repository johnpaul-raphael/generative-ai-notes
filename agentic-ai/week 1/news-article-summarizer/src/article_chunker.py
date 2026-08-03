from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

class ArticleChunker:
    
    def chunk(self, article: str) -> list[str]:
        
        if article:
            length = len(article)
            if length < 500:
                return [article]
            elif length > 500 and length < 1000:
                # A fixed-size chunking strategy is a text-splitting method 
                # that divides documents into uniform segments based on a preset number 
                # of characters, words, or tokens
                return self._fixed_text_splitter(article=article)
                
            else:
                # Recursive Chunking
                return self._recursive_text_splitter(article=article)

    
    def _fixed_text_splitter(self, article) -> list[str]:
        print("fixed size text splitter")
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        fixed_chunks = splitter.split_text(article)
        return fixed_chunks
    
    def _recursive_text_splitter(self, article: str) -> list[str]:
        # It splits text using a tiered list of separators
        # (like paragraphs \n\n, lines \n, spaces  , and characters  )
        # to keep related text like sentences and paragraphs together
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50, separators=["\n\n","\n"])
        chunks = splitter.split_text(article)
        print(f"recursive text splitter chunker completed and chunk size: {len(chunks)}")
        return chunks