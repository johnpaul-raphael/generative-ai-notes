class ArticleValidator:
    
    def is_valid_length(self, article: str) -> bool:
        return len(article.strip()) > 0  # Returns True if not empty, False if has no content

            
        