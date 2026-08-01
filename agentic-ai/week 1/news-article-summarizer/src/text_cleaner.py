import re

class TextCleaner:
    
    def remove_html_tags(self, article: str) -> str:
         # Matches any text between < and > non-greedily and replaces it with nothing
        clean_text = re.sub(r'<.*?>', '', article)
        return clean_text
        
    def remove_urls(self, article: str) -> str:
        # Matches http://, https://, and www. links followed by non-whitespace characters
        pattern = r'https?://\S+|www\.\S+'
        clean_text = re.sub(pattern, '', article)
        return clean_text
        
    def remove_special_characters(self, article: str) -> str:
        # Matches anything that is NOT a letter, number, or space
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', article) 
        return clean_text
        
    def remove_spaces(self, article: str) -> str:
        # 1. Collapse multiple spaces/tabs on the same line into one space
        article = re.sub(r'[ \t]+', ' ', article)
        
        # 2. Collapse three or more consecutive newlines into a single or double newline
        # This removes empty lines while keeping paragraph structures intact
        article = re.sub(r'\n{3,}', '\n\n', article)
        
        # 3. Strip trailing/leading spaces from the start and end of every line
        article = '\n'.join(line.strip() for line in article.splitlines())
        return article.strip()