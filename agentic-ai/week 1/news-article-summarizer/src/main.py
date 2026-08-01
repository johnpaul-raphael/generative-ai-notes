from text_cleaner import TextCleaner
from article_validator import ArticleValidator
from article_summarizer import ArticleSummarizer


class NewsArticleSummarizerApp:
    
    def __init__(self):
        self.validator = ArticleValidator()
        self.cleaner = TextCleaner()
        self.summarizer = ArticleSummarizer()

    # input
    # validate input
    # if validation looks good then 
    #       process input file to remove invalid letters using regex
    #       Once we get the only the string
    #           check if the string is within 500
    #               if yes the no chunking directly call llm for summarise
    #           check if the string is greater than 500 and within 2000
    #               if yes then do 2 to 3 chunk and then call llm for summary and then append the summary and do final summary
    #           check if the string is greater than 2000 and with in 50000
    #               if yes then do dynamic chunking and then call llm for summary and then append the summary and do final summary
    
    # finally format the output and send as a response
    
    def run(self, article: str) -> None:
        is_valid = self.validator.is_valid_length(article)
        if is_valid:
            article = self.cleaner.remove_html_tags(article=article)
            print(f"after remove html tag: {article}")
            article = self.cleaner.remove_urls(article=article)
            print(f"after remove urls: {article}")
            article = self.cleaner.remove_special_characters(article=article)
            print(f"after remove special char: {article}")
            article = self.cleaner.remove_spaces(article=article)
            print(f"after remove spaces: {article}")
            print(f"after remove all special character the article would be {article}")

            # calling llm for summary
            print("="*50)
            print()
            
            # chunking process
            # if len(article) < 500:
            # elif len(article) > 500:
            
            print("="*50)
            print()
            print("calling llm for summary")
            print()
            result = self.summarizer.summarize(article=article)
            print(f"llm response: {result}")
        else:
            raise ValueError("invalid article")
def main():
    app = NewsArticleSummarizerApp()
    sample_article = """
        <h1>Breaking News!!!</h1>   

        Scientists    at NASA  announced   today that they've discovered   a new exoplanet.
            
        The discovery was made using the <b>James Webb</b> telescope. Read more at https://www.nasa.gov/exoplanet-discovery or visit www.space.com/news for details!!!


        <p>According to Dr. Smith (@nasa_official), "This is #groundbreaking research!!"</p>

        
            
        Contact us at: info@nasa.gov *** for more info $$$ ###



        <div class="footer">Copyright © 2026 - All rights reserved</div>
        """
        
    sample_article_800 = """
        Artificial Intelligence Transforms Healthcare Industry

        Artificial intelligence is revolutionizing the healthcare industry in ways that were once thought impossible. From diagnostic imaging to drug discovery, AI technologies are enabling medical professionals to deliver better patient outcomes while reducing costs and improving efficiency across the board.

        The integration of machine learning algorithms into clinical decision support systems has proven remarkably effective. Hospitals worldwide are now deploying AI-powered tools that analyze medical imaging, such as X-rays, MRIs, and CT scans, with accuracy rates that often match or exceed human radiologists. These systems can detect tumors, fractures, and other abnormalities in seconds, allowing doctors to focus on treatment planning rather than spending hours reviewing images manually.

        One of the most promising applications of AI in healthcare is personalized medicine. By analyzing vast amounts of patient data, including genetic information, medical history, and lifestyle factors, AI algorithms can predict disease risk and recommend tailored treatment plans. This approach has shown particular success in oncology, where AI systems help oncologists select the most effective cancer treatments based on individual tumor characteristics and patient profiles.

        Drug discovery represents another transformative area. Traditionally, bringing a new pharmaceutical drug to market took 10-15 years and cost billions of dollars. AI is accelerating this process dramatically by identifying promising drug candidates in a fraction of the time. Machine learning models can screen millions of molecular combinations and predict their effectiveness against specific diseases, dramatically reducing the number of failed experiments and failed trials that pharmaceutical companies must conduct.

        Administrative efficiency has also improved significantly through AI implementation. Healthcare providers are using natural language processing to extract relevant information from patient records, automating billing processes, and scheduling appointments more intelligently. These systems reduce paperwork burden on medical staff, allowing them to dedicate more time to direct patient care.

        However, the integration of AI into healthcare is not without challenges. Data privacy and security remain paramount concerns, as AI systems require access to sensitive patient information. Healthcare organizations must implement robust cybersecurity measures to protect against breaches and ensure compliance with regulations like HIPAA. Additionally, there are concerns about algorithmic bias, where AI systems trained on non-representative data might perform poorly for certain patient populations.

        The regulatory landscape for AI in healthcare continues to evolve. The FDA has established pathways for approving AI-based medical devices, but questions remain about how to ensure these systems remain accurate and safe as they're deployed in real-world settings. Healthcare providers and AI developers must work together with regulators to establish clear standards and validation procedures.

        Training and adoption present additional hurdles. Many healthcare professionals grew up in an era before AI and may feel hesitant about trusting automated systems with clinical decisions. Medical schools and continuing education programs are beginning to incorporate AI literacy into their curricula, ensuring that future doctors understand both the capabilities and limitations of these powerful tools.

        Despite these challenges, the momentum behind AI adoption in healthcare shows no signs of slowing. Investment in healthcare AI startups reached record levels, with billions of dollars flowing into companies developing innovative solutions. Major technology companies like Google, IBM, and Microsoft are all investing heavily in healthcare AI initiatives, recognizing both the enormous potential and the lucrative market opportunity.

        Looking forward, experts predict that AI will become increasingly integrated into every aspect of healthcare delivery. From predicting patient deterioration in real-time to optimizing hospital resource allocation, AI systems will handle routine tasks, freeing healthcare professionals to focus on complex cases requiring human judgment and empathy. The future healthcare system will likely be a seamless collaboration between human clinicians and intelligent machines, each playing to their strengths.

        The ethical implications of AI in healthcare deserve serious consideration. As these systems become more powerful and widespread, questions about accountability, transparency, and patient autonomy become more pressing. Who is responsible if an AI system makes a mistake? How can patients understand and trust the recommendations made by algorithms they cannot see? These are conversations that healthcare institutions, technologists, and society at large must continue to have.

        In conclusion, artificial intelligence is poised to be one of the most significant innovations in modern healthcare. By augmenting human expertise, automating routine tasks, and enabling personalized medicine, AI has the potential to save millions of lives and improve quality of life for billions of people worldwide. While challenges remain in areas like regulation, bias mitigation, and workforce adaptation, the benefits are too substantial to ignore. The healthcare industry's future will undoubtedly be shaped by how successfully we integrate and govern AI technologies in service of better patient care.
        """

    app.run(sample_article_800)


if __name__ == "__main__":
    main()