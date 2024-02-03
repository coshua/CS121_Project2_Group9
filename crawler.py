import os
import io
import logging
from logging.handlers import RotatingFileHandler
import re
from urllib.parse import urlparse, urljoin, urldefrag, parse_qs
from lxml import html
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')  # For tokenization
nltk.download('stopwords') 

log_directory = "logs"
info_log_file = os.path.join(log_directory, "info_crawler.log")
debug_log_file = os.path.join(log_directory, "debug_crawler.log")
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

info_file_handler = RotatingFileHandler(info_log_file, maxBytes=1024*1024*5, backupCount=5)
info_file_handler.setLevel(logging.INFO)
logger.addHandler(info_file_handler)

class DebugOnlyFilter(logging.Filter):
    def filter(self, record):
        # Only log messages with DEBUG level
        return record.levelno == logging.DEBUG
debug_filter = DebugOnlyFilter()
debug_file_handler = RotatingFileHandler(debug_log_file, maxBytes=1024*1024*5, backupCount=5)
debug_file_handler.setLevel(logging.DEBUG)  # Set to DEBUG
debug_formatter = logging.Formatter('%(levelname)s - %(message)s')
debug_file_handler.setFormatter(debug_formatter)
debug_file_handler.addFilter(debug_filter)
logger.addHandler(debug_file_handler)

STOP_WORDS = set({
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
    "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 
    'each', 'few', 'for', 'from', 'further', 
    'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 
    'him', 'himself', 'his', 'how', "how's", 
    'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 
    "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
    'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
    'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 
    'this', 'those', 'through', 'to', 'too', 
    'under', 'until', 'up', 'very', 
    'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 
    'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
})

class Crawler:
    """
    This class is responsible for scraping urls from the next available link in frontier and adding the scraped links to
    the frontier
    """

    def __init__(self, frontier, corpus):
        self.frontier = frontier
        self.corpus = corpus
        self.subdomain_url_count = defaultdict(int)
        self.valid_urls = set()
        self.invalid_urls = set()
        self.static_urls = defaultdict(int)
        self.traps = set()
        self.max_valid_link = [0, None] # [Num of most valid links, page url]
        self.page_most_words = [0, None] # [Num of words in the page, page url]
        self.word_counter = Counter() # Find the 50 most frequent words
        
    def write_analysis(self, directory, filename, contents):
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, filename)
        try:
            with io.open(path, 'w', encoding='utf-8', buffering=1024 * 1024) as f:
                if isinstance(contents, list) or isinstance(contents, set):
                    f.write(f'number of items: {len(contents)}\n')
                    for item in contents:
                        f.write(f'{item}\n')
                elif isinstance(contents, dict):
                    f.write(f'number of items: {len(contents)}\n')
                    for item in contents:
                        f.write(f'{item}, {contents[item]}\n')
                else:
                    f.write(contents)
        except Exception as e:
            print(f'Error occurred while writing txt outputs: {filename}, {e}')

    def count_words(self, word_counter, content):
        words = word_tokenize(content)
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in STOP_WORDS]
        word_counter.update(words)

    def start_crawling(self):
        """
        This method starts the crawling process which is scraping urls from the next available link in frontier and adding
        the scraped links to the frontier
        """
        while self.frontier.has_next_url():
            url = self.frontier.get_next_url()
            logger.info("%s ... Fetched: %s, Queue size: %s", url, self.frontier.fetched, len(self.frontier))
            
            url_data = self.corpus.fetch_url(url)
            for next_link in self.extract_next_links(url_data):
                if self.is_valid(next_link):
                    # Keep track of the subdomains visited
                    parsed = urlparse(next_link)
                    subdomain = parsed.hostname
                    self.subdomain_url_count[subdomain] += 1

                    # List of Downloaded URLs
                    self.valid_urls.add(next_link)

                    try:
                        content = url_data['content']
                        text_content = html.document_fromstring(content).text_content()
                        word_count = len(text_content.split())
                        if word_count > self.page_most_words[0]:
                            self.page_most_words = [word_count, next_link]
                        
                        # Count the word frequencies in the valid page
                        self.count_words(self.word_counter, text_content)
                    except Exception:
                        print("Exception occured")

                    if self.corpus.get_file_name(next_link) is not None:
                        self.frontier.add_url(next_link)

                else:
                    self.invalid_urls.add(next_link)


        print(self.subdomain_url_count)
        print(f'The page with most valid out links: {self.max_valid_link}')
        print(f'The page with most words: {self.page_most_words}')
        self.write_analysis("analysis", "ValidURLs.txt", self.valid_urls)
        self.write_analysis("analysis", "InvalidURLs.txt", self.invalid_urls)
        self.write_analysis("analysis", "Subdomains_Count.txt", self.frontier.subdomain_count)
        self.write_analysis("analysis", "Traps.txt", self.traps)
        
        most_common_words = [f'{word}, {cnt}' for word, cnt in self.word_counter.most_common(50)]
        self.write_analysis("analysis", "frequentWords.txt", most_common_words)
        self.write_analysis("analysis", "PagewithMostLinks.txt", self.max_valid_link)
        self.write_analysis("analysis", "PagewithMostWords.txt", self.page_most_words)
    def extract_next_links(self, url_data):
        """ 
        The url_data coming from the fetch_url method will be given as a parameter to this method. url_data contains the
        fetched url, the url content in binary format, and the size of the content in bytes. This method should return a
        list of urls in their absolute form (some links in the content are relative and needs to be converted to the
        absolute form). Validation of links is done later via is_valid method. It is not required to remove duplicates
        that have already been fetched. The frontier takes care of that.

        Suggested library: lxml
        """
        
        if url_data['content'] is None or url_data['http_code'] == 404:
            return []
        
        base_url = url_data['final_url'] if url_data['is_redirected'] else url_data['url']

        # Convert binary content to string
        try:
            content = url_data['content']
            document = html.fromstring(content, base_url=base_url)
        except Exception as e:
            # Handle decoding error if any
            print(f'Failed to decode content: {e}')
            return []

        # Parse the content using lxml
        
        
        # Extract absolute links
        extracted_links = []
        for element, attribute, link, pos in document.iterlinks():
            if element.tag == 'a' and 'href' in attribute:
                # Normalize the link and add to the list
                normalized_link = urljoin(base_url, link)
                extracted_links.append(normalized_link)
        
        if len(extracted_links) > self.max_valid_link[0]:
            self.max_valid_link = [len(extracted_links), base_url]
        return extracted_links


    def is_valid(self, url):
        """
        Function returns True or False based on whether the url has to be fetched or not. This is a great place to
        filter out crawler traps. Duplicated urls will be taken care of by frontier. You don't need to check for duplication
        in this method
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme not in set(["http", "https"]):
                logger.debug(f"This URL is not http or https: {url}")
                return False
        
            if ".ics.uci.edu" not in parsed.hostname:
                logger.debug(f"This URL is not a subdomain of ics.uci.edu {url}")
                return False
            
            # Filter out URL with any query
            url, _ = urldefrag(url)
            # Check for repeating subdirectories
            path_parts = [part for part in parsed.path.split('/') if part]
            if len(path_parts) != len(set(path_parts)):
                logger.debug(f"This URL contains repetitive subdirectories: {url} with {path_parts}")
                return False
            
            # Check for excessively long URLs (longer than a reasonable limit, e.g., 200 characters)
            if len(url) > 200:
                logger.debug(f"This URL is excessively long: {url}.")
                return False
            
            query_params = parse_qs(parsed.query)
            if len(query_params) > 10:
                logger.debug(f"This URL has too many query parameters: {url}")
                return False
            
            static_portion = parsed.scheme + "://" + parsed.netloc + parsed.path
            self.static_urls[static_portion] += 1
            if self.static_urls[static_portion] > 1000:
                self.traps.add(static_portion)
                logger.debug(f"This URL has been visited more than 1000 times, possible trap : {url}")
                return False
            
            
            return ".ics.uci.edu" in parsed.hostname \
                    and not re.match(".*\.(css|js|bmp|gif|jpe?g|ico" + "|png|tiff?|mid|mp2|mp3|mp4" \
                                    + "|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf" \
                                    + "|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso|epub|dll|cnf|tgz|sha1" \
                                    + "|thmx|mso|arff|rtf|jar|csv" \
                                    + "|rm|smil|wmv|swf|wma|zip|rar|gz|pdf)$", parsed.path.lower())
        except TypeError:
            print("TypeError for ", parsed)
            return False
