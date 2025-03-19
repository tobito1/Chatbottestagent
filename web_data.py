# web_data.py
import logging
import csv
import time
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

logger = logging.getLogger(__name__)

class WebDataCollector:
    def __init__(self):
        # Initialize and cache the summarization pipeline once
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    def collect_web_data(self, url):
        logger.info(f"Collecting data from URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            logger.error(f"Error fetching data from {url}: {e}")
            return ""
    
    def chunk_text(self, text, max_chunk_words=400):
        words = text.split()
        return [' '.join(words[i:i+max_chunk_words]) for i in range(0, len(words), max_chunk_words)]
    
    def summarize_content(self, text, max_length=20, min_length=10):
        logger.info("Summarizing content.")
        try:
            chunks = self.chunk_text(text)
            summaries = []
            for chunk in chunks:
                if chunk.strip():
                    # Use the cached summarizer pipeline
                    summary_list = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                    summaries.append(summary_list[0]['summary_text'])
            combined_summary = " ".join(summaries)
            return combined_summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""
    
    def process_urls_from_csv(self, input_csv, output_csv, summarize_flag=True):
        results = []
        try:
            with open(input_csv, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                if 'url' not in reader.fieldnames:
                    raise ValueError("CSV file must contain a column named 'url'.")
                for row in reader:
                    url = row['url'].strip()
                    if url:
                        text = self.collect_web_data(url)
                        summary = self.summarize_content(text) if summarize_flag and text else ""
                        results.append({'url': url, 'content': text, 'summary': summary})
        except Exception as e:
            logger.error(f"Error processing CSV {input_csv}: {e}")
        
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
                fieldnames = ['url', 'content', 'summary']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"Processed {len(results)} URLs. Data saved to {output_csv}")
        except Exception as e:
            logger.error(f"Error writing output CSV {output_csv}: {e}")
