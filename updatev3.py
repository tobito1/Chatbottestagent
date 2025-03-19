import os
import ssl
import csv
import time
import requests
import json
import random
import logging
from itertools import permutations
from difflib import SequenceMatcher

import nltk
from nltk.corpus import wordnet

from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from transformers import pipeline
from transformers import pipeline as hf_pipeline

# Incomplete LangChain integration (keep as-is for now)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document as LC_Document

# Configure SSL to allow unverified HTTPS context if needed
ssl._create_default_https_context = ssl._create_unverified_context

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('wordnet')

# -----------------------------
# Configuration Class
# -----------------------------
class Config:
    """Configuration class that loads sensitive data and parameters from environment variables."""
    CHATBOT_ENDPOINT = os.getenv("CHATBOT_ENDPOINT", "https://rag.myscheme.in/rag-service/v3/chat")
    APP_ID = os.getenv("CHATBOT_APP_ID", "6t4rIofAaIAEgu2P9lmtD")
    APP_TYPE = os.getenv("CHATBOT_APP_TYPE", "system")
    ENTITY_ID = os.getenv("CHATBOT_ENTITY_ID", "64b15bf4c3a58e12cb335ec0")
    SESSION_ID = os.getenv("CHATBOT_SESSION_ID", "BmytObehoRGNNhZw5vNDC")
    AUTHORIZATION = os.getenv("CHATBOT_AUTHORIZATION", 
                              "Basic bXlzY2hlbWVfcmFnX3NlcnZpY2U6a2pqZ2V0dXBieDg5MDY3ODc4")
    X_API_KEY = os.getenv("CHATBOT_X_API_KEY", "MEpg04&*@%^Vng!KL")
    X_FP_ID = os.getenv("CHATBOT_X_FP_ID", "b373adbbe0a20d818712bcd8a3e95bfa4bddf68bd0261146ac879b05851ddf8d")
    ORIGIN = os.getenv("CHATBOT_ORIGIN", "https://aistore.myscheme.in")
    CSV_URL_INPUT = os.getenv("CSV_URL_INPUT", "url.csv")
    CSV_URL_OUTPUT = os.getenv("CSV_URL_OUTPUT", "CollectedWebData.csv")
    CSV_TEST_INPUT = os.getenv("CSV_TEST_INPUT", "Myagentdata.csv")
    CSV_TEST_DYNAMIC = os.getenv("CSV_TEST_DYNAMIC", "DynamicTestCases.csv")
    CSV_LOG_FILE = os.getenv("CSV_LOG_FILE", "test_results.csv")
    CSV_ADAPTED_TESTS = os.getenv("CSV_ADAPTED_TESTS", "AdaptedDynamicTests.csv")
    DEMO_URL = os.getenv("DEMO_URL", "https://www.example.com")
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "2"))
    DELAY_BETWEEN_ATTEMPTS = int(os.getenv("DELAY_BETWEEN_ATTEMPTS", "1"))
    DELAY_BETWEEN_TESTS = int(os.getenv("DELAY_BETWEEN_TESTS", "5"))
    MAX_QUERIES = int(os.getenv("MAX_QUERIES", "10"))
    EVALUATION_THRESHOLD = float(os.getenv("EVALUATION_THRESHOLD", "0.7"))

# -----------------------------
# Web Data Collection Module
# -----------------------------
class WebDataCollector:
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
        chunks = []
        for i in range(0, len(words), max_chunk_words):
            chunk = ' '.join(words[i:i+max_chunk_words])
            chunks.append(chunk)
        return chunks

    def summarize_content(self, text, max_length=20, min_length=10):
        logger.info("Summarizing content.")
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            chunks = self.chunk_text(text)
            summaries = []
            for chunk in chunks:
                if chunk.strip():
                    summary_list = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
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
                        summary = ""
                        if summarize_flag and text:
                            summary = self.summarize_content(text)
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

# -----------------------------
# Test Generation Module
# -----------------------------
class TestGenerator:
    def generate_synonyms(self, sentence):
        words = sentence.split()
        variations = set()
        for i, word in enumerate(words):
            synonyms = wordnet.synsets(word)
            if synonyms:
                for syn in synonyms[:2]:
                    synonym = syn.lemmas()[0].name().replace('_', ' ')
                    new_sentence = words[:i] + [synonym] + words[i+1:]
                    variations.add(' '.join(new_sentence))
        return list(variations)

    def introduce_typos(self, sentence):
        words = sentence.split()
        typo_variations = set()
        for i, word in enumerate(words):
            if len(word) > 3:
                typo_word = list(word)
                swap_idx = random.randint(0, len(typo_word) - 2)
                typo_word[swap_idx], typo_word[swap_idx + 1] = typo_word[swap_idx + 1], typo_word[swap_idx]
                typo_sentence = words[:i] + [''.join(typo_word)] + words[i+1:]
                typo_variations.add(' '.join(typo_sentence))
        return list(typo_variations)

    def generate_permutations(self, sentence):
        words = sentence.split()
        if len(words) > 3:
            permuted_sentences = [' '.join(p) for p in permutations(words, min(4, len(words)))]
            return permuted_sentences[:2]
        return []

    def generate_test_variations(self, input_text):
        variations = set()
        variations.update(self.generate_synonyms(input_text))
        variations.update(self.introduce_typos(input_text))
        variations.update(self.generate_permutations(input_text))
        return list(variations)

    def is_similar(self, a, b, threshold=0.9):
        return SequenceMatcher(None, a, b).ratio() > threshold

    def generate_dynamic_tests(self, input_csv, output_csv, include_original=True):
        new_test_cases = []
        seen = set()
        try:
            if include_original:
                with open(input_csv, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        input_text = row['input'].strip()
                        expected_output = row['expected'].strip()
                        if input_text and input_text not in seen:
                            new_test_cases.append({'input': input_text, 'expected': expected_output})
                            seen.add(input_text)
            with open(input_csv, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    original_text = row['input'].strip()
                    expected_output = row['expected'].strip()
                    variations = self.generate_test_variations(original_text)
                    for variation in variations:
                        variation = variation.strip()
                        if variation and variation != original_text and variation not in seen and not self.is_similar(original_text, variation, threshold=0.9):
                            new_test_cases.append({'input': variation, 'expected': expected_output})
                            seen.add(variation)
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['input', 'expected']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(new_test_cases)
            logger.info(f"Generated {len(new_test_cases)} dynamic test cases saved to {output_csv}")
        except Exception as e:
            logger.error(f"Error generating dynamic tests: {e}")

    def update_dynamic_tests_from_failures(self, log_file, output_csv, failure_threshold=0.5):
        failing_tests = []
        try:
            with open(log_file, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        score = float(row['score'])
                    except ValueError:
                        score = 0.0
                    if row['result'].strip().lower() != 'true' or score < failure_threshold:
                        failing_tests.append({
                            'input': row['input'].strip(),
                            'expected': row['expected'].strip()
                        })
        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {e}")
        
        new_dynamic_tests = []
        seen = set()
        for test in failing_tests:
            input_text = test['input']
            expected = test['expected']
            variations = self.generate_test_variations(input_text)
            for variation in variations:
                variation = variation.strip()
                if variation and variation != input_text and variation not in seen:
                    new_dynamic_tests.append({'input': variation, 'expected': expected})
                    seen.add(variation)
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['input', 'expected']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(new_dynamic_tests)
            logger.info(f"Adapted dynamic test cases generated from failing tests: {len(new_dynamic_tests)} saved to {output_csv}")
        except Exception as e:
            logger.error(f"Error writing adapted dynamic tests to {output_csv}: {e}")

# -----------------------------
# Test Evaluation Module
# -----------------------------
class TestEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate_response(self, actual, expected, method='exact', threshold=0.8):
        if method == 'exact':
            result = actual.strip() == expected.strip()
            score = 1.0 if result else 0.0
            return result, score
        elif method == 'semantic':
            embedding_actual = self.model.encode(actual, convert_to_tensor=True)
            embedding_expected = self.model.encode(expected, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding_actual, embedding_expected)
            similarity_score = similarity.item()
            result = similarity_score >= threshold
            logger.info(f"Normalized actual response: {actual.strip()}")
            logger.info(f"Normalized expected response: {expected.strip()}")
            logger.info(f"Similarity score: {similarity_score}")
            return result, similarity_score
        else:
            raise ValueError("Unknown evaluation method. Please choose 'exact' or 'semantic'.")

    def load_test_cases_from_file(self, file_path):
        test_cases = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test case file '{file_path}' not found.")
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_columns = {'input', 'expected'}
            if not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"CSV file must contain columns: {required_columns}. Found: {reader.fieldnames}")
            for row in reader:
                test_case = {
                    'input': row['input'].strip(),
                    'expected': row['expected'].strip()
                }
                test_cases.append(test_case)
        return test_cases

    def send_query(self, query):
        payload = {
            "messages": [{"role": "user", "content": query}],
            "app_id": Config.APP_ID,
            "app_type": Config.APP_TYPE,
            "entity_id": Config.ENTITY_ID,
            "session_id": Config.SESSION_ID
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': Config.AUTHORIZATION,
            'x-api-key': Config.X_API_KEY,
            'x-fp-id': Config.X_FP_ID,
            'Origin': Config.ORIGIN,
            'Accept': '*/*'
        }
        attempts = Config.RETRY_ATTEMPTS
        for attempt in range(attempts):
            try:
                response = requests.post(Config.CHATBOT_ENDPOINT, json=payload, headers=headers, timeout=10)
                if response.status_code == 429:
                    logger.warning(f"Attempt {attempt+1}: Rate limited. Waiting for {Config.DELAY_BETWEEN_ATTEMPTS} seconds before retrying...")
                    time.sleep(Config.DELAY_BETWEEN_ATTEMPTS)
                    continue
                response.raise_for_status()
                if not response.text.strip():
                    logger.warning("Response text is empty.")
                else:
                    logger.info(f"Raw response text: {response.text}")
                content_type = response.headers.get("Content-Type", "")
                if content_type.startswith("application/json"):
                    try:
                        json_response = response.json()
                        return json_response
                    except json.decoder.JSONDecodeError:
                        logger.error("Failed to decode JSON; falling back to raw text.")
                        return {"answer": response.text}
                else:
                    return {"answer": response.text}
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {e}")
                if attempt < attempts - 1:
                    logger.info("Retrying once more...")
                    time.sleep(Config.DELAY_BETWEEN_ATTEMPTS)
        logger.error("Exceeded retry attempts. Moving to next query.")
        return {}

    def log_test_results(self, results, log_file=Config.CSV_LOG_FILE):
        file_exists = os.path.isfile(log_file)
        try:
            with open(log_file, mode='a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['test_case', 'input', 'expected', 'actual', 'evaluation_method', 'result', 'score']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(results)
        except Exception as e:
            logger.error(f"Error logging test results: {e}")

    def generate_summary_report(self, results, report_file='test_summary.json'):
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['result'])
        average_score = sum(r['score'] for r in results) / total_tests if total_tests > 0 else 0
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0,
            "average_score": round(average_score, 2)
        }
        try:
            with open(report_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(summary, jsonfile, indent=4)
            logger.info(f"Test Summary Report saved to {report_file}")
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")

    def run_test_runner(self, csv_file, evaluation_method='exact', threshold=0.8, max_queries=10):
        try:
            test_cases = self.load_test_cases_from_file(csv_file)
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            return []
        results = []
        for idx, case in enumerate(test_cases, start=1):
            if idx > max_queries:
                logger.info(f"Reached maximum query limit of {max_queries}. Stopping further queries.")
                break
            query = case['input']
            expected = case['expected']
            logger.info(f"Running Test Case {idx}: Query: {query}")
            response_data = self.send_query(query)
            actual = response_data.get('answer', "")
            result, score = self.evaluate_response(actual, expected, method=evaluation_method, threshold=threshold)
            test_result = {
                'test_case': idx,
                'input': query,
                'expected': expected,
                'actual': actual,
                'evaluation_method': evaluation_method,
                'result': result,
                'score': score
            }
            results.append(test_result)
            logger.info(f"Test Case {idx} -> Result: {result}, Score: {score}")
            time.sleep(Config.DELAY_BETWEEN_TESTS)
        passed = sum(1 for r in results if r['result'])
        total = len(results)
        logger.info(f"Test Summary: {passed} out of {total} test cases passed.")
        return results

    def run_test_runner_with_dynamic(self, original_csv, dynamic_csv, evaluation_method='exact', threshold=0.8, max_queries=70):
        tg = TestGenerator()
        tg.generate_dynamic_tests(original_csv, dynamic_csv, include_original=True)
        results = self.run_test_runner(dynamic_csv, evaluation_method, threshold, max_queries)
        return results

# -----------------------------
# Agent Modules
# -----------------------------
def model_based_reflex_agent(prompt, internal_model):
    state = internal_model.get("state", "unknown")
    last_action = internal_model.get("last_action", "none")
    response = (f"[Reflex Agent] Current state: {state}, Last action: {last_action}. "
                f"Based on your prompt '{prompt}', I respond reflexively.")
    return response

def learning_agent(prompt, performance_history):
    if "improve" in prompt.lower() or "learn" in prompt.lower():
        response = ("[Learning Agent] I've been learning from past interactions. "
                    "I am updating my strategies to better handle your requests.")
    else:
        avg_accuracy = performance_history.get("accuracy", 0.0)
        response = (f"[Learning Agent] My current performance accuracy is {avg_accuracy*100:.1f}%. "
                    f"I received your prompt: '{prompt}'.")
    return response

# Global configurations for agents
internal_model = {
    "state": "stable",
    "last_action": "collected web data"
}
performance_history = {
    "accuracy": 0.85,
    "feedback": "overall good performance"
}

# -----------------------------
# LLM-based Task Extraction (via Hugging Face)
# -----------------------------
llm_pipeline = hf_pipeline("text2text-generation", model="google/flan-t5-base")

def extract_tasks_llm(prompt):
    instruction = (
        "Extract only a JSON array of task names from the following text. "
        "The available task names are: \"web_data_collection\", \"test_generation\", "
        "\"chatbot_testing\", \"reflex_agent\", \"learning_agent\". "
        "Output only the JSON array without any additional text or explanation. "
        f"Text: \"{prompt}\""
    )
    result = llm_pipeline(instruction, max_length=128)[0]['generated_text']
    try:
        tasks = json.loads(result)
    except json.JSONDecodeError:
        tasks = []
    return tasks

def extract_tasks(prompt):
    tasks = extract_tasks_llm(prompt)
    if tasks:
        return tasks
    prompt_lower = prompt.lower()
    fallback_tasks = []
    if "web" in prompt_lower or "data" in prompt_lower:
        fallback_tasks.append("web_data_collection")
    if "test" in prompt_lower or "generate" in prompt_lower:
        fallback_tasks.append("test_generation")
    if "chatbot" in prompt_lower or "interaction" in prompt_lower:
        fallback_tasks.append("chatbot_testing")
    if "reflex" in prompt_lower or "model" in prompt_lower:
        fallback_tasks.append("reflex_agent")
    if "learn" in prompt_lower or "improve" in prompt_lower:
        fallback_tasks.append("learning_agent")
    return fallback_tasks

# -----------------------------
# Agentic Routing Function (Retains Incomplete LangChain Part)
# -----------------------------
def advanced_agent_routing():
    user_prompt = input("Enter your prompt: ")
    tasks_to_run = extract_tasks(user_prompt)
    logger.info(f"Extracted tasks: {tasks_to_run}")
    
    for task in tasks_to_run:
        if task == "web_data_collection":
            logger.info("Executing Web Data Collection & Summarization Module...")
            collector = WebDataCollector()
            collector.process_urls_from_csv(Config.CSV_URL_INPUT, Config.CSV_URL_OUTPUT, summarize_flag=True)
        elif task == "test_generation":
            logger.info("Executing Dynamic Test Generation Module...")
            tg = TestGenerator()
            tg.generate_dynamic_tests(Config.CSV_TEST_INPUT, Config.CSV_TEST_DYNAMIC, include_original=True)
        elif task == "chatbot_testing":
            logger.info("Executing Chatbot Interaction & Evaluation Module...")
            evaluator = TestEvaluator()
            final_results = evaluator.run_test_runner(Config.CSV_TEST_INPUT, evaluation_method='semantic', 
                                                        threshold=Config.EVALUATION_THRESHOLD, 
                                                        max_queries=Config.MAX_QUERIES)
            evaluator.log_test_results(final_results)
            evaluator.generate_summary_report(final_results)
        elif task == "reflex_agent":
            logger.info("Executing Model-based Reflex Agent Module...")
            reflex_response = model_based_reflex_agent(user_prompt, internal_model)
            logger.info("Reflex Agent Response:")
            logger.info(reflex_response)
        elif task == "learning_agent":
            logger.info("Executing Learning Agent Module...")
            learning_response = learning_agent(user_prompt, performance_history)
            logger.info("Learning Agent Response:")
            logger.info(learning_response)
    
    if not tasks_to_run:
        logger.info("No matching tasks found in the prompt. Please refine your query.")

# -----------------------------
# Main Execution Block
# -----------------------------
if __name__ == '__main__':
    def define_testing_objectives():
        objectives = {
            "objective_1": "Ensure the chatbot responds accurately to user queries.",
            "objective_2": "Enhance test coverage by dynamically generating test variations.",
            "objective_3": "Continuously improve test cases using active learning from failures.",
            "objective_4": "Collect relevant web data and summarize content to simulate real-world queries.",
            "objective_5": "Monitor and iterate the test suite based on performance metrics."
        }
        logger.info("Testing Objectives:")
        for key, value in objectives.items():
            logger.info(f" - {value}")
        return objectives

    # Display testing objectives
    define_testing_objectives()
    logger.info("Agentic AI Mode: Awaiting your prompt")
    advanced_agent_routing()
    
    run_core = input("Would you also like to run the full core framework? (y/n): ").strip().lower()
    if run_core == 'y':
        collector = WebDataCollector()
        logger.info("Executing Web Data Collection & Summarization Demo...")
        web_text = collector.collect_web_data(Config.DEMO_URL)
        if web_text:
            summary = collector.summarize_content(web_text)
            logger.info("Summary of the webpage content:")
            logger.info(summary)
        else:
            logger.info("No content fetched from the web.")
        use_web_csv = True
        if use_web_csv:
            collector.process_urls_from_csv(Config.CSV_URL_INPUT, Config.CSV_URL_OUTPUT, summarize_flag=True)
        evaluator = TestEvaluator()
        USE_DYNAMIC_TESTS = True
        if USE_DYNAMIC_TESTS:
            final_results = evaluator.run_test_runner_with_dynamic(Config.CSV_TEST_INPUT, Config.CSV_TEST_DYNAMIC, 
                                                                   evaluation_method='semantic', 
                                                                   threshold=Config.EVALUATION_THRESHOLD, 
                                                                   max_queries=Config.MAX_QUERIES)
        else:
            final_results = evaluator.run_test_runner(Config.CSV_TEST_INPUT, evaluation_method='semantic', 
                                                      threshold=Config.EVALUATION_THRESHOLD, 
                                                      max_queries=Config.MAX_QUERIES)
        evaluator.log_test_results(final_results)
        evaluator.generate_summary_report(final_results)
        tg = TestGenerator()
        tg.update_dynamic_tests_from_failures(Config.CSV_LOG_FILE, Config.CSV_ADAPTED_TESTS, failure_threshold=0.5)
