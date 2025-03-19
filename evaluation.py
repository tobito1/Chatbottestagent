# evaluation.py
import csv
import json
import time
import os
import logging
import requests
from sentence_transformers import SentenceTransformer, util
from config import Config
from test_generation import TestGenerator

logger = logging.getLogger(__name__)

class TestEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate_response(self, actual, expected, method='exact', threshold=0.7):
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
                test_cases.append({
                    'input': row['input'].strip(),
                    'expected': row['expected'].strip()
                })
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
        
        max_attempts = Config.RETRY_ATTEMPTS
        base_delay = Config.DELAY_BETWEEN_ATTEMPTS
        backoff_factor = 2  # Exponential backoff factor
        
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(Config.CHATBOT_ENDPOINT, json=payload, headers=headers, timeout=10)
                # Raise error for bad status codes
                response.raise_for_status()
                
                # Handle rate limiting explicitly
                if response.status_code == 429:
                    raise requests.exceptions.HTTPError("Rate limited", response=response)
                
                if not response.text.strip():
                    logger.warning("Response text is empty.")
                else:
                    logger.info(f"Raw response text: {response.text}")
                
                if response.headers.get("Content-Type", "").startswith("application/json"):
                    try:
                        return response.json()
                    except json.decoder.JSONDecodeError:
                        logger.error("Failed to decode JSON; falling back to raw text.")
                        return {"answer": response.text}
                else:
                    return {"answer": response.text}
            
            except requests.exceptions.HTTPError as http_err:
                # Specific handling for HTTP errors
                if response.status_code == 429:
                    logger.warning(f"Attempt {attempt}: Rate limited. Retrying with exponential backoff...")
                else:
                    logger.error(f"HTTP error occurred: {http_err}")
                    break  # Exit loop for non-rate-limit HTTP errors
            except requests.exceptions.Timeout as timeout_err:
                logger.error(f"Timeout error occurred: {timeout_err}")
            except requests.exceptions.ConnectionError as conn_err:
                logger.error(f"Connection error occurred: {conn_err}")
            except Exception as e:
                logger.error(f"Unexpected error occurred: {e}")
            
            if attempt < max_attempts:
                sleep_time = base_delay * (backoff_factor ** (attempt - 1))
                logger.info(f"Retrying after {sleep_time} seconds (attempt {attempt}/{max_attempts})...")
                time.sleep(sleep_time)
        
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
            results.append({
                'test_case': idx,
                'input': query,
                'expected': expected,
                'actual': actual,
                'evaluation_method': evaluation_method,
                'result': result,
                'score': score
            })
            logger.info(f"Test Case {idx} -> Result: {result}, Score: {score}")
            time.sleep(Config.DELAY_BETWEEN_TESTS)
        logger.info(f"Test Summary: {sum(1 for r in results if r['result'])} out of {len(results)} test cases passed.")
        return results

    def run_test_runner_with_dynamic(self, original_csv, dynamic_csv, evaluation_method='exact', threshold=0.8, max_queries=70):
        tg = TestGenerator()
        tg.generate_dynamic_tests(original_csv, dynamic_csv, include_original=True)
        return self.run_test_runner(dynamic_csv, evaluation_method, threshold, max_queries)
