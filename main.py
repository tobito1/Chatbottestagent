# main.py
import logging
from config import Config
from web_data import WebDataCollector
from evaluation import TestEvaluator
from agent_routing import advanced_agent_routing

# Setup basic logging if not already configured
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def main():
    # Define testing objectives
    define_testing_objectives()

    # Global state encapsulated in objects instead of globals
    internal_model = {"state": "stable", "last_action": "collected web data"}
    performance_history = {"accuracy": 0.85, "feedback": "overall good performance"}
    
    # Run agentic routing which will ask the user for a prompt and execute accordingly
    logger.info("Agentic AI Mode: Awaiting your prompt")
    advanced_agent_routing(internal_model, performance_history)

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

if __name__ == '__main__':
    main()
