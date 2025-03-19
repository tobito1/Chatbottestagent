# agent_routing.py
import json
import logging
from transformers import pipeline as hf_pipeline
from config import Config
from web_data import WebDataCollector
from test_generation import TestGenerator
from evaluation import TestEvaluator

logger = logging.getLogger(__name__)

# Incomplete LangChain integration is kept as-is
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document as LC_Document

# Initialize the LLM-based extraction pipeline
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
        return json.loads(result)
    except json.JSONDecodeError:
        return []

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

def model_based_reflex_agent(prompt, internal_model):
    state = internal_model.get("state", "unknown")
    last_action = internal_model.get("last_action", "none")
    return (f"[Reflex Agent] Current state: {state}, Last action: {last_action}. "
            f"Based on your prompt '{prompt}', I respond reflexively.")

def learning_agent(prompt, performance_history):
    if "improve" in prompt.lower() or "learn" in prompt.lower():
        return ("[Learning Agent] I've been learning from past interactions. "
                "I am updating my strategies to better handle your requests.")
    avg_accuracy = performance_history.get("accuracy", 0.0)
    return (f"[Learning Agent] My current performance accuracy is {avg_accuracy*100:.1f}%. "
            f"I received your prompt: '{prompt}'.")

def advanced_agent_routing(internal_model, performance_history):
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
