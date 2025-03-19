# config.py
import os

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
