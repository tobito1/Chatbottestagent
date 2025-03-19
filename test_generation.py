# test_generation.py
import csv
import random
import logging
from itertools import permutations
from difflib import SequenceMatcher
from nltk.corpus import wordnet

logger = logging.getLogger(__name__)

class TestGenerator:
    def __init__(self):
        # Cache to store computed variations for each input text
        self._variation_cache = {}
    
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
        # Limit permutations for sentences that are too long
        if len(words) > 10:
            return []
        if len(words) > 3:
            permuted_sentences = [' '.join(p) for p in permutations(words, min(4, len(words)))]
            # Return only a limited number of permutations
            return permuted_sentences[:2]
        return []

    def is_similar(self, a, b, threshold=0.9):
        return SequenceMatcher(None, a, b).ratio() > threshold

    def generate_test_variations(self, input_text):
        # Check the cache first
        if input_text in self._variation_cache:
            return self._variation_cache[input_text]
        
        variations = set()
        variations.update(self.generate_synonyms(input_text))
        variations.update(self.introduce_typos(input_text))
        variations.update(self.generate_permutations(input_text))
        result = list(variations)
        # Cache the result
        self._variation_cache[input_text] = result
        return result

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
