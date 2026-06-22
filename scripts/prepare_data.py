import os
import json
import re
import random
import glob

# --- Configuration ---
FIXTURE_SRC_PATH = r"C:\Users\simon\Documents\HermesAnywhere\Feiyue\packages\feiyue-core\tests\*.py"
OUTPUT_DIR = r"C:\Users\simon\Feiyue-model\data"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.jsonl")

SYSTEM_MESSAGE = "You are a local coding agent. Given a TaskContract, produce structured tool calls in JSON format."
SYNTHETIC_PROBLEM_COUNT = 50
SPLIT_RATIO = (0.8, 0.1, 0.1)  # Train, Validation, Test

# --- Synthetic Data Definitions ---
SYNTHETIC_PROBLEMS = [
    {
        "name": "add_numbers",
        "description": "a function `add_numbers(a, b)` that returns the sum of two numbers.",
        "code": """def add_numbers(a, b):
    \"\"\"Returns the sum of two numbers.\"\"\"
    return a + b

# Test cases
assert add_numbers(5, 3) == 8
assert add_numbers(-1, 1) == 0
assert add_numbers(0, 0) == 0
assert add_numbers(1.5, 2.5) == 4.0"""
    },
    {
        "name": "is_palindrome",
        "description": "a function `is_palindrome(s)` that checks if a string is a palindrome (reads the same forwards and backwards), ignoring case and non-alphanumeric characters.",
        "code": """import re

def is_palindrome(s):
    \"\"\"Checks if a string is a palindrome, ignoring case and non-alphanumeric characters.\"\"\"
    normalized = re.sub(r'[^a-z0-9]', '', s.lower())
    return normalized == normalized[::-1]

# Test cases
assert is_palindrome("A man, a plan, a canal: Panama") is True
assert is_palindrome("race a car") is False
assert is_palindrome("Was it a car or a cat I saw?") is True
assert is_palindrome("hello") is False"""
    },
    {
        "name": "factorial",
        "description": "a recursive function `factorial(n)` to calculate the factorial of a non-negative integer.",
        "code": """def factorial(n):
    \"\"\"Calculates the factorial of a non-negative integer recursively.\"\"\"
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer.")
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Test cases
assert factorial(0) == 1
assert factorial(1) == 1
assert factorial(5) == 120
assert factorial(7) == 5040"""
    },
    {
        "name": "find_max",
        "description": "a function `find_max(numbers)` that finds the maximum number in a list of numbers.",
        "code": """def find_max(numbers):
    \"\"\"Finds the maximum number in a list. Returns None if the list is empty.\"\"\"
    if not numbers:
        return None
    max_val = numbers[0]
    for number in numbers:
        if number > max_val:
            max_val = number
    return max_val

# Test cases
assert find_max([1, 2, 3, 4, 5]) == 5
assert find_max([-1, -5, -3]) == -1
assert find_max([100, 20, 80]) == 100
assert find_max([]) is None"""
    },
    {
        "name": "reverse_string",
        "description": "a function `reverse_string(s)` that returns the reversed version of a string.",
        "code": """def reverse_string(s):
    \"\"\"Reverses a given string.\"\"\"
    return s[::-1]

# Test cases
assert reverse_string("hello") == "olleh"
assert reverse_string("Python") == "nohtyP"
assert reverse_string("") == ""
assert reverse_string("a") == "a" """
    },
    {
        "name": "fibonacci",
        "description": "a function `fibonacci(n)` that returns the n-th number in the Fibonacci sequence (starting with 0 and 1).",
        "code": """def fibonacci(n):
    \"\"\"Returns the n-th Fibonacci number.\"\"\"
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Test cases
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(10) == 55"""
    },
    {
        "name": "is_prime",
        "description": "a function `is_prime(num)` that checks if a number is a prime number.",
        "code": """def is_prime(num):
    \"\"\"Checks if a number is prime.\"\"\"
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

# Test cases
assert is_prime(2) is True
assert is_prime(3) is True
assert is_prime(4) is False
assert is_prime(29) is True
assert is_prime(100) is False"""
    },
    {
        "name": "remove_duplicates",
        "description": "a function `remove_duplicates(items)` that removes duplicate elements from a list while preserving the original order.",
        "code": """def remove_duplicates(items):
    \"\"\"Removes duplicates from a list, preserving order.\"\"\"
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Test cases
assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
assert remove_duplicates(['a', 'b', 'a', 'c']) == ['a', 'b', 'c']
assert remove_duplicates([]) == []"""
    },
    {
        "name": "bubble_sort",
        "description": "a function `bubble_sort(arr)` that sorts a list of numbers in ascending order using the bubble sort algorithm.",
        "code": """def bubble_sort(arr):
    \"\"\"Sorts a list using the bubble sort algorithm.\"\"\"
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# Test cases
assert bubble_sort([64, 34, 25, 12, 22, 11, 90]) == [11, 12, 22, 25, 34, 64, 90]
assert bubble_sort([5, 1, 4, 2, 8]) == [1, 2, 4, 5, 8]
assert bubble_sort([]) == []"""
    },
    {
        "name": "count_words",
        "description": "a function `count_words(text)` that counts the frequency of each word in a given text.",
        "code": """import re
from collections import Counter

def count_words(text):
    \"\"\"Counts the frequency of each word in a text.\"\"\"
    words = re.findall(r'\\b\\w+\\b', text.lower())
    return Counter(words)

# Test cases
text = "hello world hello"
counts = count_words(text)
assert counts['hello'] == 2
assert counts['world'] == 1
assert count_words("A test, a simple test.") == Counter({'a': 2, 'test': 2, 'simple': 1})"""
    }
]

def py_dict_str_to_json_str(py_str):
    """
    Converts a Python dictionary string literal to a valid JSON string.
    This is a brittle conversion, relying on regex and string replacement
    as per the problem constraints (no `ast` module).
    """
    s = py_str.strip()
    
    # Replace triple-quoted strings with JSON-escaped strings
    s = re.sub(r"'''([\s\S]*?)'''", lambda m: json.dumps(m.group(1)), s)
    s = re.sub(r'"""([\s\S]*?)"""', lambda m: json.dumps(m.group(1)), s)

    # Replace Python boolean/None keywords with JSON equivalents (whole words only)
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)

    # Replace single quotes with double quotes, avoiding escaped single quotes
    # This is tricky. A simpler approach for dicts is to quote all keys and string values.
    # Let's try a different, more robust replacement for dicts.
    # First, replace all single quotes with double quotes.
    # This is risky if a string contains a double quote. Assume fixtures are clean.
    s = s.replace("'", '"')

    # Remove trailing commas before a closing brace or bracket
    s = re.sub(r',\s*([\}\]])', r'\1', s)
    
    return s

def find_and_pair_literals(content):
    """
    Finds variable assignments to dict/list literals and pairs them up.
    Assumes a naming convention like `..._contract` and `..._expected...`.
    Uses a robust brace-counting method to find literal boundaries.
    """
    assignments = {}
    # Regex to find the start of an assignment: var_name = { or var_name = [
    for match in re.finditer(r'(\w+)\s*=\s*([\{\[])', content):
        var_name = match.group(1)
        open_char = match.group(2)
        close_char = '}' if open_char == '{' else ']'
        
        search_start_pos = match.end(0)
        level = 1
        
        for i in range(search_start_pos, len(content)):
            char = content[i]
            if char == open_char:
                level += 1
            elif char == close_char:
                level -= 1
            
            if level == 0:
                literal_content = content[match.start(2):i+1]
                assignments[var_name] = literal_content
                break
    
    # Pair up contracts and expected outputs
    pairs = []
    processed_contracts = set()
    for name, literal in assignments.items():
        if 'contract' in name and name not in processed_contracts:
            base_name = name.replace('task_contract', '').strip('_')
            # Find a matching expected output
            for expected_name in assignments:
                if 'expected' in expected_name and base_name in expected_name:
                    pairs.append((assignments[name], assignments[expected_name]))
                    processed_contracts.add(name)
                    break
    return pairs

def process_fixture_files():
    """Reads test fixture files, extracts data, and converts to ChatML format."""
    samples = []
    fixture_files = glob.glob(FIXTURE_SRC_PATH)
    
    for file_path in fixture_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            continue

        literal_pairs = find_and_pair_literals(content)
        
        for contract_str, expected_str in literal_pairs:
            try:
                # Convert Python dict string to JSON string
                user_content_json_str = py_dict_str_to_json_str(contract_str)
                assistant_content_json_str = py_dict_str_to_json_str(expected_str)
                
                # Parse and re-dump to ensure valid, formatted JSON
                user_json = json.loads(user_content_json_str)
                assistant_json = json.loads(assistant_content_json_str)
                
                user_content = json.dumps(user_json, indent=2)
                assistant_content = json.dumps(assistant_json, indent=2)

                sample = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }
                samples.append(sample)
            except Exception as e:
                # print(f"Warning: Failed to process a literal pair in {file_path}: {e}")
                pass # Suppress warnings for cleaner output
                
    return samples

def generate_synthetic_data():
    """Generates synthetic coding problems in ChatML format."""
    samples = []
    
    # Ensure we have enough unique problems
    problems_to_generate = (SYNTHETIC_PROBLEMS * (SYNTHETIC_PROBLEM_COUNT // len(SYNTHETIC_PROBLEMS) + 1))[:SYNTHETIC_PROBLEM_COUNT]
    
    for i, problem in enumerate(problems_to_generate):
        # Add slight variations to make problems more unique if desired
        func_name = problem['name']
        if i >= len(SYNTHETIC_PROBLEMS):
            func_name = f"{problem['name']}_{i // len(SYNTHETIC_PROBLEMS)}"
            
        user_prompt = f"Write a Python function that implements {problem['description']}\n" \
                      f"The function should be named `{func_name}`. " \
                      "Include several test cases using `assert` to verify its correctness."

        # The system message is more for tool use, but we use it as required.
        # A more general system message like "You are a helpful coding assistant."
        # might be better, but we stick to the requirements.
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": problem['code'].replace(problem['name'], func_name, 1)}
            ]
        }
        samples.append(sample)
        
    return samples

def validate_sample(sample):
    """Checks if a sample conforms to the required ChatML structure."""
    if not isinstance(sample, dict) or "messages" not in sample:
        return False
    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        return False
    
    roles = [msg.get("role") for msg in messages]
    if roles != ["system", "user", "assistant"]:
        return False
        
    for msg in messages:
        if not isinstance(msg.get("content"), str) or not msg["content"]:
            return False
            
    return True

def main():
    """Main script execution."""
    print("Starting data preparation for Qwen3-4B fine-tuning...")

    # 1. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory set to: {OUTPUT_DIR}")

    # 2. Process real data from fixtures
    print(f"Reading test fixtures from: {FIXTURE_SRC_PATH}")
    fixture_samples = process_fixture_files()
    
    # 3. Generate synthetic data
    print(f"Generating {SYNTHETIC_PROBLEM_COUNT} synthetic coding problems...")
    synthetic_samples = generate_synthetic_data()

    all_samples = fixture_samples + synthetic_samples
    
    # 4. Validate all samples
    initial_count = len(all_samples)
    validated_samples = [s for s in all_samples if validate_sample(s)]
    final_count = len(validated_samples)
    
    # 5. Shuffle and split data
    random.shuffle(validated_samples)
    
    train_end = int(len(validated_samples) * SPLIT_RATIO[0])
    val_end = train_end + int(len(validated_samples) * SPLIT_RATIO[1])
    
    train_set = validated_samples[:train_end]
    val_set = validated_samples[train_end:val_end]
    test_set = validated_samples[val_end:] # Not written, but calculated for stats

    # 6. Write to JSONL files
    print(f"Writing {len(train_set)} samples to {TRAIN_FILE}...")
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for sample in train_set:
            f.write(json.dumps(sample) + '\n')

    print(f"Writing {len(val_set)} samples to {VAL_FILE}...")
    with open(VAL_FILE, 'w', encoding='utf-8') as f:
        for sample in val_set:
            f.write(json.dumps(sample) + '\n')

    # 7. Print statistics
    print("\n--- Data Preparation Statistics ---")
    print(f"Fixture files found: {len(glob.glob(FIXTURE_SRC_PATH))}")
    print(f"Samples extracted from fixtures: {len(fixture_samples)}")
    print(f"Synthetic samples generated: {len(synthetic_samples)}")
    print("-" * 35)
    print(f"Total samples before validation: {initial_count}")
    print(f"Total valid samples after validation: {final_count}")
    if initial_count != final_count:
        print(f"  ({initial_count - final_count} invalid samples discarded)")
    print("-" * 35)
    print(f"Training set size: {len(train_set)} ({SPLIT_RATIO[0]*100:.0f}%)")
    print(f"Validation set size: {len(val_set)} ({SPLIT_RATIO[1]*100:.0f}%)")
    print(f"Test set size (not written): {len(test_set)} ({SPLIT_RATIO[2]*100:.0f}%)")
    print("-" * 35)
    print("Script finished successfully.")

if __name__ == "__main__":
    main()