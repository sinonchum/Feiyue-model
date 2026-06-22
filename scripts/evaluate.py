import argparse
import json
import os
import time
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- Problem Definitions ---

CODING_PROBLEMS = [
    {"name": "is_prime", "prompt": "Write a Python function `is_prime(n)` that returns True if n is prime.", "test": "assert is_prime(2)\nassert is_prime(29)\nassert not is_prime(1)\nassert not is_prime(15)"},
    {"name": "fibonacci", "prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number.", "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55"},
    {"name": "flatten_list", "prompt": "Write a Python function `flatten_list(nested)` that flattens a list of lists.", "test": "assert flatten_list([[1,2],[3,4]]) == [1,2,3,4]"},
    {"name": "merge_sorted", "prompt": "Write a Python function `merge_sorted(a, b)` that merges two sorted lists.", "test": "assert merge_sorted([1,3,5],[2,4,6]) == [1,2,3,4,5,6]"},
    {"name": "caesar_cipher", "prompt": "Write `caesar_cipher(text, shift)` implementing Caesar cipher.", "test": "assert caesar_cipher('xyz', 3) == 'abc'"},
    {"name": "is_palindrome", "prompt": "Write `is_palindrome(s)` checking palindrome ignoring case.", "test": "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('race a car')"},
    {"name": "max_subarray", "prompt": "Write `max_subarray(nums)` returning max contiguous subarray sum.", "test": "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6"},
    {"name": "reverse_string", "prompt": "Write `reverse_string(s)` reversing a string.", "test": "assert reverse_string('hello') == 'olleh'"},
    {"name": "factorial", "prompt": "Write `factorial(n)` computing factorial.", "test": "assert factorial(0) == 1\nassert factorial(5) == 120"},
    {"name": "binary_search", "prompt": "Write `binary_search(arr, target)` returning index or -1.", "test": "assert binary_search([1,2,3,4,5], 4) == 3\nassert binary_search([1,2,3], 7) == -1"},
    {"name": "remove_dupes", "prompt": "Write `remove_dupes(arr)` removing duplicates preserving order.", "test": "assert remove_dupes([1,2,2,3,1,4]) == [1,2,3,4]"},
    {"name": "count_words", "prompt": "Write `count_words(s)` returning dict of word counts.", "test": "assert count_words('hello world hello') == {'hello': 2, 'world': 1}"},
    {"name": "matrix_rotate", "prompt": "Write `rotate_90(matrix)` rotating a matrix 90 degrees clockwise.", "test": "assert rotate_90([[1,2],[3,4]]) == [[3,1],[4,2]]"},
    {"name": "two_sum", "prompt": "Write `two_sum(nums, target)` returning indices of two numbers that add to target.", "test": "r = two_sum([2,7,11,15], 9)\nassert sorted(r) == [0,1]"},
    {"name": "valid_parens", "prompt": "Write `valid_parens(s)` checking if parentheses are valid.", "test": "assert valid_parens('()[]{}')\nassert not valid_parens('(]')"},
]

TOOL_CALLING_PROBLEMS = [
    {"name": "file_read", "prompt": "TaskContract: Read the file 'config.yaml'.\nRespond with JSON tool call.", "expected_tool": "file_read"},
    {"name": "run_tests", "prompt": "TaskContract: Run pytest on 'tests/test_utils.py'.\nRespond with JSON tool call.", "expected_tool": "run_tests"},
    {"name": "apply_patch", "prompt": "TaskContract: Fix the import error in 'utils.py'.\nRespond with JSON tool calls.", "expected_tool": "apply_patch"},
    {"name": "search_files", "prompt": "TaskContract: Find all files containing 'TODO'.\nRespond with JSON tool call.", "expected_tool": "search_files"},
    {"name": "multi_step", "prompt": "TaskContract: Read 'main.py', then run linter on it.\nRespond with JSON tool calls.", "expected_tool": "file_read"},
]


def extract_code(response):
    """Extract Python code from model response."""
    # Try to find code block
    import re
    match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find function definition
    match = re.search(r'(def \w+.*?)(?=\n\S|\Z)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try class definition
    match = re.search(r'(class \w+.*?)(?=\n\S|\Z)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_json_tool_calls(response):
    """Check if response contains valid JSON tool calls."""
    import re
    # Find JSON-like content
    match = re.search(r'\{[^{}]*"tool_calls?"[^{}]*\}|\{[^{}]*"name"[^{}]*\}', response)
    if match:
        try:
            json.loads(match.group())
            return True
        except:
            pass
    # Also accept if response contains tool-like keywords
    tool_names = ["file_read", "file_write", "run_tests", "run_linter", "apply_patch", "search_files", "shell_exec"]
    return any(t in response for t in tool_names)


def load_model(model_name, adapter_path=None):
    """Load model with 4-bit quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=512, temperature=0.3):
    """Generate response from model."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Respond with clean, correct code."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
    latency = time.time() - start
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response, latency


def evaluate_coding(model, tokenizer):
    """Evaluate coding problems."""
    results = []
    for prob in CODING_PROBLEMS:
        try:
            response, latency = generate(model, tokenizer, prob["prompt"])
            code = extract_code(response)

            # Syntax check
            compile(code, '<test>', 'exec')
            syntax_ok = True

            # Functional check
            test_code = code + "\n" + prob["test"]
            exec(test_code, {})
            functional_ok = True
        except SyntaxError:
            syntax_ok = False
            functional_ok = False
        except Exception:
            syntax_ok = syntax_ok if 'syntax_ok' in dir() else False
            functional_ok = False

        results.append({
            "name": prob["name"],
            "syntax_ok": syntax_ok,
            "functional_ok": functional_ok,
            "latency": latency,
        })
        status = "PASS" if functional_ok else ("SYNTAX" if syntax_ok else "FAIL")
        print(f"  {prob['name']}: {status} ({latency:.1f}s)")

    return results


def evaluate_tools(model, tokenizer):
    """Evaluate tool-calling problems."""
    results = []
    for prob in TOOL_CALLING_PROBLEMS:
        try:
            response, latency = generate(model, tokenizer, prob["prompt"], max_new_tokens=256)
            format_ok = extract_json_tool_calls(response)
            has_expected_tool = prob["expected_tool"] in response
        except Exception:
            format_ok = False
            has_expected_tool = False
            latency = 0

        results.append({
            "name": prob["name"],
            "format_ok": format_ok,
            "has_expected_tool": has_expected_tool,
            "latency": latency,
        })
        status = "PASS" if (format_ok and has_expected_tool) else ("FORMAT" if format_ok else "FAIL")
        print(f"  {prob['name']}: {status} ({latency:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    if args.adapter:
        print(f"Loading adapter: {args.adapter}")

    model, tokenizer = load_model(args.model, args.adapter)

    vram_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"Peak VRAM: {vram_mb:.0f} MB")

    print("\n--- Coding Evaluation (15 problems) ---")
    coding_results = evaluate_coding(model, tokenizer)

    print("\n--- Tool-Calling Evaluation (5 problems) ---")
    tool_results = evaluate_tools(model, tokenizer)

    # Aggregate
    coding_pass = sum(1 for r in coding_results if r["functional_ok"]) / len(coding_results)
    tool_pass = sum(1 for r in tool_results if r["format_ok"] and r["has_expected_tool"]) / len(tool_results)
    avg_latency = sum(r["latency"] for r in coding_results + tool_results) / (len(coding_results) + len(tool_results))

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "coding_pass_rate": coding_pass,
        "tool_format_rate": tool_pass,
        "avg_latency": avg_latency,
        "peak_vram_mb": vram_mb,
        "coding_details": coding_results,
        "tool_details": tool_results,
    }

    os.makedirs("evals", exist_ok=True)
    out_path = "evals/baseline.json" if not args.adapter else "evals/finetuned.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Coding pass rate:  {coding_pass:.1%}")
    print(f"Tool format rate:  {tool_pass:.1%}")
    print(f"Avg latency:       {avg_latency:.1f}s")
    print(f"Peak VRAM:         {vram_mb:.0f} MB")
    print(f"Saved to:          {out_path}")


if __name__ == "__main__":
    main()
