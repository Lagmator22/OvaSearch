import subprocess
import time
import json
import sys

# Mix of questions to test different capabilities
base_questions = [
    "what does monkey.txt say?",
    "what is the secret password in test.txt?",
    "describe cat image",
    "is there a dog in the knowledge base?",
    "what is intel openvino?",
    "summarize all documents",
    "what is the color of the cat?",
    "does monkey.txt mention a banana?",
    "where is test.txt located?",
    "how many files are loaded?",
    "who wrote test.txt?",
    "can you see a car in any image?",
    "what is the main topic of the dataset?",
    "is there any code in the text files?",
    "what does the hi message say?",
    "is the cat indoor or outdoor?",
    "what is the password again?",
    "is there an elephant in the dataset?",
    "what format is the cat image?",
    "what is the first word in monkey.txt?"
]

# We need 100 prompts, so we multiply the base set by 5
prompts = base_questions * 5

def run_tests():
    print("Starting OvaSearch Engine for 100-prompt stress test...")
    # Add a small delay so we don't conflict if ports/models take a second to free up from previous run
    time.sleep(2)
    
    process = subprocess.Popen(
        ['./build/ovasearch'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    def read_until_prompt():
        output = ""
        while True:
            char = process.stdout.read(1)
            if not char: # EOF
                break
            output += char
            if output.endswith("❯ "):
                break
        return output

    print("Waiting for engine to load models and index...")
    initial_output = read_until_prompt()
    print("Engine ready. Starting prompts...")

    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/100] Sending: {prompt}")
        start_time = time.time()
        
        process.stdin.write(prompt + "\n")
        process.stdin.flush()
        
        response = read_until_prompt()
        elapsed = time.time() - start_time
        
        # Clean up the response text from the prompt we sent
        clean_resp = response.replace(prompt + "\n", "").strip()
        if clean_resp.endswith("❯"):
             clean_resp = clean_resp[:-1].strip()
        
        results.append({
            "id": i + 1,
            "prompt": prompt,
            "response": clean_resp,
            "time_seconds": round(elapsed, 2)
        })
        
        print(f"[{i+1}/100] Done in {elapsed:.2f}s")
        
    process.stdin.write("exit\n")
    process.stdin.flush()
    process.wait()
    
    with open("stress_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Stress test complete. Results saved to stress_test_results.json")

    # Quick Analysis
    crashes = 0
    total_time = 0
    for r in results:
        if "segmentation fault" in r['response'].lower() or "error" in r['response'].lower():
             crashes += 1
        total_time += r['time_seconds']
        
    print(f"\n--- Quick Summary ---")
    print(f"Total Prompts: 100")
    print(f"Crashes/Errors Detected in Output: {crashes}")
    print(f"Average Time per Prompt: {total_time/100:.2f}s")
    print(f"Total Time: {total_time:.2f}s")

if __name__ == "__main__":
    run_tests()
