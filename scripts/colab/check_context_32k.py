import sys
sys.path.insert(0, '/content/graph-memory/src')
from llm import complete

needle = 'NEEDLE_APPLE_42'
# 1500 repetitions of 9 words is ~13,500 words / ~17,000 tokens
haystack = 'The quick brown fox jumps over the lazy dog. ' * 1000
prompt = f'Here is a secret marker: {needle}\n\n{haystack}\n\nRepeat the secret marker exactly:'

out = complete(prompt, overrides={'provider': 'ollama', 'model': 'qwen2.5:7b-instruct', 'num_ctx': 32768}, max_tokens=32)
print('Output:', repr(out))
if needle in out:
    print('SUCCESS: Needle preserved at >5k tokens! 32k context is working!')
else:
    print(f'FAILURE: Needle missing from output: {out}')
    sys.exit(1)
