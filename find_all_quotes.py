import re

with open('c:/python/app_streamlit.py', 'r', encoding='utf-8') as f:
    content = f.read()

matches = list(re.finditer(r'"""', content))
print(f"Total: {len(matches)}")
for m in matches:
    line = content[:m.start()].count('\n') + 1
    print(f"Line {line}: {m.group()}")
