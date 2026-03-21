content = open('c:/python/app_streamlit.py', 'r', encoding='utf-8').read()
lines = content.split('\n')
in_s = False
in_d = False
for i, line in enumerate(lines):
    line_num = i + 1
    if line_num < 1142: continue
    if line_num > 1380: break
    
    start_s, start_d = in_s, in_d
    
    j = 0
    while j < len(line):
        if line[j:j+3] == '"""' or line[j:j+3] == "'''":
            j += 3
            continue
        if line[j] == '\\':
            j += 2
            continue
        if line[j] == '"' and not in_s:
            in_d = not in_d
        elif line[j] == "'" and not in_d:
            in_s = not in_s
        j += 1
    
    if in_s != start_s or in_d != start_d:
        print(f"Line {line_num} changed state to Single={in_s}, Double={in_d}: {line.strip()}")
    
    if in_s or in_d:
        # If it STAYS true across lines, it's a candidate for the bug
        pass
