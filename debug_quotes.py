def check_all_quotes(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    
    i = 0
    n = len(content)
    
    def get_line(idx):
        return content[:idx].count('\n') + 1

    while i < n:
        line_num = get_line(i)
        
        # Check triple double
        if not in_single and not in_double and not in_triple_single:
            if content.startswith('"""', i):
                in_triple_double = not in_triple_double
                print(f"Line {line_num}: Triple Double toggled to {in_triple_double}")
                i += 3
                continue
        
        # Check triple single
        if not in_single and not in_double and not in_triple_double:
            if content.startswith("'''", i):
                in_triple_single = not in_triple_single
                print(f"Line {line_num}: Triple Single toggled to {in_triple_single}")
                i += 3
                continue
        
        # Skip escape
        if content[i] == '\\':
            i += 2
            continue

content = open('c:/python/app_streamlit.py', 'r', encoding='utf-8').read()
content = content.replace('"""', '   ')
count = content.count('"')
print(f"Total single double-quotes: {count}")
