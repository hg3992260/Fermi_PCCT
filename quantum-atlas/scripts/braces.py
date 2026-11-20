import io, re

p = r"I:\Fermi\quantum-atlas\main.cjs"
lines = io.open(p, encoding="utf-8").read().split("\n")

depth = 0
opened = []
for i, ln in enumerate(lines, 1):
    s = ln
    if s.lstrip().startswith("//"):
        continue
    s = re.sub(r"`[^`]*`", "``", s)
    s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s)
    s = re.sub(r"'(?:[^'\\]|\\.)*'", "''", s)
    s = re.sub(r"//.*$", "", s)
    for ch in s:
        if ch == "{":
            depth += 1
            opened.append(i)
        elif ch == "}":
            depth -= 1
            if opened:
                opened.pop()

print("final depth", depth)
print("last unclosed:", opened[-5:])
for i in opened[-3:]:
    print(i, lines[i - 1][:110])
