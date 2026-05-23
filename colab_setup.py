"""Colab automation: paste energivanu code cells into a notebook"""
import socket, time, json, os, base64

HOST, PORT = "127.0.0.1", 9999

def sc(cmd, arg=""):
    s = socket.socket()
    s.settimeout(120)
    s.connect((HOST, PORT))
    msg = cmd + (" " + arg if arg else "")
    s.send(msg.encode())
    r = s.recv(65536).decode()
    s.close()
    return r

def js(code):
    is_stmt = (code.startswith("try") or code.startswith("var ") or code.startswith("let ") or
               code.startswith("const ") or code.startswith("if") or code.startswith("for") or
               code.startswith("while") or code.startswith("switch") or code.startswith("{"))
    if is_stmt:
        code = "return (function(){" + code + "})()"
    elif not code.startswith("return"):
        code = "return " + code
    r = sc("js", code)
    if r.startswith("ERROR:"):
        print("  JS Error:", r)
    return r

def wait(n):
    sc("wait", str(n))

# Read cells from energivanu_colab.py
COLAB_FILE = os.path.join(os.path.dirname(__file__), "energivanu_colab.py")
with open(COLAB_FILE, "r", encoding="utf-8") as f:
    content = f.read()

cells = []
current = []
for line in content.split("\n"):
    if line.strip().startswith("# CELL"):
        if current:
            cells.append("\n".join(current).strip())
            current = []
    current.append(line)
if current:
    cells.append("\n".join(current).strip())

print("Found", len(cells), "cells")
for i, cell in enumerate(cells):
    print("  Cell", i+1, ":", len(cell), "chars")

# Remove docstring from each cell
def strip_docstring(text):
    lines = text.split("\n")
    result = []
    in_ds = False
    for line in lines:
        if '"""' in line:
            in_ds = not in_ds
            continue
        if not in_ds:
            result.append(line)
    return "\n".join(result).strip()

# Test server
print("\nConnecting to browser server...")
for attempt in range(5):
    try:
        r = sc("ping")
        if r and r.startswith("pong"):
            print("Connected:", r)
            break
    except:
        pass
    print("  Waiting for server (attempt", attempt+1, ")...")
    time.sleep(2)
else:
    print("ERROR: Server not running. Start: python browser_control.py start")
    exit(1)

# Navigate to Colab new notebook
print("\n1. Opening Colab new notebook...")
sc("get", "https://colab.research.google.com/#create=true")
wait(8)
title = js("return document.title")
print("   Title:", title)

# Paste code into cells using json.dumps for proper JS string escaping
def set_cell_code(code, cell_index=0):
    """Set cell code using Monaco API"""
    code_json = json.dumps(code)  # produces proper JS string literal
    p = (
        "try{"
        "var m=monaco.editor.getModels();"
        "if(m.length>" + str(cell_index) + "){"
        "m[" + str(cell_index) + "].setValue(" + code_json + ");"
        "return 'OK';"
        "}else{return 'no model';}"
        "}catch(e){return 'ERROR:'+e.message;}"
    )
    return js(p)

def add_cell_via_api():
    """Add new cell using Colab's internal API"""
    p = (
        "try{"
        "var api=document.querySelector('colab-notebook');"
        "if(api){api.addCell();return'added';}"
        "return'no colab-notebook found';"
        "}catch(e){return'ERROR:'+e.message;}"
    )
    return js(p)

# Try to find monaco
print("\n2. Checking Monaco editor availability...")
r = js("try{return typeof monaco.editor.getModels}catch(e){return 'err: '+e.message}")
print("   Monaco check:", r)

wait(2)

# Set cell 1
print("\n3. Setting Cell 1...")
cell1 = strip_docstring(cells[0])
r = set_cell_code(cell1, 0)
print("   Result:", r)
wait(1)

# Add and set remaining cells
for i in range(1, len(cells)):
    print("\n4." + str(i) + " Adding Cell", i+1, "...")
    
    # Add new cell
    r = add_cell_via_api()
    print("   Add result:", r)
    wait(2)
    
    # Set code
    code = strip_docstring(cells[i])
    r = set_cell_code(code, i)
    print("   Set result:", r)
    wait(1)

print("\nDone! All", len(cells), "cells pasted.")
