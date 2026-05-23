"""
Browser Control via Selenium + Brave (Persistent Server Mode)

  python browser_control.py start          - Start Brave browser (background)
  python browser_control.py stop           - Stop Brave and server
  python browser_control.py <cmd> [args]   - Run command on running browser

Commands: get, open, click, type, search, scroll, screenshot, js, close, list, switch, wait
"""
import sys, os, json, subprocess, time, socket, threading

BRAVE_PATH = r"C:\Users\admin\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"
USER_DATA = r"C:\Users\admin\AppData\Local\BraveSoftware\Brave-Browser\User Data"
DEBUG_PORT = 9999
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "_browser_server.py")

def ensure_server():
    if not os.path.exists(SERVER_SCRIPT):
        _create_server_script()

def _create_server_script():
    with open(SERVER_SCRIPT, "w") as f:
        f.write(r'''import sys, os, json, socket, threading, time, subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BRAVE = r"C:\Users\admin\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"
UD = r"C:\Users\admin\AppData\Local\BraveSoftware\Brave-Browser\User Data"
PORT = 9999
HOST = "127.0.0.1"

driver = None
wait = None

def start_browser():
    global driver, wait
    subprocess.run("taskkill /F /IM brave.exe >NUL 2>NUL", shell=True)
    time.sleep(1)
    opts = Options()
    opts.binary_location = BRAVE
    opts.add_argument(f"--user-data-dir={UD}")
    opts.add_argument("--profile-directory=Default")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_experimental_option("detach", True)
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=opts)
    wait = WebDriverWait(driver, 10)

def handle(cmd, arg):
    global driver, wait
    if cmd == "start":
        start_browser()
        return "OK"
    elif cmd == "stop":
        if driver: driver.quit()
        driver = None
        return "OK"
    elif cmd == "get":
        driver.get(arg)
        return f"OK: {driver.current_url}"
    elif cmd == "open":
        driver.execute_script(f"window.open('{arg}','_blank');")
        return "OK"
    elif cmd == "search":
        driver.get("https://www.google.com")
        box = wait.until(EC.element_to_be_clickable((By.NAME, "q")))
        box.clear(); box.send_keys(arg); box.submit()
        return "OK"
    elif cmd == "click":
        for xp in [f"//*[text()='{arg}']", f"//*[contains(text(),'{arg}')]"]:
            try:
                wait.until(EC.element_to_be_clickable((By.XPATH, xp))).click()
                return "OK"
            except: continue
        return "ERROR: not found"
    elif cmd == "type":
        sel, txt = arg.split(" ", 1)
        el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
        el.clear(); el.send_keys(txt)
        return "OK"
    elif cmd == "scroll":
        driver.execute_script(f"window.scrollBy(0,{arg});")
        return "OK"
    elif cmd == "screenshot":
        driver.save_screenshot(arg)
        return "OK"
    elif cmd == "js":
        r = driver.execute_script(arg)
        return json.dumps(r)
    elif cmd == "close":
        driver.close()
        hs = driver.window_handles
        if hs: driver.switch_to.window(hs[0])
        return "OK"
    elif cmd == "list":
        hs = []
        for i, h in enumerate(driver.window_handles):
            driver.switch_to.window(h)
            hs.append(f"[{i}] {driver.title}")
        return "|".join(hs)
    elif cmd == "switch":
        hs = driver.window_handles
        driver.switch_to.window(hs[int(arg)])
        return f"OK: {driver.title}"
    elif cmd == "wait":
        time.sleep(float(arg))
        return "OK"
    return "ERROR: unknown cmd"

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))
sock.listen()
while True:
    conn, _ = sock.accept()
    data = conn.recv(65536).decode().strip()
    if not data: conn.close(); continue
    parts = data.split(" ", 1)
    cmd, arg = parts[0], parts[1] if len(parts) > 1 else ""
    try:
        res = handle(cmd, arg)
    except Exception as e:
        res = f"ERROR: {e}"
    conn.send(res.encode())
    conn.close()
''')
    print(f"Server script created: {SERVER_SCRIPT}")

def send_cmd(cmd, arg=""):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(30)
    try:
        s.connect(("127.0.0.1", DEBUG_PORT))
        msg = cmd + (" " + arg if arg else "")
        s.send(msg.encode())
        return s.recv(65536).decode()
    except ConnectionRefusedError:
        return None
    finally:
        s.close()

if __name__ == "__main__":
    ensure_server()
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    action = sys.argv[1].lower()
    rest = sys.argv[2] if len(sys.argv) > 2 else ""

    if action == "start":
        r = send_cmd("start")
        if r is None:
            subprocess.Popen([sys.executable, SERVER_SCRIPT], creationflags=subprocess.CREATE_NO_WINDOW)
            time.sleep(3)
            r = send_cmd("start")
        print(r or "Server started")
    elif action == "stop":
        print(send_cmd("stop") or "Server not running")
    else:
        r = send_cmd(action, rest)
        if r is None:
            print("ERROR: Browser not running. Start first: python browser_control.py start")
        else:
            print(r)
