import sys, os, json, socket, threading, time, subprocess
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
    elif cmd == "ping":
        return "pong " + str(driver is not None)
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
        code = arg.strip()
        r = driver.execute_script(code)
        try:
            return json.dumps(r)
        except:
            return str(r)
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
