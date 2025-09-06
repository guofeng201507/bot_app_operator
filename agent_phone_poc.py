import os, io, time, json, base64, subprocess, math
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import requests
from appium.options.android.uiautomator2.base import UiAutomator2Options
from appium import webdriver

from PIL import Image

# ---------- 环境 ----------
load_dotenv()
ADB = os.getenv("ADB_HOST_PORT")
APPIUM = os.getenv("APPIUM_ENDPOINT", "http://127.0.0.1:4723/")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-vl-plus")
QWEN_URL = os.getenv("QWEN_URL")

AGENT_GOAL = os.getenv("AGENT_GOAL", "Open the Settings app.")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))

ADB_DIR = os.getenv("ADB_DIR")
if os.path.isdir(ADB_DIR) and ADB_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ADB_DIR + os.pathsep + os.environ.get("PATH", "")

import os, time, json, base64, requests, certifi

# 让 requests 明确使用 certifi 的根证书，避免公司代理/系统证书问题
os.environ.setdefault("SSL_CERT_FILE", certifi.where())


# ---------- 工具 ----------
def run(cmd):
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{cp.stdout}")
    return cp.stdout


def adb_connect(host_port):
    try:
        run(["adb", "disconnect", host_port])
    except Exception:
        pass
    print(run(["adb", "connect", host_port]))


def build_driver(adb_host_port):
    caps = {
        "platformName": "Android",
        "automationName": "UiAutomator2",
        "udid": adb_host_port,
        "newCommandTimeout": 300,
        "autoGrantPermissions": True,
        "unicodeKeyboard": True,
        "resetKeyboard": True,
        "skipServerInstallation": True
    }
    options = UiAutomator2Options().load_capabilities(caps)
    return webdriver.Remote(APPIUM, options=options)


def screenshot_png(driver) -> bytes:
    return driver.get_screenshot_as_png()


def _encode_image_for_api(png_bytes: bytes, max_side=1280, target_kb=1500) -> dict:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    # 二分法压缩到目标大小附近
    q_low, q_high = 60, 95
    best_b = None
    while q_low <= q_high:
        q = (q_low + q_high) // 2
        buff = io.BytesIO()
        img.save(buff, format="JPEG", quality=q, optimize=True)
        size_kb = buff.tell() / 1024
        if size_kb > target_kb:
            q_high = q - 1
        else:
            best_b = buff.getvalue()
            q_low = q + 1
    jpeg_bytes = best_b if best_b else io.BytesIO()
    if not best_b:
        buff = io.BytesIO()
        img.save(buff, format="JPEG", quality=80, optimize=True)
        jpeg_bytes = buff.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode()
    return {"media_type": "image/jpeg", "data": b64}


def screen_size(driver):
    s = driver.get_window_size()
    return s["width"], s["height"]


def clamp_bbox(b, W, H):
    # b: [x,y,w,h]，裁剪到屏幕范围，避免越界
    x, y, w, h = b
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return [x, y, w, h]


def center_of(b):  # bbox中心
    x, y, w, h = b
    return int(x + w / 2), int(y + h / 2)


# ---------- 与 VLM 通信 ----------
SYS_PROMPT = """You are a mobile UI agent. You see Android screenshots and a natural-language goal.
You must reason step-by-step internally and output ONLY a STRICT JSON action with this schema:

{
  "action": "tap|long_tap|swipe|back|home|type|done|fail",
  "bbox": [x,y,w,h],           // required for tap/long_tap/type; omit for others
  "text": "string",            // required for type
  "swipe": "up|down|left|right", // required for swipe
  "reason": "short why this action helps"
}

Rules:
- Always output valid JSON and nothing else.
- Prefer tapping clearly labeled buttons/icons that progress toward the goal.
- If a search field is visible and relevant, choose type with bbox and give the query text.
- If the current screen already satisfies the goal, output {"action":"done", ...}.
- If you are certain the goal cannot be achieved from here, output {"action":"fail", ...}.
"""

VERIFY_PROMPT = """You are a verifier. Given the same goal and a new screenshot, return STRICT JSON:
{
  "progress": 0..100,     // how close we are to the goal
  "done": true|false,
  "hint": "short hint on next step"
}
Only JSON. Be concise and robust to language differences in UI.
"""


def call_qwen(prompt_text: str, img_png: bytes) -> str:
    if not QWEN_API_KEY:
        raise RuntimeError("QWEN_API_KEY not set")

    img_payload = _encode_image_for_api(img_png)

    # qwen_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    qwen_url = QWEN_URL
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": QWEN_MODEL,
        "input": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "source": {"type": "base64", **img_payload}}
            ]
        }],
        "parameters": {"result_format": "json"}
    }
    r = requests.post(qwen_url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    # 兼容不同返回结构
    txt = data.get("output", {}).get("text") or (data.get("choices", [{}])[0].get("message", {}).get("content"))
    if not txt:
        raise RuntimeError(f"Unexpected model response: {data}")
    return txt.strip().strip("```json").strip("```").strip()


def think_action(goal: str, screenshot: bytes) -> Dict[str, Any]:
    prompt = f"{SYS_PROMPT}\n\nGoal: {goal}\nReturn JSON only."
    out = call_qwen(prompt, screenshot)
    return json.loads(out)


def verify_progress(goal: str, screenshot: bytes) -> Dict[str, Any]:
    prompt = f"{VERIFY_PROMPT}\n\nGoal: {goal}\nJSON only."
    out = call_qwen(prompt, screenshot)
    return json.loads(out)


# ---------- 执行动作 ----------
def act(driver, action: Dict[str, Any]):
    W, H = screen_size(driver)
    a = action.get("action")
    if a in ("tap", "long_tap", "type"):
        bbox = clamp_bbox(action.get("bbox", [0, 0, 10, 10]), W, H)
        x, y = center_of(bbox)
        duration = 600 if a == "long_tap" else 80
        driver.execute_script("mobile: clickGesture", {"x": x, "y": y, "duration": duration})
        if a == "type":
            text = action.get("text", "").strip()
            if text:
                # 轻等待聚焦输入框
                time.sleep(0.3)
                # 用 adb 直接输入更稳：避免键盘布局问题
                run(["adb", "-s", os.getenv("ADB_HOST_PORT"), "shell", "input", "text", text.replace(" ", "%s")])
    elif a == "swipe":
        dir = action.get("swipe", "down")
        sx = int(W * 0.5);
        ex = sx
        sy = int(H * 0.75);
        ey = int(H * 0.25)
        if dir == "up":
            sx, sy, ex, ey = int(W * 0.5), int(H * 0.7), int(W * 0.5), int(H * 0.3)
        elif dir == "down":
            sx, sy, ex, ey = int(W * 0.5), int(H * 0.3), int(W * 0.5), int(H * 0.7)
        elif dir == "left":
            sx, sy, ex, ey = int(W * 0.7), int(H * 0.5), int(W * 0.3), int(H * 0.5)
        elif dir == "right":
            sx, sy, ex, ey = int(W * 0.3), int(H * 0.5), int(W * 0.7), int(H * 0.5)
        driver.swipe(sx, sy, ex, ey, 300)
    elif a == "back":
        driver.back()
    elif a == "home":
        driver.press_keycode(3)
    elif a in ("done", "fail"):
        pass
    else:
        # 未知动作：忽略
        pass
    time.sleep(0.6)  # 动作后等待界面稳定


# ---------- 主循环 ----------
def main():
    if not ADB: raise RuntimeError("请在 .env 设置 ADB_HOST_PORT")
    print("[AGENT] goal:", AGENT_GOAL)
    print("[SETUP] connect ADB:", ADB)
    adb_connect(ADB)
    driver = build_driver(ADB)

    try:
        # 起步回到桌面，避免卡在奇怪界面
        driver.press_keycode(3);
        time.sleep(1.2)

        progress = 0
        for step in range(1, MAX_STEPS + 1):
            print(f"\n[STEP {step}] observe")
            img = screenshot_png(driver)

            print("[STEP] think")
            try:
                action = think_action(AGENT_GOAL, img)
            except Exception as e:
                print("[ERROR] think failed:", e)
                # 简单自愈：尝试下滑刷新
                driver.swipe(300, 500, 300, 1200, 300)
                continue

            print("[ACTION]", action)

            if action.get("action") == "done":
                print("[DONE] model认为已达成");
                break
            if action.get("action") == "fail":
                print("[FAIL] model认为无法达成");
                break

            print("[STEP] act")
            try:
                act(driver, action)
            except Exception as e:
                print("[ERROR] act failed:", e)
                # 退一步：按返回
                driver.back();
                time.sleep(0.6)

            print("[STEP] verify")
            img2 = screenshot_png(driver)
            try:
                v = verify_progress(AGENT_GOAL, img2)
                print("[VERIFY]", v)
                progress = max(progress, int(v.get("progress", 0)))
                if v.get("done") is True or progress >= 95:
                    print("[DONE] verify达成");
                    break
            except Exception as e:
                print("[WARN] verify failed:", e)

        print(f"\n[RESULT] progress≈{progress}%")

    finally:
        try:
            driver.quit()
        except:
            pass


if __name__ == "__main__":
    main()
