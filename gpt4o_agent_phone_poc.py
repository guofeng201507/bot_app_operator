import os, io, re, json, base64, subprocess, time
from typing import Dict, Any
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ---------- 配置 ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

SYS_PROMPT = """You are an agent that sees the phone screen.
Return STRICT JSON with fields such as {"action": "tap", "bbox": [x1,y1,x2,y2], "reason": "..."}.
No explanations, no markdown fences."""
VERIFY_PROMPT = """You are verifying progress.
Return STRICT JSON like {"status":"done"} or {"status":"not_done","reason":"..."}.
No explanations, no markdown fences."""

# ---------- adb 工具 ----------
ADB_DIR = os.getenv("ADB_DIR")
if ADB_DIR and os.path.isdir(ADB_DIR) and ADB_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ADB_DIR + os.pathsep + os.environ.get("PATH", "")


def run(cmd):
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{cp.stdout}")
    return cp.stdout


def adb_connect(host_port: str):
    return run(["adb", "connect", host_port])


def adb_screencap(host_port: str) -> bytes:
    out = subprocess.check_output(["adb", "-s", host_port, "exec-out", "screencap", "-p"])
    return out


def adb_input(host_port: str, cmd: list):
    return run(["adb", "-s", host_port, "shell", "input"] + cmd)


# ---------- 图像工具 ----------
def _png_to_jpeg_dataurl(png_bytes, max_side=1024, quality=85) -> str:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _force_parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.S).strip()
    try:
        return json.loads(t)
    except:
        m = re.search(r"\{[\s\S]*\}", t)
        if m: return json.loads(m.group(0))
        raise ValueError(f"Model did not return JSON. preview={t[:200]}")


# ---------- OpenAI 调用 ----------
def call_openai(prompt_text: str, img_png: bytes) -> dict:
    assert OPENAI_API_KEY, "请先设置 OPENAI_API_KEY"
    data_url = _png_to_jpeg_dataurl(img_png)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 可改成 gpt-4o
        temperature=0,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
    )
    text = resp.choices[0].message.content
    return _force_parse_json(text)


# ---------- 高层逻辑 ----------
def think_action(goal: str, screenshot: bytes) -> Dict[str, Any]:
    prompt = f"{SYS_PROMPT}\n\nGoal: {goal}\nReturn JSON only."
    return call_openai(prompt, screenshot)


def verify_progress(goal: str, screenshot: bytes) -> Dict[str, Any]:
    prompt = f"{VERIFY_PROMPT}\n\nGoal: {goal}\nReturn JSON only."
    return call_openai(prompt, screenshot)


def act(host_port: str, action: Dict[str, Any]):
    a = action.get("action")
    if a == "tap":
        x, y = action["bbox"][:2]
        adb_input(host_port, ["tap", str(x), str(y)])
    elif a == "swipe":
        direction = action.get("swipe", "up")
        if direction == "up":
            adb_input(host_port, ["swipe", "300", "800", "300", "200", "300"])
        elif direction == "down":
            adb_input(host_port, ["swipe", "300", "200", "300", "800", "300"])
        elif direction == "left":
            adb_input(host_port, ["swipe", "600", "400", "100", "400", "300"])
        elif direction == "right":
            adb_input(host_port, ["swipe", "100", "400", "600", "400", "300"])
    elif a == "type":
        adb_input(host_port, ["text", action["text"]])
    else:
        print("Unknown action:", action)


# ---------- 主循环 ----------
def main():
    host_port = os.getenv("ADB_HOST_PORT", "127.0.0.1:7555")
    goal = os.getenv("GOAL", "Open Settings app")

    print("[STEP 1] observe")
    adb_connect(host_port)

    for step in range(10):
        print(f"[STEP] think")
        screenshot = adb_screencap(host_port)
        try:
            action = think_action(goal, screenshot)
            print("Action:", action)
        except Exception as e:
            print("[ERROR] think failed:", e)
            break

        act(host_port, action)
        time.sleep(2)

        print("[STEP] verify")
        screenshot = adb_screencap(host_port)
        try:
            result = verify_progress(goal, screenshot)
            print("Verify:", result)
            if result.get("status") == "done":
                print("Goal achieved ✅")
                break
        except Exception as e:
            print("[ERROR] verify failed:", e)
            break


if __name__ == "__main__":
    main()
