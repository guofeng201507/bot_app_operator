import os, io, re, json, base64, subprocess, time
from typing import Dict, Any
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- 配置 ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# SYS_PROMPT = """You are an agent that sees the phone screen.
# Return STRICT JSON with fields such as {"action": "tap", "bbox": [x1,y1,x2,y2], "reason": "..."}.
# No explanations, no markdown fences."""
# VERIFY_PROMPT = """You are verifying progress.
# Return STRICT JSON like {"status":"done"} or {"status":"not_done","reason":"..."}.
# No explanations, no markdown fences."""

SYS_PROMPT = """
You are a mobile UI agent operating an Cloud Phone (Android 12). 
You SEE a screenshot and must choose ONE next action. 
First REFLECT briefly (internally) to evaluate options, then OUTPUT ONLY a STRICT JSON decision. 
No markdown fences, no extra text outside JSON.

# Operating context (Cloud Phone)
- Network/ADB latency may be high; taps/scrolls might not apply immediately.
- UI language can be Chinese or English (e.g., “设置/Settings”, “允许/Allow”, “同意/Agree”).
- Permission dialogs are common (Android 12–14): e.g., “始终允许/仅在使用期间允许/不允许”, “允许/拒绝”, “确定/取消”.
- If the goal is not achieved, just click home button to return to home
- The launcher may require swiping UP from home to open the app drawer (or left/right page swipes on some launchers).

# Action schema (STRICT JSON)
Return exactly this schema with only the keys needed for the chosen action:
{
  "action": "tap|long_tap|swipe|type|back|home|keyevent|wait|done|fail",
  "bbox": [x,y,w,h],          // required for tap/long_tap/type when targeting a UI element (absolute pixels on the given screenshot)
  "tap_point": [x,y],         // optional alternative to bbox when a point is clearer than a box
  "swipe": "up|down|left|right",  // required for swipe
  "text": "string",           // required for type (ASCII; spaces OK)
  "keycode": 3|4|66|67,       // required for keyevent (examples: 3=HOME, 4=BACK, 66=ENTER, 67=DEL)
  "wait_ms": 300-2000,        // optional: if UI needs time to settle
  "reason": "≤120 chars concise why this action helps", // keep short; no step lists
  "confidence": 0-100         // self-estimate of decision quality
}

# Bbox/coordinates
- Coordinates are ABSOLUTE pixel values on this screenshot. 
- For tap/long_tap, click the CENTER of bbox. Keep bbox tight to avoid mis-taps.

# Goals & strategy (reflect-then-decide)
- You have a high-level goal from the user. Think in this order:
  1) Is the goal already visible/completed? If yes, output {"action":"done","reason":"..."}.
  2) If a permission/security dialog blocks progress (e.g., “允许/Allow”, “同意/Agree”, “确定/OK”), tap the safest allow/continue variant when it clearly unblocks the flow. Avoid destructive options (wipe/reset).
  3) Prefer clear, labeled controls that advance toward the goal (e.g., “搜索/Search”, “设置/Settings”, magnifier icon).
  4) If the target app/icon isn’t visible on home, try a single swipe: usually "swipe":"up" to open the app drawer; else swipe left/right on paged launchers.
  5) To enter text, pick the visible search/input field bbox and use {"action":"type","text":"..."} (keep short, no quotes/emoji).
  6) If you reach an unexpected page, try {"action":"back"} once; if still blocked, try a directional {"action":"swipe"}.
- Make only ONE action per decision. Keep a steady, safe progression.

# Hazards to avoid
- Do NOT open developer options, factory reset, airplane mode, or uninstall flows.
- Do NOT grant dangerous permissions unless clearly required to reach the stated goal.
- Do NOT press power/reboot.
- Do NOT tap ads or unrelated apps.

# Chinese/English labels (examples)
- Allow dialogs: “允许/Allow”, “同意/Agree”, “确定/OK”, “始终允许/Always allow”, “仅在使用期间允许/Allow only while using”
- Deny/Cancel: “拒绝/Deny”, “取消/Cancel”, “稍后/Later”
- System nav: “返回/Back”, “主页/Home”, “搜索/Search/查找”, “设置/Settings”

# Output rules
- Internally REFLECT, then output ONLY one JSON object (no code fences, no prose).
- If the screen content is ambiguous, choose the lowest-risk exploratory action (short swipe or back) with a brief reason.
- If truly impossible to proceed, return {"action":"fail","reason":"why"}.
"""

VERIFY_PROMPT = """
You are a verifier for the same Cloud Phone. 
Given a new screenshot and the user goal, decide if the goal is achieved.

Return ONLY STRICT JSON:
{
  "status": "done|not_done",
  "progress": 0-100,                 // rough closeness to goal
  "hint": "≤120 chars next best action if not done"
}

Guidance:
- “done” only if the target app/page/state is clearly visible/active.
- If a permission dialog is blocking, status = not_done with a short hint like “Tap 允许/Allow”.
- Be robust to Chinese/English UI.
- No code fences, no extra text.
"""

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
    print
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
    goal = os.getenv("AGENT_GOAL", "Click home")
    MAX_STEPS = os.getenv("MAX_STEPS", "10")

    adb_connect(host_port)

    for _no_step in range(int(MAX_STEPS)):
        print(f"[STEP {_no_step}] observe & think")
        screenshot = adb_screencap(host_port)
        try:
            action = think_action(goal, screenshot)
            print("Action instructed by AI Brain:", action)
        except Exception as e:
            print("[ERROR] think failed:", e)
            break

        act(host_port, action)
        time.sleep(2)

        print(f"[STEP {_no_step}] verify")
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
