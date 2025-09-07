import os, io, re, json, base64, subprocess, time
from typing import Dict, Any
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- 配置 ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

SYS_PROMPT = """
You are a mobile UI agent operating an Alibaba Cloud Cloud Phone (Android).
You SEE a screenshot and must choose ONE next action.
First REFLECT internally on possible actions, then OUTPUT ONLY a STRICT JSON decision.
Do not include markdown fences or extra text outside JSON.

# Operating context (Aliyun Cloud Phone)
- Network/ADB latency may be high; actions may take time.
- UI language can be Chinese or English (e.g., “设置/Settings”, “允许/Allow”, “同意/Agree”).
- Permission dialogs (Android 12–14) often show: “始终允许/Always allow”, “仅在使用期间允许/Allow only while using”, “不允许/Deny”, “取消/Cancel”.
- If goal already visible/achieved → return {"action":"done","reason":"..."}.
- If blocked by permission/security dialog → choose the safest allow/continue option (允许/Allow/同意/OK). Avoid destructive actions.
- If target app not visible on home → {"action":"swipe","swipe":"up"} (to open app drawer) or left/right to navigate pages.
- For input → pick an input field and {"action":"type","text":"..."}.
- If stuck or unclear → {"action":"back"} or small swipe.
- Only one action per step.

# Action schema (STRICT JSON)
{
  "action": "tap|long_tap|swipe|type|back|home|keyevent|wait|done|fail",
  "norm_bbox": [x0,y0,x1,y1],   // normalized [0,1], prefer this
  "norm_point": [x,y],          // normalized [0,1]
  "bbox": [x,y,w,h],            // optional absolute in screenshot
  "tap_point": [x,y],           // optional absolute point
  "swipe": "up|down|left|right",
  "text": "string",            // for type
  "keycode": 3|4|66|67,         // for keyevent
  "wait_ms": 300-2000,          // optional wait
  "reason": "≤120 chars why",
  "confidence": 0-100
}
"""

VERIFY_PROMPT = """
You are a verifier for the same Alibaba Cloud Cloud Phone.
Given a new screenshot and the user goal, decide if the goal is achieved.

Return ONLY STRICT JSON:
{
  "status": "done|not_done",
  "progress": 0-100,
  "hint": "≤120 chars next best action if not done"
}

Guidance:
- “done” only if the target app/page/state is clearly visible/active.
- If a permission dialog blocks → {"status":"not_done","hint":"Tap 允许/Allow"}.
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
    return subprocess.check_output(["adb","-s",host_port,"exec-out","screencap","-p"])

def adb_input(host_port: str, cmd: list):
    return run(["adb","-s",host_port,"shell","input"] + cmd)

# ---------- 图像工具 ----------
def png_to_jpeg_dataurl_and_sizes(png_bytes, max_side=1024, quality=85):
    src = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    W, H = src.size
    s = min(1.0, max_side / max(W, H))
    if s < 1.0:
        dst = src.resize((int(W * s), int(H * s)))
    else:
        dst = src
    buf = io.BytesIO(); dst.save(buf, format="JPEG", quality=quality, optimize=True)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return data_url, (W, H), dst.size

def denorm_point(norm_xy, W, H):
    x = int(round(float(norm_xy[0]) * W))
    y = int(round(float(norm_xy[1]) * H))
    return x, y

def denorm_bbox(norm_xyxy, W, H):
    x0 = int(round(float(norm_xyxy[0]) * W))
    y0 = int(round(float(float(norm_xyxy[1]) * H)))
    x1 = int(round(float(norm_xyxy[2]) * W))
    y1 = int(round(float(norm_xyxy[3]) * H))
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    return [x0, y0, w, h]

def center_of_bbox(b):
    x, y, w, h = b
    return int(x + w/2), int(y + h/2)

def map_action_coords_to_device(action: dict, orig_size, res_size):
    W, H = orig_size; w_res, h_res = res_size
    sx, sy = W/float(w_res), H/float(h_res)

    if "norm_point" in action:
        return {"tap_px": denorm_point(action["norm_point"], W, H)}
    if "norm_bbox" in action:
        b = denorm_bbox(action["norm_bbox"], W, H)
        return {"tap_px": center_of_bbox(b), "bbox_px": b}
    if "tap_point" in action:
        rx, ry = action["tap_point"]
        return {"tap_px": (int(round(rx*sx)), int(round(ry*sy)))}
    if "bbox" in action:
        x,y,w,h = action["bbox"]
        b = [int(round(x*sx)), int(round(y*sy)), int(round(w*sx)), int(round(h*sy))]
        return {"tap_px": center_of_bbox(b), "bbox_px": b}
    return {}

# ---------- OpenAI 调用 ----------
def _force_parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\\s*|\\s*```$","",t,flags=re.I|re.S).strip()
    try:
        return json.loads(t)
    except:
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Model did not return JSON. preview={t[:200]}")

def call_openai(prompt_text: str, img_png: bytes) -> dict:
    assert OPENAI_API_KEY, "请先设置 OPENAI_API_KEY"
    data_url, orig_size, res_size = png_to_jpeg_dataurl_and_sizes(img_png)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 或 gpt-4o
        temperature=0,
        messages=[
            {"role":"system","content":SYS_PROMPT},
            {"role":"user","content":[
                {"type":"text","text": f"Goal: {prompt_text} (orig={orig_size}, resized={res_size}). Return JSON only."},
                {"type":"image_url","image_url":{"url":data_url}}
            ]}
        ]
    )
    text = resp.choices[0].message.content
    return _force_parse_json(text)

# ---------- 高层逻辑 ----------
def think_action(goal: str, screenshot: bytes) -> Dict[str, Any]:
    return call_openai(goal, screenshot)

def verify_progress(goal: str, screenshot: bytes) -> Dict[str, Any]:
    return call_openai(f"{VERIFY_PROMPT}\nGoal: {goal}", screenshot)

def act(host_port: str, action: Dict[str, Any], orig_size, res_size):
    a = action.get("action")
    mapped = map_action_coords_to_device(action, orig_size, res_size)

    if a == "tap" and "tap_px" in mapped:
        x,y = mapped["tap_px"]
        adb_input(host_port,["tap",str(x),str(y)])
    elif a == "swipe":
        direction = action.get("swipe","up")
        if direction=="up":
            adb_input(host_port,["swipe","300","800","300","200","300"])
        elif direction=="down":
            adb_input(host_port,["swipe","300","200","300","800","300"])
        elif direction=="left":
            adb_input(host_port,["swipe","600","400","100","400","300"])
        elif direction=="right":
            adb_input(host_port,["swipe","100","400","600","400","300"])
    elif a == "type":
        adb_input(host_port,["text",action["text"]])
    elif a == "home":
        adb_input(host_port,["keyevent","3"])
    elif a == "back":
        adb_input(host_port,["keyevent","4"])
    elif a == "keyevent":
        adb_input(host_port,["keyevent",str(action["keycode"])])
    elif a == "wait":
        time.sleep(int(action.get("wait_ms",1000))/1000.0)
    elif a == "done":
        print("Goal achieved ✅")
    else:
        print("Unknown action:", action)

# ---------- 主循环 ----------
def main():
    host_port = os.getenv("ADB_HOST_PORT","127.0.0.1:7555")
    goal = os.getenv("AGENT_GOAL","Open Settings app")
    MAX_STEPS = int(os.getenv("MAX_STEPS","10"))

    adb_connect(host_port)
    for step in range(MAX_STEPS):
        print(f"[STEP {step}] observe & think")
        screenshot = adb_screencap(host_port)
        orig_size = Image.open(io.BytesIO(screenshot)).size

        try:
            action = call_openai(goal, screenshot)
            print("Action:", action)
        except Exception as e:
            print("[ERROR] think failed:", e)
            break

        act(host_port, action, orig_size, (1024, int(1024*orig_size[1]/max(orig_size))))
        time.sleep(2)

        print(f"[STEP {step}] verify")
        screenshot = adb_screencap(host_port)
        try:
            result = verify_progress(goal, screenshot)
            print("Verify:", result)
            if result.get("status") == "done":
                print("🎉 Goal completed!")
                break
        except Exception as e:
            print("[ERROR] verify failed:", e)
            break

if __name__ == "__main__":
    main()
