import os, io, re, json, base64, subprocess, time
from typing import Dict, Any
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- 配置 ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def load_prompt(filename: str) -> str:
    """Load prompt from prompts directory"""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

SYS_PROMPT = load_prompt("system_prompt.txt")
VERIFY_PROMPT = load_prompt("verify_prompt.txt")

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
    return subprocess.check_output(["adb", "-s", host_port, "exec-out", "screencap", "-p"])


def adb_input(host_port: str, cmd: list):
    return run(["adb", "-s", host_port, "shell", "input"] + cmd)


# ---------- 图像工具 ----------
def png_to_jpeg_dataurl_and_sizes(png_bytes, max_side=1024, quality=85):
    src = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    W, H = src.size
    s = min(1.0, max_side / max(W, H))
    if s < 1.0:
        dst = src.resize((int(W * s), int(H * s)))
    else:
        dst = src
    buf = io.BytesIO();
    dst.save(buf, format="JPEG", quality=quality, optimize=True)
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
    return int(x + w / 2), int(y + h / 2)


def map_action_coords_to_device(action: dict, orig_size, res_size):
    W, H = orig_size;
    w_res, h_res = res_size
    sx, sy = W / float(w_res), H / float(h_res)

    # ---- tap / long_tap mapping ----
    if "norm_point" in action:
        return {"tap_px": denorm_point(action["norm_point"], W, H)}
    if "norm_bbox" in action:
        b = denorm_bbox(action["norm_bbox"], W, H)
        return {"tap_px": center_of_bbox(b), "bbox_px": b}
    if "tap_point" in action:
        rx, ry = action["tap_point"]
        return {"tap_px": (int(round(rx * sx)), int(round(ry * sy)))}
    if "bbox" in action:
        x, y, w, h = action["bbox"]
        b = [int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))]
        return {"tap_px": center_of_bbox(b), "bbox_px": b}

    # ---- swipe mapping (optional points) ----
    swipe_from = swipe_to = None
    if "swipe_norm_from" in action and "swipe_norm_to" in action:
        swipe_from = denorm_point(action["swipe_norm_from"], W, H)
        swipe_to = denorm_point(action["swipe_norm_to"], W, H)
    elif "swipe_px_from" in action and "swipe_px_to" in action:
        fx, fy = action["swipe_px_from"];
        tx, ty = action["swipe_px_to"]
        swipe_from = (int(round(fx * sx)), int(round(fy * sy)))
        swipe_to = (int(round(tx * sx)), int(round(ty * sy)))

    out = {}
    if swipe_from and swipe_to:
        out["swipe_from_px"] = swipe_from
        out["swipe_to_px"] = swipe_to
    return out


HOME_GUARD_WINDOW = 1  # 只有在“上一次动作是 HOME”的接下来 1 步内，才允许 swipe
_last_home_step = -999  # 记录最近执行 HOME 的步号


def enforce_home_first(proposed: dict, step: int) -> dict:
    """
    如果模型想直接 swipe，但最近没有执行过 HOME，则强制改为 HOME。
    一旦提议/执行了 HOME，就记录步号，下一步才允许 swipe。
    """
    global _last_home_step
    act = (proposed or {}).get("action")

    # 模型提议 HOME：接受并记录时间窗
    if act == "home":
        _last_home_step = step
        return proposed

    # 模型提议 SWIPE：若上一动作不是刚按过 HOME，则覆盖为 HOME
    if act == "swipe":
        if step - _last_home_step > HOME_GUARD_WINDOW:
            return {
                "action": "home",
                "reason": "Policy: go to home before swiping to find apps",
                "confidence": 100
            }

    # 其他动作原样返回
    return proposed


# ---------- OpenAI 调用 ----------
def _force_parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", t, flags=re.I | re.S).strip()
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
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": [
                {"type": "text",
                 "text": f"Goal: {prompt_text} (orig={orig_size}, resized={res_size}). Return JSON only."},
                {"type": "image_url", "image_url": {"url": data_url}}
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

    if a in ("tap", "long_tap") and "tap_px" in mapped:
        x, y = mapped["tap_px"]
        if a == "long_tap":
            # 长按：按住 500ms（部分 ROM 需要更长可调）
            adb_input(host_port, ["swipe", str(x), str(y), str(x), str(y), "500"])
        else:
            adb_input(host_port, ["tap", str(x), str(y)])

    elif a == "swipe":
        # 优先使用坐标化滑动；否则用方向滑动回退
        if "swipe_from_px" in mapped and "swipe_to_px" in mapped:
            x0, y0 = mapped["swipe_from_px"]
            x1, y1 = mapped["swipe_to_px"]
            adb_input(host_port, ["swipe", str(x0), str(y0), str(x1), str(y1), "300"])
        else:
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

    elif a == "home":
        adb_input(host_port, ["keyevent", "3"])

    elif a == "back":
        adb_input(host_port, ["keyevent", "4"])

    elif a == "keyevent":
        adb_input(host_port, ["keyevent", str(action["keycode"])])

    elif a == "wait":
        time.sleep(int(action.get("wait_ms", 1000)) / 1000.0)

    elif a == "done":
        print("Goal achieved ✅")

    else:
        print("Unknown action:", action)


# ---------- 主循环 ----------
def main():
    host_port = os.getenv("ADB_HOST_PORT", "127.0.0.1:7555")
    goal = os.getenv("AGENT_GOAL", "Open Settings app")
    MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))

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

        act(host_port, action, orig_size, (1024, int(1024 * orig_size[1] / max(orig_size))))
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
