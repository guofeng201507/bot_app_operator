import os, io, json, time, base64, subprocess, uuid
from datetime import datetime
from dotenv import load_dotenv
import requests

from appium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import oss2

load_dotenv()

ADB_HOST_PORT     = os.getenv("ADB_HOST_PORT")
APPIUM_ENDPOINT   = os.getenv("APPIUM_ENDPOINT", "http://127.0.0.1:4723/wd/hub")

APK_PATH          = os.getenv("APK_PATH")  # 可空
APP_PACKAGE       = os.getenv("APP_PACKAGE")  # 可空
APP_ACTIVITY      = os.getenv("APP_ACTIVITY") # 可空

OSS_ENDPOINT      = os.getenv("OSS_ENDPOINT")
OSS_BUCKET        = os.getenv("OSS_BUCKET")
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET")

QWEN_API_KEY      = os.getenv("QWEN_API_KEY")
QWEN_MODEL        = os.getenv("QWEN_MODEL", "qwen-vl-plus")

# ---------- Utils ----------
def run(cmd):
    print(f"$ {' '.join(cmd)}")
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(cp.stdout)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return cp.stdout

def adb_connect(adb_host_port: str):
    # 先断再连，避免“already connected”残留
    try:
        run(["adb", "disconnect", adb_host_port])
    except Exception:
        pass
    run(["adb", "connect", adb_host_port])

def adb_install(apk_path: str, adb_host_port: str):
    if not apk_path or not os.path.exists(apk_path):
        print("[adb_install] APK 路径为空或不存在，跳过安装。")
        return
    run(["adb", "-s", adb_host_port, "install", "-r", apk_path])

def connect_appium(adb_host_port: str):
    """
    通过 uiautomator2 驱动连接云手机。
    如果提供 APP_PACKAGE/APP_ACTIVITY，则直接启动；否则如果提供 APK_PATH，交给 Appium 安装启动。
    """
    caps = {
        "platformName": "Android",
        "automationName": "UiAutomator2",
        "udid": adb_host_port,              # 关键：指向云手机的 ADB 目标
        "newCommandTimeout": 300,
        "autoGrantPermissions": True,
        "unicodeKeyboard": True,
        "resetKeyboard": True,
    }

    if APP_PACKAGE and APP_ACTIVITY:
        caps["appPackage"]  = APP_PACKAGE
        caps["appActivity"] = APP_ACTIVITY
    elif APK_PATH and os.path.exists(APK_PATH):
        caps["app"] = APK_PATH
    else:
        print("[connect_appium] 未提供 app 信息，将仅连接设备，不自动启动 App。")

    print("[connect_appium] caps =", caps)
    driver = webdriver.Remote(APPIUM_ENDPOINT, caps)
    return driver

def oss_uploader(local_bytes: bytes, key_prefix: str="screenshots/") -> str:
    """
    把字节流上传到 OSS，返回可访问的 object key（或拼接 URL）。
    """
    if not all([OSS_ENDPOINT, OSS_BUCKET, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET]):
        raise RuntimeError("OSS 环境变量未配置完整。")

    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, f"https://{OSS_ENDPOINT}", OSS_BUCKET)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    obj_key = f"{key_prefix}{ts}-{uuid.uuid4().hex}.png"
    print(f"[oss] uploading to oss://{OSS_BUCKET}/{obj_key}")
    bucket.put_object(obj_key, local_bytes)

    # 这里返回 OSS 的 object key；若有 CDN/自建域名可自行拼接 URL
    return obj_key

def qwen_vl_analyze_png(png_bytes: bytes) -> dict:
    """
    直接用 HTTP 调用 DashScope 多模态对话接口（无需 SDK）。
    参考：https://dashscope.aliyun.com （不同模型/版本可能略有差异）
    """
    if not QWEN_API_KEY:
        raise RuntimeError("QWEN_API_KEY 未配置。")

    b64 = base64.b64encode(png_bytes).decode()
    prompt = (
        "You are an app visual QA checker.\n"
        "Check:\n"
        "1) Mixed-language on a single page (should be single language unless proper bilingual labels).\n"
        "2) Text misalignment.\n"
        "3) Overlapping elements.\n"
        "4) Text truncation/overflow.\n"
        "Return strict JSON only:\n"
        "{\"mixed_language\": true/false, \"page_language\":\"zh|en|other\", "
        "\"issues\": [{\"type\":\"overlap|misalign|truncate|other\",\"desc\":\"...\",\"bbox\":[x,y,w,h]}]}"
    )

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": QWEN_MODEL,
        "input": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
            ]
        }],
        "parameters": {
            "result_format": "json"  # 要求只返回 JSON
        }
    }

    print("[qwen] request ->", {"model": QWEN_MODEL, "bytes": len(png_bytes)})
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # 不同版本可能字段名不同，这里尽量兼容常见返回格式
    # 例如 data.get("output", {}).get("text") 或 choices[0].message.content
    result_json_str = None
    if "output" in data and "text" in data["output"]:
        result_json_str = data["output"]["text"]
    elif "choices" in data:
        # 兼容 OpenAI 风格
        try:
            result_json_str = data["choices"][0]["message"]["content"]
        except Exception:
            pass

    if not result_json_str:
        raise RuntimeError(f"Unexpected response format from Qwen: {data}")

    try:
        result = json.loads(result_json_str)
    except json.JSONDecodeError:
        # 万一不是严格 JSON，再做一次兜底清洗
        cleaned = result_json_str.strip().strip("```json").strip("```").strip()
        result = json.loads(cleaned)

    return result

# ---------- PoC 主流程 ----------
def main():
    if not ADB_HOST_PORT:
        raise RuntimeError("请在 .env 设置 ADB_HOST_PORT，如 1.2.3.4:5555")

    print("== 1) 连接无影云手机 ADB ==")
    adb_connect(ADB_HOST_PORT)

    # 可选：先用 adb 安装，确保设备已有 app（若你习惯让 Appium 安装，可跳过）
    if APK_PATH and os.path.exists(APK_PATH):
        print("== 2) 通过 ADB 安装 APK ==")
        adb_install(APK_PATH, ADB_HOST_PORT)

    print("== 3) 连接 Appium，启动 App ==")
    driver = connect_appium(ADB_HOST_PORT)

    try:
        wait = WebDriverWait(driver, 20)

        # —— 示例操作：等待/点击。如果你有明确的资源 ID，请替换：
        # 这里尽量用“存在就截图”的最小化链路，避免找不到元素导致失败。
        time.sleep(3)  # 等待首页渲染

        # 截图 #1：启动后
        png1 = driver.get_screenshot_as_png()
        key1 = oss_uploader(png1, "screenshots/")
        print(f"[poc] screenshot#1 uploaded: {key1}")

        # 尝试做一点交互（若没有元素 ID，就模拟一次 tap/swipe）
        try:
            # 示例：如果知道某个按钮 ID => 替换 "com.demo.app:id/btn_login"
            # btn = wait.until(EC.element_to_be_clickable((By.ID, "com.demo.app:id/btn_login")))
            # btn.click()
            # 否则：做一次坐标点击（全屏中间），仅为了 PoC 产生第二张截图
            size = driver.get_window_size()
            x, y = size["width"] // 2, size["height"] // 2
            driver.execute_script("mobile: clickGesture", {"x": x, "y": y})
            time.sleep(2)
        except Exception as e:
            print("[warn] 交互步骤失败，继续后续截图。", e)

        # 截图 #2：交互后
        png2 = driver.get_screenshot_as_png()
        key2 = oss_uploader(png2, "screenshots/")
        print(f"[poc] screenshot#2 uploaded: {key2}")

        print("== 4) 调用 Qwen-VL 分析（以截图#2 为例）==")
        result = qwen_vl_analyze_png(png2)
        print("[qwen result]", json.dumps(result, ensure_ascii=False, indent=2))

        # 这里你也可以把结果写回数据库；PoC 直接打印即可
    finally:
        print("== 5) 关闭会话 ==")
        try:
            driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()
