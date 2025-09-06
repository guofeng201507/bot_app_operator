from dotenv import load_dotenv
from appium import webdriver
from appium.options.android.uiautomator2.base import UiAutomator2Options
import os

# ---------- 环境 ----------
load_dotenv()
ADB = os.getenv("ADB_HOST_PORT")
APPIUM = os.getenv("APPIUM_ENDPOINT", "http://127.0.0.1:4723")

# 你的能力
caps = {
    "platformName": "Android",
    "automationName": "UiAutomator2",
    "udid": ADB,  # 云手机 ADB 地址
    # "app": r"C:\path\to\your.apk",       # 二选一：给 app 路径
    # "appPackage": "com.demo.app",        # 或者给包名 + Activity
    # "appActivity": ".MainActivity",
    "newCommandTimeout": 300,
    "autoGrantPermissions": True,
    "unicodeKeyboard": True,
    "resetKeyboard": True,
    "skipServerInstallation": True
}

options = UiAutomator2Options().load_capabilities(caps)

# 注意：新版本 Appium 不要求 /wd/hub，两个都能用
driver = webdriver.Remote(APPIUM, options=options)
print("session:", driver.session_id)
driver.quit()
