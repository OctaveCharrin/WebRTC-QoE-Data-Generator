from __future__ import annotations

"""
Selenium WebDriver management for controlling Chrome browsers in Docker.

Replaces the Java browser automation from the original project:
  - ElasTestRemoteControlParent (constructor, initDriver, script injection)
  - startRecording / stopRecording / getRecording methods
  - waitForJsObject polling pattern
"""

import base64
import logging
import time
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)


class BrowserController:
    """
    Controls a remote Chrome browser via Selenium WebDriver.

    Each instance manages one browser (sender or receiver) running inside
    a Docker selenium/standalone-chrome container.
    """

    def __init__(self, selenium_url: str, role: str):
        """
        Args:
            selenium_url: Selenium Grid URL (e.g., http://localhost:4444)
            role: 'sender' or 'receiver' — used for logging
        """
        self.selenium_url = selenium_url
        self.role = role
        self.driver: Optional[webdriver.Remote] = None

    def connect(self, chrome_flags: Optional[list[str]] = None) -> None:
        """
        Connect to the remote Chrome via Selenium WebDriver.

        Args:
            chrome_flags: Chrome command-line flags to pass. For the sender,
                          this includes the fake media device flags.
        """
        options = Options()
        if chrome_flags:
            for flag in chrome_flags:
                options.add_argument(flag)

        logger.info(f"[{self.role}] Connecting to {self.selenium_url}...")
        self.driver = webdriver.Remote(
            command_executor=self.selenium_url,
            options=options,
        )
        self.driver.set_page_load_timeout(60)
        self.driver.set_script_timeout(60)
        logger.info(f"[{self.role}] Connected (session: {self.driver.session_id})")

    def navigate(self, url: str) -> None:
        """Navigate the browser to a URL."""
        logger.info(f"[{self.role}] Navigating to {url}")
        self.driver.get(url)

    def wait_for_connection(self, timeout_sec: int = 30) -> bool:
        """
        Poll window.__connectionState until WebRTC is connected.

        This uses the same polling pattern as the original Java code in
        ElasTestRemoteControlParent.waitForJsObject(), which polled every
        500ms for up to 30 seconds.
        """
        deadline = time.time() + timeout_sec
        last_state = None

        while time.time() < deadline:
            state = self.execute_script("return window.__connectionState")
            if state != last_state:
                logger.info(f"[{self.role}] WebRTC state: {state}")
                last_state = state
            if state == "connected":
                return True
            if state == "failed":
                logger.error(f"[{self.role}] WebRTC connection failed")
                return False
            time.sleep(0.5)

        logger.error(f"[{self.role}] WebRTC connection timeout ({timeout_sec}s)")
        return False

    # ---- Recording (receiver only) -----------------------------------------

    def start_recording(self) -> None:
        """
        Start MediaRecorder on the receiver.

        Calls the global startRecording() function exposed by index.html.
        This replaces the original RecordRTC-based approach.
        """
        result = self.execute_script("return window.startRecording()")
        if not result:
            raise RuntimeError("startRecording() returned falsy")
        logger.info(f"[{self.role}] Recording started")

    def stop_recording(self) -> None:
        """
        Stop MediaRecorder and wait for it to finish.

        Uses execute_async_script because stopRecording() returns a Promise.
        """
        self.driver.execute_async_script("""
            var callback = arguments[arguments.length - 1];
            window.stopRecording()
                .then(function() { callback(true); })
                .catch(function(e) { callback('ERROR: ' + e.message); });
        """)
        logger.info(f"[{self.role}] Recording stopped")

    def save_recording(self, output_path: Path) -> Path:
        """
        Retrieve the recording as base64 and save to disk.

        This mirrors the pattern from the original project:
          - elastest-remote-control.js recordingToData() converted blob to data URL
          - ElasTestRemoteControlParent.getRecording() decoded base64 and wrote to file
        """
        logger.info(f"[{self.role}] Retrieving recording (base64 transfer)...")
        data_url = self.driver.execute_async_script("""
            var callback = arguments[arguments.length - 1];
            window.getRecordingBase64()
                .then(function(result) { callback(result); })
                .catch(function(e) { callback('ERROR: ' + e.message); });
        """)

        if isinstance(data_url, str) and data_url.startswith("ERROR:"):
            raise RuntimeError(f"Recording retrieval failed: {data_url}")

        # Strip the data URL prefix: "data:video/webm;base64,..."
        base64_data = data_url.split(",", 1)[1]
        raw_bytes = base64.b64decode(base64_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw_bytes)

        size_mb = len(raw_bytes) / (1024 * 1024)
        logger.info(f"[{self.role}] Recording saved: {output_path.name} ({size_mb:.2f} MB)")
        return output_path

    # ---- Utilities ---------------------------------------------------------

    def execute_script(self, script: str):
        """Execute JavaScript synchronously and return the result."""
        return self.driver.execute_script(script)

    def refresh(self) -> None:
        """Refresh the page (reset state for next experiment)."""
        self.driver.refresh()
        time.sleep(1)

    def quit(self) -> None:
        """Close the browser session."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            logger.info(f"[{self.role}] Browser closed")
