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

    def connect(
        self,
        chrome_flags: Optional[list[str]] = None,
        max_retries: int = 10,
        retry_delay: float = 3,
    ) -> None:
        """
        Connect to the remote Chrome via Selenium WebDriver.

        Retries if the Selenium container isn't ready yet (common when
        containers were just started).

        Args:
            chrome_flags: Chrome command-line flags to pass. For the sender,
                          this includes the fake media device flags.
            max_retries: Number of connection attempts before giving up.
            retry_delay: Seconds between retries.
        """
        options = Options()
        options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
        if chrome_flags:
            for flag in chrome_flags:
                options.add_argument(flag)

        logger.info(f"[{self.role}] Connecting to {self.selenium_url}...")
        for attempt in range(1, max_retries + 1):
            try:
                self.driver = webdriver.Remote(
                    command_executor=self.selenium_url,
                    options=options,
                )
                break
            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"[{self.role}] Failed to connect after {max_retries} attempts"
                    )
                    raise
                logger.info(
                    f"[{self.role}] Selenium not ready (attempt {attempt}/{max_retries}), "
                    f"retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)

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
                self._log_diagnostics()
                return False
            time.sleep(0.5)

        logger.error(f"[{self.role}] WebRTC connection timeout ({timeout_sec}s)")
        self._log_diagnostics()
        return False

    def _log_diagnostics(self) -> None:
        """Capture browser-side diagnostics to help debug connection issues."""
        try:
            diag = self.execute_script("""
                return {
                    wsState: window.__wsState,
                    connectionState: window.__connectionState,
                    error: window.__error,
                    log: (window.__signalingLog || []).slice(-20),
                    url: window.location.href,
                    title: document.title
                };
            """)
            if diag:
                logger.error(f"[{self.role}] Diagnostics:")
                logger.error(f"  URL: {diag.get('url')}")
                logger.error(f"  WS state: {diag.get('wsState')}")
                logger.error(f"  Connection state: {diag.get('connectionState')}")
                logger.error(f"  Error: {diag.get('error')}")
                for entry in (diag.get("log") or []):
                    logger.error(f"  JS: {entry}")
        except Exception as e:
            logger.error(f"[{self.role}] Could not retrieve diagnostics: {e}")

        # Also try to get browser console logs
        try:
            browser_logs = self.driver.get_log("browser")
            if browser_logs:
                logger.error(f"[{self.role}] Browser console logs:")
                for entry in browser_logs[-20:]:
                    logger.error(f"  [{entry.get('level')}] {entry.get('message')}")
        except Exception:
            pass  # Some Selenium versions don't support get_log

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
