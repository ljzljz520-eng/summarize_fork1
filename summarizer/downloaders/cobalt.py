"""Cobalt downloader for non-YouTube platforms."""

import os
import tempfile
import uuid
from typing import Optional
from urllib.parse import urlparse
from ..exceptions import AudioProcessingError, TranscriptError
from ..handlers import process_audio_file
from ..progress import ProgressSpinner, print_status
from ..security.httpguards import guarded_request, preflight_url
from ..security.netpolicy import (
    OutboundPolicyError,
    PURPOSE_COBALT_API,
    PURPOSE_COBALT_DOWNLOAD,
    PURPOSE_GENERIC_DOWNLOAD,
    get_policy,
)
from .base import BaseDownloader


class CobaltDownloader(BaseDownloader):
    """Downloader that proxies through a Cobalt instance."""

    def __init__(self, base_url: str):
        self.base_url = (base_url or "").rstrip("/")

    def supports(self, url: str) -> bool:
        parsed = urlparse(url or "")
        return parsed.scheme in ("http", "https")

    def _resolve_download_url(
        self, url: str, verbose: bool, mode: str = "audio"
    ) -> str:
        if not self.base_url:
            raise TranscriptError("Cobalt base URL not configured")

        # The user-supplied target travels inside the POST body and would
        # otherwise be fetched by the Cobalt server from ITS network position,
        # turning Cobalt into an SSRF open relay. Apply the same public-host
        # policy here (scheme/port/suffix/IP classification, resolving every
        # address now) before asking Cobalt to fetch anything.
        policy = get_policy()
        if policy.enforce:
            origin = preflight_url(policy, url, PURPOSE_GENERIC_DOWNLOAD)
            policy.resolve_and_check(origin.host, purpose=PURPOSE_GENERIC_DOWNLOAD)

        spinner = ProgressSpinner("Requesting Cobalt download", verbose)
        try:
            spinner.start()
            if mode == "audio":
                # Ask Cobalt for audio-only media. Explicitly request opus/64kbps to
                # avoid the mp3/128kbps defaults and reduce download size. Metadata is
                # stripped since it is discarded during our own re-encode step anyway.
                request_payload = {
                    "url": url,
                    "downloadMode": "audio",
                    "audioFormat": "opus",
                    "audioBitrate": "64",
                    "disableMetadata": True,
                }
            else:
                # Video mode: let Cobalt pick the best available format
                request_payload = {
                    "url": url,
                    "downloadMode": "auto",
                    "disableMetadata": True,
                }
            # Exact registered Cobalt origin; redirects are disabled for this
            # purpose by the guarded session.
            response = guarded_request(
                "post",
                f"{self.base_url}/",
                purpose=PURPOSE_COBALT_API,
                json=request_payload,
                headers={"Accept": "application/json"},
                timeout=60,
            )
            # Cobalt v11 returns HTTP 4xx with a structured JSON error body. Try
            # to parse the body before raising so the caller sees a meaningful
            # error code instead of a generic 'Bad Request'.
            try:
                payload = response.json()
            except Exception:
                response.raise_for_status()
                raise TranscriptError("Cobalt returned non-JSON response")
            spinner.stop()
        except OutboundPolicyError as exc:
            spinner.stop()
            # Never echo internal hostnames/IPs from the policy error.
            raise TranscriptError(
                "Cobalt request blocked by the outbound security policy."
            ) from exc
        except Exception as e:
            spinner.stop()
            raise TranscriptError(f"Cobalt request failed: {str(e)}")

        if isinstance(payload, dict):
            if payload.get("status") == "error":
                err = payload.get("error")
                if isinstance(err, dict):
                    msg = err.get("code") or "Cobalt returned an error"
                    ctx = err.get("context") or {}
                    if ctx:
                        msg = f"{msg} ({ctx})"
                else:
                    msg = payload.get("text") or "Cobalt returned an error"
                raise TranscriptError(msg)

            download_url = (
                payload.get("url")
                or payload.get("download")
                or payload.get("audio")
                or payload.get("file")
            )
            if not download_url and isinstance(payload.get("links"), list):
                first_link = payload["links"][0] if payload["links"] else {}
                download_url = first_link.get("url")

            if download_url:
                print_status("Cobalt download link ready", "SUCCESS", verbose)
                return download_url

        raise TranscriptError("Cobalt response did not include a download URL")

    def download_audio(
        self,
        url: str,
        temp_dir: Optional[str] = None,
        verbose: bool = False,
        audio_speed: float = 1.0,
        use_proxy: bool = False,
    ) -> str:
        try:
            download_url = self._resolve_download_url(url, verbose, mode="audio")
        except OutboundPolicyError as exc:
            raise AudioProcessingError(
                "Cobalt request blocked by the outbound security policy."
            ) from exc
        except Exception as e:
            raise AudioProcessingError(f"Cobalt audio download failed: {str(e)}") from e
        temp_root = temp_dir or tempfile.gettempdir()
        temp_name = f"cobalt_audio_{uuid.uuid4().hex}"
        temp_path = os.path.join(temp_root, f"{temp_name}.bin")
        processed_path = os.path.join(temp_root, f"{temp_name}.mp3")

        spinner = ProgressSpinner("Downloading audio from Cobalt", verbose)
        try:
            spinner.start()
            with guarded_request(
                "get",
                download_url,
                purpose=PURPOSE_COBALT_DOWNLOAD,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            spinner.stop()
            print_status("Cobalt download completed", "SUCCESS", verbose)

            spinner = ProgressSpinner("Processing audio file", verbose)
            spinner.start()
            process_audio_file(temp_path, processed_path, playback_speed=audio_speed)
            spinner.stop()
            print_status("Audio processing completed", "SUCCESS", verbose)
            os.remove(temp_path)
            return processed_path
        except OutboundPolicyError as exc:
            spinner.stop()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(processed_path):
                os.remove(processed_path)
            raise AudioProcessingError(
                "Cobalt download blocked by the outbound security policy."
            ) from exc
        except Exception as e:
            spinner.stop()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(processed_path):
                os.remove(processed_path)
            raise AudioProcessingError(f"Cobalt audio download failed: {str(e)}")

    def download_video(
        self,
        url: str,
        temp_dir: Optional[str] = None,
        verbose: bool = False,
        use_proxy: bool = False,
    ) -> str:
        try:
            download_url = self._resolve_download_url(url, verbose, mode="video")
        except OutboundPolicyError as exc:
            raise AudioProcessingError(
                "Cobalt request blocked by the outbound security policy."
            ) from exc
        except Exception as e:
            raise AudioProcessingError(f"Cobalt video download failed: {str(e)}") from e
        temp_root = temp_dir or tempfile.gettempdir()
        temp_name = f"cobalt_video_{uuid.uuid4().hex}"
        temp_path = os.path.join(temp_root, f"{temp_name}.bin")

        spinner = ProgressSpinner("Downloading video from Cobalt", verbose)
        try:
            spinner.start()
            with guarded_request(
                "get",
                download_url,
                purpose=PURPOSE_COBALT_DOWNLOAD,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            spinner.stop()
            print_status("Cobalt video download completed", "SUCCESS", verbose)
            return temp_path
        except OutboundPolicyError as exc:
            spinner.stop()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise AudioProcessingError(
                "Cobalt download blocked by the outbound security policy."
            ) from exc
        except Exception as e:
            spinner.stop()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise AudioProcessingError(f"Cobalt video download failed: {str(e)}")
