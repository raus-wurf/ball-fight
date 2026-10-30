"""
VaultMind Ball Fight — automated daily TikTok pipeline
────────────────────────────────────────────────────────
Simulates a short physics "ball fight" (pymunk), writes short bursts of
AI commentary (Groq) tied to real in-sim events, mixes them with an
always-epic synthesized music bed + collision SFX, appends a "<COLOR>
WINS!" end card with a follow/like call-to-action, color-grades the
result, and Direct-Posts it to TikTok via the Content Posting API.

No burned-in captions — the video speaks (literally, sparingly) for
itself.
"""

import os, json, time, random, logging, tempfile
import subprocess, shutil, math
from pathlib import Path
from typing import Optional

import pymunk

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment

from groq import Groq
import edge_tts, asyncio
import colorsys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vaultmind")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
VIDEO_W, VIDEO_H   = 1080, 1920
FPS                = 30
MUSIC_DUCK_DB      = -16
FONT_PATH          = "font_bold.ttf"

# Visual identity + ffmpeg color-grade curve for the Ball Fight niche.
NICHES = {
    "ballfight": {
        "color": "#39FF14", "glow": "#8CFF66", "bg": "#04050A",
        "color_grade": (
            "curves=r='0/0 0.4/0.42 0.8/0.85 1/1.0':"
            "g='0/0 0.4/0.50 0.8/0.92 1/1.0':"
            "b='0/0 0.4/0.35 0.8/0.70 1/0.90',"  # punchy neon arcade look
            "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.8"
        ),
    },
}


def ensure_font() -> str:
    """Download Impact/Montserrat Black font if not present."""
    if Path(FONT_PATH).exists():
        return FONT_PATH
    # Try system fonts first
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            shutil.copy(p, FONT_PATH)
            log.info("Font: using system %s", p)
            return FONT_PATH
    # Download Montserrat Black from Google Fonts CDN
    try:
        url  = "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Black.ttf"
        data = requests.get(url, timeout=15).content
        Path(FONT_PATH).write_bytes(data)
        log.info("Font: Montserrat Black downloaded")
        return FONT_PATH
    except Exception:
        pass
    log.warning("Font: falling back to PIL default")
    return ""


def load_font(size: int) -> ImageFont.FreeTypeFont:
    fp = FONT_PATH if Path(FONT_PATH).exists() else None
    if fp:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — SCRIPT (aggressive viral prompts)
# ══════════════════════════════════════════════════════════════════════

def _elevenlabs_tts(text: str, path: str) -> bool:
    key = os.getenv("ELEVENLABS_KEY")
    if not key:
        log.info("ElevenLabs: no key set, skipping")
        return False
    voice_id = os.getenv("ELEVEN_VOICE_ID", "")
    if not voice_id:
        try:
            r = requests.get("https://api.elevenlabs.io/v1/voices",
                             headers={"xi-api-key": key}, timeout=10)
            if r.status_code != 200:
                log.warning(
                    "ElevenLabs: voice list failed (%d) — %s. "
                    "FIX: enable 'voices_read' permission for this API key "
                    "at elevenlabs.io → Settings → API Keys, or set "
                    "ELEVEN_VOICE_ID secret to a specific voice ID manually.",
                    r.status_code, r.text[:150]
                )
                return False
            voices = r.json().get("voices", [])
            if not voices:
                log.warning("ElevenLabs: account has zero voices available")
                return False
            voice_id = voices[0]["voice_id"]
            log.info("ElevenLabs: auto-selected voice '%s'", voices[0].get("name"))
        except Exception as e:
            log.warning("ElevenLabs: voice discovery error: %s", e)
            return False

    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json={"text": text, "model_id": "eleven_turbo_v2_5",
                  "voice_settings": {"stability": 0.30, "similarity_boost": 0.82,
                                     "style": 0.55, "use_speaker_boost": True}},
            timeout=90,
        )
    except Exception as e:
        log.warning("ElevenLabs: request error: %s", e)
        return False

    if r.status_code == 200:
        Path(path).write_bytes(r.content)
        log.info("ElevenLabs TTS OK (%d bytes)", len(r.content))
        return True
    # Surface the actual reason — quota, invalid key, etc.
    log.warning("ElevenLabs failed (%d): %s", r.status_code, r.text[:200])
    return False


def _openai_tts(text: str, path: str) -> bool:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        log.info("OpenAI TTS: no key set, skipping")
        return False
    try:
        r = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "tts-1-hd", "voice": "onyx", "input": text, "speed": 1.08},
            timeout=90,
        )
    except Exception as e:
        log.warning("OpenAI TTS: request error: %s", e)
        return False

    if r.status_code == 200:
        Path(path).write_bytes(r.content)
        log.info("OpenAI TTS OK (%d bytes)", len(r.content))
        return True
    log.warning("OpenAI TTS failed (%d): %s", r.status_code, r.text[:200])
    return False


def _edgetts_tts(text: str, path: str) -> bool:
    """
    edge-tts fallback. Uses a standard (non-multilingual) voice, since
    multilingual variants are more prone to 'NoAudioReceived' errors in
    CI environments. Retries up to 3 times with backoff — Microsoft's
    endpoint occasionally rate-limits/blocks datacenter IPs (GitHub Actions).
    """
    voices_to_try = ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-EricNeural"]

    for voice in voices_to_try:
        for attempt in range(3):
            try:
                async def _run():
                    comm = edge_tts.Communicate(text, voice, rate="+8%", volume="+10%")
                    await comm.save(path)
                asyncio.run(_run())
                if Path(path).exists() and Path(path).stat().st_size > 1000:
                    log.info("edge-tts OK (voice=%s, attempt=%d)", voice, attempt + 1)
                    return True
            except Exception as e:
                log.warning("edge-tts failed (voice=%s, attempt=%d): %s",
                           voice, attempt + 1, e)
                time.sleep(2 * (attempt + 1))
    log.warning("edge-tts: all voices/retries exhausted")
    return False


def _gtts_fallback(text: str, path: str) -> bool:
    """
    Absolute last resort: Google Translate TTS (gTTS).
    Free, no API key, but lower quality and has a ~100-char chunk limit
    internally (gTTS handles chunking automatically). Robotic but reliable.
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(path)
        if Path(path).exists() and Path(path).stat().st_size > 1000:
            log.info("gTTS OK (final fallback)")
            return True
    except Exception as e:
        log.error("gTTS fallback also failed: %s", e)
    return False


def generate_voiceover(text: str, path: str) -> None:
    """
    Voice fallback chain (first success wins):
    ElevenLabs → OpenAI TTS → edge-tts (3 voices × 3 retries) → gTTS
    Raises RuntimeError only if every single option fails.
    """
    if _elevenlabs_tts(text, path):
        return
    if _openai_tts(text, path):
        return
    if _edgetts_tts(text, path):
        return
    if _gtts_fallback(text, path):
        return
    raise RuntimeError(
        "All TTS providers failed (ElevenLabs, OpenAI, edge-tts, gTTS). "
        "Check API keys and network/firewall settings."
    )


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — ASSEMBLYAI TRANSCRIPTION (raw HTTP)
# ══════════════════════════════════════════════════════════════════════

def _hex(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))


# ══════════════════════════════════════════════════════════════════════
# STEP 11 — YOUTUBE UPLOAD
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# STEP 11b — TIKTOK UPLOAD, DIRECT (Content Posting API v2) — fallback when no
# ZERNIO_API_KEY is set; unaudited apps can only post privately this way
#
#  Flow: refresh access token → query creator_info (allowed privacy
#  levels) → POST /video/init/ (FILE_UPLOAD) → PUT raw video bytes to
#  the returned upload_url → poll /status/fetch/ until PUBLISH_COMPLETE.
#
#  IMPORTANT CAVEAT (real TikTok platform constraint, not a bug here):
#  until this TikTok developer app has passed TikTok's content-posting
#  audit, every video posted via the API is forced to SELF_ONLY privacy
#  (visible only to the account owner) regardless of what's requested.
#  Once the app is audited/approved, PUBLIC_TO_EVERYONE becomes an
#  available option and this code will pick it up automatically (it
#  reads the allowed levels fresh from creator_info on every run).
# ══════════════════════════════════════════════════════════════════════

TIKTOK_API = "https://open.tiktokapis.com/v2"

def _tiktok_refresh_token() -> Optional[str]:
    """Exchange the long-lived TIKTOK_REFRESH_TOKEN for a fresh access token."""
    client_key    = os.getenv("TIKTOK_CLIENT_KEY")
    client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
    refresh_token = os.getenv("TIKTOK_REFRESH_TOKEN")
    if not (client_key and client_secret and refresh_token):
        log.warning("TikTok: missing TIKTOK_CLIENT_KEY/SECRET/REFRESH_TOKEN — skipping upload")
        return None
    try:
        r = requests.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"client_key": client_key, "client_secret": client_secret,
                  "grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=20,
        )
        data = r.json()
        if "access_token" in data:
            log.info("TikTok: access token refreshed (expires in %ss)", data.get("expires_in"))
            return data["access_token"]
        log.warning("TikTok: token refresh failed: %s", data)
    except Exception as e:
        log.warning("TikTok: token refresh error: %s", e)
    return None


def _tiktok_creator_info(token: str) -> dict:
    try:
        r = requests.post(f"{TIKTOK_API}/post/publish/creator_info/query/",
                          headers={"Authorization": f"Bearer {token}"}, timeout=20)
        body = r.json()
        log.info("TikTok creator_info raw response [%s]: %s", r.status_code, body)
        return body.get("data", {})
    except Exception as e:
        log.warning("TikTok: creator_info query failed: %s", e)
        return {}


def _upload_tiktok_direct(video_path: str, title: str, hashtags: Optional[list[str]] = None) -> Optional[str]:
    """
    Direct-Post a video to TikTok via the Content Posting API (FILE_UPLOAD).
    Returns a status string (publish_id) since TikTok's API does not hand
    back a public post URL synchronously — check the account for the live
    post once PUBLISH_COMPLETE is reached.
    """
    token = _tiktok_refresh_token()
    if not token:
        return None

    info          = _tiktok_creator_info(token)
    allowed_priv  = info.get("privacy_level_options") or ["SELF_ONLY"]
    # Prefer public if the app's audit status allows it; otherwise SELF_ONLY
    # is the only privacy_level unaudited clients are actually permitted to
    # post with — don't just grab allowed_priv[0], since a private account's
    # options can list FOLLOWER_OF_CREATOR/MUTUAL_FOLLOW_FRIENDS first, and
    # TikTok rejects those for unaudited apps even though they're "valid"
    # options for the account in general.
    if "PUBLIC_TO_EVERYONE" in allowed_priv:
        privacy = "PUBLIC_TO_EVERYONE"
    elif "SELF_ONLY" in allowed_priv:
        privacy = "SELF_ONLY"
    else:
        privacy = allowed_priv[0]
    if privacy == "SELF_ONLY":
        log.warning("TikTok: app not yet audited for public posting — "
                    "video will upload as private (visible only to the account owner)")
    log.info("TikTok: creator=%s privacy_level_options=%s -> using privacy=%s",
              info.get("creator_username"), allowed_priv, privacy)

    caption = title[:150]
    if hashtags:
        tag_line = " ".join(h if h.startswith("#") else f"#{h}" for h in hashtags)
        caption  = f"{caption} {tag_line}"[:2200]

    size = Path(video_path).stat().st_size
    try:
        init = requests.post(
            f"{TIKTOK_API}/post/publish/video/init/",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "post_info": {
                    "title": caption,
                    "privacy_level": privacy,
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                    "video_cover_timestamp_ms": 1000,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": size,
                    "chunk_size": size,
                    "total_chunk_count": 1,
                },
            },
            timeout=30,
        ).json()
    except Exception as e:
        log.warning("TikTok: init request failed: %s", e)
        return None

    data = init.get("data", {})
    upload_url, publish_id = data.get("upload_url"), data.get("publish_id")
    if not upload_url:
        log.warning("TikTok: init response missing upload_url: %s", init)
        return None

    try:
        with open(video_path, "rb") as f:
            put = requests.put(
                upload_url,
                headers={"Content-Type": "video/mp4",
                         "Content-Range": f"bytes 0-{size-1}/{size}"},
                data=f, timeout=180,
            )
        if put.status_code not in (200, 201):
            log.warning("TikTok: video PUT failed (%d): %s", put.status_code, put.text[:200])
            return None
    except Exception as e:
        log.warning("TikTok: video upload error: %s", e)
        return None

    # Poll publish status (async on TikTok's side)
    for _ in range(20):
        time.sleep(3)
        try:
            status = requests.post(
                f"{TIKTOK_API}/post/publish/status/fetch/",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"publish_id": publish_id}, timeout=15,
            ).json().get("data", {})
        except Exception as e:
            log.warning("TikTok: status poll error: %s", e)
            break
        st = status.get("status")
        if st == "PUBLISH_COMPLETE":
            log.info("TikTok upload complete: %s (privacy=%s)", publish_id, privacy)
            return publish_id
        if st == "FAILED":
            log.warning("TikTok: publish failed: %s", status)
            return None
    log.info("TikTok: upload accepted, still processing (publish_id=%s)", publish_id)
    return publish_id


# ══════════════════════════════════════════════════════════════════════
# STEP 12 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# STEP 11c — TIKTOK UPLOAD VIA ZERNIO (preferred: public posts, no own app audit)
#
#  Zernio (zernio.com) is a posting API that already holds an audited TikTok
#  app, so videos can go out PUBLICLY without waiting for our own TikTok
#  Developer review. One-time setup:
#    1. Sign up at zernio.com (free tier) and connect the TikTok account
#    2. Create an API key -> GitHub secret ZERNIO_API_KEY
#    3. Set the TikTok account back to PUBLIC (a private account can't post
#       publicly)
#  Without ZERNIO_API_KEY the direct TikTok API above is used instead.
# ══════════════════════════════════════════════════════════════════════

ZERNIO_API = "https://zernio.com/api/v1"
ZERNIO_AI_LABEL = True  # synthetic voice + AI-written commentary -> disclose it on TikTok


def _zernio_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _zernio_tiktok_account_id(api_key: str) -> Optional[str]:
    override = (os.getenv("ZERNIO_TIKTOK_ACCOUNT_ID") or "").strip()
    if override:
        return override
    try:
        r = requests.get(f"{ZERNIO_API}/accounts", headers=_zernio_headers(api_key), timeout=20)
        if not r.ok:
            log.warning("Zernio: listing accounts failed [%s]: %s", r.status_code, r.text[:300])
            return None
        data = r.json()
        accounts = data.get("accounts", []) if isinstance(data, dict) else data
        tiktok = [a for a in accounts
                  if str(a.get("platform", "")).lower() == "tiktok" and a.get("isActive", True) is not False]
        if not tiktok:
            log.warning("Zernio: no TikTok account connected — connect it in the Zernio dashboard")
            return None
        return tiktok[0].get("_id") or tiktok[0].get("id")
    except Exception as e:
        log.warning("Zernio: account lookup error: %s", e)
        return None


def _zernio_creator_levels(api_key: str, account_id: str) -> list:
    """Privacy levels TikTok currently allows for this creator (empty if unknown)."""
    try:
        r = requests.get(f"{ZERNIO_API}/accounts/{account_id}/tiktok/creator-info",
                         params={"mediaType": "video"}, headers=_zernio_headers(api_key), timeout=20)
        log.info("Zernio creator-info [%s]: %s", r.status_code, r.text[:500])
        if not r.ok:
            return []
        out = []
        for lvl in (r.json().get("privacyLevels") or []):
            if isinstance(lvl, str):
                out.append(lvl)
            elif isinstance(lvl, dict):
                v = lvl.get("value") or lvl.get("privacy_level") or lvl.get("level") or lvl.get("id")
                if v:
                    out.append(str(v))
        return out
    except Exception as e:
        log.warning("Zernio: creator-info error: %s", e)
        return []


def _zernio_pick_privacy(levels: list) -> str:
    for pref in ("PUBLIC_TO_EVERYONE", "FOLLOWER_OF_CREATOR", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"):
        if pref in levels:
            return pref
    return "PUBLIC_TO_EVERYONE"  # creator-info unavailable -> try public


def _zernio_create_post(headers: dict, payload: dict) -> tuple:
    """POST /posts. Returns (post_id or None, error_text). Success means HTTP 2xx
    AND the TikTok platform entry didn't come back as failed."""
    try:
        r = requests.post(f"{ZERNIO_API}/posts", headers=headers, json=payload, timeout=120)
    except Exception as e:
        return None, f"request error: {e}"
    if not r.ok:
        return None, f"HTTP {r.status_code}: {r.text[:600]}"
    try:
        post = (r.json().get("post") or {})
    except Exception:
        return None, f"unreadable response: {r.text[:300]}"
    plats = post.get("platforms") or []
    status = str((plats[0] if plats else {}).get("status") or post.get("status") or "").lower()
    if status in ("failed", "error"):
        return None, f"platform status {status}: {r.text[:600]}"
    pid = post.get("_id") or post.get("id")
    log.info("Zernio: TikTok post %s accepted (status=%s)", pid, status or "?")
    return pid, ""


def _upload_tiktok_zernio(video_path: str, title: str, hashtags: Optional[list[str]] = None) -> Optional[str]:
    api_key = (os.getenv("ZERNIO_API_KEY") or "").strip()
    account_id = _zernio_tiktok_account_id(api_key)
    if not account_id:
        return None
    headers = _zernio_headers(api_key)

    levels = _zernio_creator_levels(api_key, account_id)
    privacy = _zernio_pick_privacy(levels)
    if levels and "PUBLIC_TO_EVERYONE" not in levels:
        log.warning("Zernio: TikTok only offers %s — the account is probably set to PRIVATE. "
                    "Switch it to public in TikTok (Settings -> Privacy) for public posts.", levels)

    # 1) presigned upload -> 2) PUT the file -> 3) create the post
    try:
        pre = requests.post(f"{ZERNIO_API}/media/presign", headers=headers, timeout=30,
                            json={"filename": f"ballfight_{int(time.time())}.mp4",
                                  "contentType": "video/mp4", "size": os.path.getsize(video_path)})
        if not pre.ok:
            log.warning("Zernio: presign failed [%s]: %s", pre.status_code, pre.text[:300])
            return None
        upload_url, public_url = pre.json().get("uploadUrl"), pre.json().get("publicUrl")
        if not (upload_url and public_url):
            log.warning("Zernio: presign response missing urls: %s", pre.text[:300])
            return None
        with open(video_path, "rb") as f:
            put = requests.put(upload_url, data=f, headers={"Content-Type": "video/mp4"}, timeout=300)
        if not put.ok:
            log.warning("Zernio: video upload failed [%s]: %s", put.status_code, put.text[:300])
            return None
    except Exception as e:
        log.warning("Zernio: media upload error: %s", e)
        return None

    tags = " ".join(f"#{h.lstrip('#')}" for h in (hashtags or [])[:8])
    caption = f"{title[:150]} {tags}".strip()[:2200]
    settings = {"privacy_level": privacy, "allow_comment": True, "allow_duet": True,
                "allow_stitch": True, "content_preview_confirmed": True,
                "express_consent_given": True, "video_made_with_ai": ZERNIO_AI_LABEL}

    for attempt in range(1, 4):
        payload = {"content": caption,
                   "mediaItems": [{"type": "video", "url": public_url}],
                   "platforms": [{"platform": "tiktok", "accountId": account_id}],
                   "tiktokSettings": settings, "publishNow": True}
        pid, err = _zernio_create_post(headers, payload)
        if pid:
            return pid
        log.warning("Zernio: post attempt %d failed: %s", attempt, err)
        low = err.lower()
        if any(k in low for k in ("duet", "stitch", "comment")) and settings["allow_duet"]:
            settings.update(allow_comment=False, allow_duet=False, allow_stitch=False)
        elif "privacy" in low and levels:
            tried = settings["privacy_level"]
            nxt = next((l for l in levels if l != tried), None)
            if not nxt:
                break
            settings["privacy_level"] = nxt
        elif any(k in low for k in ("capacity", "active user", "limit", "cap")) and not settings.get("draft"):
            log.warning("Zernio: TikTok capacity reached — sending to the creator inbox as a draft instead")
            settings["draft"] = True
        else:
            break
    return None


def upload_tiktok(video_path: str, title: str, hashtags: Optional[list[str]] = None) -> Optional[str]:
    """Publish to TikTok: via Zernio when ZERNIO_API_KEY is set (public posts),
    otherwise through the direct Content Posting API (private until audited)."""
    if (os.getenv("ZERNIO_API_KEY") or "").strip():
        return _upload_tiktok_zernio(video_path, title, hashtags)
    log.info("ZERNIO_API_KEY not set — using the direct TikTok Content Posting API")
    return _upload_tiktok_direct(video_path, title, hashtags)


# ══════════════════════════════════════════════════════════════════════
# POST-RENDER STEP B — CINEMATIC COLOR GRADE (ffmpeg)
#
#  Teal-orange grade: the single most recognisable "cinematic" look.
#  Also adds subtle unsharp mask (makes text pop harder).
#  ~2–3% retention increase based on A/B tests in similar channels.
# ══════════════════════════════════════════════════════════════════════

def apply_color_grade(video_path: str, output_path: str,
                      niche: str = "fact") -> str:
    """
    Apply niche-specific cinematic color grade via ffmpeg.
    Each niche has its own curated color curve (defined in NICHES dict).
    Falls back to teal-orange universal grade if niche grade missing.
    """
    # Get niche-specific grade, fall back to universal teal-orange
    vf = NICHES.get(niche, {}).get("color_grade") or (
        "curves="
        "r='0/0 0.3/0.27 0.7/0.74 1/0.95':"
        "g='0/0 0.3/0.30 0.7/0.72 1/0.96':"
        "b='0/0.04 0.3/0.33 0.7/0.68 1/0.85',"
        "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.7"
    )
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log.info("Color grade applied [%s] → %s", niche, output_path)
        return output_path
    except subprocess.CalledProcessError as e:
        log.warning("Color grade failed: %s", e.stderr.decode()[:200])
        return video_path


# ══════════════════════════════════════════════════════════════════════
# POST-RENDER STEP C — GOOGLE DRIVE BACKUP (optional)
#
#  Uploads finished video + SRT to Drive folder "VaultMind".
#  Only runs if google-auth Drive scope is available.
# ══════════════════════════════════════════════════════════════════════

BF_W, BF_H, BF_FPS = VIDEO_W, VIDEO_H, FPS
BF_MODES = ["duel_1v1", "group_battle", "boss_battle", "color_infection", "gravity_race", "spike_accumulator"]
BF_PATTERNS = ["solid", "stripes", "dots", "eyes", "gradient", "neon"]


def _bf_vivid_color(h: float, s: float = 0.85, v: float = 0.98) -> tuple:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _bf_make_sprite(diameter: int, color: tuple, pattern: str, rng: random.Random) -> Image.Image:
    """Pre-render one ball as an antialiased RGBA sprite (supersampled 4x)."""
    ss = 4
    d = diameter
    big = Image.new("RGBA", (d * ss, d * ss), (0, 0, 0, 0))
    dr = ImageDraw.Draw(big)
    pad = 2 * ss
    dr.ellipse([pad, pad, d * ss - pad, d * ss - pad], fill=color + (255,))
    # subtle highlight for a "bouncy toy" look
    hl_r = d * ss * 0.18
    dr.ellipse([d * ss * 0.28 - hl_r, d * ss * 0.26 - hl_r,
                d * ss * 0.28 + hl_r, d * ss * 0.26 + hl_r], fill=(255, 255, 255, 90))
    if pattern == "stripes":
        for i in range(-d, d, max(6, d // 4)):
            dr.line([(i * ss, 0), ((i + d) * ss, d * ss)], fill=(255, 255, 255, 80), width=ss * 2)
    elif pattern == "dots":
        for _ in range(5):
            cx = rng.uniform(0.25, 0.75) * d * ss
            cy = rng.uniform(0.25, 0.75) * d * ss
            rr = rng.uniform(0.05, 0.09) * d * ss
            dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(255, 255, 255, 110))
    elif pattern == "eyes":
        ex, ey, er = d * ss * 0.5, d * ss * 0.42, d * ss * 0.11
        for sign in (-1, 1):
            cx = ex + sign * er * 1.2
            dr.ellipse([cx - er, ey - er, cx + er, ey + er], fill=(255, 255, 255, 235))
            dr.ellipse([cx - er * 0.45, ey - er * 0.3, cx + er * 0.45, ey + er * 0.7],
                       fill=(20, 20, 20, 255))
        dr.arc([ex - er * 1.6, ey + er * 0.5, ex + er * 1.6, ey + er * 2.4],
               start=15, end=165, fill=(30, 30, 30, 255), width=max(2, int(ss * 1.1)))
    elif pattern == "gradient":
        cx, cy = d * ss / 2, d * ss / 2
        for s in range(6, 0, -1):
            frac = s / 6
            rr = (d * ss / 2 - pad) * frac
            light = tuple(min(255, int(c + (255 - c) * 0.55 * (1 - frac))) for c in color)
            dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=light + (255,))
    elif pattern == "neon":
        cx, cy = d * ss / 2, d * ss / 2
        for rr, alpha in ((d * ss * 0.42, 55), (d * ss * 0.30, 95), (d * ss * 0.16, 160)):
            dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(255, 255, 255, alpha))
    return big.resize((d, d), Image.LANCZOS)


def _bf_ffmpeg_writer(path: str, w: int, h: int, fps: int) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{w}x{h}", "-framerate", str(fps),
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
         "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ─────────────────────────────────────────────────────────────────────
# VISUAL STYLE SYSTEM — rotates between distinct looks per video, modeled
# on real viral "ball battle" formats: pastel mobile-game HUD (big HP
# numbers, thick black arena), dark neon with permanent bounce-trail webs,
# and a neon hybrid with segmented pip health bars + VS title cards.
# ─────────────────────────────────────────────────────────────────────

BF_VISUAL_STYLES = {
    "pastel_hud": {
        "bg": (250, 240, 200), "arena": "square", "border_color": (25, 25, 25),
        "border_w": 7, "hp": "number", "text_color": (25, 25, 25),
        "text_outline": (255, 255, 255),
    },
    "dark_web": {
        "bg": (6, 6, 10), "arena": "circle_thin", "border_color": (240, 240, 245),
        "border_w": 2, "hp": "none", "text_color": (240, 240, 245),
        "text_outline": (0, 0, 0), "web_trails": True,
    },
    "neon_hybrid": {
        "bg": None, "arena": "circle_glow", "border_color": None,
        "border_w": 3, "hp": "pips", "text_color": (255, 255, 255),
        "text_outline": (0, 0, 0),
    },
}


def _bf_style_background(style: dict, w: int, h: int, bg_hex: str, accent: tuple) -> Image.Image:
    if style["bg"] is None:
        return _bf_background(w, h, bg_hex, accent)
    return Image.new("RGB", (w, h), style["bg"])


def _bf_draw_arena(dr: ImageDraw.ImageDraw, style: dict, center: tuple, R: float,
                   accent: tuple, t: float) -> None:
    col = style["border_color"] or accent
    if style["arena"] == "square":
        dr.rectangle([center[0] - R, center[1] - R, center[0] + R, center[1] + R],
                    outline=col, width=style["border_w"])
    elif style["arena"] == "circle_thin":
        dr.ellipse([center[0] - R, center[1] - R, center[0] + R, center[1] + R],
                   outline=col, width=style["border_w"])
    else:  # circle_glow — pulsing colored ring, matches the existing neon branding
        pulse = 6 + int(4 * math.sin(t * 2))
        dr.ellipse([center[0] - R - pulse, center[1] - R - pulse,
                   center[0] + R + pulse, center[1] + R + pulse], outline=accent, width=10)
        dr.ellipse([center[0] - R, center[1] - R, center[0] + R, center[1] + R],
                   outline=(255, 255, 255), width=3)


def _bf_draw_hp(style: dict, img: Image.Image, dr: ImageDraw.ImageDraw, pos: tuple,
                r: float, hp_frac: float, color: tuple, width_mult: float = 2.2) -> None:
    hp_frac = max(0.0, min(1.0, hp_frac))
    mode = style["hp"]
    if mode == "none":
        return
    if mode == "number":
        _bf_hud_text(img, (pos[0], pos[1]), str(int(round(hp_frac * 100))), int(r * 0.62),
                    color=(20, 20, 20) if sum(color) > 400 else (255, 255, 255))
        return
    bar_w = r * width_mult
    x, y = pos[0] - bar_w / 2, pos[1] - r - 24
    if mode == "pips":
        n_pips, gap = 5, 3
        seg_w = (bar_w - gap * (n_pips - 1)) / n_pips
        filled = round(hp_frac * n_pips)
        for i in range(n_pips):
            px = x + i * (seg_w + gap)
            col = (60, 220, 90) if i < filled else (55, 55, 60)
            dr.rectangle([px, y, px + seg_w, y + 10], fill=col, outline=(255, 255, 255), width=1)
        return
    # fallback: smooth bar
    dr.rectangle([x, y, x + bar_w, y + 9], fill=(35, 35, 40))
    barcol = (60, 220, 90) if hp_frac > 0.5 else (240, 195, 40) if hp_frac > 0.25 else (230, 60, 60)
    dr.rectangle([x, y, x + bar_w * hp_frac, y + 9], fill=barcol)
    dr.rectangle([x, y, x + bar_w, y + 9], outline=(255, 255, 255), width=1)


def _bf_vs_title_card(w: int, h: int, style: dict, left_name: str, left_color: tuple,
                      right_name: str, right_color: tuple) -> Image.Image:
    """A few frozen intro frames announcing the matchup, mimicking the
    clickbait 'X vs Y — who will win?' title cards every reference uses."""
    img = Image.new("RGB", (w, h), style["bg"] or (8, 8, 12))
    dr = ImageDraw.Draw(img)
    ty = h * 0.30
    _bf_hud_text(img, (w / 2, ty - 90), "WHO WILL WIN?", 56, color=style["text_color"])
    r = 70
    lx, rx = w * 0.28, w * 0.72
    dr.ellipse([lx - r, ty - r, lx + r, ty + r], fill=left_color, outline=(255, 255, 255), width=5)
    dr.ellipse([rx - r, ty - r, rx + r, ty + r], fill=right_color, outline=(255, 255, 255), width=5)
    _bf_hud_text(img, (w / 2, ty), "VS", 60, color=style["text_color"])
    _bf_hud_text(img, (lx, ty + r + 55), left_name.upper(), 38, color=left_color)
    _bf_hud_text(img, (rx, ty + r + 55), right_name.upper(), 38, color=right_color)
    return img


def _bf_background(w: int, h: int, bg_hex: str, accent: tuple) -> Image.Image:
    base = _hex(bg_hex)
    img = Image.new("RGB", (w, h), base)
    dr = ImageDraw.Draw(img)
    cx, cy = w / 2, h * 0.42
    for i, r in enumerate(range(int(w * 0.75), 0, -40)):
        fade = i / (w * 0.75 / 40)
        col = tuple(int(base[k] + (accent[k] - base[k]) * 0.05 * (1 - fade)) for k in range(3))
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    return img


def _bf_hud_text(img: Image.Image, xy: tuple, text: str, size: int,
                 color: tuple = (255, 255, 255), anchor: str = "mm") -> None:
    dr = ImageDraw.Draw(img)
    font = load_font(size)
    # cheap outline for legibility over busy footage
    x, y = xy
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        dr.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0), anchor=anchor)
    dr.text(xy, text, font=font, fill=color, anchor=anchor)


def _bf_gen_balls(n: int, radius_range: tuple, rng: random.Random,
                  palette: Optional[list] = None) -> list:
    balls = []
    for i in range(n):
        r = rng.uniform(*radius_range)
        color = palette[i % len(palette)] if palette else _bf_vivid_color(rng.random())
        pattern = rng.choice(BF_PATTERNS)
        sprite = _bf_make_sprite(int(r * 2), color, pattern, rng)
        balls.append({"id": i, "r": r, "color": color, "pattern": pattern, "sprite": sprite})
    return balls


def _bf_paste_sprite(img: Image.Image, sprite: Image.Image, pos: tuple, angle_rad: float) -> None:
    rotated = sprite.rotate(-math.degrees(angle_rad), resample=Image.BICUBIC, expand=False)
    x, y = pos
    img.paste(rotated, (int(x - sprite.width / 2), int(y - sprite.height / 2)), rotated)


# ─────────────────────────────────────────────────────────────────────
# SHARED FX — motion trails + shard-burst "shatter" effect, used by
# every sim mode below for a punchier, more "satisfying-ball-video" look
# ─────────────────────────────────────────────────────────────────────

def _bf_blend(base_bg: tuple, color: tuple, frac: float) -> tuple:
    frac = max(0.0, min(1.0, frac))
    return tuple(int(base_bg[k] + (color[k] - base_bg[k]) * frac) for k in range(3))


def _bf_update_trail(history: dict, ball_id, pos: tuple, max_len: int = 7) -> None:
    hist = history.setdefault(ball_id, [])
    hist.append((pos[0], pos[1]))
    if len(hist) > max_len:
        hist.pop(0)


def _bf_draw_trails(dr: ImageDraw.ImageDraw, history: dict, colors: dict, radii: dict,
                    base_bg: tuple) -> None:
    for bid, hist in history.items():
        n = len(hist)
        if n < 2 or bid not in colors:
            continue
        color, r = colors[bid], radii[bid]
        for i, (x, y) in enumerate(hist[:-1]):
            frac = (i + 1) / n
            blended = _bf_blend(base_bg, color, 0.30 * frac)
            rr = max(2, r * 0.55 * frac)
            dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=blended)


def _bf_spawn_burst(bursts: list, pos: tuple, color: tuple, t: float) -> None:
    bursts.append({"x": pos[0], "y": pos[1], "color": color, "t0": t})


def _bf_draw_bursts(dr: ImageDraw.ImageDraw, bursts: list, t: float, base_bg: tuple,
                    life: float = 0.32, n_shards: int = 8) -> list:
    """Draws + prunes active shard-burst effects (a ball 'shattering' into
    fading fragments); returns the still-active subset."""
    active = []
    for burst in bursts:
        age = t - burst["t0"]
        if age > life:
            continue
        active.append(burst)
        frac = age / life
        jitter = (burst["x"] * 0.017) % (2 * math.pi)
        for k in range(n_shards):
            ang = (2 * math.pi / n_shards) * k + jitter
            dist = 8 + frac * 60
            bx = burst["x"] + math.cos(ang) * dist
            by = burst["y"] + math.sin(ang) * dist
            blended = _bf_blend(base_bg, burst["color"], 1 - frac)
            sz = max(1, 5 * (1 - frac))
            dr.ellipse([bx - sz, by - sz, bx + sz, by + sz], fill=blended)
    return active


# ─────────────────────────────────────────────────────────────────────
# COMBAT SYSTEM — abilities, damage resolution, HP bars.
# Shared by duel_1v1 / group_battle / boss_battle.
# ─────────────────────────────────────────────────────────────────────

BF_ABILITIES = {
    "tank":      {"tag": "TANK",  "hp_mult": 1.7,  "speed_mult": 0.78, "dmg_mult": 1.0,  "weapon": "hammer"},
    "speedster": {"tag": "FAST",  "hp_mult": 0.65, "speed_mult": 1.55, "dmg_mult": 0.7,  "weapon": "dagger"},
    "bomber":    {"tag": "BOMB",  "hp_mult": 1.0,  "speed_mult": 1.0,  "dmg_mult": 1.0,  "weapon": "bomb"},
    "vampire":   {"tag": "VAMP",  "hp_mult": 0.9,  "speed_mult": 1.05, "dmg_mult": 0.85, "weapon": "dagger"},
    "berserker": {"tag": "RAGE",  "hp_mult": 1.0,  "speed_mult": 1.0,  "dmg_mult": 1.0,  "weapon": "mace"},
    "shield":    {"tag": "GUARD", "hp_mult": 1.0,  "speed_mult": 0.95, "dmg_mult": 1.0,  "weapon": "shield"},
}

# Pixel-art weapon sprites: small blocky grids drawn as flat-color squares and
# orbited around each fighter like the "Wrench"/"Orbital" ball-fight reference
# videos, instead of abstract rings or lines.
_BF_WEAPON_GRIDS = {
    "hammer": [
        "....DDDDDD..", "....DMMMMD..", "....DMMMMD..", "....DMMMMD..",
        "....DDDDDD..", "......HH....", "......HH....", "......HH....",
        "......HH....", "......HH....", "......HH....", ".....HHHH...",
    ],
    "dagger": [
        ".......M....", "......MMM...", "......MMM...", "......MMM...",
        "......MMM...", "......MMM...", "..DGGGGGGGD.", "..DGGGGGGGD.",
        "......H.....", "......H.....", "......H.....", "......H.....",
    ],
    "mace": [
        "....DD......", "...DMMD.....", "..DMMMMD....", "..DMMMMD....",
        "..DMMMMD....", "...DMMD.....", "....DD......", ".....H......",
        ".....H......", ".....H......", ".....H......", ".....H......",
    ],
    "bomb": [
        "......F.....", ".....FD.....", "....OOO.....", "...OBBBO....",
        "..OBWBBBO...", "..OBWBBBO...", ".OBBBBBBBO..", ".OBBBBBBBO..",
        ".OBBBBBBBO..", "..OBBBBBO...", "..OBBBBBO...", "...OOOOO....",
    ],
    "shield": [
        "DDDDDDDDDDDD", "DSSSSSSSSSSD", "DSSSSSSSSSSD", "DSSSSSSSSSSD",
        "DSSSSSSSSSSD", "DSSSSSSSSSSD", ".DSSSSSSSSD.", ".DSSSSSSSSD.",
        "..DSSSSSSD..", "..DSSSSSSD..", "...DSSSSD...", "....DDDD....",
    ],
}
_BF_WEAPON_COLORS = {
    "H": (120, 80, 45, 255), "M": (195, 198, 208, 255), "D": (75, 77, 88, 255),
    "G": (230, 180, 40, 255), "F": (255, 170, 40, 255), "B": (48, 49, 60, 255),
    "O": (178, 181, 196, 255), "W": (240, 240, 250, 255),
    "S": (70, 145, 225, 255),
}


def _bf_make_pixel_sprite(grid: list, px: int = 6) -> Image.Image:
    h, w = len(grid), len(grid[0])
    img = Image.new("RGBA", (w * px, h * px), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch != ".":
                dr.rectangle([x * px, y * px, x * px + px - 1, y * px + px - 1],
                            fill=_BF_WEAPON_COLORS[ch])
    return img


_BF_WEAPON_SPRITES = {name: _bf_make_pixel_sprite(grid) for name, grid in _BF_WEAPON_GRIDS.items()}


def _bf_make_fighter(fid: int, r: float, color: tuple, ability: str, rng: random.Random,
                     base_hp: float = 100.0) -> dict:
    ab = BF_ABILITIES[ability]
    max_hp = base_hp * ab["hp_mult"] * _BF_HP_SCALE
    return {"id": fid, "r": r, "color": color, "ability": ability,
            "hp": max_hp, "max_hp": max_hp, "alive": True, "shield_up": ability == "shield",
            "orbit_n": 1 if ability == "vampire" else 0,
            "orbit_spin": rng.uniform(0, 2 * math.pi)}


def _bf_draw_flat_fighter(dr: ImageDraw.ImageDraw, f: dict) -> None:
    """Plain flat-fill circle with a white outline — matches how every
    reference video (airballsfight, battleofballs, bluejason10, shadow_recall)
    actually renders its fighters, instead of a patterned/textured sprite."""
    p, r = f["body"].position, f["r"]
    dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=tuple(f["color"]),
              outline=(255, 255, 255), width=max(2, int(r * 0.06)))


def _bf_draw_heart(dr: ImageDraw.ImageDraw, x: float, y: float, s: float,
                   color: tuple = (255, 55, 95)) -> None:
    r = s * 0.5
    dr.ellipse([x - r * 1.6, y - r * 0.8, x - r * 0.2, y + r * 0.8], fill=color)
    dr.ellipse([x + r * 0.2, y - r * 0.8, x + r * 1.6, y + r * 0.8], fill=color)
    dr.polygon([(x - r * 1.5, y + r * 0.15), (x + r * 1.5, y + r * 0.15), (x, y + r * 2.0)], fill=color)


def _bf_fighter_stat_text(f: dict, t: float) -> str:
    """A number that visibly climbs over the fight — the exact bottom-HUD
    quirk from the reference stat-battle videos (Max Speed / Dodge Chance
    creeping up over time)."""
    if f["ability"] == "vampire":
        return f"ORBITS {f['orbit_n']}"
    return f"PWR {1.0 + t * 0.12:.2f}"


def _bf_resolve_clash(a: dict, b: dict, impact_speed: float, t: float,
                      events: list, bursts: list, mult_a: float = 1.0, mult_b: float = 1.0) -> None:
    """Both fighters hit each other from the same collision, resolved
    simultaneously (damage computed from pre-hit HP/shield state for both
    sides before anything is applied) — so whichever fighter happens to be
    iterated first doesn't get a free kill before the other can retaliate.
    mult_a/mult_b scale a's/b's outgoing damage (e.g. a boss hitting harder)."""
    def compute(attacker, defender, mult):
        dmg = max(3.0, impact_speed * 0.05) * BF_ABILITIES[attacker["ability"]]["dmg_mult"] * mult
        if attacker["ability"] == "berserker":
            dmg *= 1 + (1 - attacker["hp"] / attacker["max_hp"]) * 0.9
        blocked = defender["ability"] == "shield" and defender["shield_up"]
        return dmg, blocked

    dmg_a, blocked_b = compute(a, b, mult_a)  # a hits b
    dmg_b, blocked_a = compute(b, a, mult_b)  # b hits a

    if blocked_b:
        b["shield_up"] = False
        events.append({"t": round(t, 2), "type": "block", "id": b["id"], "color": b["color"]})
        dmg_a = 0
    if blocked_a:
        a["shield_up"] = False
        events.append({"t": round(t, 2), "type": "block", "id": a["id"], "color": a["color"]})
        dmg_b = 0

    b["hp"] -= dmg_a
    a["hp"] -= dmg_b
    if a["ability"] == "vampire" and dmg_a > 0:
        a["hp"] = min(a["max_hp"], a["hp"] + dmg_a * 0.4)
        a["orbit_n"] = min(10, a["orbit_n"] + 1)
    if b["ability"] == "vampire" and dmg_b > 0:
        b["hp"] = min(b["max_hp"], b["hp"] + dmg_b * 0.4)
        b["orbit_n"] = min(10, b["orbit_n"] + 1)

    for f in (a, b):
        if f["hp"] <= 0 and f["alive"]:
            f["alive"] = False
            events.append({"t": round(t, 2), "type": "eliminated", "id": f["id"], "color": f["color"]})
            _bf_spawn_burst(bursts, f["body"].position, f["color"], t)


_BF_WEAPON_DMG = 2.4   # base damage of one weapon hit (balance-tested)
_BF_WEAPON_CD = 0.6    # seconds between hits per attacker/target pair
_BF_HP_SCALE = 1.5     # global multiplier on every fighter's HP pool

_BF_SPIN = {"tank": 0.065, "speedster": 0.15, "bomber": 0.09, "vampire": 0.10,
            "berserker": 0.11, "shield": 0.08}


def _bf_weapon_slots(f: dict) -> list:
    """World position, hit radius and angle of every weapon orbiting fighter f."""
    ability = f["ability"]
    if ability == "shield" and not f["shield_up"]:
        return []
    count = f["orbit_n"] if ability == "vampire" else 1
    if count <= 0:
        return []
    r = f["r"]
    wsize = max(22, int(r * 1.35))
    orbit_dist = r + wsize * 0.5 + 8
    x, y = f["body"].position
    return [(x + math.cos(f["orbit_spin"] + (2 * math.pi / count) * i) * orbit_dist,
             y + math.sin(f["orbit_spin"] + (2 * math.pi / count) * i) * orbit_dist,
             wsize * 0.42, f["orbit_spin"] + (2 * math.pi / count) * i, wsize)
            for i in range(count)]


def _bf_step_weapons(fighters: list) -> None:
    for f in fighters:
        rate = _BF_SPIN[f["ability"]]
        if f["ability"] == "berserker":
            rate *= 1 + (1 - f["hp"] / f["max_hp"]) * 0.8  # swings faster the angrier it gets
        f["orbit_spin"] += rate


def _bf_weapon_strike(a: dict, d: dict, t: float, events: list, bursts: list, pt: tuple) -> None:
    dmg = _BF_WEAPON_DMG * BF_ABILITIES[a["ability"]]["dmg_mult"] * a.get("weapon_mult", 1.0)
    if a["ability"] == "berserker":
        dmg *= 1 + (1 - a["hp"] / a["max_hp"]) * 0.9
    if d["ability"] == "shield" and d["shield_up"]:
        d["shield_up"] = False
        events.append({"t": round(t, 2), "type": "block", "id": d["id"], "color": d["color"]})
        return
    d["hp"] -= dmg
    if a["ability"] == "vampire":
        a["hp"] = min(a["max_hp"], a["hp"] + dmg * 0.4)
        a["orbit_n"] = min(10, a["orbit_n"] + 1)
    events.append({"t": round(t, 2), "type": "hit", "impact": 0.55})
    _bf_spawn_burst(bursts, pt, a["color"], t)
    if d["hp"] <= 0 and d["alive"]:
        d["alive"] = False
        events.append({"t": round(t, 2), "type": "eliminated", "id": d["id"], "color": d["color"]})
        _bf_spawn_burst(bursts, d["body"].position, d["color"], t)


def _bf_weapon_hits(fighters: list, enemies_of, t: float, events: list, bursts: list,
                    cooldowns: dict) -> None:
    """A weapon that physically touches an enemy ball hurts it — so what you
    see swinging is what deals the damage (0.5 s cooldown per attacker/target)."""
    for f in fighters:
        if not f["alive"]:
            continue
        for (wx, wy, wr, _ang, _ws) in _bf_weapon_slots(f):
            for e in enemies_of(f):
                if not e["alive"]:
                    continue
                ex, ey = e["body"].position
                if math.hypot(wx - ex, wy - ey) < e["r"] + wr:
                    key = (f["id"], e["id"])
                    if t - cooldowns.get(key, -9.0) < _BF_WEAPON_CD:
                        continue
                    cooldowns[key] = t
                    _bf_weapon_strike(f, e, t, events, bursts, (wx, wy))


def _bf_cull_dead(space, fighters: list) -> None:
    """Remove eliminated fighters from the physics world — otherwise their
    invisible bodies keep bouncing the survivors around."""
    for f in fighters:
        if not f["alive"] and not f.get("removed"):
            space.remove(f["body"], f["shape"])
            f["removed"] = True


def _bf_draw_ability_geo(img: Image.Image, dr: ImageDraw.ImageDraw, f: dict, t: float) -> None:
    """Pixel-art weapons swinging on a dotted tether around the ball, like the
    'Wrench'/'Orbital' ball-fight reference videos. Berserker keeps a pulsing
    rage aura underneath its mace."""
    r, ability = f["r"], f["ability"]
    x, y = f["body"].position

    if ability == "berserker":
        rage = 1 - f["hp"] / f["max_hp"]
        if rage > 0.15:
            pulse = int(6 + 10 * rage * (0.6 + 0.4 * math.sin(t * 10)))
            aura = tuple(min(255, int(c + (255 - c) * 0.3)) for c in (200, 30, 30))
            dr.ellipse([x - r - pulse, y - r - pulse, x + r + pulse, y + r + pulse],
                      outline=aura, width=3)

    weapon = _BF_WEAPON_SPRITES[BF_ABILITIES[ability]["weapon"]]
    flip = 180 if ability == "shield" else 0
    for i, (wx, wy, _wr, ang, wsize) in enumerate(_bf_weapon_slots(f)):
        # dotted tether from the ball's edge out to the weapon (pixel squares)
        sq = max(3, int(r * 0.06))
        for k in range(4):
            frac = (k + 0.5) / 4
            d = r + 2 + (math.hypot(wx - x, wy - y) - r - wsize * 0.35) * frac
            tx, ty = x + math.cos(ang) * d, y + math.sin(ang) * d
            dr.rectangle([tx - sq / 2, ty - sq / 2, tx + sq / 2, ty + sq / 2], fill=(90, 92, 104))
        wobble = 18 * math.sin(t * 7 + i * 1.7)
        spr = weapon.resize((wsize, wsize), Image.NEAREST)
        spun = spr.rotate(-math.degrees(ang) - 90 + flip + wobble, resample=Image.NEAREST, expand=True)
        img.paste(spun, (int(wx - spun.width / 2), int(wy - spun.height / 2)), spun)


def _bf_draw_hp_bar(dr: ImageDraw.ImageDraw, pos: tuple, r: float, hp_frac: float,
                    width_mult: float = 2.2, height: int = 9) -> None:
    hp_frac = max(0.0, min(1.0, hp_frac))
    bar_w = r * width_mult
    x, y = pos[0] - bar_w / 2, pos[1] - r - height - 16
    dr.rectangle([x, y, x + bar_w, y + height], fill=(35, 35, 40))
    col = (60, 220, 90) if hp_frac > 0.5 else (240, 195, 40) if hp_frac > 0.25 else (230, 60, 60)
    dr.rectangle([x, y, x + bar_w * hp_frac, y + height], fill=col)
    dr.rectangle([x, y, x + bar_w, y + height], outline=(255, 255, 255), width=1)


# ─────────────────────────────────────────────────────────────────────
# MODE 1 — 1v1 DUEL (two fighters, HP bars, abilities, last one standing)
# ─────────────────────────────────────────────────────────────────────

def simulate_duel_1v1(target_dur: float, out_path: str, rng: random.Random, style: dict) -> tuple:
    space = pymunk.Space(); space.gravity = (0, 0)
    center = (BF_W / 2, BF_H * 0.44)
    R = BF_W * 0.44
    accent = NICHES["ballfight"]["glow"]
    base_bg_col = style["bg"] or _hex(NICHES["ballfight"]["bg"])

    abilities = rng.sample(list(BF_ABILITIES.keys()), 2)
    hues = rng.sample([i / 6 for i in range(6)], 2)
    fighters = []
    for i in range(2):
        color = _bf_vivid_color(hues[i])
        f = _bf_make_fighter(i, 78, color, abilities[i], rng, base_hp=100.0)
        body = pymunk.Body(3.0, pymunk.moment_for_circle(3.0, 0, f["r"]))
        ang = i * math.pi
        body.position = (center[0] + math.cos(ang) * R * 0.5, center[1] + math.sin(ang) * R * 0.5)
        speed = 170 * BF_ABILITIES[f["ability"]]["speed_mult"]
        vang = rng.uniform(0, 2 * math.pi)
        body.velocity = (math.cos(vang) * speed, math.sin(vang) * speed)
        shape = pymunk.Circle(body, f["r"]); shape.elasticity = 1.0; shape.friction = 0.02
        space.add(body, shape)
        f["body"], f["shape"] = body, shape
        fighters.append(f)

    events = [{"t": 0.0, "type": "start",
              "fighters": [{"color": f["color"], "ability": f["ability"]} for f in fighters]}]
    touching, bursts, trail_hist, web_pts = set(), [], {}, {0: [], 1: []}
    proc = _bf_ffmpeg_writer(out_path, BF_W, BF_H, BF_FPS)
    bg = _bf_style_background(style, BF_W, BF_H, NICHES["ballfight"]["bg"], _hex(accent))
    n_frames = int(target_dur * BF_FPS)
    winner, finish_hold = None, None
    heart, next_heart_t = None, rng.uniform(3.5, 6.0)
    weapon_cd = {}

    title = _bf_vs_title_card(BF_W, BF_H, style,
                              BF_ABILITIES[fighters[0]["ability"]]["tag"], fighters[0]["color"],
                              BF_ABILITIES[fighters[1]["ability"]]["tag"], fighters[1]["color"])
    for _ in range(int(BF_FPS * 1.4)):
        proc.stdin.write(np.array(title).tobytes())

    for fr in range(n_frames):
        t = fr / BF_FPS
        live = [f for f in fighters if f["alive"]]
        if len(live) == 2:
            a, b = live
            for me, opp in ((a, b), (b, a)):
                direction = opp["body"].position - me["body"].position
                if direction.length > 1:
                    me["body"].velocity += direction.normalized() * 9
                    cap = 260 * BF_ABILITIES[me["ability"]]["speed_mult"]
                    if me["body"].velocity.length > cap:
                        me["body"].velocity = me["body"].velocity.normalized() * cap
        for _ in range(2):
            space.step(1 / (BF_FPS * 2))
        live = [f for f in fighters if f["alive"]]
        for f in live:
            p = f["body"].position
            d = math.hypot(p[0] - center[0], p[1] - center[1])
            if d + f["r"] > R:
                nx, ny = (p[0] - center[0]) / max(d, 1e-6), (p[1] - center[1]) / max(d, 1e-6)
                vx, vy = f["body"].velocity
                dot = vx * nx + vy * ny
                if dot > 0:
                    f["body"].velocity = (vx - 2 * dot * nx, vy - 2 * dot * ny)
                    web_pts[f["id"]].append((p[0], p[1]))
        _bf_step_weapons(live)
        _bf_weapon_hits(live, lambda f: [o for o in live if o is not f], t, events, bursts, weapon_cd)
        _bf_cull_dead(space, fighters)
        live = [f for f in fighters if f["alive"]]
        if len(live) == 2:
            a, b = live
            pa, pb = a["body"].position, b["body"].position
            dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            if dist < a["r"] + b["r"]:
                if "hit" not in touching:
                    speed = math.hypot(*a["body"].velocity) + math.hypot(*b["body"].velocity)
                    events.append({"t": round(t, 2), "type": "hit", "impact": min(1.0, speed / 900)})
                    _bf_resolve_clash(a, b, speed, t, events, bursts)
                    touching.add("hit")
            else:
                touching.discard("hit")

        live = [f for f in fighters if f["alive"]]
        if len(live) <= 1 and winner is None:
            winner = live[0] if live else max(fighters, key=lambda f: f["hp"])
            events.append({"t": round(t, 2), "type": "winner", "color": winner["color"]})
            finish_hold = t + 1.8

        if heart is None and winner is None and t >= next_heart_t:
            heart = {"x": rng.uniform(center[0] - R * 0.55, center[0] + R * 0.55),
                    "y": rng.uniform(center[1] - R * 0.55, center[1] + R * 0.55)}
        if heart:
            for f in live:
                p = f["body"].position
                if math.hypot(p[0] - heart["x"], p[1] - heart["y"]) < f["r"] + 22:
                    f["hp"] = min(f["max_hp"], f["hp"] + f["max_hp"] * 0.18)
                    events.append({"t": round(t, 2), "type": "heal", "id": f["id"], "color": f["color"]})
                    heart = None
                    next_heart_t = t + rng.uniform(4.5, 7.5)
                    break

        img = bg.copy()
        dr = ImageDraw.Draw(img)
        _bf_draw_arena(dr, style, center, R, _hex(accent), t)
        if style.get("web_trails"):
            for f in live:
                pts = web_pts[f["id"]]
                if pts:
                    px, py = f["body"].position
                    for (bx, by) in pts:
                        dr.line([bx, by, px, py], fill=f["color"], width=2)
        else:
            for f in live:
                _bf_update_trail(trail_hist, f["id"], f["body"].position)
            _bf_draw_trails(dr, trail_hist, {f["id"]: f["color"] for f in live},
                            {f["id"]: f["r"] for f in live}, base_bg_col)
        bursts = _bf_draw_bursts(dr, bursts, t, base_bg_col)
        if heart:
            _bf_draw_heart(dr, heart["x"], heart["y"], 30)
        for f in live:
            _bf_draw_flat_fighter(dr, f)
            _bf_draw_ability_geo(img, dr, f, t)
            _bf_draw_hp(style, img, dr, f["body"].position, f["r"], f["hp"] / f["max_hp"], f["color"])
            _bf_hud_text(img, (f["body"].position[0], f["body"].position[1] + f["r"] + 34),
                        BF_ABILITIES[f["ability"]]["tag"], 24, color=f["color"])
        _bf_hud_text(img, (BF_W / 2, 90), "FIGHT!" if winner is None else "WINNER!", 78,
                    color=style["text_color"])
        if style["hp"] == "number":  # pastel_hud — the escalating bottom stat line
            remaining = max(0, (finish_hold or target_dur) - t)
            _bf_hud_text(img, (center[0] - R, center[1] + R + 40), f"TIME LEFT: {remaining:.0f}s",
                        26, color=style["text_color"], anchor="lm")
            stat_line = "    ".join(_bf_fighter_stat_text(f, t) for f in live)
            _bf_hud_text(img, (center[0] + R, center[1] + R + 40), stat_line,
                        26, color=style["text_color"], anchor="rm")
        proc.stdin.write(np.array(img).tobytes())

        if finish_hold and t >= finish_hold:
            break

    proc.stdin.close(); proc.wait()
    if winner is None:
        winner = max(fighters, key=lambda f: f["hp"])
        events.append({"t": round(target_dur, 2), "type": "winner", "color": winner["color"]})
    return events, {"mode": "duel_1v1", "winner_color": winner["color"],
                     "duration": (finish_hold or target_dur) + 1.4}


# ─────────────────────────────────────────────────────────────────────
# MODE 2 — GROUP BATTLE (team vs. team, abilities, HP bars, last team standing)
# ─────────────────────────────────────────────────────────────────────

def simulate_group_battle(target_dur: float, out_path: str, rng: random.Random, style: dict) -> tuple:
    space = pymunk.Space(); space.gravity = (0, 0)
    center = (BF_W / 2, BF_H * 0.44)
    R = BF_W * 0.45
    accent = NICHES["ballfight"]["glow"]
    base_bg_col = style["bg"] or _hex(NICHES["ballfight"]["bg"])
    per_team = rng.randint(3, 4)
    team_colors = [_bf_vivid_color(0.0), _bf_vivid_color(0.55)]  # red-ish vs blue-ish, always readable

    fighters, fid = [], 0
    for i in range(per_team):
        for team, tcol in enumerate(team_colors):
            ability = rng.choice(list(BF_ABILITIES.keys()))
            f = _bf_make_fighter(fid, 40, tcol, ability, rng, base_hp=70.0)
            f["team"] = team
            ang = rng.uniform(0, 2 * math.pi); dist = rng.uniform(0, R * 0.6)
            body = pymunk.Body(1.5, pymunk.moment_for_circle(1.5, 0, f["r"]))
            body.position = (center[0] + math.cos(ang) * dist, center[1] + math.sin(ang) * dist)
            speed = rng.uniform(130, 210) * BF_ABILITIES[ability]["speed_mult"]
            vang = rng.uniform(0, 2 * math.pi)
            body.velocity = (math.cos(vang) * speed, math.sin(vang) * speed)
            shape = pymunk.Circle(body, f["r"]); shape.elasticity = 1.0; shape.friction = 0.02
            space.add(body, shape)
            f["body"], f["shape"] = body, shape
            fighters.append(f); fid += 1

    events = [{"t": 0.0, "type": "start", "teams": 2, "n": len(fighters)}]
    touching, bursts, trail_hist = set(), [], {}
    web_pts = {f["id"]: [] for f in fighters}
    weapon_cd = {}
    proc = _bf_ffmpeg_writer(out_path, BF_W, BF_H, BF_FPS)
    bg = _bf_style_background(style, BF_W, BF_H, NICHES["ballfight"]["bg"], _hex(accent))
    n_frames = int(target_dur * BF_FPS)
    winner_color, finish_hold = None, None

    title = _bf_vs_title_card(BF_W, BF_H, style, "TEAM RED", team_colors[0], "TEAM BLUE", team_colors[1])
    for _ in range(int(BF_FPS * 1.4)):
        proc.stdin.write(np.array(title).tobytes())

    for fr in range(n_frames):
        t = fr / BF_FPS
        live = [f for f in fighters if f["alive"]]
        for f in live:
            enemies = [e for e in live if e["team"] != f["team"]]
            if enemies:
                nearest = min(enemies, key=lambda e: (e["body"].position - f["body"].position).length)
                direction = nearest["body"].position - f["body"].position
                if direction.length > 1:
                    f["body"].velocity += direction.normalized() * 14
                    cap = 230 * BF_ABILITIES[f["ability"]]["speed_mult"]
                    if f["body"].velocity.length > cap:
                        f["body"].velocity = f["body"].velocity.normalized() * cap
        for _ in range(2):
            space.step(1 / (BF_FPS * 2))
        live = [f for f in fighters if f["alive"]]
        for f in live:
            p = f["body"].position
            d = math.hypot(p[0] - center[0], p[1] - center[1])
            if d + f["r"] > R:
                nx, ny = (p[0] - center[0]) / max(d, 1e-6), (p[1] - center[1]) / max(d, 1e-6)
                vx, vy = f["body"].velocity
                dot = vx * nx + vy * ny
                if dot > 0:
                    f["body"].velocity = (vx - 2 * dot * nx, vy - 2 * dot * ny)
                    web_pts[f["id"]].append((p[0], p[1]))
        _bf_step_weapons(live)
        _bf_weapon_hits(live, lambda f: [e for e in live if e["team"] != f["team"]], t, events, bursts, weapon_cd)
        _bf_cull_dead(space, fighters)
        live = [f for f in fighters if f["alive"]]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                pa, pb = a["body"].position, b["body"].position
                dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                key = (a["id"], b["id"])
                if dist < a["r"] + b["r"]:
                    if a["team"] == b["team"]:
                        continue  # friendly bounce, no damage
                    if key not in touching:
                        speed = math.hypot(*a["body"].velocity) + math.hypot(*b["body"].velocity)
                        _bf_resolve_clash(a, b, speed, t, events, bursts)
                    touching.add(key)
                else:
                    touching.discard(key)

        live = [f for f in fighters if f["alive"]]
        teams_left = {f["team"] for f in live}
        if winner_color is None and len(teams_left) <= 1:
            winner_team = next(iter(teams_left)) if teams_left else 0
            winner_color = list(team_colors[winner_team])
            events.append({"t": round(t, 2), "type": "winner", "color": winner_color})
            finish_hold = t + 1.8

        img = bg.copy()
        dr = ImageDraw.Draw(img)
        _bf_draw_arena(dr, style, center, R, _hex(accent), t)
        if style.get("web_trails"):
            for f in live:
                pts = web_pts[f["id"]]
                if pts:
                    px, py = f["body"].position
                    for (bx, by) in pts:
                        dr.line([bx, by, px, py], fill=f["color"], width=1)
        else:
            for f in live:
                _bf_update_trail(trail_hist, f["id"], f["body"].position)
            _bf_draw_trails(dr, trail_hist, {f["id"]: f["color"] for f in live},
                            {f["id"]: f["r"] for f in live}, base_bg_col)
        bursts = _bf_draw_bursts(dr, bursts, t, base_bg_col)
        for f in live:
            _bf_draw_flat_fighter(dr, f)
            _bf_draw_ability_geo(img, dr, f, t)
            _bf_draw_hp(style, img, dr, f["body"].position, f["r"], f["hp"] / f["max_hp"], f["color"],
                       width_mult=1.8)
        n0 = sum(1 for f in live if f["team"] == 0); n1 = sum(1 for f in live if f["team"] == 1)
        _bf_hud_text(img, (BF_W / 2, 90),
                    f"{n0} vs {n1}" if winner_color is None else "WINNER!", 68, color=style["text_color"])
        proc.stdin.write(np.array(img).tobytes())

        if finish_hold and t >= finish_hold:
            break

    proc.stdin.close(); proc.wait()
    if winner_color is None:
        live = [f for f in fighters if f["alive"]]
        def team_score(team):
            members = [f for f in live if f["team"] == team]
            return (len(members), sum(f["hp"] for f in members))
        best_team = max((0, 1), key=team_score)
        winner_color = list(team_colors[best_team])
        events.append({"t": round(target_dur, 2), "type": "winner", "color": winner_color})
    return events, {"mode": "group_battle", "winner_color": winner_color,
                     "n_balls": len(fighters), "duration": (finish_hold or target_dur) + 1.4}


# ─────────────────────────────────────────────────────────────────────
# MODE 3 — BOSS BATTLE (one huge boss vs. a team of heroes, boss HP bar
# up top, heroes must bring it down before it wipes the team)
# ─────────────────────────────────────────────────────────────────────

def simulate_boss_battle(target_dur: float, out_path: str, rng: random.Random, style: dict) -> tuple:
    space = pymunk.Space(); space.gravity = (0, 0)
    center = (BF_W / 2, BF_H * 0.46)
    R = BF_W * 0.46
    accent = NICHES["ballfight"]["glow"]
    base_bg_col = style["bg"] or _hex(NICHES["ballfight"]["bg"])
    boss_color = (215, 30, 40)

    n_heroes = rng.randint(4, 6)
    boss = _bf_make_fighter(0, 150, boss_color, "tank", rng, base_hp=n_heroes * 58.0)
    boss["team"] = "boss"
    boss["weapon_mult"] = 1.4
    bbody = pymunk.Body(20.0, pymunk.moment_for_circle(20.0, 0, boss["r"]))
    bbody.position = center
    bbody.velocity = (rng.choice([-1, 1]) * 90, rng.choice([-1, 1]) * 90)
    bshape = pymunk.Circle(bbody, boss["r"]); bshape.elasticity = 1.0; bshape.friction = 0.02
    space.add(bbody, bshape)
    boss["body"], boss["shape"] = bbody, bshape

    heroes = []
    hero_hue = rng.random()
    for i in range(n_heroes):
        ability = rng.choice(list(BF_ABILITIES.keys()))
        color = _bf_vivid_color(hero_hue + i * 0.11)
        h = _bf_make_fighter(i + 1, 34, color, ability, rng, base_hp=100.0)
        h["team"] = "hero"
        ang = rng.uniform(0, 2 * math.pi); dist = R * rng.uniform(0.55, 0.8)
        body = pymunk.Body(1.2, pymunk.moment_for_circle(1.2, 0, h["r"]))
        body.position = (center[0] + math.cos(ang) * dist, center[1] + math.sin(ang) * dist)
        speed = rng.uniform(160, 240) * BF_ABILITIES[ability]["speed_mult"]
        vang = rng.uniform(0, 2 * math.pi)
        body.velocity = (math.cos(vang) * speed, math.sin(vang) * speed)
        shape = pymunk.Circle(body, h["r"]); shape.elasticity = 1.0; shape.friction = 0.02
        space.add(body, shape)
        h["body"], h["shape"] = body, shape
        heroes.append(h)

    events = [{"t": 0.0, "type": "start", "n": n_heroes + 1}]
    touching, bursts, trail_hist = set(), [], {}
    weapon_cd = {}
    proc = _bf_ffmpeg_writer(out_path, BF_W, BF_H, BF_FPS)
    bg = _bf_style_background(style, BF_W, BF_H, NICHES["ballfight"]["bg"], _hex(accent))
    n_frames = int(target_dur * BF_FPS)
    outcome, finish_hold = None, None  # "heroes" or "boss"

    title = _bf_vs_title_card(BF_W, BF_H, style, "HEROES", heroes[0]["color"], "BOSS", boss_color)
    for _ in range(int(BF_FPS * 1.4)):
        proc.stdin.write(np.array(title).tobytes())

    for fr in range(n_frames):
        t = fr / BF_FPS
        # gentle homing force so the boss actively hunts the nearest hero
        live_heroes = [h for h in heroes if h["alive"]]
        if live_heroes and boss["alive"]:
            nearest = min(live_heroes, key=lambda h: (h["body"].position - bbody.position).length)
            direction = (nearest["body"].position - bbody.position)
            if direction.length > 1:
                direction = direction.normalized()
                bbody.velocity += direction * 14
                speed_cap = 130
                if bbody.velocity.length > speed_cap:
                    bbody.velocity = bbody.velocity.normalized() * speed_cap
        if boss["alive"]:
            for h in live_heroes:
                to_boss = bbody.position - h["body"].position
                if to_boss.length > 1:
                    h["body"].velocity += to_boss.normalized() * 10
                    hcap = 210 * BF_ABILITIES[h["ability"]]["speed_mult"]
                    if h["body"].velocity.length > hcap:
                        h["body"].velocity = h["body"].velocity.normalized() * hcap
        for _ in range(2):
            space.step(1 / (BF_FPS * 2))

        all_live = ([boss] if boss["alive"] else []) + [h for h in heroes if h["alive"]]
        for f in all_live:
            p = f["body"].position
            d = math.hypot(p[0] - center[0], p[1] - center[1])
            if d + f["r"] > R:
                nx, ny = (p[0] - center[0]) / max(d, 1e-6), (p[1] - center[1]) / max(d, 1e-6)
                vx, vy = f["body"].velocity
                dot = vx * nx + vy * ny
                if dot > 0:
                    f["body"].velocity = (vx - 2 * dot * nx, vy - 2 * dot * ny)

        now_live = ([boss] if boss["alive"] else []) + [h for h in heroes if h["alive"]]
        _bf_step_weapons(now_live)
        _bf_weapon_hits(now_live,
                        lambda f: ([boss] if boss["alive"] else []) if f["team"] == "hero"
                        else [h for h in heroes if h["alive"]],
                        t, events, bursts, weapon_cd)
        _bf_cull_dead(space, [boss] + heroes)
        live_heroes = [h for h in heroes if h["alive"]]
        if boss["alive"]:
            for h in live_heroes:
                pa, pb = boss["body"].position, h["body"].position
                dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                key = ("boss", h["id"])
                if dist < boss["r"] + h["r"]:
                    if key not in touching:
                        speed = math.hypot(*boss["body"].velocity) + math.hypot(*h["body"].velocity)
                        _bf_resolve_clash(boss, h, speed, t, events, bursts, mult_a=1.1, mult_b=1.0)
                    touching.add(key)
                else:
                    touching.discard(key)
        live_heroes = [h for h in heroes if h["alive"]]

        live_heroes = [h for h in heroes if h["alive"]]
        if outcome is None:
            if not boss["alive"]:
                outcome = "heroes"
                win_color_now = live_heroes[0]["color"] if live_heroes else heroes[0]["color"]
                events.append({"t": round(t, 2), "type": "winner", "color": list(win_color_now)})
                finish_hold = t + 2.0
            elif not live_heroes:
                outcome = "boss"
                events.append({"t": round(t, 2), "type": "winner", "color": list(boss_color)})
                finish_hold = t + 2.0

        img = bg.copy()
        dr = ImageDraw.Draw(img)
        _bf_draw_arena(dr, style, center, R, _hex(accent), t)
        for f in ([boss] if boss["alive"] else []) + live_heroes:
            _bf_update_trail(trail_hist, f["id"], f["body"].position)
        _bf_draw_trails(dr, trail_hist,
                        {f["id"]: f["color"] for f in ([boss] if boss["alive"] else []) + live_heroes},
                        {f["id"]: f["r"] for f in ([boss] if boss["alive"] else []) + live_heroes}, base_bg_col)
        bursts = _bf_draw_bursts(dr, bursts, t, base_bg_col)
        for h in live_heroes:
            _bf_draw_flat_fighter(dr, h)
            _bf_draw_ability_geo(img, dr, h, t)
            _bf_draw_hp(style, img, dr, h["body"].position, h["r"], h["hp"] / h["max_hp"], h["color"],
                       width_mult=1.8)
        if boss["alive"]:
            _bf_draw_flat_fighter(dr, boss)
            _bf_draw_ability_geo(img, dr, boss, t)
        # big boss HP bar across the top — classic boss-fight UI, kept consistent across all styles
        bx0, bx1, by = 60, BF_W - 60, 140
        boss_frac = max(0.0, boss["hp"] / boss["max_hp"])
        dr.rectangle([bx0, by, bx1, by + 26], fill=(35, 35, 40))
        dr.rectangle([bx0, by, bx0 + (bx1 - bx0) * boss_frac, by + 26], fill=(215, 40, 50))
        dr.rectangle([bx0, by, bx1, by + 26], outline=(255, 255, 255), width=2)
        _bf_hud_text(img, (BF_W / 2, 115), "BOSS", 34, color=style["text_color"])
        _bf_hud_text(img, (BF_W / 2, 200),
                    f"{len(live_heroes)} HEROES LEFT" if outcome is None else
                    ("HEROES WIN!" if outcome == "heroes" else "BOSS WINS!"), 46, color=style["text_color"])
        proc.stdin.write(np.array(img).tobytes())

        if finish_hold and t >= finish_hold:
            break

    proc.stdin.close(); proc.wait()
    if outcome is None:
        outcome = "heroes" if live_heroes else "boss"
    win_color = list(heroes[0]["color"]) if outcome == "heroes" and heroes else list(boss_color)
    return events, {"mode": "boss_battle", "winner_color": win_color,
                     "winner_label": "HEROES WIN!" if outcome == "heroes" else "BOSS WINS!",
                     "duration": (finish_hold or target_dur) + 1.4}


# ─────────────────────────────────────────────────────────────────────
# MODE 4 — COLOR INFECTION (team paint-tag, non-elimination)
# ─────────────────────────────────────────────────────────────────────

def simulate_color_infection(target_dur: float, out_path: str, rng: random.Random, style: dict) -> tuple:
    space = pymunk.Space(); space.gravity = (0, 0)
    center = (BF_W / 2, BF_H * 0.42)
    R = BF_W * 0.44
    n_teams = rng.choice([3, 4, 5])
    per_team = rng.randint(3, 5)
    team_colors = [_bf_vivid_color(i / n_teams) for i in range(n_teams)]
    accent = NICHES["ballfight"]["glow"]

    balls = []
    bid = 0
    for tcol in team_colors:
        for _ in range(per_team):
            r = 30
            body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, r))
            ang = rng.uniform(0, 2 * math.pi); dist = rng.uniform(0, R * 0.7)
            body.position = (center[0] + math.cos(ang) * dist, center[1] + math.sin(ang) * dist)
            speed = rng.uniform(120, 220); vang = rng.uniform(0, 2 * math.pi)
            body.velocity = (math.cos(vang) * speed, math.sin(vang) * speed)
            shape = pymunk.Circle(body, r); shape.elasticity = 0.95; shape.friction = 0.05
            space.add(body, shape)
            balls.append({"id": bid, "r": r, "color": list(tcol), "body": body, "shape": shape})
            bid += 1

    total = len(balls)
    events = [{"t": 0.0, "type": "start", "teams": n_teams, "n": total}]
    touching = set()
    trail_hist = {}
    base_bg_col = style["bg"] or _hex(NICHES["ballfight"]["bg"])
    proc = _bf_ffmpeg_writer(out_path, BF_W, BF_H, BF_FPS)
    bg = _bf_style_background(style, BF_W, BF_H, NICHES["ballfight"]["bg"], _hex(accent))
    n_frames = int(target_dur * BF_FPS)
    sudden_death = target_dur - 4.0

    for f in range(n_frames):
        t = f / BF_FPS
        for _ in range(2):
            space.step(1 / (BF_FPS * 2))
        for b in balls:
            p = b["body"].position
            d = math.hypot(p[0] - center[0], p[1] - center[1])
            if d + b["r"] > R:
                nx, ny = (p[0] - center[0]) / max(d, 1e-6), (p[1] - center[1]) / max(d, 1e-6)
                vx, vy = b["body"].velocity
                dot = vx * nx + vy * ny
                if dot > 0:
                    b["body"].velocity = (vx - 2 * dot * nx, vy - 2 * dot * ny)
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                a, b = balls[i], balls[j]
                pa, pb = a["body"].position, b["body"].position
                dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                key = (i, j)
                if dist < a["r"] + b["r"]:
                    if key not in touching and a["color"] != b["color"]:
                        sa, sb = math.hypot(*a["body"].velocity), math.hypot(*b["body"].velocity)
                        w, l = (a, b) if sa >= sb else (b, a)
                        l["color"] = list(w["color"])
                        events.append({"t": round(t, 2), "type": "convert", "color": w["color"]})
                    touching.add(key)
                else:
                    touching.discard(key)
        # sudden-death: force convergence to majority color near the end
        if t >= sudden_death:
            from collections import Counter
            counts = Counter(tuple(b["color"]) for b in balls)
            majority = counts.most_common(1)[0][0]
            for b in balls:
                b["color"] = list(majority)

        img = bg.copy()
        dr = ImageDraw.Draw(img)
        _bf_draw_arena(dr, style, center, R, _hex(accent), t)
        for b in balls:
            _bf_update_trail(trail_hist, b["id"], b["body"].position)
        _bf_draw_trails(dr, trail_hist,
                        {b["id"]: b["color"] for b in balls},
                        {b["id"]: b["r"] for b in balls}, base_bg_col)
        for b in balls:
            p = b["body"].position; r = b["r"]
            dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=tuple(b["color"]),
                       outline=(255, 255, 255), width=2)
        # territory bar
        from collections import Counter
        counts = Counter(tuple(b["color"]) for b in balls)
        bx = 40
        for col, cnt in counts.items():
            bw = (BF_W - 80) * (cnt / total)
            dr.rectangle([bx, 40, bx + bw, 70], fill=col)
            bx += bw
        proc.stdin.write(np.array(img).tobytes())

    proc.stdin.close(); proc.wait()
    from collections import Counter
    final_counts = Counter(tuple(b["color"]) for b in balls)
    win_color = final_counts.most_common(1)[0][0]
    events.append({"t": target_dur, "type": "winner", "color": list(win_color)})
    return events, {"mode": "color_infection", "winner_color": list(win_color),
                     "n_balls": total, "n_teams": n_teams, "duration": target_dur}


# ─────────────────────────────────────────────────────────────────────
# MODE 3 — GRAVITY RACE (pachinko drop race)
# ─────────────────────────────────────────────────────────────────────

def simulate_gravity_race(target_dur: float, out_path: str, rng: random.Random, style: dict) -> tuple:
    space = pymunk.Space(); space.gravity = (0, 1500)
    margin = 60
    accent = NICHES["ballfight"]["glow"]
    walls = [pymunk.Segment(space.static_body, (margin, 0), (margin, BF_H), 4),
             pymunk.Segment(space.static_body, (BF_W - margin, 0), (BF_W - margin, BF_H), 4)]
    for w_ in walls:
        w_.elasticity = 0.4; w_.friction = 0.3
    space.add(*walls)

    peg_positions = []
    rows = rng.randint(11, 15)
    row_gap = (BF_H * 0.62) / rows
    for row in range(rows):
        y = BF_H * 0.14 + row * row_gap
        offset = 34 if row % 2 == 0 else 0
        col_gap = 78
        col = 0
        while True:
            x = margin + 40 + offset + col * col_gap
            if x > BF_W - margin - 40:
                break
            peg_positions.append((x, y))
            shape = pymunk.Circle(space.static_body, 9, (x, y))
            shape.elasticity = 0.55; shape.friction = 0.15
            space.add(shape)
            col += 1

    finish_y = BF_H * 0.86
    n = rng.randint(6, 10)
    palette = [_bf_vivid_color(i / n) for i in range(n)]
    balls_meta = _bf_gen_balls(n, (26, 26), rng, palette=palette)
    balls = []
    for i, bm in enumerate(balls_meta):
        body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, bm["r"]))
        body.position = (BF_W / 2 + rng.uniform(-70, 70), -80)
        shape = pymunk.Circle(body, bm["r"]); shape.elasticity = 0.5; shape.friction = 0.2
        space.add(body, shape)
        balls.append({**bm, "body": body, "shape": shape,
                      "dropped": False, "finished": False, "drop_t": i * 0.18})

    events = [{"t": 0.0, "type": "start", "n": n}]
    trail_hist = {}
    base_bg_col = style["bg"] or _hex(NICHES["ballfight"]["bg"])
    proc = _bf_ffmpeg_writer(out_path, BF_W, BF_H, BF_FPS)
    bg = _bf_style_background(style, BF_W, BF_H, NICHES["ballfight"]["bg"], _hex(accent))
    n_frames = int(target_dur * BF_FPS)
    winner = None
    finish_hold = None
    peg_color = style["border_color"] if style["arena"] == "square" else (130, 130, 150)

    for f in range(n_frames):
        t = f / BF_FPS
        for b in balls:
            if not b["dropped"] and t >= b["drop_t"]:
                b["dropped"] = True
                b["body"].position = (BF_W / 2 + random.uniform(-70, 70), -40)
        for _ in range(2):
            space.step(1 / (BF_FPS * 2))
        for b in balls:
            if not b["dropped"]:
                continue
            if not b["finished"] and b["body"].position[1] >= finish_y:
                b["finished"] = True
                events.append({"t": round(t, 2), "type": "finish", "id": b["id"], "color": b["color"]})
                if winner is None:
                    winner = b
                    events.append({"t": round(t, 2), "type": "winner", "id": b["id"], "color": b["color"]})
                    finish_hold = t + 2.5

        img = bg.copy()
        dr = ImageDraw.Draw(img)
        for (x, y) in peg_positions:
            dr.ellipse([x - 9, y - 9, x + 9, y + 9], fill=peg_color)
        dr.line([(margin, finish_y), (BF_W - margin, finish_y)], fill=_hex(accent), width=6)
        _bf_hud_text(img, (BF_W / 2, finish_y - 30), "FINISH", 42, color=style["text_color"])
        dropped_balls = [b for b in balls if b["dropped"]]
        for b in dropped_balls:
            _bf_update_trail(trail_hist, b["id"], b["body"].position, max_len=9)
        _bf_draw_trails(dr, trail_hist,
                        {b["id"]: b["color"] for b in dropped_balls},
                        {b["id"]: b["r"] for b in dropped_balls}, base_bg_col)
        for b in dropped_balls:
            _bf_paste_sprite(img, b["sprite"], b["body"].position, b["body"].angle)
        if winner:
            _bf_hud_text(img, (BF_W / 2, 90), "WINNER!", 78, color=style["text_color"])
        proc.stdin.write(np.array(img).tobytes())

        if finish_hold and t >= finish_hold:
            break

    proc.stdin.close(); proc.wait()
    return events, {"mode": "gravity_race", "winner_color": winner["color"] if winner else None,
                     "n_balls": n, "duration": (finish_hold or target_dur)}


# ─────────────────────────────────────────────────────────────────────
# MODE 4 — SPIKE ACCUMULATOR (touch the spiked ring, freeze in place —
# inspired by ballsimulator.com's "Accumulation" mode: tension builds as
# balls stick to the spikes one by one; last color still moving wins)
# ─────────────────────────────────────────────────────────────────────

def simulate_spike_accumulator(target_dur: float, out_path: str, rng: random.Random, style: dict) -> tuple:
    space = pymunk.Space(); space.gravity = (0, 0)
    center = (BF_W / 2, BF_H * 0.44)
    r0 = BF_W * 0.46
    r1 = BF_W * 0.20
    grace = target_dur * 0.15
    sudden_death = target_dur - 3.0
    n_teams = rng.choice([3, 4])
    per_team = rng.randint(3, 5)
    team_colors = [_bf_vivid_color(i / n_teams) for i in range(n_teams)]
    accent = NICHES["ballfight"]["glow"]

    def radius_at(t):
        if t < grace:
            return r0
        if t < sudden_death:
            frac = (t - grace) / max(0.01, sudden_death - grace)
            return r0 - (r0 - r1) * frac
        return r1

    balls, bid = [], 0
    for tcol in team_colors:
        for _ in range(per_team):
            r = rng.uniform(26, 32)
            body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, r))
            ang = rng.uniform(0, 2 * math.pi); dist = rng.uniform(0, r0 * 0.5)
            body.position = (center[0] + math.cos(ang) * dist, center[1] + math.sin(ang) * dist)
            speed = rng.uniform(150, 260); vang = rng.uniform(0, 2 * math.pi)
            body.velocity = (math.cos(vang) * speed, math.sin(vang) * speed)
            shape = pymunk.Circle(body, r); shape.elasticity = 0.98; shape.friction = 0.02
            space.add(body, shape)
            balls.append({"id": bid, "r": r, "color": list(tcol), "body": body, "shape": shape,
                          "frozen": False, "frozen_pos": None})
            bid += 1

    total = len(balls)
    events = [{"t": 0.0, "type": "start", "teams": n_teams, "n": total}]
    touching, bursts, trail_hist = set(), [], {}
    base_bg_col = style["bg"] or _hex(NICHES["ballfight"]["bg"])
    proc = _bf_ffmpeg_writer(out_path, BF_W, BF_H, BF_FPS)
    bg = _bf_style_background(style, BF_W, BF_H, NICHES["ballfight"]["bg"], _hex(accent))
    n_frames = int(target_dur * BF_FPS)
    winner_color, finish_hold = None, None

    def colors_free():
        return {tuple(b["color"]) for b in balls if not b["frozen"]}

    for f in range(n_frames):
        t = f / BF_FPS
        for _ in range(2):
            space.step(1 / (BF_FPS * 2))
        R = radius_at(t)
        n_spikes = max(18, int(R / 13))
        free_balls = [b for b in balls if not b["frozen"]]
        for b in free_balls:
            p = b["body"].position
            d = math.hypot(p[0] - center[0], p[1] - center[1])
            if d + b["r"] > R:
                nx, ny = (p[0] - center[0]) / max(d, 1e-6), (p[1] - center[1]) / max(d, 1e-6)
                snap_d = R - b["r"] * 0.3
                frozen_pos = (center[0] + nx * snap_d, center[1] + ny * snap_d)
                b["frozen"] = True; b["frozen_pos"] = frozen_pos
                space.remove(b["body"], b["shape"])
                _bf_spawn_burst(bursts, frozen_pos, b["color"], t)
                events.append({"t": round(t, 2), "type": "frozen", "id": b["id"], "color": b["color"]})

        free_balls = [b for b in balls if not b["frozen"]]
        for i in range(len(free_balls)):
            for j in range(i + 1, len(free_balls)):
                a, b2 = free_balls[i], free_balls[j]
                pa, pb = a["body"].position, b2["body"].position
                dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                key = (a["id"], b2["id"])
                if dist < a["r"] + b2["r"]:
                    if key not in touching:
                        speed = math.hypot(*a["body"].velocity) + math.hypot(*b2["body"].velocity)
                        events.append({"t": round(t, 2), "type": "collision", "impact": min(1.0, speed / 800)})
                    touching.add(key)
                else:
                    touching.discard(key)

        remaining = colors_free()
        if winner_color is None and len(remaining) <= 1 and t > 1.0:
            winner_color = list(next(iter(remaining))) if remaining else balls[-1]["color"]
            events.append({"t": round(t, 2), "type": "winner", "color": winner_color})
            finish_hold = t + 2.2

        img = bg.copy()
        dr = ImageDraw.Draw(img)
        dr.ellipse([center[0] - R, center[1] - R, center[0] + R, center[1] + R],
                   outline=(255, 255, 255), width=3)
        for i in range(n_spikes):
            ang = 2 * math.pi * i / n_spikes + t * 0.15
            x1 = center[0] + math.cos(ang) * R
            y1 = center[1] + math.sin(ang) * R
            x2 = center[0] + math.cos(ang) * (R + 16)
            y2 = center[1] + math.sin(ang) * (R + 16)
            dr.line([x1, y1, x2, y2], fill=_hex(accent), width=4)
        for b in balls:
            if b["frozen"]:
                fx, fy = b["frozen_pos"]
                dr.ellipse([fx - b["r"], fy - b["r"], fx + b["r"], fy + b["r"]],
                          fill=tuple(b["color"]), outline=(255, 255, 255), width=2)
        for b in free_balls:
            _bf_update_trail(trail_hist, b["id"], b["body"].position)
        _bf_draw_trails(dr, trail_hist,
                        {b["id"]: b["color"] for b in free_balls},
                        {b["id"]: b["r"] for b in free_balls}, base_bg_col)
        bursts = _bf_draw_bursts(dr, bursts, t, base_bg_col)
        for b in free_balls:
            p = b["body"].position; r = b["r"]
            dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=tuple(b["color"]),
                       outline=(255, 255, 255), width=2)
        _bf_hud_text(img, (BF_W / 2, 90),
                    f"{len(free_balls)} FREE" if winner_color is None else "WINNER!", 68,
                    color=style["text_color"])
        proc.stdin.write(np.array(img).tobytes())

        if finish_hold and t >= finish_hold:
            break

    proc.stdin.close(); proc.wait()
    if winner_color is None:
        remaining = colors_free()
        winner_color = list(next(iter(remaining))) if remaining else (balls[0]["color"] if balls else None)
        events.append({"t": target_dur, "type": "winner", "color": winner_color})
    return events, {"mode": "spike_accumulator", "winner_color": winner_color,
                     "n_balls": total, "n_teams": n_teams, "duration": (finish_hold or target_dur)}


BF_SIM_FUNCS = {"duel_1v1": simulate_duel_1v1,
                "group_battle": simulate_group_battle,
                "boss_battle": simulate_boss_battle,
                "color_infection": simulate_color_infection,
                "gravity_race": simulate_gravity_race,
                "spike_accumulator": simulate_spike_accumulator}


# ─────────────────────────────────────────────────────────────────────
# MUSIC — 4 distinct synthesized moods, one picked at random per video
# (epic is always part of the pool)
# ─────────────────────────────────────────────────────────────────────

def _bf_synth_arcade_chiptune(path: str, dur_s: float, rng: random.Random) -> None:
    sr = 44100; n = int(sr * dur_s); t = np.linspace(0, dur_s, n)
    root = rng.choice([220, 246.94, 261.63, 293.66])
    notes = [root, root * 1.125, root * 1.25, root * 1.5]
    bpm = rng.choice([128, 140, 150])
    beat = 60 / bpm
    wave = np.zeros(n)
    for i in range(int(dur_s / beat)):
        f = notes[i % len(notes)]
        s0, s1 = int(i * beat * sr), int(min(dur_s, (i + 1) * beat) * sr)
        seg_t = t[s0:s1] - t[s0]
        square = np.sign(np.sin(2 * np.pi * f * seg_t))
        env = np.exp(-6 * seg_t)
        wave[s0:s1] += square * env * 0.5
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * 20000
    seg = AudioSegment(wave.astype(np.int16).tobytes(), frame_rate=sr, sample_width=2, channels=1)
    seg.fade_in(200).fade_out(600).export(path, format="mp3")


def _bf_synth_epic_hype(path: str, dur_s: float, rng: random.Random) -> None:
    sr = 44100; n = int(sr * dur_s); t = np.linspace(0, dur_s, n)
    root = rng.choice([65.4, 73.4, 82.4, 87.3, 98.0, 110.0])
    pulse = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2.2 * t))
    wave = (0.35 * np.sin(2 * np.pi * root * t) + 0.2 * np.sin(2 * np.pi * root * 2 * t)) * pulse
    wave += 0.12 * np.sin(2 * np.pi * root * 4 * t)
    swell = np.clip(t / (dur_s * 0.3), 0, 1)
    wave *= (0.5 + 0.5 * swell)
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * 21000
    seg = AudioSegment(wave.astype(np.int16).tobytes(), frame_rate=sr, sample_width=2, channels=1)
    seg.fade_in(400).fade_out(800).export(path, format="mp3")


def _bf_synth_comedy_bounce(path: str, dur_s: float, rng: random.Random) -> None:
    sr = 44100; n = int(sr * dur_s); t = np.linspace(0, dur_s, n)
    root = rng.choice([329.63, 349.23, 392.0])
    seq = [1, 1.2, 1.5, 1.2]
    bpm = rng.choice([110, 120, 130]); beat = 60 / bpm
    wave = np.zeros(n)
    for i in range(int(dur_s / beat)):
        f = root * seq[i % len(seq)]
        s0, s1 = int(i * beat * sr), int(min(dur_s, (i + 1) * beat) * sr)
        seg_t = t[s0:s1] - t[s0]
        tone = np.sin(2 * np.pi * f * seg_t) * np.exp(-8 * seg_t)
        wave[s0:s1] += tone * 0.6
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * 19000
    seg = AudioSegment(wave.astype(np.int16).tobytes(), frame_rate=sr, sample_width=2, channels=1)
    seg.fade_in(150).fade_out(500).export(path, format="mp3")


def _bf_synth_retro_synth(path: str, dur_s: float, rng: random.Random) -> None:
    sr = 44100; n = int(sr * dur_s); t = np.linspace(0, dur_s, n)
    root = rng.choice([146.83, 164.81, 174.61])
    wave = (0.3 * np.sin(2 * np.pi * root * t) + 0.22 * np.sin(2 * np.pi * root * 1.5 * t) +
           0.15 * np.sin(2 * np.pi * root * 2 * t))
    lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    wave *= lfo
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * 20000
    seg = AudioSegment(wave.astype(np.int16).tobytes(), frame_rate=sr, sample_width=2, channels=1)
    seg.fade_in(300).fade_out(700).export(path, format="mp3")


BF_MOODS = {
    "arcade": _bf_synth_arcade_chiptune,
    "epic": _bf_synth_epic_hype,
    "comedy": _bf_synth_comedy_bounce,
    "retro": _bf_synth_retro_synth,
}


def ensure_ballfight_music(dur_s: float, out_path: str, rng: random.Random) -> str:
    mood = rng.choice(list(BF_MOODS.keys()))
    BF_MOODS[mood](out_path, dur_s + 5, rng)
    log.info("Ballfight music: mood='%s'", mood)
    return out_path


def _bf_make_blip_segment(freq: float, dur: float = 0.09) -> AudioSegment:
    """Build a short synth blip directly in memory — no file/ffmpeg round-trip,
    which is both faster and avoids occasional corrupt-file failures when
    spawning many ffmpeg subprocesses back-to-back for dozens of events."""
    sr = 44100
    n = max(1, int(sr * dur))
    t = np.linspace(0, dur, n, endpoint=False)
    env = np.exp(-14 * t)
    wave = (np.sin(2 * np.pi * freq * t) * env * 24000).astype(np.int16)
    return AudioSegment(wave.tobytes(), frame_rate=sr, sample_width=2, channels=1)


def mix_ballfight_audio(events: list, total_dur: float, voice_lines: list,
                        music_path: str, tmp_dir: Path) -> str:
    """Layer synthesized collision/elimination blips + a looped ducked epic
    music bed, then overlay each short commentary burst at its resolved
    timestamp (bursts are sparse — most of the runtime stays music+SFX only,
    not continuously narrated)."""
    bed = AudioSegment.silent(duration=int(total_dur * 1000) + 500)

    music = AudioSegment.from_file(music_path)
    music = (music * (int(len(bed) / len(music)) + 1))[:len(bed)]
    music = music + MUSIC_DUCK_DB
    bed = bed.overlay(music)

    rng = random.Random()
    for ev in events:
        if ev["type"] not in ("collision", "eliminated", "convert", "finish", "winner", "frozen", "hit", "block", "heal"):
            continue
        base_freq = {"collision": 520, "eliminated": 220, "convert": 700,
                     "finish": 380, "winner": 300, "frozen": 260, "hit": 460, "block": 640,
                     "heal": 880}[ev["type"]]
        freq = base_freq * rng.uniform(0.85, 1.25)
        dur = 0.22 if ev["type"] in ("eliminated", "winner", "frozen", "heal") else 0.09
        blip = _bf_make_blip_segment(freq, dur)
        impact = ev.get("impact", 0.7)
        blip = blip + (6 * impact - 6)  # louder on harder impacts
        pos_ms = int(ev["t"] * 1000)
        if pos_ms < len(bed):
            bed = bed.overlay(blip, position=pos_ms)

    for line in voice_lines:
        voice = AudioSegment.from_file(line["path"])
        # duck the music a bit further under each spoken burst so it's clear
        pos_ms = int(line["start_s"] * 1000)
        if pos_ms < len(bed):
            bed = bed.overlay(voice, position=pos_ms)

    out_path = str(tmp_dir / "ballfight_audio.mp3")
    bed.export(out_path, format="mp3")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# COMMENTARY — a handful of short reaction bursts tied to real events,
# plus one closing line for the winner/CTA end card
# ─────────────────────────────────────────────────────────────────────

def generate_ballfight_commentary(mode: str, events: list, meta: dict, target_dur: float) -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    key_events = [e for e in events if e["type"] in
                  ("start", "eliminated", "convert", "finish", "winner", "frozen", "block", "heal")][:24]
    event_summary = "\n".join(
        f"- t={e['t']}s: {e['type']}" + (f" (ball/color {e.get('color') or e.get('id')})" if e.get('color') or 'id' in e else "")
        for e in key_events
    )
    mode_desc = {
        "duel_1v1": "a 1v1 duel between two balls, each with a random combat ability, fighting with HP bars until one is knocked out",
        "group_battle": "a team vs. team ball battle with HP bars and abilities — last team with a fighter standing wins",
        "boss_battle": "a boss battle — a team of small hero balls vs. one huge boss ball with a big HP bar, heroes win if they bring the boss to 0 HP, boss wins if it wipes the team first",
        "color_infection": "a color-infection game — balls 'infect' each other on contact, one team color takes over the whole arena",
        "gravity_race": "a gravity pachinko drop-race — balls fall through pegs and race to a finish line",
        "spike_accumulator": "a spike-accumulator game — balls freeze in place the instant they touch the spiked ring, tension builds as fewer stay free, last color with a free ball left wins",
    }[mode]
    winner_name = meta.get("winner_label") or _bf_color_name(meta.get("winner_color"))

    system = (
        "You are a hyped sports commentator for physics-simulation TikTok videos "
        "(satisfying marble-race / ball-battle content). Return ONLY valid JSON, no markdown: "
        "{title, hashtags: [string], lines: [{t: number, text: string}], winner_line: string}. "
        "Pick 4 to 6 of the most exciting moments from the real event list and write ONE short, "
        "punchy reaction line for each (5-12 words, playful/hyped/a little absurd — like a caster "
        "who is way too invested in some bouncing balls). The 't' value for each line MUST be "
        "copied EXACTLY from one of the given event timestamps — never invent a new number. "
        "These lines are short bursts spoken over an otherwise mostly-instrumental video, NOT "
        "continuous narration — so keep every line short and punchy, not a play-by-play. "
        "Also write 'winner_line': one hyped line (8-15 words) announcing the winner and telling "
        f"viewers to follow and like — it plays over the end card announcing: {winner_name}. No timestamp. "
        "title: short catchy TikTok title. hashtags: 6-10 tags mixing #ballfight #satisfying "
        "#physics #fyp with a couple specific to the game mode."
    )
    prompt = (
        f"Game mode: {mode_desc}\n"
        f"Real events from this exact simulation, in order:\n{event_summary}\n"
        f"Winner: {winner_name}\n"
        "Comment on only a handful of the punchiest moments, spaced out across the video."
    )

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=1.0, max_tokens=800,
            )
            raw = resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
            data = json.loads(raw)
            lines = [l for l in data.get("lines", [])
                     if isinstance(l, dict) and l.get("text") and isinstance(l.get("t"), (int, float))]
            if lines and data.get("winner_line"):
                data["lines"] = lines
                return data
        except Exception as e:
            log.warning("Ballfight commentary attempt %d: %s", attempt + 1, e)
        time.sleep(2 ** attempt)
    # Fallback if Groq fails entirely — still produces a postable video
    mid_t = key_events[len(key_events)//2]["t"] if key_events else target_dur/2
    return {"title": f"Ball {mode.replace('_',' ').title()}!",
            "hashtags": ["ballfight", "satisfying", "physics", "fyp", "foryou"],
            "lines": [{"t": 0.3, "text": "Here we go!"},
                      {"t": mid_t, "text": "This is getting intense!"}],
            "winner_line": f"{winner_name}! Follow for more, and drop a like!"}


# ─────────────────────────────────────────────────────────────────────
# END CARD — "<COLOR> WINS!" + follow/like CTA, held for a few seconds
# ─────────────────────────────────────────────────────────────────────

_BF_COLOR_NAMES = [
    (15, "RED"), (45, "ORANGE"), (65, "YELLOW"), (150, "GREEN"),
    (195, "CYAN"), (250, "BLUE"), (280, "PURPLE"), (325, "PINK"), (360, "RED"),
]

def _bf_color_name(rgb: Optional[tuple]) -> str:
    if not rgb:
        return "EVERYONE"
    r, g, b = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.15:
        return "WHITE" if v > 0.7 else "GRAY"
    deg = h * 360
    for hi, name in _BF_COLOR_NAMES:
        if deg < hi:
            return name
    return "RED"


def _bf_fit_font_size(text: str, max_width: int, start_size: int, min_size: int = 44) -> int:
    """Largest font size (down to min_size) at which `text` still fits
    within max_width, so longer color names don't overflow the canvas."""
    size = start_size
    dummy = Image.new("RGB", (1, 1))
    dr = ImageDraw.Draw(dummy)
    while size > min_size:
        font = load_font(size)
        w = dr.textbbox((0, 0), text, font=font)[2]
        if w <= max_width:
            break
        size -= 6
    return size


def _bf_render_outro_card(winner_color: Optional[tuple], dur_s: float,
                          out_mp4_path: str, tmp_dir: Path, style: dict,
                          winner_label: Optional[str] = None) -> str:
    accent = tuple(winner_color) if winner_color else _hex(NICHES["ballfight"]["glow"])
    bg = _bf_style_background(style, BF_W, BF_H, NICHES["ballfight"]["bg"], accent)
    dr = ImageDraw.Draw(bg)
    cx, cy, r = BF_W / 2, BF_H * 0.36, 150
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent, outline=(255, 255, 255), width=8)
    if winner_label:
        title_text = winner_label
    else:
        name = _bf_color_name(winner_color)
        title_text = "IT'S A DRAW!" if name == "EVERYONE" else f"{name} WINS!"
    title_size = _bf_fit_font_size(title_text, int(BF_W * 0.9), 145)
    _bf_hud_text(bg, (cx, cy + r + 125), title_text, title_size, color=accent)
    _bf_hud_text(bg, (cx, cy + r + 250), "FOLLOW + LIKE FOR MORE", 62, color=style["text_color"])
    png_path = str(tmp_dir / "outro.png")
    bg.save(png_path)
    subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", png_path, "-t", f"{dur_s:.2f}",
         "-r", str(BF_FPS), "-pix_fmt", "yuv420p", "-vf", f"scale={BF_W}:{BF_H}",
         out_mp4_path],
        check=True, capture_output=True,
    )
    return out_mp4_path


def _bf_concat_videos(video_paths: list, out_path: str) -> str:
    args = ["ffmpeg", "-y"]
    for p in video_paths:
        args += ["-i", p]
    n = len(video_paths)
    filt = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]"
    args += ["-filter_complex", filt, "-map", "[outv]",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
              "-pix_fmt", "yuv420p", out_path]
    subprocess.run(args, check=True, capture_output=True)
    return out_path


# ─────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────

def run_ballfight_pipeline() -> None:
    mode = random.choice(BF_MODES)
    style_name = random.choice(list(BF_VISUAL_STYLES.keys()))
    style = BF_VISUAL_STYLES[style_name]
    target_dur = random.uniform(24, 36)
    rng = random.Random()
    log.info("═══ VaultMind Ballfight — mode: %s, style: %s (~%.0fs) ═══", mode, style_name, target_dur)

    ensure_font()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Simulate + render silent physics footage
        raw_video = str(tmp / "raw.mp4")
        events, meta = BF_SIM_FUNCS[mode](target_dur, raw_video, rng, style)
        real_dur = meta.get("duration", target_dur)
        winner_color = meta.get("winner_color")
        log.info("Sim done: %d events, winner=%s", len(events), winner_color)

        # 2. Short commentary bursts tied to real events + one winner/CTA line
        script    = generate_ballfight_commentary(mode, events, meta, real_dur)
        title     = script.get("title", "Ball Fight!")
        hashtags  = script.get("hashtags", ["ballfight", "satisfying", "fyp"])
        raw_lines = script["lines"]
        winner_text = script["winner_line"]

        # Snap each line's timestamp to the nearest real event (defends
        # against the model drifting off the given numbers) and clamp it
        # inside the sim's actual duration.
        valid_ts = sorted(e["t"] for e in events)
        def _snap(t):
            return min(valid_ts, key=lambda v: abs(v - t)) if valid_ts else 0.0

        # 3. Synth each burst, resolve non-overlapping start times
        voice_lines, cur_end = [], 0.0
        for raw in sorted(raw_lines, key=lambda l: l["t"]):
            t = min(_snap(raw["t"]), max(0.0, real_dur - 0.6))
            start = max(t, cur_end + 0.4) if cur_end else t
            if start >= real_dur - 0.3:
                continue  # ran out of room before the sim ends — drop it
            path = str(tmp / f"line_{len(voice_lines)}.mp3")
            generate_voiceover(raw["text"], path)
            dur = len(AudioSegment.from_file(path)) / 1000
            voice_lines.append({"start_s": start, "path": path})
            cur_end = start + dur

        # 4. Winner line — synth first so we know how long the end card
        #    needs to stay on screen.
        winner_path = str(tmp / "winner_line.mp3")
        generate_voiceover(winner_text, winner_path)
        winner_dur = len(AudioSegment.from_file(winner_path)) / 1000
        outro_dur  = max(2.6, winner_dur + 1.2)
        winner_start = real_dur + 0.3
        voice_lines.append({"start_s": winner_start, "path": winner_path})

        # 5. Build the "<COLOR> WINS!" + follow/like end card and append it
        outro_video = str(tmp / "outro.mp4")
        _bf_render_outro_card(winner_color, outro_dur, outro_video, tmp, style,
                              winner_label=meta.get("winner_label"))
        combined_video = str(tmp / "combined.mp4")
        _bf_concat_videos([raw_video, outro_video], combined_video)
        total_dur = real_dur + outro_dur

        # 6. Always-epic music bed + collision/elimination SFX + the sparse
        #    commentary bursts (incl. winner line), mixed to total_dur.
        music_path = str(tmp / "music.mp3")
        ensure_ballfight_music(total_dur, music_path, rng)
        audio_path = mix_ballfight_audio(events, total_dur, voice_lines, music_path, tmp)

        # 7. Mux video (sim + end card) with the mixed audio bed
        muxed = str(tmp / "muxed.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", combined_video, "-i", audio_path,
                        "-c:v", "copy", "-c:a", "aac", "-shortest", muxed],
                       check=True, capture_output=True)

        # 8. Color grade — no captions burned in anymore
        final_video = apply_color_grade(muxed, str(tmp / "final.mp4"), niche="ballfight")

        # 9. Upload to TikTok
        tiktok_id = upload_tiktok(final_video, title, hashtags)

    log.info("═══ Ballfight complete: %s ═══", tiktok_id or "upload failed/skipped")


if __name__ == "__main__":
    run_ballfight_pipeline()
