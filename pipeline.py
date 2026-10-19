"""
VaultMind Auto Video Pipeline — v4
Major upgrades over v3:
  ┌─────────────────────────────────────────────────────────┐
  │  • Pexels VIDEOS as backgrounds (not static images)     │
  │  • Word-by-word animated captions (viral MrBeast style) │
  │  • Creatomate cloud render (optional, premium quality)   │
  │  • Zoom-punch transitions via ffmpeg                     │
  │  • Aggressive viral script prompts                       │
  │  • Auto font download (Impact/Montserrat Black)          │
  └─────────────────────────────────────────────────────────┘
"""

import os, json, time, base64, pickle, random, logging, tempfile, textwrap
import subprocess, shutil, math
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pymunk

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.signal
from pydub import AudioSegment
from pydub.silence import detect_silence

from moviepy import (
    VideoFileClip, ImageClip, AudioFileClip,
    CompositeVideoClip, concatenate_videoclips,
    CompositeAudioClip, concatenate_audioclips,
)
from moviepy.video.fx import FadeIn, FadeOut, CrossFadeIn
# ── Beautiful Captions (animated viral-style subtitles) ────────────────
try:
    from beautiful_captions import Video as BCVideo, CaptionConfig
    BEAUTIFUL_CAPTIONS_OK = True
except Exception:
    BEAUTIFUL_CAPTIONS_OK = False
    log_warn = print  # pre-logger fallback

from groq import Groq
import edge_tts, asyncio
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vaultmind")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
VIDEO_W, VIDEO_H   = 1080, 1920
FPS                = 30
MIN_SCENES         = 20        # more scenes = faster cuts even at 80s+ (viral pacing)
TARGET_DUR         = 85        # >60s required for Shorts monetization eligibility
TRANSITION_DUR     = 0.22      # snappier cuts — matches faster viral edit rhythm
FADE_IN_DUR        = 0.3
FADE_OUT_DUR       = 0.5
MUSIC_DUCK_DB      = -16
CAPTION_WORDS      = 3        # words shown per caption chunk
CAPTION_FONT_SIZE  = 95       # big & punchy

GAMEPLAY_PATH = os.getenv("GAMEPLAY_PATH", "gameplay_bg.mp4")
MUSIC_PATH    = os.getenv("MUSIC_PATH",    "bg_music.mp3")
SFX_WHOOSH    = "sfx_whoosh.mp3"
SFX_IMPACT    = "sfx_impact.mp3"
PIXABAY_KEY   = os.getenv("PIXABAY_KEY", "")
FONT_PATH     = "font_bold.ttf"

NICHES = {
    # Each niche has:
    #  color/glow/bg — visual identity
    #  color_grade   — ffmpeg curves filter for cinematic look
    #  video_queries — pool of Pexels search terms rotated randomly
    #                  (avoids same footage appearing every video)
    "reddit": {
        "color": "#FF4500", "glow": "#FF6633", "bg": "#1A1A2E",
        "color_grade": (
            "curves=r='0/0 0.5/0.55 1/0.95':g='0/0 0.5/0.48 1/0.88':b='0/0 0.5/0.44 1/0.80',"
            "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.6"
        ),
        "video_queries": [
            "person alone dark room", "dramatic phone screen night",
            "crowded subway strangers", "rainy window city night",
            "person walking alone", "shadow dramatic lighting",
            "empty hallway dark", "late night city",
        ],
    },
    "dating": {
        "color": "#FF69B4", "glow": "#FF90C8", "bg": "#1A1A2E",
        "color_grade": (
            "curves=r='0/0.02 0.5/0.58 1/0.98':g='0/0 0.5/0.46 1/0.88':b='0/0 0.5/0.50 1/0.90',"
            "unsharp=luma_msize_x=3:luma_msize_y=3:luma_amount=0.4"
        ),
        "video_queries": [
            "romantic couple bokeh", "love hands holding",
            "couple coffee date", "sunset romantic silhouette",
            "first date restaurant", "flowers bokeh soft",
            "couple walking city", "heartbreak alone rain",
        ],
    },
    "rich": {
        "color": "#FFD700", "glow": "#FFEC80", "bg": "#0D0D0D",
        "color_grade": (
            "curves=r='0/0 0.3/0.32 0.7/0.78 1/1.0':g='0/0 0.3/0.30 0.7/0.72 1/0.96':"
            "b='0/0 0.3/0.22 0.7/0.58 1/0.78',"  # warm golden tone, pull blue
            "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.7"
        ),
        "video_queries": [
            "luxury mansion interior", "private jet interior",
            "sports car driving", "rolex watch close",
            "penthouse city view", "yacht ocean luxury",
            "cash money briefcase", "monaco luxury lifestyle",
        ],
    },
    "lifehack": {
        "color": "#00FF88", "glow": "#66FFBB", "bg": "#111111",
        "color_grade": (
            "curves=r='0/0 0.5/0.50 1/0.95':g='0/0 0.5/0.55 1/1.0':b='0/0 0.5/0.48 1/0.88',"
            "unsharp=luma_msize_x=3:luma_msize_y=3:luma_amount=0.5"
        ),
        "video_queries": [
            "person working laptop coffee", "productivity desk setup",
            "person writing notes", "organized workspace minimal",
            "hands typing keyboard", "brainstorm whiteboard",
            "morning routine person", "phone productivity app",
        ],
    },
    "fact": {
        "color": "#00BFFF", "glow": "#66D9FF", "bg": "#0A0A1A",
        "color_grade": (
            "curves=r='0/0 0.3/0.27 0.7/0.72 1/0.92':"
            "g='0/0 0.3/0.30 0.7/0.74 1/0.95':"
            "b='0/0.05 0.3/0.38 0.7/0.78 1/1.0',"  # cool blue science feel
            "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.8"
        ),
        "video_queries": [
            "galaxy stars universe timelapse", "ocean deep underwater",
            "earth from space", "human brain neurons",
            "microscope cells biology", "lightning storm dramatic",
            "volcano eruption lava", "coral reef ocean life",
        ],
    },
    "scary": {
        "color": "#FF2222", "glow": "#FF6666", "bg": "#050505",
        "color_grade": (
            "curves=r='0/0 0.3/0.35 0.7/0.75 1/0.95':"
            "g='0/0 0.3/0.22 0.7/0.55 1/0.78':"
            "b='0/0 0.3/0.22 0.7/0.55 1/0.78',"  # desaturated red horror
            "unsharp=luma_msize_x=7:luma_msize_y=7:luma_amount=1.0"
        ),
        "video_queries": [
            "dark forest fog night", "abandoned building corridor",
            "horror shadows flickering", "empty road night fog",
            "old door creaking dark", "basement dark stairs",
            "person alone dark house", "storm lightning night",
        ],
    },
    "motivation": {
        "color": "#FF8C00", "glow": "#FFAA44", "bg": "#0D0D0D",
        "color_grade": (
            "curves=r='0/0 0.3/0.35 0.7/0.80 1/1.0':"
            "g='0/0 0.3/0.30 0.7/0.72 1/0.92':"
            "b='0/0 0.3/0.20 0.7/0.55 1/0.78',"  # warm orange energy
            "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.7"
        ),
        "video_queries": [
            "athlete training gym intense", "runner marathon sunrise",
            "person climbing mountain", "winner podium celebration",
            "bodybuilder workout sweat", "sunrise horizon epic",
            "boxing training fight", "marathon finish line",
        ],
    },
    "conspiracy": {
        "color": "#9B59B6", "glow": "#C285E0", "bg": "#070713",
        "color_grade": (
            "curves=r='0/0 0.3/0.25 0.7/0.65 1/0.85':"
            "g='0/0 0.3/0.22 0.7/0.58 1/0.80':"
            "b='0/0.04 0.3/0.35 0.7/0.70 1/0.95',"  # dark purple mystery
            "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.9"
        ),
        "video_queries": [
            "government building shadows", "cctv surveillance camera",
            "secret document papers", "person shadows mystery",
            "eye surveillance dramatic", "underground tunnel dark",
            "newspaper headlines dramatic", "shadow figure silhouette",
        ],
    },
    "learn": {
        "color": "#1ABC9C", "glow": "#4DDFC4", "bg": "#0D1B2A",
        "color_grade": (
            "curves=r='0/0 0.3/0.28 0.7/0.72 1/0.93':"
            "g='0/0 0.3/0.32 0.7/0.75 1/0.97':"
            "b='0/0.02 0.3/0.33 0.7/0.70 1/0.92',"  # clean teal academic
            "unsharp=luma_msize_x=3:luma_msize_y=3:luma_amount=0.5"
        ),
        "video_queries": [
            "student studying library", "books open knowledge",
            "science experiment lab", "technology innovation future",
            "whiteboard teaching class", "person reading focus",
            "coding computer screen", "universe stars learning",
        ],
    },
    # ── BALL FIGHT (physics-sim niche — no stock footage, self-rendered) ──
    "ballfight": {
        "color": "#39FF14", "glow": "#8CFF66", "bg": "#04050A",
        "color_grade": (
            "curves=r='0/0 0.4/0.42 0.8/0.85 1/1.0':"
            "g='0/0 0.4/0.50 0.8/0.92 1/1.0':"
            "b='0/0 0.4/0.35 0.8/0.70 1/0.90',"  # punchy neon arcade look
            "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.8"
        ),
        "video_queries": [],  # unused — ballfight renders its own physics footage
    },
}

# Niches eligible for the classic stock-footage/YouTube pipeline.
# "ballfight" is excluded — it has its own generation + upload path (TikTok).
STOCKFOOTAGE_NICHES = [k for k in NICHES if k != "ballfight"]


# ══════════════════════════════════════════════════════════════════════
# FONT BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════

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

def generate_script(niche: str) -> dict:
    """
    Script generation tuned to 2026 viral-format research:
    - Outcome/result-first hooks (or Question/Why-How — both top performers)
    - Third-person narration (narrate, don't star)
    - FAST CUT PACING: many short scenes (2.5-4.5s each) even though total
      runtime stays 80-95s for monetization — this mimics the rapid scene-
      change rhythm of actually-viral Shorts instead of slow 5-6s scenes
    - 8-12 hashtags generated as part of the script (5-8 niche + 2-4 generic)
    - Mid-video "re-hooks" every ~15-20s to fight drop-off on longer videos
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    hook_styles = [
        "OUTCOME-FIRST: open by showing/stating the shocking result or fact "
        "immediately — viewer must know the payoff within 2 seconds",
        "QUESTION/WHY-HOW: open with a sharp specific question the viewer "
        "needs answered ('Why does X happen?', 'How is X even possible?')",
        "CONTRARIAN: open by contradicting a widely-held belief ('Everyone "
        "thinks X. They're wrong.')",
    ]
    chosen_hook = random.choice(hook_styles)

    system = (
        "You are a viral YouTube Shorts / TikTok scriptwriter optimized for "
        "2026 platform research data. Return ONLY valid JSON, no markdown, "
        "no preamble. "
        "Format: {title, hook, hashtags: [string], scenes: [{text, duration, pexels_query}]} "
        ""
        "HOOK RULE — use this exact style for the opening line: "
        f"{chosen_hook} "
        ""
        "PACING RULE (critical — this is what separates viral from flat): "
        "- Generate 20-26 scenes minimum "
        "- EACH scene is 2.5-4.5 seconds, ONE short punchy sentence or fragment "
        "- Never let two consecutive scenes run long — vary rhythm like a "
        "real edit: short-short-medium-short "
        "- Total spoken duration must sum to 80-95 seconds "
        "- Every 4-5 scenes, insert a mini cliffhanger or re-hook line "
        "('But here's the part nobody talks about...', 'And it gets weirder...') "
        "to fight viewer drop-off across the longer runtime "
        ""
        "NARRATION RULE: third-person narrator voice — describe the fact/story, "
        "never 'I' or personal anecdotes, like a documentary voiceover "
        ""
        "HASHTAG RULE: generate exactly 8-12 hashtags — 5-8 specific to the "
        "niche/topic content, 2-4 generic high-traffic tags (#shorts #fyp "
        "#viral #trending) "
        ""
        "pexels_query: 3-word visual search term per scene for stock footage"
    )
    # Load patterns from previous viral video analysis (free Groq Vision)
    patterns     = load_viral_patterns()
    pattern_hint = patterns_to_prompt_hints(patterns, niche)

    prompt = (
        f"Write a viral {niche} YouTube Shorts script. "
        "Make it genuinely surprising, controversial or emotionally charged — "
        "something people will want to comment on or share. "
        "Fast pacing, short scenes, frequent re-hooks. "
        "This must feel like a tightly-edited 80+ second video, not a slow one."
        + (f"\n\n{pattern_hint}" if pattern_hint else "")
    )

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": prompt}],
                temperature=0.95, max_tokens=3000,
            )
            raw  = resp.choices[0].message.content.strip()
            raw  = raw.strip("```json").strip("```").strip()
            data = json.loads(raw)

            n_scenes  = len(data.get("scenes", []))
            total_dur = sum(float(s.get("duration", 0)) for s in data.get("scenes", []))
            hashtags  = data.get("hashtags", [])

            if n_scenes >= MIN_SCENES and total_dur >= 60:
                log.info(
                    "Script OK: %d scenes, %.0fs, %d hashtags — \"%s\"",
                    n_scenes, total_dur, len(hashtags), data.get("title", "")
                )
                return data
            log.warning(
                "Script rejected (scenes=%d, dur=%.0fs) — retrying…",
                n_scenes, total_dur
            )
        except Exception as e:
            log.warning("Script attempt %d: %s", attempt + 1, e)
        time.sleep(2 ** attempt)
    raise RuntimeError("Script generation failed after 5 attempts")


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — VOICEOVER (ElevenLabs → OpenAI → edge-tts)
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

def transcribe_with_assemblyai(audio_path: str) -> dict:
    key = os.environ["ASSEMBLYAI_API_KEY"]
    hdr = {"authorization": key}

    log.info("Uploading to AssemblyAI…")
    with open(audio_path, "rb") as f:
        up = requests.post("https://api.assemblyai.com/v2/upload",
                           headers=hdr, data=f, timeout=60)
    up.raise_for_status()
    audio_url = up.json()["upload_url"]

    sub = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={**hdr, "content-type": "application/json"},
        json={"audio_url": audio_url, "speech_models": ["universal-2"],
              "punctuate": True, "format_text": True, "disfluencies": False},
        timeout=30,
    )
    sub.raise_for_status()
    tid = sub.json()["id"]

    poll = f"https://api.assemblyai.com/v2/transcript/{tid}"
    for i in range(120):
        time.sleep(3)
        r = requests.get(poll, headers=hdr, timeout=10).json()
        if r["status"] == "completed": break
        if r["status"] == "error":     raise RuntimeError(f"AAI: {r.get('error')}")
        if i % 10 == 0: log.info("AAI polling… %ds", i*3)

    words = [{"text": w["text"], "start_ms": w["start"],
              "end_ms": w["end"], "confidence": w.get("confidence",1.0)}
             for w in (r.get("words") or [])]

    # Build utterances from punctuation
    utterances, cur = [], []
    for w in words:
        cur.append(w)
        if w["text"].rstrip().endswith((".", "!", "?")):
            utterances.append({"text": " ".join(x["text"] for x in cur),
                                "start_ms": cur[0]["start_ms"],
                                "end_ms":   cur[-1]["end_ms"]})
            cur = []
    if cur:
        utterances.append({"text": " ".join(x["text"] for x in cur),
                            "start_ms": cur[0]["start_ms"],
                            "end_ms":   cur[-1]["end_ms"]})

    pauses = [{"start_ms": utterances[i]["end_ms"],
               "end_ms":   utterances[i+1]["start_ms"]}
              for i in range(len(utterances)-1)
              if utterances[i+1]["start_ms"] - utterances[i]["end_ms"] >= 200]

    total_ms = words[-1]["end_ms"] if words else 0
    log.info("AAI: %d words, %d utterances, %d pauses (%.1fs)",
             len(words), len(utterances), len(pauses), total_ms/1000)
    return {"words": words, "utterances": utterances,
            "pauses": pauses, "total_ms": total_ms}


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — SMART CUT POINTS (pydub + AAI)
# ══════════════════════════════════════════════════════════════════════

def detect_cut_points(audio_path: str, aai_data: dict, n_scenes: int,
                      silence_thresh_db: int = -40,
                      min_silence_ms: int = 300) -> list[float]:
    """
    Returns n_scenes-1 cut points (seconds), prioritizing real silence/
    pauses but FILLING IN additional cuts snapped to word-end boundaries
    when natural pauses aren't enough to support the target scene count.

    Without this fallback, a script with e.g. 22 scenes but only 5 real
    pauses would collapse down to 6 giant scenes — defeating the entire
    point of fast viral-style cut pacing.
    """
    audio    = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    words    = aai_data.get("words", [])

    # ── 1. "Hard" cuts — real silence/pauses (highest quality) ────────
    pydub_mids = [(s+e)//2 for s,e in
                  detect_silence(audio, min_silence_len=min_silence_ms,
                                 silence_thresh=silence_thresh_db)]
    aai_mids   = [(p["start_ms"]+p["end_ms"])//2
                  for p in aai_data.get("pauses", [])]

    hard, last = [], -9999
    for t in sorted(set(pydub_mids + aai_mids)):
        if t - last > 500 and 3000 < t < total_ms - 3000:
            hard.append(t)
            last = t

    n_cuts = n_scenes - 1
    log.info("Natural pause candidates: %d (need %d cuts)", len(hard), n_cuts)

    if len(hard) >= n_cuts:
        # Plenty of real pauses — pick the ones closest to even spacing
        ideal = total_ms / n_scenes
        pool, chosen = hard.copy(), []
        for i in range(1, n_scenes):
            target  = i * ideal
            closest = min(pool, key=lambda t: abs(t-target))
            chosen.append(closest)
            pool.remove(closest)
        result = sorted(t/1000.0 for t in chosen)
        log.info("Cut points (%d, all natural): %s",
                 len(result), [f"{c:.1f}s" for c in result])
        return result

    # ── 2. Not enough natural pauses — fill gaps with word-boundary cuts ─
    # Build a sorted list of every word-end timestamp as a fallback
    # candidate. These are never mid-word (they're exactly where one
    # word finishes), so audio never gets cut inside a syllable.
    word_ends = sorted(w["end_ms"] for w in words
                       if 3000 < w["end_ms"] < total_ms - 3000)

    ideal    = total_ms / n_scenes
    targets  = [i * ideal for i in range(1, n_scenes)]
    chosen   = []
    used_hard, used_words = set(), set()

    for target in targets:
        # Prefer a natural pause near this target position
        nearby_hard = [t for t in hard if t not in used_hard
                      and abs(t - target) < ideal * 0.5]
        if nearby_hard:
            pick = min(nearby_hard, key=lambda t: abs(t-target))
            used_hard.add(pick)
            chosen.append(pick)
            continue
        # Otherwise snap to the nearest unused word-end boundary
        candidates = [w for w in word_ends if w not in used_words]
        if candidates:
            pick = min(candidates, key=lambda t: abs(t-target))
            used_words.add(pick)
            chosen.append(pick)

    result = sorted(set(t/1000.0 for t in chosen))
    log.info("Cut points (%d, %d natural + %d word-snapped): %s",
             len(result), len(used_hard), len(used_words),
             [f"{c:.1f}s" for c in result])
    return result


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — PEXELS VIDEO BACKGROUNDS (NEW — replaces static images)
# ══════════════════════════════════════════════════════════════════════

_pexels_video_cache: dict = {}

# ══════════════════════════════════════════════════════════════════════
# GROQ-POWERED SMART PEXELS QUERY (free — Groq has no cost)
#
#  Instead of using the generic 3-word pexels_query from the script,
#  this function uses Groq to analyze the scene text + niche and
#  generate the most visually specific search term possible.
#  Falls back to the niche's rotating video_queries pool instantly
#  if Groq is unavailable, so zero extra latency risk.
# ══════════════════════════════════════════════════════════════════════

_groq_query_cache: dict = {}

def smart_pexels_query(scene_text: str, niche: str,
                       fallback_query: str = "") -> str:
    """
    Use Groq to generate the most visually relevant Pexels search query
    for a given scene. Caches results to avoid duplicate API calls.
    Falls back to niche rotation pool if Groq fails.
    """
    cache_key = scene_text[:60]
    if cache_key in _groq_query_cache:
        return _groq_query_cache[cache_key]

    # Niche fallback pool — always available instantly
    pool = NICHES[niche].get("video_queries", ["cinematic nature landscape"])
    niche_fallback = random.choice(pool)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return niche_fallback

    try:
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "system",
                "content": (
                    "You are a video director. Given a short script line and a niche, "
                    "output ONLY a 3-5 word Pexels video search query that would find "
                    "the most visually striking, emotionally relevant portrait-oriented "
                    "footage to play behind this voiceover. "
                    "No explanation. No punctuation. Just the search query."
                )
            }, {
                "role": "user",
                "content": f"Niche: {niche}\nLine: {scene_text}\nQuery:"
            }],
            temperature=0.7,
            max_tokens=20,
        )
        query = resp.choices[0].message.content.strip().strip('"').strip("'")
        if 2 <= len(query.split()) <= 6:
            _groq_query_cache[cache_key] = query
            log.info("Smart query: '%s' → '%s'", scene_text[:40], query)
            return query
    except Exception as e:
        log.debug("Smart query failed: %s — using pool fallback", e)

    _groq_query_cache[cache_key] = niche_fallback
    return niche_fallback


def fetch_pexels_video(query: str, min_dur: float = 4.0,
                       download: bool = False) -> Optional[str]:
    """
    Fetch a Pexels portrait video for a query.
    download=False (default): returns the direct CDN URL (for Creatomate).
    download=True: downloads to /tmp and returns local path (for MoviePy).
    """
    key = os.getenv("PEXELS_KEY")
    if not key:
        return None

    cache_key = f"{query.lower()[:30]}_{download}"
    if cache_key in _pexels_video_cache:
        return _pexels_video_cache[cache_key]

    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": key},
            params={"query": query, "per_page": 15,
                    "orientation": "portrait", "size": "medium"},
            timeout=15,
        )
        r.raise_for_status()
        videos = r.json().get("videos", [])
        good   = [v for v in videos if v.get("duration", 0) >= min_dur] or videos
        if not good:
            return None

        video  = random.choice(good)
        files  = sorted(video.get("video_files", []),
                        key=lambda f: f.get("height", 0) or 0, reverse=True)
        best   = next((f for f in files if (f.get("height") or 0) >= 1080), files[0])
        url    = best["link"]

        if not download:
            _pexels_video_cache[cache_key] = url
            log.info("Pexels URL: '%s' → %s", query, url[:60])
            return url

        # Download for local MoviePy fallback
        tmp_p = f"/tmp/pex_{video['id']}.mp4"
        if not Path(tmp_p).exists():
            data = requests.get(url, timeout=60, stream=True)
            with open(tmp_p, "wb") as f:
                for chunk in data.iter_content(1024*1024): f.write(chunk)
        _pexels_video_cache[cache_key] = tmp_p
        log.info("Pexels download: '%s' → %s", query, tmp_p)
        return tmp_p

    except Exception as e:
        log.warning("Pexels failed for '%s': %s", query, e)
        return None


def make_solid_bg_clip(color_hex: str, duration: float) -> VideoFileClip:
    """Fallback: solid color clip when Pexels unavailable."""
    r, g, b = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0,2,4))
    arr     = np.full((VIDEO_H, VIDEO_W, 3), [r,g,b], dtype=np.uint8)
    clip    = ImageClip(arr).with_duration(duration).with_fps(FPS)
    return clip


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — WORD-BY-WORD ANIMATED CAPTIONS  (viral MrBeast style)
#
#  Groups words into chunks of CAPTION_WORDS.
#  Each chunk becomes an ImageClip overlay timed to the audio.
#  Current chunk: accent color + scale-up.  Rest: white.
#  All text: bold, black stroke, center screen at ~60% height.
# ══════════════════════════════════════════════════════════════════════

def _hex(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))


def render_caption_frame(
    chunk_words:   list[str],
    active_idx:    int,        # which word in chunk is currently spoken
    accent_color:  str,
    frame_w:       int = VIDEO_W,
    frame_h:       int = VIDEO_H,
) -> Image.Image:
    """
    Renders one caption frame: chunk of words, active word in accent color.
    Returns RGBA image (transparent outside text area).
    """
    frame = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(frame)
    font  = load_font(CAPTION_FONT_SIZE)
    ac    = _hex(accent_color)

    line  = " ".join(chunk_words)
    bb    = draw.textbbox((0, 0), line, font=font)
    lw    = bb[2] - bb[0]
    lh    = bb[3] - bb[1]
    x0    = (frame_w - lw) // 2
    y0    = int(frame_h * 0.60)

    # Measure each word individually for per-word coloring
    x_cur = x0
    for i, word in enumerate(chunk_words):
        wb    = draw.textbbox((0, 0), word, font=font)
        ww    = wb[2] - wb[0]
        space = draw.textbbox((0, 0), " ", font=font)
        sw    = space[2] - space[0]

        # Black outline / stroke (4px in 8 directions)
        for dx, dy in [(-4,0),(4,0),(0,-4),(0,4),(-3,-3),(3,-3),(-3,3),(3,3)]:
            draw.text((x_cur+dx, y0+dy), word, font=font, fill=(0,0,0,255))

        # Word color: accent if active, white otherwise
        color = (*ac, 255) if i == active_idx else (255,255,255,255)
        draw.text((x_cur, y0), word, font=font, fill=color)

        x_cur += ww + sw

    return frame


def build_caption_clips(aai_data: dict, niche: str, total_dur: float) -> list:
    """
    Returns list of (ImageClip, start_time) tuples for all word chunks.
    Each clip is a transparent RGBA overlay timed to spoken words.
    """
    words   = aai_data.get("words", [])
    if not words:
        return []
    accent  = NICHES[niche]["color"]
    clips   = []
    i       = 0

    while i < len(words):
        chunk = words[i : i + CAPTION_WORDS]
        if not chunk:
            break
        chunk_texts  = [w["text"] for w in chunk]
        chunk_start  = chunk[0]["start_ms"] / 1000.0
        chunk_end    = chunk[-1]["end_ms"]   / 1000.0
        chunk_dur    = max(chunk_end - chunk_start, 0.15)

        # For each word in chunk, figure out its sub-timing
        # We render one frame per word that is "active" inside the chunk
        for j, word in enumerate(chunk):
            w_start = word["start_ms"] / 1000.0
            w_end   = word["end_ms"]   / 1000.0
            w_dur   = max(w_end - w_start, 0.08)

            img  = render_caption_frame(chunk_texts, j, accent)
            arr  = np.array(img)
            clip = (ImageClip(arr, is_mask=False)
                    .with_duration(w_dur)
                    .with_fps(FPS))
            clips.append((clip, w_start))

        i += CAPTION_WORDS

    log.info("Caption clips built: %d word segments", len(clips))
    return clips


# ══════════════════════════════════════════════════════════════════════
# STEP 7 — SFX & MUSIC BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════

def _synth_whoosh(path: str) -> None:
    sr, dur = 44100, 0.28
    n = int(sr*dur)
    t = np.linspace(0,dur,n)
    freq = np.exp(np.linspace(np.log(3200),np.log(180),n))
    phase = np.cumsum(2*np.pi*freq/sr)
    env = np.exp(-4*t/dur)
    wave = (np.sin(phase)*env*26000).astype(np.int16)
    AudioSegment(wave.tobytes(),frame_rate=sr,sample_width=2,channels=1).fade_out(60).export(path,format="mp3")

def _synth_impact(path: str) -> None:
    sr,n = 44100, int(44100*0.12)
    noise = np.random.uniform(-1,1,n)
    b,a = scipy.signal.butter(4,700/(sr/2),btype="low")
    f = scipy.signal.lfilter(b,a,noise)
    dec = np.exp(-np.linspace(0,9,n))
    wave = (f*dec*30000).astype(np.int16)
    AudioSegment(wave.tobytes(),frame_rate=sr,sample_width=2,channels=1).fade_out(40).export(path,format="mp3")

def _synth_ambient(path: str, dur_s: float = 120) -> None:
    sr = 44100; n = int(sr*dur_s); t = np.linspace(0,dur_s,n)
    wave = (0.30*np.sin(2*np.pi*110*t)+0.22*np.sin(2*np.pi*165*t)+
            0.16*np.sin(2*np.pi*220*t)+0.10*np.sin(2*np.pi*262*t)+
            0.07*np.sin(2*np.pi*330*t))
    wave *= (0.72+0.28*np.sin(2*np.pi*0.4*t))
    wave  = wave/(np.max(np.abs(wave))+1e-9)*22000
    seg   = AudioSegment(wave.astype(np.int16).tobytes(),frame_rate=sr,sample_width=2,channels=1)
    seg.fade_in(2000).fade_out(2000).export(path,format="mp3")
    log.info("Ambient pad synthesised (%.0fs)",dur_s)

def ensure_music(duration_s: float = 120) -> str:
    if Path(MUSIC_PATH).exists():
        return MUSIC_PATH
    if PIXABAY_KEY:
        for q in ("cinematic background","lofi ambient","dark dramatic"):
            try:
                r = requests.get("https://pixabay.com/api/videos/music/",
                                 params={"key":PIXABAY_KEY,"q":q,"per_page":5},timeout=10)
                hits = r.json().get("hits",[])
                if hits:
                    url  = random.choice(hits)["audio"]["url"]
                    data = requests.get(url,timeout=30).content
                    Path(MUSIC_PATH).write_bytes(data)
                    log.info("Pixabay music: '%s'",q)
                    return MUSIC_PATH
            except Exception as e:
                log.warning("Pixabay: %s",e)
    _synth_ambient(MUSIC_PATH, duration_s+10)
    return MUSIC_PATH

def ensure_sfx() -> tuple[str,str]:
    if not Path(SFX_WHOOSH).exists(): _synth_whoosh(SFX_WHOOSH)
    if not Path(SFX_IMPACT).exists(): _synth_impact(SFX_IMPACT)
    return SFX_WHOOSH, SFX_IMPACT


# ══════════════════════════════════════════════════════════════════════
# STEP 8 — CREATOMATE CLOUD RENDER  (PRIMARY render engine)
#
#  This is the "external Schnitt-KI" — Creatomate handles:
#   • Ken Burns zoom per scene (scale animation)
#   • Crossfade transitions between scenes
#   • Word-appear text animations
#   • AUTO-TRANSCRIPT viral captions (transcript_effect: highlight)
#     → transcribes the voiceover itself and renders word-by-word
#       highlighted captions, the exact MrBeast/TikTok style
#   • Progress bar per scene
#   • Ducked background music
#
#  Falls back to local MoviePy render if CREATOMATE_API_KEY not set
#  or if the API call fails for any reason.
# ══════════════════════════════════════════════════════════════════════

def _upload_to_cdn(file_path: str) -> Optional[str]:
    """
    Upload a file to 0x0.st (free, no API key) and return its public URL.
    Creatomate needs publicly reachable URLs for audio/video sources.
    """
    try:
        with open(file_path, "rb") as f:
            r = requests.post("https://0x0.st", files={"file": f}, timeout=60)
        url = r.text.strip()
        if url.startswith("http"):
            log.info("CDN upload OK: %s", url)
            return url
        log.warning("CDN upload returned unexpected response: %s", url[:100])
    except Exception as e:
        log.warning("CDN upload failed: %s", e)
    return None


def creatomate_render(
    scenes:      list[dict],
    audio_path:  str,
    niche:       str,
    output_path: str,
    music_path:  str = "",
) -> Optional[str]:
    """
    Full Creatomate JSON-to-video render.
    Returns output_path on success, None to trigger MoviePy fallback.
    """
    api_key = os.getenv("CREATOMATE_API_KEY")
    if not api_key:
        log.info("CREATOMATE_API_KEY not set — using local MoviePy render")
        return None

    cfg = NICHES[niche]
    ac  = cfg["color"]

    # ── 1. Upload voiceover to CDN (Creatomate needs a public URL) ────
    audio_url = _upload_to_cdn(audio_path)
    if not audio_url:
        log.warning("Creatomate: voice CDN upload failed — falling back")
        return None

    # ── 2. Build per-scene elements ────────────────────────────────────
    elements = []
    t_cursor = 0.0

    for i, scene in enumerate(scenes):
        dur  = float(scene.get("duration", 5))
        text = scene.get("text", "")
        q    = scene.get("pexels_query", text[:40])

        q = smart_pexels_query(scene.get("text",""), niche, fallback_query=q)
        vid_url = fetch_pexels_video(q, min_dur=dur, download=False)
        if vid_url:
            anims = [{
                "type": "scale", "scope": "element",
                "easing": "linear", "from": "100%", "to": "108%",
            }]
            if i > 0:
                anims.append({
                    "type": "fade", "scope": "element",
                    "easing": "linear", "duration": TRANSITION_DUR,
                })
            elements.append({
                "type":     "video",
                "id":       f"bg-{i}",
                "source":   vid_url,
                "time":     round(t_cursor, 3),
                "duration": round(dur, 3),
                "volume":   0,
                "fit":      "cover",
                "animations": anims,
            })

        # Dark overlay for caption readability
        elements.append({
            "type": "shape", "shape_type": "rectangle",
            "time": round(t_cursor, 3), "duration": round(dur, 3),
            "width": "100%", "height": "100%",
            "fill_color": "rgba(0,0,0,0.42)",
        })

        # Scene headline text (word-appear)
        elements.append({
            "type":          "text",
            "value":         text,
            "time":          round(t_cursor, 3),
            "duration":      round(dur, 3),
            "y":             "62%",
            "width":         "88%",
            "height":        "28%",
            "x_alignment":   "50%",
            "y_alignment":   "0%",
            "font_family":   "Montserrat",
            "font_weight":   "900",
            "font_size":     "7.5 vmin",
            "fill_color":    "#FFFFFF",
            "stroke_color":  "#000000",
            "stroke_width":  "1.2 vmin",
            "animations": [{
                "type": "text-appear", "scope": "split", "split": "word",
                "easing": "back-out", "duration": min(0.4, dur * 0.15),
            }],
        })

        # Per-scene progress bar
        progress = (i + 1) / len(scenes)
        elements.append({
            "type": "shape", "shape_type": "rectangle",
            "time": round(t_cursor, 3), "duration": round(dur, 3),
            "width": f"{progress*100:.1f}%", "height": "0.6%",
            "x": "0%", "y": "99%",
            "x_alignment": "0%", "y_alignment": "100%",
            "fill_color": ac,
        })

        t_cursor += dur

    # ── 3. Voiceover audio track ───────────────────────────────────────
    voice_id = "voiceover-audio"
    elements.append({
        "type": "audio", "id": voice_id, "source": audio_url,
        "time": 0, "duration": round(t_cursor, 3), "volume": "100%",
    })

    # ── 4. AUTO-TRANSCRIPT viral captions (Creatomate transcribes itself) ─
    elements.append({
        "type":                       "text",
        "transcript_source":          voice_id,
        "transcript_effect":          "highlight",
        "transcript_color":           ac,
        "transcript_maximum_length":  12,
        "time":                       0,
        "duration":                   round(t_cursor, 3),
        "y":                          "82%",
        "width":                      "84%",
        "height":                     "14%",
        "x_alignment":                "50%",
        "y_alignment":                "50%",
        "font_family":                "Montserrat",
        "font_weight":                "700",
        "font_size":                  "8.5 vmin",
        "fill_color":                 "#FFFFFF",
        "stroke_color":               "#000000",
        "stroke_width":               "1.5 vmin",
    })

    # ── 5. Background music (ducked) ───────────────────────────────────
    mp = music_path or MUSIC_PATH
    if Path(mp).exists():
        music_url = _upload_to_cdn(mp)
        if music_url:
            elements.append({
                "type": "audio", "source": music_url,
                "time": 0, "duration": round(t_cursor, 3), "volume": "12%",
            })

    # ── 6. Submit render job ───────────────────────────────────────────
    payload = {
        "output_format": "mp4",
        "width":  VIDEO_W,
        "height": VIDEO_H,
        "frame_rate": FPS,
        "elements": elements,
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type":  "application/json"}

    log.info("Creatomate: submitting %d elements, %.0fs total…",
             len(elements), t_cursor)
    r = requests.post("https://api.creatomate.com/v1/renders",
                      headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        log.error("Creatomate submit failed (%d): %s", r.status_code, r.text[:300])
        return None

    render = r.json()
    if isinstance(render, list):
        render = render[0]
    render_id = render["id"]
    log.info("Creatomate render ID: %s", render_id)

    # ── 7. Poll until done ──────────────────────────────────────────────
    poll_url = f"https://api.creatomate.com/v1/renders/{render_id}"
    for attempt in range(120):
        time.sleep(5)
        sr     = requests.get(poll_url, headers=headers, timeout=15).json()
        status = sr.get("status", "")
        if attempt % 6 == 0:
            log.info("Creatomate status: %s (%ds elapsed)", status, attempt*5)
        if status == "succeeded":
            video_url = sr["url"]
            data = requests.get(video_url, timeout=180).content
            Path(output_path).write_bytes(data)
            log.info("Creatomate render complete: %s (%.1fMB)",
                     output_path, len(data)/1e6)
            return output_path
        if status in ("failed", "error"):
            log.error("Creatomate render failed: %s", sr.get("error", ""))
            return None

    log.error("Creatomate timeout")
    return None


# ══════════════════════════════════════════════════════════════════════
# STEP 8b — GOOGLE DRIVE CDN UPLOAD
#   Uploads audio/assets to Google Drive and returns a public URL.
#   Used by JSON2Video and Shotstack which need public URLs.
#   Requires YouTube OAuth token (already in YOUTUBE_TOKEN_B64) —
#   the same account has Drive access on the same token.
# ══════════════════════════════════════════════════════════════════════

def drive_cdn_upload(file_path: str, filename: str) -> Optional[str]:
    """
    Upload a file to Google Drive 'VaultMind/cdn' folder,
    set it to public, and return the direct download URL.
    Falls back to 0x0.st CDN if Drive upload fails.
    """
    try:
        from googleapiclient.discovery import build as gb
        token_b64  = os.getenv("YOUTUBE_TOKEN_B64")
        if not token_b64:
            raise ValueError("No token")
        creds_data = pickle.loads(base64.b64decode(token_b64))
        if isinstance(creds_data, Credentials):
            creds = creds_data
        elif isinstance(creds_data, str):
            creds = Credentials.from_authorized_user_info(json.loads(creds_data))
        else:
            creds = Credentials.from_authorized_user_info(creds_data)

        drive = gb("drive", "v3", credentials=creds)

        # Find or create VaultMind/cdn folder
        def find_or_create_folder(name, parent_id=None):
            q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            if parent_id:
                q += f" and '{parent_id}' in parents"
            res = drive.files().list(q=q, fields="files(id)").execute()
            if res.get("files"):
                return res["files"][0]["id"]
            meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
            if parent_id:
                meta["parents"] = [parent_id]
            return drive.files().create(body=meta, fields="id").execute()["id"]

        vault_id = find_or_create_folder("VaultMind")
        cdn_id   = find_or_create_folder("cdn", vault_id)

        # Upload file
        mime = "audio/mpeg" if file_path.endswith(".mp3") else "video/mp4"
        meta = {"name": filename, "parents": [cdn_id]}
        media = MediaFileUpload(file_path, mimetype=mime, resumable=True)
        f = drive.files().create(body=meta, media_body=media,
                                  fields="id").execute()
        fid = f["id"]

        # Make public
        drive.permissions().create(
            fileId=fid,
            body={"type": "anyone", "role": "reader"}
        ).execute()

        url = f"https://drive.google.com/uc?export=download&id={fid}"
        log.info("Drive CDN: %s → %s", filename, url)
        return url
    except Exception as e:
        log.warning("Drive CDN failed (%s) — using 0x0.st", e)
        return _upload_to_cdn(file_path)


# ══════════════════════════════════════════════════════════════════════
# STEP 8c — JSON2VIDEO RENDER  (FREE — 600 seconds no credit card)
#
#   json2video.com → free account → copy API key → JSON2VIDEO_API_KEY
#
#   Features used:
#   • Portrait 1080×1920 MP4
#   • Per-scene video elements (Pexels URLs)
#   • Text overlays with built-in animation styles
#   • Voiceover audio track
#   • Auto-generated subtitles (included free)
#   • Background music track
# ══════════════════════════════════════════════════════════════════════

def json2video_render(
    scenes:      list[dict],
    audio_path:  str,
    niche:       str,
    output_path: str,
    music_path:  str = "",
) -> Optional[str]:
    api_key = os.getenv("JSON2VIDEO_API_KEY")
    if not api_key:
        log.info("JSON2VIDEO_API_KEY not set — skipping")
        return None

    cfg = NICHES[niche]
    ac  = cfg["color"]

    # Upload audio to CDN (Drive → 0x0.st fallback)
    audio_url = drive_cdn_upload(audio_path, "voice.mp3")
    if not audio_url:
        log.warning("JSON2Video: audio CDN failed")
        return None

    # Build scenes array
    j2v_scenes = []
    for i, scene in enumerate(scenes):
        dur  = float(scene.get("duration", 5))
        text = scene.get("text", "")
        q    = scene.get("pexels_query", text[:40])

        elements = []

        # Background video from Pexels
        q = smart_pexels_query(scene.get("text",""), niche, fallback_query=q)
        vid_url = fetch_pexels_video(q, min_dur=dur, download=False)
        if vid_url:
            elements.append({
                "type":     "video",
                "src":      vid_url,
                "duration": int(dur * 1000),  # ms
                "volume":   0,
            })
        else:
            # Solid color background fallback
            elements.append({
                "type":       "rectangle",
                "width":      "100%",
                "height":     "100%",
                "background": cfg["bg"],
                "duration":   int(dur * 1000),
            })

        # Dark overlay
        elements.append({
            "type":       "rectangle",
            "width":      "100%",
            "height":     "100%",
            "background": "rgba(0,0,0,0.40)",
            "duration":   int(dur * 1000),
        })

        # Scene text
        elements.append({
            "type":       "text",
            "text":       text,
            "duration":   int(dur * 1000),
            "x":          "center",
            "y":          "62%",
            "width":      "88%",
            "font-size":  72,
            "font-family":"Montserrat",
            "font-weight":"900",
            "color":      "#FFFFFF",
            "stroke":     "#000000",
            "stroke-width": 8,
            "style":      "004",    # built-in word-reveal animation
        })

        # Progress bar
        prog_w = int(1080 * (i+1) / len(scenes))
        elements.append({
            "type":       "rectangle",
            "x":          0, "y": "99.4%",
            "width":      prog_w,
            "height":     12,
            "background": ac,
            "duration":   int(dur * 1000),
        })

        j2v_scenes.append({
            "comment":  f"Scene {i+1}",
            "duration": int(dur * 1000),
            "elements": elements,
        })

    # Global audio tracks (added outside scenes)
    global_elements = [
        {
            "type":     "audio",
            "src":      audio_url,
            "duration": int(sum(float(s.get("duration",5)) for s in scenes) * 1000),
            "volume":   100,
        }
    ]

    # Background music
    mp = music_path or MUSIC_PATH
    if Path(mp).exists():
        music_url = drive_cdn_upload(mp, "music.mp3")
        if music_url:
            global_elements.append({
                "type":     "audio",
                "src":      music_url,
                "duration": int(sum(float(s.get("duration",5)) for s in scenes) * 1000),
                "volume":   12,
            })

    payload = {
        "width":    VIDEO_W,
        "height":   VIDEO_H,
        "fps":      FPS,
        "quality":  "high",
        "exports": [{
            "format":  "mp4",
            "start":   0,
        }],
        "scenes":   j2v_scenes,
        "elements": global_elements,
    }

    log.info("JSON2Video: submitting %d scenes…", len(j2v_scenes))
    r = requests.post(
        "https://api.json2video.com/v2/movies",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=payload, timeout=30,
    )
    if r.status_code not in (200, 201):
        log.error("JSON2Video submit failed (%d): %s", r.status_code, r.text[:300])
        return None

    movie_id = r.json().get("movie", {}).get("id") or r.json().get("id")
    log.info("JSON2Video movie ID: %s", movie_id)

    # Poll
    for attempt in range(120):
        time.sleep(5)
        pr = requests.get(
            f"https://api.json2video.com/v2/movies?project={movie_id}",
            headers={"x-api-key": api_key}, timeout=15
        )
        status = pr.json().get("movie", {}).get("status", "")
        if attempt % 6 == 0:
            log.info("JSON2Video: %s (%ds)", status, attempt*5)
        if status == "done":
            url  = pr.json()["movie"]["exports"][0]["url"]
            data = requests.get(url, timeout=180).content
            Path(output_path).write_bytes(data)
            log.info("JSON2Video render done: %.1fMB", len(data)/1e6)
            return output_path
        if status == "error":
            log.error("JSON2Video error: %s", pr.json())
            return None

    log.error("JSON2Video timeout")
    return None


# ══════════════════════════════════════════════════════════════════════
# STEP 8d — SHOTSTACK RENDER  (free developer sandbox)
#
#   shotstack.io → sign up → API key → SHOTSTACK_API_KEY
#   Uses stage (sandbox) endpoint — unlimited renders in sandbox.
# ══════════════════════════════════════════════════════════════════════

def shotstack_render(
    scenes:      list[dict],
    audio_path:  str,
    niche:       str,
    output_path: str,
    music_path:  str = "",
) -> Optional[str]:
    api_key = os.getenv("SHOTSTACK_API_KEY")
    if not api_key:
        log.info("SHOTSTACK_API_KEY not set — skipping")
        return None

    cfg      = NICHES[niche]
    ac       = cfg["color"]
    base_url = "https://api.shotstack.io/edit/v2"
    headers  = {"x-api-key": api_key, "Content-Type": "application/json"}

    # CDN upload
    audio_url = drive_cdn_upload(audio_path, "voice_shotstack.mp3")
    if not audio_url:
        log.warning("Shotstack: audio CDN failed")
        return None

    # Build timeline tracks
    video_clips  = []
    text_clips   = []
    t_cursor     = 0.0

    for i, scene in enumerate(scenes):
        dur  = float(scene.get("duration", 5))
        text = scene.get("text", "")
        q    = scene.get("pexels_query", text[:40])

        q = smart_pexels_query(scene.get("text",""), niche, fallback_query=q)
        vid_url = fetch_pexels_video(q, min_dur=dur, download=False)
        if vid_url:
            video_clips.append({
                "asset":  {"type": "video", "src": vid_url, "volume": 0},
                "start":  round(t_cursor, 3),
                "length": round(dur, 3),
                "fit":    "cover",
                "effect": "zoomIn",     # Shotstack built-in Ken Burns
                "transition": {"in": "fade", "out": "fade"} if i > 0 else {},
            })

        text_clips.append({
            "asset": {
                "type":     "html",
                "html":     f'<p style="font-family:Montserrat,Arial;font-weight:900;font-size:72px;color:#fff;text-shadow:4px 4px 8px #000;text-align:center;width:900px;word-wrap:break-word;">{text}</p>',
                "width":    960,
                "height":   600,
            },
            "start":  round(t_cursor, 3),
            "length": round(dur, 3),
            "position": "center",
            "offset":   {"y": 0.12},
            "transition": {"in": "slideUp"},
        })

        t_cursor += dur

    # Audio track
    audio_clips = [{
        "asset":  {"type": "audio", "src": audio_url, "volume": 1},
        "start":  0,
        "length": round(t_cursor, 3),
    }]

    # Music
    mp = music_path or MUSIC_PATH
    if Path(mp).exists():
        music_url = drive_cdn_upload(mp, "music_shotstack.mp3")
        if music_url:
            audio_clips.append({
                "asset":  {"type": "audio", "src": music_url, "volume": 0.12},
                "start":  0,
                "length": round(t_cursor, 3),
            })

    payload = {
        "timeline": {
            "background": cfg["bg"],
            "tracks": [
                {"clips": text_clips},
                {"clips": video_clips},
                {"clips": audio_clips},
            ],
        },
        "output": {
            "format":      "mp4",
            "resolution":  "1080",
            "aspectRatio": "9:16",
            "fps":         FPS,
            "quality":     "high",
        },
    }

    log.info("Shotstack: submitting %d scenes…", len(scenes))
    r = requests.post(f"{base_url}/render", headers=headers,
                      json=payload, timeout=30)
    if r.status_code not in (200, 201):
        log.error("Shotstack submit failed (%d): %s", r.status_code, r.text[:300])
        return None

    render_id = r.json().get("response", {}).get("id")
    log.info("Shotstack render ID: %s", render_id)

    for attempt in range(120):
        time.sleep(5)
        sr     = requests.get(f"{base_url}/render/{render_id}",
                              headers=headers, timeout=15).json()
        status = sr.get("response", {}).get("status", "")
        if attempt % 6 == 0:
            log.info("Shotstack: %s (%ds)", status, attempt*5)
        if status == "done":
            url  = sr["response"]["url"]
            data = requests.get(url, timeout=180).content
            Path(output_path).write_bytes(data)
            log.info("Shotstack done: %.1fMB", len(data)/1e6)
            return output_path
        if status in ("failed", "error"):
            log.error("Shotstack failed: %s", sr)
            return None

    log.error("Shotstack timeout")
    return None


# ══════════════════════════════════════════════════════════════════════
# STEP 9 — MOVIEPY VIDEO ASSEMBLY (local render)
#   Scene video clips with Ken Burns + caption overlays + SFX at cuts
# ══════════════════════════════════════════════════════════════════════

def _trim_video_clip(path: str, duration: float) -> VideoFileClip:
    """Load a Pexels video, smart-crop to 9:16, trim/loop to duration."""
    clip = VideoFileClip(path, audio=False)
    # Smart crop to portrait
    cw, ch = clip.size
    target_ratio = VIDEO_W / VIDEO_H
    src_ratio    = cw / ch
    if src_ratio > target_ratio:
        new_w = int(ch * target_ratio)
        x1    = (cw - new_w) // 2
        clip  = clip.cropped(x1=x1, x2=x1+new_w)
    else:
        new_h = int(cw / target_ratio)
        clip  = clip.cropped(y1=0, y2=new_h)

    clip = clip.resized((VIDEO_W, VIDEO_H))

    # Loop if clip is shorter than needed duration
    if clip.duration < duration:
        reps = int(np.ceil(duration / clip.duration))
        clip = concatenate_videoclips([clip] * reps)

    return clip.subclipped(0, duration)


def _ken_burns(clip: VideoFileClip, scale_to: float = 1.07) -> VideoFileClip:
    """Slow Ken Burns zoom on a video clip."""
    dur = clip.duration
    def zoom_frame(frame, t):
        scale = 1.0 + (scale_to - 1.0) * (t / dur)
        h, w  = frame.shape[:2]
        new_w, new_h = int(w*scale), int(h*scale)
        resized = np.array(Image.fromarray(frame).resize((new_w,new_h),Image.LANCZOS))
        x0, y0 = (new_w-w)//2, (new_h-h)//2
        return resized[y0:y0+h, x0:x0+w]
    return clip.transform(zoom_frame)


def build_video_moviepy(
    scenes:          list[dict],
    niche:           str,
    audio_path:      str,
    aai_data:        dict,
    cut_points_s:    list[float],
    output_path:     str,
    music_path:      str = "",
    sfx_whoosh_path: str = "",
    sfx_impact_path: str = "",
) -> str:
    log.info("▶ MoviePy v4 render — %d scenes, %dx%d", len(scenes), VIDEO_W, VIDEO_H)
    cfg = NICHES[niche]

    # Scene durations from cut points
    total_s    = aai_data["total_ms"] / 1000.0
    bounds     = [0.0] + cut_points_s + [total_s]
    durations  = [bounds[i+1]-bounds[i] for i in range(len(bounds)-1)]
    while len(scenes) < len(durations): scenes.append(scenes[-1])
    scenes = scenes[:len(durations)]
    n      = len(scenes)

    # ── Build one video clip per scene ────────────────────────────────
    scene_clips = []
    for idx, (scene, dur) in enumerate(zip(scenes, durations)):
        dur = max(dur, 0.5)
        q   = scene.get("pexels_query", scene["text"][:40])
        # Groq generates a more visually specific search query per scene
        q   = smart_pexels_query(scene.get("text",""), niche, fallback_query=q)

        # Background: Pexels video → solid color fallback
        vid_path = fetch_pexels_video(q, min_dur=dur)
        if vid_path and Path(vid_path).exists():
            try:
                bg = _trim_video_clip(vid_path, dur)
                bg = _ken_burns(bg)
            except Exception as e:
                log.warning("Video clip error (%s), using solid BG", e)
                bg = make_solid_bg_clip(cfg["bg"], dur)
        else:
            bg = make_solid_bg_clip(cfg["bg"], dur)

        # Dark vignette overlay (ImageClip)
        vig_arr  = _make_vignette(dur)
        vig_clip = ImageClip(vig_arr).with_duration(dur).with_fps(FPS)

        scene_comp = CompositeVideoClip([bg, vig_clip],
                                        size=(VIDEO_W, VIDEO_H)).with_duration(dur)
        scene_clips.append(scene_comp)
        log.info("  Scene %d/%d (%.1fs) — %s", idx+1, n, dur, q[:35])

    # ── Crossfade transitions ─────────────────────────────────────────
    trans = [scene_clips[0]]
    for c in scene_clips[1:]:
        trans.append(c.with_effects([CrossFadeIn(TRANSITION_DUR)]))
    final = concatenate_videoclips(trans, method="compose", padding=-TRANSITION_DUR)
    final = final.with_effects([FadeIn(FADE_IN_DUR), FadeOut(FADE_OUT_DUR)])

    # ── Caption overlays (word-by-word animated) ──────────────────────
    cap_clips = build_caption_clips(aai_data, niche, final.duration)
    if cap_clips:
        cap_composites = []
        for clip, start_t in cap_clips:
            cap_composites.append(clip.with_start(start_t))
        final = CompositeVideoClip([final] + cap_composites,
                                   size=(VIDEO_W, VIDEO_H))
        log.info("Caption overlays composited (%d clips)", len(cap_clips))

    # ── Audio mix ─────────────────────────────────────────────────────
    vid_dur  = final.duration
    voice    = AudioFileClip(audio_path).subclipped(0, min(vid_dur, total_s))
    tracks   = [voice]

    mp = music_path or MUSIC_PATH
    if Path(mp).exists():
        music_raw = AudioSegment.from_file(mp) + MUSIC_DUCK_DB
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mtmp = f.name
        music_raw.export(mtmp, format="mp3")
        mc = AudioFileClip(mtmp)
        if mc.duration < vid_dur:
            mc = concatenate_audioclips(
                [mc] * int(np.ceil(vid_dur / mc.duration)))
        tracks.append(mc.subclipped(0, vid_dur))
        log.info("Music mixed (%ddB duck)", MUSIC_DUCK_DB)

    wp = sfx_whoosh_path or SFX_WHOOSH
    ip = sfx_impact_path or SFX_IMPACT
    if Path(wp).exists() and Path(ip).exists():
        sfx_seg = AudioSegment.silent(duration=int(vid_dur*1000)+500)
        whoosh  = AudioSegment.from_file(wp) - 5
        impact  = AudioSegment.from_file(ip) - 3
        sfx_seg = sfx_seg.overlay(impact, position=0)
        for cs in cut_points_s:
            sfx_seg = sfx_seg.overlay(whoosh, position=max(0,int(cs*1000)-130))
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            stmp = f.name
        sfx_seg.export(stmp, format="mp3")
        tracks.append(AudioFileClip(stmp).subclipped(0, vid_dur))
        log.info("SFX mixed (%d cuts)", len(cut_points_s))

    final = final.with_audio(CompositeAudioClip(tracks))

    # ── Export ────────────────────────────────────────────────────────
    log.info("Exporting %dx%d @ %dfps → %s", VIDEO_W, VIDEO_H, FPS, output_path)
    final.write_videofile(
        output_path, fps=FPS,
        codec="libx264", audio_codec="aac",
        temp_audiofile="tmp_audio.m4a", remove_temp=True,
        preset="medium",
        ffmpeg_params=["-crf","20","-movflags","+faststart",
                       "-vf", f"scale={VIDEO_W}:{VIDEO_H}"],
        logger=None,
    )
    log.info("Export complete: %s", output_path)
    return output_path


def _make_vignette(duration: float) -> np.ndarray:
    """Dark gradient overlay (bottom 55%) as RGBA numpy array."""
    img  = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    start = int(VIDEO_H * 0.38)
    for y in range(start, VIDEO_H):
        alpha = int(200 * ((y-start)/(VIDEO_H-start))**0.55)
        draw.line([(0,y),(VIDEO_W,y)], fill=(0,0,0,alpha))
    return np.array(img)


# ══════════════════════════════════════════════════════════════════════
# STEP 10 — SRT EXPORT
# ══════════════════════════════════════════════════════════════════════

def export_srt(aai_data: dict, out_path: str) -> None:
    def fmt(s):
        h=int(s//3600); m=int((s%3600)//60); ss=s%60
        return f"{h:02d}:{m:02d}:{ss:06.3f}".replace(".",",")
    words  = aai_data.get("words",[])
    lines  = []
    i, idx = 0, 1
    while i < len(words):
        chunk = words[i:i+4]
        lines += [str(idx),
                  f"{fmt(chunk[0]['start_ms']/1000)} --> {fmt(chunk[-1]['end_ms']/1000)}",
                  " ".join(w["text"] for w in chunk),""]
        i += 4; idx += 1
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════
# STEP 11 — YOUTUBE UPLOAD
# ══════════════════════════════════════════════════════════════════════

def upload_youtube(video_path: str, title: str, description: str,
                   srt_path: Optional[str] = None,
                   tags: Optional[list[str]] = None) -> Optional[str]:
    token_b64 = os.getenv("YOUTUBE_TOKEN_B64")
    if not token_b64:
        log.warning("YOUTUBE_TOKEN_B64 not set")
        return None
    creds_data = pickle.loads(base64.b64decode(token_b64))
    if isinstance(creds_data, Credentials):
        creds = creds_data
    elif isinstance(creds_data, str):
        creds = Credentials.from_authorized_user_info(json.loads(creds_data))
    else:
        creds = Credentials.from_authorized_user_info(creds_data)

    yt        = build("youtube","v3",credentials=creds,cache_discovery=False)
    pub_at    = (datetime.utcnow()+timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    tag_list  = [t.lstrip("#") for t in (tags or ["shorts", "viral"])][:15]
    body      = {"snippet":  {"title": title[:100], "description": description,
                             "tags": tag_list, "categoryId": "22"},
               "status":   {"privacyStatus": "private", "publishAt": pub_at,
                             "selfDeclaredMadeForKids": False}}
    media   = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
    req     = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp    = None
    while resp is None:
        _, resp = req.next_chunk()
    vid_url = f"https://youtu.be/{resp['id']}"
    log.info("YouTube upload: %s", vid_url)

    if srt_path and Path(srt_path).exists():
        try:
            yt.captions().insert(
                part="snippet",
                body={"snippet":{"videoId":resp["id"],"language":"en",
                                  "name":"Auto","isDraft":False}},
                media_body=MediaFileUpload(srt_path, mimetype="text/plain"),
            ).execute()
        except Exception as e:
            log.warning("Caption upload skipped: %s", e)
    return vid_url


# ══════════════════════════════════════════════════════════════════════
# STEP 11b — TIKTOK UPLOAD via Zernio (unified posting API, free tier)
#
#  Zernio already holds its own audited TikTok app, so posting is three
#  plain REST calls from here — no TikTok developer app, no OAuth dance,
#  no refresh tokens on our side. Free tier: 2 connected accounts,
#  unlimited posts, full API access — comfortably covers one daily post.
#  (TikTok's own Content Posting API explicitly does not approve
#  single-account personal-use tools — this sidesteps that restriction
#  the same way any scheduler like Buffer/Later does.)
#
#  Setup (one-time, in Zernio's dashboard, not in this code):
#    1. Sign up at zernio.com (free, no credit card)
#    2. Connect your TikTok account (one-click OAuth button)
#    3. Copy the API key into the ZERNIO_API_KEY secret
# ══════════════════════════════════════════════════════════════════════

ZERNIO_API = "https://zernio.com/api/v1"

def _zernio_tiktok_account_id(api_key: str) -> Optional[str]:
    """Auto-resolve the connected TikTok account so no manual ID copy-paste
    is needed — falls back to ZERNIO_TIKTOK_ACCOUNT_ID if set explicitly."""
    override = os.getenv("ZERNIO_TIKTOK_ACCOUNT_ID")
    if override:
        return override
    try:
        r = requests.get(f"{ZERNIO_API}/accounts",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        accounts = r.json()
        accounts = accounts.get("accounts", accounts) if isinstance(accounts, dict) else accounts
        for acc in accounts:
            if acc.get("platform") == "tiktok":
                return acc.get("id") or acc.get("_id")
    except Exception as e:
        log.warning("Zernio: could not list accounts: %s", e)
    return None


def upload_tiktok(video_path: str, title: str, hashtags: Optional[list[str]] = None) -> Optional[str]:
    """
    Post a video to TikTok via Zernio (presign -> upload bytes -> create
    post). Returns the Zernio post id on success, or None (upload
    skipped/failed) — callers already treat a falsy return as "not posted".
    """
    api_key = os.getenv("ZERNIO_API_KEY")
    if not api_key:
        log.warning("Zernio: missing ZERNIO_API_KEY — skipping upload")
        return None

    account_id = _zernio_tiktok_account_id(api_key)
    if not account_id:
        log.warning("Zernio: no connected TikTok account found — skipping upload")
        return None

    caption = title[:150]
    if hashtags:
        tag_line = " ".join(h if h.startswith("#") else f"#{h}" for h in hashtags)
        caption  = f"{caption} {tag_line}"[:2200]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        presign = requests.post(
            f"{ZERNIO_API}/media/presign", headers=headers,
            json={"filename": Path(video_path).name, "contentType": "video/mp4"},
            timeout=30,
        ).json()
        upload_url, public_url = presign.get("uploadUrl"), presign.get("publicUrl")
        if not upload_url:
            log.warning("Zernio: presign failed: %s", presign)
            return None

        with open(video_path, "rb") as f:
            put = requests.put(upload_url, data=f,
                               headers={"Content-Type": "video/mp4"}, timeout=300)
        if put.status_code not in (200, 201):
            log.warning("Zernio: video PUT failed (%d)", put.status_code)
            return None

        post = requests.post(
            f"{ZERNIO_API}/posts", headers=headers,
            json={
                "content": caption,
                "mediaItems": [{"type": "video", "url": public_url}],
                "platforms": [{"platform": "tiktok", "accountId": account_id}],
            },
            timeout=60,
        ).json()
    except Exception as e:
        log.warning("Zernio: upload failed: %s", e)
        return None

    post_id = (post.get("post") or {}).get("_id") or post.get("_id") or post.get("id")
    if not post_id:
        log.warning("Zernio: unexpected post response: %s", post)
        return None
    log.info("Zernio: TikTok post created — %s", post_id)
    return post_id


# ══════════════════════════════════════════════════════════════════════
# STEP 12 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════

def update_dashboard(niche, title, url, ok, platform="youtube"):
    p    = Path("dashboard.json")
    data = json.loads(p.read_text()) if p.exists() else {"videos":[]}
    data["videos"].insert(0,{"niche":niche,"title":title,"url":url,
                              "ok":ok,"platform":platform,
                              "ts":datetime.utcnow().isoformat()})
    data["videos"] = data["videos"][:50]
    p.write_text(json.dumps(data,indent=2))


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# POST-RENDER STEP A — ANIMATED CAPTIONS (beautiful-captions)
#
#  Takes the raw rendered video + word-level SRT and burns in
#  Montserrat bounce-animated captions — exactly the viral TikTok look.
#  Falls back gracefully to static ffmpeg subtitles if unavailable.
# ══════════════════════════════════════════════════════════════════════

def apply_beautiful_captions(
    video_path:  str,
    srt_path:    str,
    output_path: str,
    accent:      str = "#FFD700",
    words_per_line: int = 2,
) -> str:
    """
    Burn animated word-by-word captions onto video.
    Returns output_path (or video_path unchanged if both methods fail).
    """
    if not Path(srt_path).exists():
        log.warning("SRT missing — skipping caption burn")
        return video_path

    # ── Method 1: beautiful-captions (bounce animation) ───────────────
    if BEAUTIFUL_CAPTIONS_OK:
        try:
            cfg = CaptionConfig(
                animation={"enabled": True, "type": "bounce", "keyframes": 12},
                style={
                    "font":              "Montserrat",
                    "font_size":         140,
                    "color":             "yellow",
                    "outline_color":     "black",
                    "outline_thickness": 12,
                    "verticle_position": 0.50,   # center-screen (note: their typo)
                    "max_words_per_line": words_per_line,
                    "auto_scale_font":   True,
                },
                diarization={"enabled": False},
            )
            vid = BCVideo(video_path, config=cfg)
            vid.add_captions(
                srt_input_path=srt_path,
                output_path=output_path,
                add_styling=True,
                cuda=False,
            )
            log.info("beautiful-captions: bounce captions applied → %s", output_path)
            return output_path
        except Exception as e:
            log.warning("beautiful-captions failed (%s) — falling back to ASS", e)

    # ── Method 2: ffmpeg ASS karaoke subtitles (fallback) ─────────────
    try:
        ass_path = srt_path.replace(".srt", ".ass")
        _srt_to_ass_karaoke(srt_path, ass_path, accent)
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"ass={ass_path}",
            "-c:a", "copy", "-preset", "fast", output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        log.info("ASS karaoke captions applied → %s", output_path)
        return output_path
    except Exception as e:
        log.warning("ASS fallback also failed (%s) — returning raw video", e)
        return video_path


def _srt_to_ass_karaoke(srt_path: str, ass_path: str, accent: str) -> None:
    """
    Convert SRT to ASS with karaoke-style word highlighting.
    Each word highlights in accent color as it is spoken.
    """
    def ms_to_ass(ms: int) -> str:
        h  = ms // 3600000
        m  = (ms % 3600000) // 60000
        s  = (ms % 60000) // 1000
        cs = (ms % 1000) // 10
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    # Convert accent hex to ASS BGR
    r, g, b  = _hex(accent)
    ass_col  = f"&H00{b:02X}{g:02X}{r:02X}&"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {VIDEO_W}
PlayResY: {VIDEO_H}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,ScaleX,ScaleY,Spacing,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV
Style: Default,Arial,85,&H00FFFFFF,{ass_col},&H00000000,&H99000000,-1,0,0,100,100,0,1,6,2,2,60,60,220

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = Path(srt_path).read_text().strip().split('\n\n')
    events = []
    for block in lines:
        parts = block.strip().split('\n')
        if len(parts) < 3:
            continue
        # Parse timing line: 00:00:00,000 --> 00:00:02,500
        try:
            timing = parts[1]
            start_str, end_str = timing.split(" --> ")
            def parse_t(t):
                t = t.replace(",", ".")
                h, m, s = t.split(":")
                return int((int(h)*3600 + int(m)*60 + float(s)) * 1000)
            start_ms  = parse_t(start_str.strip())
            end_ms    = parse_t(end_str.strip())
            text      = " ".join(parts[2:])
            words     = text.split()
            dur_ms    = end_ms - start_ms
            word_dur  = dur_ms // max(len(words), 1)
            # Build karaoke tags: {\k<centiseconds>}word
            _k        = chr(92) + "k"
            karaoke   = " ".join("{" + _k + str(word_dur // 10) + "}" + w for w in words)
            events.append(
                f"Dialogue: 0,{ms_to_ass(start_ms)},{ms_to_ass(end_ms)},"
                f"Default,,0,0,0,,{karaoke}"
            )
        except Exception:
            continue
    Path(ass_path).write_text(header + '\n'.join(events), encoding='utf-8')


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

def backup_to_drive(video_path: str, srt_path: str, title: str) -> Optional[str]:
    """
    Upload finished video to Google Drive → 'VaultMind' folder.
    Returns Drive file URL or None.
    """
    try:
        from googleapiclient.discovery import build as gdrive_build
        token_b64 = os.getenv("YOUTUBE_TOKEN_B64")
        if not token_b64:
            return None
        creds_data = pickle.loads(base64.b64decode(token_b64))
        if isinstance(creds_data, Credentials):
            creds = creds_data
        elif isinstance(creds_data, str):
            creds = Credentials.from_authorized_user_info(json.loads(creds_data))
        else:
            creds = Credentials.from_authorized_user_info(creds_data)

        drive    = gdrive_build("drive", "v3", credentials=creds)

        # Find or create VaultMind folder
        q        = "name='VaultMind' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results  = drive.files().list(q=q, fields="files(id)").execute()
        folders  = results.get("files", [])
        if folders:
            folder_id = folders[0]["id"]
        else:
            folder_meta = {"name": "VaultMind",
                           "mimeType": "application/vnd.google-apps.folder"}
            folder_id = drive.files().create(body=folder_meta,
                                              fields="id").execute()["id"]

        # Upload video
        vid_meta  = {"name": f"{title}.mp4", "parents": [folder_id]}
        vid_media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
        vid_file  = drive.files().create(
            body=vid_meta, media_body=vid_media, fields="id,webViewLink"
        ).execute()
        link = vid_file.get("webViewLink", "")
        log.info("Drive backup: %s", link)
        return link
    except Exception as e:
        log.warning("Drive backup skipped: %s", e)
        return None



# ══════════════════════════════════════════════════════════════════════
# SELF-LEARNING: VIRAL VIDEO ANALYZER
#
#  Runs BEFORE script generation on each pipeline run.
#  Uses only FREE tools:
#    • yt-dlp   → download captions (no Whisper needed for YouTube)
#    • ffmpeg   → extract keyframes
#    • Groq Vision (llama-4-scout) → analyze frames (free tier, 1000 RPD)
#
#  Result: viral_patterns.json committed to repo.
#  Script generator reads it and adapts prompts automatically.
#
#  Curated viral Shorts per niche — all public YouTube URLs.
#  Update this list by adding YouTube Shorts links you like.
# ══════════════════════════════════════════════════════════════════════

VIRAL_REFERENCE_VIDEOS = {
    "fact":       ["https://www.youtube.com/shorts/2gpg7W7ht6A"],
    "scary":      ["https://youtube.com/shorts/JCEEvR-1b2A"],
    "motivation": ["https://youtube.com/shorts/5f7E4DQG6kk"],
    "reddit":     ["https://youtube.com/shorts/P0xBYEwcl-4"],
    "rich":       ["https://youtube.com/shorts/g1jmkEjykLk"],
    "conspiracy": ["https://youtube.com/shorts/VUyBmwGpzkE"],
    "dating":     ["https://youtube.com/shorts/2W4SDn_6WEw"],
    "lifehack":   ["https://youtube.com/shorts/klMk4UkYQ0o"],
    "learn":      [
        "https://youtube.com/shorts/kxOFEpLKNpk",
        "https://youtube.com/shorts/qHf2RSKLbsw",
    ],
}


def _extract_youtube_captions(url: str, out_path: str) -> Optional[str]:
    """Download auto-captions from YouTube via yt-dlp (free, no Whisper)."""
    try:
        import yt_dlp
        opts = {
            "skip_download":    True,
            "writeautomaticsub": True,
            "writesubtitles":   True,
            "subtitleslangs":   ["en"],
            "subtitlesformat":  "vtt",
            "outtmpl":          out_path.replace(".vtt",""),
            "quiet":            True,
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        # yt-dlp appends .en.vtt
        vtt = out_path.replace(".vtt","") + ".en.vtt"
        if Path(vtt).exists():
            # Strip VTT timing tags → plain text
            text = Path(vtt).read_text(errors="ignore")
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line and "-->" not in line and not line.startswith("WEBVTT")                         and not line.startswith("NOTE") and not line.isdigit():
                    import re
                    clean = re.sub(r"<[^>]+>", "", line)
                    if clean and clean not in lines[-3:]:
                        lines.append(clean)
            return " ".join(lines)[:3000]
    except Exception as e:
        log.debug("Caption extraction failed: %s", e)
    return None


def _extract_frames(url: str, out_dir: str, n_frames: int = 8) -> list[str]:
    """
    Download video and extract n_frames keyframes as JPEGs.
    Uses yt-dlp + ffmpeg — both already in the GitHub Actions environment.
    """
    try:
        import yt_dlp
        video_path = os.path.join(out_dir, "clip.mp4")
        opts = {
            "format":    "bestvideo[height<=720][ext=mp4]/best[height<=720]",
            "outtmpl":   video_path,
            "quiet":     True,
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        if not Path(video_path).exists():
            return []

        # Get duration
        probe = subprocess.run(
            ["ffprobe","-v","quiet","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True
        )
        duration = float(probe.stdout.strip() or "60")

        # Extract evenly-spaced frames
        frame_paths = []
        for i in range(n_frames):
            t        = duration * (i + 0.5) / n_frames
            out_jpg  = os.path.join(out_dir, f"frame_{i:02d}.jpg")
            subprocess.run(
                ["ffmpeg","-y","-ss",str(t),"-i",video_path,
                 "-frames:v","1","-q:v","3","-vf","scale=512:-1",
                 out_jpg],
                capture_output=True
            )
            if Path(out_jpg).exists():
                frame_paths.append(out_jpg)
        return frame_paths
    except Exception as e:
        log.debug("Frame extraction failed: %s", e)
        return []


def _analyze_frames_groq(frames: list[str], transcript: str,
                          niche: str) -> Optional[dict]:
    """
    Send frames + transcript to Groq Vision (llama-4-scout, free).
    Returns structured analysis dict or None on failure.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        import base64
        client = Groq(api_key=key)

        # Build image content blocks (max 6 to save tokens)
        image_blocks = []
        for fp in frames[:6]:
            with open(fp, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        prompt = (
            f'You are a viral short-form video analyst. Niche: {niche}.\n'
            f'Transcript snippet: "{transcript[:600]}"\n\n'
            'Analyze these video frames and return a JSON object with:\n'
            '- hook_style: how does the first 2 seconds grab attention?\n'
            '- caption_style: font size (small/medium/large), position (top/center/bottom), '
            'highlighted words yes/no, color\n'
            '- cut_speed: average scene length estimate in seconds (e.g. 2, 3, 4)\n'
            '- visual_style: dark/bright/colorful/minimal, any overlays or effects\n'
            '- text_animation: bounce/fade/none/pop\n'
            '- background_type: stock_footage/solid_color/gameplay/animation\n'
            '- emotional_trigger: curiosity/fear/inspiration/humor/shock\n'
            '- what_makes_it_viral: 1-2 sentences\n'
            'Return ONLY valid JSON, no explanation.'
        )

        resp = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": image_blocks + [{"type": "text", "text": prompt}]
            }],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.3,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        log.warning("Groq Vision analysis failed: %s", e)
        return None


def analyze_viral_videos(force: bool = False) -> dict:
    """
    Analyze curated viral reference videos and update viral_patterns.json.

    Skips if viral_patterns.json was updated in the last 3 days (to avoid
    hitting Groq rate limits on every run). Set force=True to override.

    Returns the loaded patterns dict.
    """
    patterns_path = Path("viral_patterns.json")

    # Load existing patterns
    if patterns_path.exists():
        try:
            existing = json.loads(patterns_path.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}

    # Skip if recently updated
    last_run = existing.get("_last_analyzed", "")
    if not force and last_run:
        try:
            age_days = (datetime.utcnow() -
                        datetime.fromisoformat(last_run)).days
            if age_days < 3:
                log.info("viral_patterns.json is %d days old — skipping analysis", age_days)
                return existing
        except Exception:
            pass

    log.info("Starting viral video analysis…")
    patterns = dict(existing)

    for niche, urls in VIRAL_REFERENCE_VIDEOS.items():
        if not urls:
            continue
        niche_results = []
        for url in urls[:2]:   # max 2 per niche per run (rate limit friendly)
            log.info("Analyzing [%s]: %s", niche, url)
            with tempfile.TemporaryDirectory() as td:
                transcript = _extract_youtube_captions(url, os.path.join(td,"caps.vtt"))
                transcript = transcript or ""
                frames     = _extract_frames(url, td, n_frames=8)
                if not frames:
                    log.warning("No frames extracted for %s", url)
                    continue
                analysis = _analyze_frames_groq(frames, transcript, niche)
                if analysis:
                    analysis["url"]       = url
                    analysis["analyzed_at"] = datetime.utcnow().isoformat()
                    niche_results.append(analysis)
                    log.info("[%s] Analysis: %s", niche,
                             json.dumps(analysis, indent=2)[:300])
                time.sleep(2)   # be gentle on rate limits

        if niche_results:
            patterns[niche] = niche_results

    patterns["_last_analyzed"] = datetime.utcnow().isoformat()
    patterns_path.write_text(json.dumps(patterns, indent=2))
    log.info("viral_patterns.json updated (%d niches)", len(patterns)-1)
    return patterns


def load_viral_patterns() -> dict:
    """Load existing viral patterns, return empty dict if unavailable."""
    try:
        p = Path("viral_patterns.json")
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {}


def patterns_to_prompt_hints(patterns: dict, niche: str) -> str:
    """
    Convert analyzed viral patterns for a niche into concrete script hints.
    These get injected into the Groq script generation prompt.
    """
    niche_data = patterns.get(niche, [])
    if not niche_data:
        return ""

    # Aggregate across multiple analyzed videos
    cut_speeds  = [d.get("cut_speed", 3) for d in niche_data if "cut_speed" in d]
    hook_styles = [d.get("hook_style","") for d in niche_data if d.get("hook_style")]
    triggers    = [d.get("emotional_trigger","") for d in niche_data if d.get("emotional_trigger")]
    viral_tips  = [d.get("what_makes_it_viral","") for d in niche_data if d.get("what_makes_it_viral")]

    avg_cut = round(sum(cut_speeds)/len(cut_speeds), 1) if cut_speeds else 3.5
    hints = [
        f"VIRAL DATA for {niche} (learned from real viral videos in this niche):",
        f"- Ideal scene length: ~{avg_cut}s per scene",
    ]
    if hook_styles:
        hints.append(f"- Hook style that works: {hook_styles[0]}")
    if triggers:
        unique_triggers = list(dict.fromkeys(triggers))
        hints.append(f"- Emotional triggers that drive retention: {', '.join(unique_triggers[:3])}")
    if viral_tips:
        hints.append(f"- What makes {niche} videos go viral: {viral_tips[0]}")
    return '\n'.join(hints)


# ══════════════════════════════════════════════════════════════════════
# BALL FIGHT ENGINE — physics-simulated arena videos for TikTok
#
#  Three self-contained game modes, chosen at random each run:
#    • battle_royale   — shrinking circular arena, last ball standing
#    • color_infection — team-color "paint tag", one color takes the arena
#    • gravity_race    — pachinko-style drop race to the finish line
#
#  Every run randomizes: ball count, colors, skins/patterns, arena size,
#  and which of the 4 synthesized music moods plays — so no two videos
#  look or sound alike, per spec ("neue Bälle", "dynamische Musik").
# ══════════════════════════════════════════════════════════════════════

import colorsys

BF_W, BF_H, BF_FPS = VIDEO_W, VIDEO_H, FPS
BF_MODES = ["battle_royale", "color_infection", "gravity_race"]
BF_PATTERNS = ["solid", "stripes", "dots", "eyes"]


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
    return big.resize((d, d), Image.LANCZOS)


def _bf_ffmpeg_writer(path: str, w: int, h: int, fps: int) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{w}x{h}", "-framerate", str(fps),
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
         "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


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
# MODE 1 — BATTLE ROYALE (shrinking circular arena, elimination)
# ─────────────────────────────────────────────────────────────────────

def simulate_battle_royale(target_dur: float, out_path: str, rng: random.Random) -> tuple:
    space = pymunk.Space(); space.gravity = (0, 0)
    center = (BF_W / 2, BF_H * 0.44)
    r0 = BF_W * 0.46
    r1 = BF_W * 0.10
    grace = target_dur * 0.18
    n = rng.randint(8, 14)
    accent = NICHES["ballfight"]["glow"]

    balls_meta = _bf_gen_balls(n, (32, 46), rng)
    live = []
    for bm in balls_meta:
        mass = (bm["r"] ** 2) * 0.02
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, bm["r"]))
        ang = rng.uniform(0, 2 * math.pi); dist = rng.uniform(0, r0 * 0.55)
        body.position = (center[0] + math.cos(ang) * dist, center[1] + math.sin(ang) * dist)
        speed = rng.uniform(140, 260)
        vang = rng.uniform(0, 2 * math.pi)
        body.velocity = (math.cos(vang) * speed, math.sin(vang) * speed)
        shape = pymunk.Circle(body, bm["r"]); shape.elasticity = 0.97; shape.friction = 0.05
        space.add(body, shape)
        live.append({**bm, "body": body, "shape": shape, "alive": True})

    def radius_at(t):
        sudden_death = target_dur - 3.0
        if t < grace:
            return r0
        if t < sudden_death:
            frac = (t - grace) / max(0.01, sudden_death - grace)
            return r0 - (r0 - r1) * frac
        return r1  # hold tight for forced finish

    events = [{"t": 0.0, "type": "start", "n": n}]
    touching, winner = set(), None
    proc = _bf_ffmpeg_writer(out_path, BF_W, BF_H, BF_FPS)
    bg = _bf_background(BF_W, BF_H, NICHES["ballfight"]["bg"], _hex(accent))
    n_frames = int(target_dur * BF_FPS)

    for f in range(n_frames):
        t = f / BF_FPS
        for _ in range(2):
            space.step(1 / (BF_FPS * 2))
        R = radius_at(t)
        alive_balls = [b for b in live if b["alive"]]
        # boundary handling
        for b in alive_balls:
            p = b["body"].position
            d = math.hypot(p[0] - center[0], p[1] - center[1])
            if d + b["r"] > R:
                if d + b["r"] > R + b["r"] * 0.7 and len(alive_balls) > 1:
                    b["alive"] = False
                    space.remove(b["body"], b["shape"])
                    events.append({"t": round(t, 2), "type": "eliminated",
                                    "id": b["id"], "color": b["color"], "left": len(alive_balls) - 1})
                else:
                    nx, ny = (p[0] - center[0]) / max(d, 1e-6), (p[1] - center[1]) / max(d, 1e-6)
                    vx, vy = b["body"].velocity
                    dot = vx * nx + vy * ny
                    if dot > 0:
                        b["body"].velocity = (vx - 2 * dot * nx, vy - 2 * dot * ny)
        # collision logging (for SFX)
        alive_balls = [b for b in live if b["alive"]]
        for i in range(len(alive_balls)):
            for j in range(i + 1, len(alive_balls)):
                a, b = alive_balls[i], alive_balls[j]
                pa, pb = a["body"].position, b["body"].position
                dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                key = (a["id"], b["id"])
                if dist < a["r"] + b["r"]:
                    if key not in touching:
                        speed = math.hypot(*a["body"].velocity) + math.hypot(*b["body"].velocity)
                        events.append({"t": round(t, 2), "type": "collision", "impact": min(1.0, speed / 800)})
                    touching.add(key)
                else:
                    touching.discard(key)

        if len(alive_balls) == 1 and winner is None:
            winner = alive_balls[0]
            events.append({"t": round(t, 2), "type": "winner", "id": winner["id"], "color": winner["color"]})

        img = bg.copy()
        dr = ImageDraw.Draw(img)
        dr.ellipse([center[0] - R - 6, center[1] - R - 6, center[0] + R + 6, center[1] + R + 6],
                   outline=_hex(accent), width=10)
        dr.ellipse([center[0] - R, center[1] - R, center[0] + R, center[1] + R],
                   outline=(255, 255, 255), width=3)
        for b in alive_balls:
            _bf_paste_sprite(img, b["sprite"], b["body"].position, b["body"].angle)
        left_n = len(alive_balls) if winner is None else 1
        _bf_hud_text(img, (BF_W / 2, 90), f"{left_n} LEFT" if winner is None else "WINNER!", 78, _hex(accent))
        proc.stdin.write(np.array(img).tobytes())

    proc.stdin.close(); proc.wait()
    if winner is None and live:
        winner = max((b for b in live), key=lambda b: b["alive"])
        events.append({"t": target_dur, "type": "winner", "id": winner["id"], "color": winner["color"]})
    return events, {"mode": "battle_royale", "winner_color": winner["color"] if winner else None,
                     "n_balls": n, "duration": target_dur}


# ─────────────────────────────────────────────────────────────────────
# MODE 2 — COLOR INFECTION (team paint-tag, non-elimination)
# ─────────────────────────────────────────────────────────────────────

def simulate_color_infection(target_dur: float, out_path: str, rng: random.Random) -> tuple:
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
    proc = _bf_ffmpeg_writer(out_path, BF_W, BF_H, BF_FPS)
    bg = _bf_background(BF_W, BF_H, NICHES["ballfight"]["bg"], _hex(accent))
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
        dr.ellipse([center[0] - R - 6, center[1] - R - 6, center[0] + R + 6, center[1] + R + 6],
                   outline=_hex(accent), width=10)
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

def simulate_gravity_race(target_dur: float, out_path: str, rng: random.Random) -> tuple:
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
    proc = _bf_ffmpeg_writer(out_path, BF_W, BF_H, BF_FPS)
    bg = _bf_background(BF_W, BF_H, NICHES["ballfight"]["bg"], _hex(accent))
    n_frames = int(target_dur * BF_FPS)
    winner = None
    finish_hold = None

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
            dr.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(130, 130, 150))
        dr.line([(margin, finish_y), (BF_W - margin, finish_y)], fill=_hex(accent), width=6)
        _bf_hud_text(img, (BF_W / 2, finish_y - 30), "FINISH", 42, _hex(accent))
        for b in balls:
            if not b["dropped"]:
                continue
            _bf_paste_sprite(img, b["sprite"], b["body"].position, b["body"].angle)
        if winner:
            _bf_hud_text(img, (BF_W / 2, 90), "WINNER!", 78, _hex(accent))
        proc.stdin.write(np.array(img).tobytes())

        if finish_hold and t >= finish_hold:
            break

    proc.stdin.close(); proc.wait()
    return events, {"mode": "gravity_race", "winner_color": winner["color"] if winner else None,
                     "n_balls": n, "duration": (finish_hold or target_dur)}


BF_SIM_FUNCS = {"battle_royale": simulate_battle_royale,
                "color_infection": simulate_color_infection,
                "gravity_race": simulate_gravity_race}


# ─────────────────────────────────────────────────────────────────────
# DYNAMIC MUSIC — 4 distinct synthesized moods, one picked at random
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
    root = rng.choice([82.4, 98.0, 110.0])
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


def mix_ballfight_audio(events: list, video_dur: float, voice_path: str,
                        music_path: str, tmp_dir: Path) -> str:
    """Layer synthesized collision/elimination blips + looped ducked music
    under the narration, matched to real event timestamps."""
    voice = AudioSegment.from_file(voice_path)
    bed = AudioSegment.silent(duration=int(video_dur * 1000) + 3000)

    music = AudioSegment.from_file(music_path)
    music = (music * (int(len(bed) / len(music)) + 1))[:len(bed)]
    music = music + MUSIC_DUCK_DB
    bed = bed.overlay(music)

    rng = random.Random()
    for ev in events:
        if ev["type"] not in ("collision", "eliminated", "convert", "finish", "winner"):
            continue
        base_freq = {"collision": 520, "eliminated": 220, "convert": 700,
                     "finish": 380, "winner": 300}[ev["type"]]
        freq = base_freq * rng.uniform(0.85, 1.25)
        dur = 0.22 if ev["type"] in ("eliminated", "winner") else 0.09
        blip = _bf_make_blip_segment(freq, dur)
        impact = ev.get("impact", 0.7)
        blip = blip + (6 * impact - 6)  # louder on harder impacts
        pos_ms = int(ev["t"] * 1000)
        if pos_ms < len(bed):
            bed = bed.overlay(blip, position=pos_ms)

    bed = bed.overlay(voice)
    out_path = str(tmp_dir / "ballfight_audio.mp3")
    bed.export(out_path, format="mp3")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# COMMENTARY — funny sports-caster script generated from real events
# ─────────────────────────────────────────────────────────────────────

def generate_ballfight_commentary(mode: str, events: list, meta: dict, target_dur: float) -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    key_events = [e for e in events if e["type"] in
                  ("start", "eliminated", "convert", "finish", "winner")][:24]
    event_summary = "\n".join(
        f"- t={e['t']}s: {e['type']}" + (f" (ball/color {e.get('color') or e.get('id')})" if e.get('color') or 'id' in e else "")
        for e in key_events
    )
    mode_desc = {
        "battle_royale": "a shrinking-arena ball battle royale — balls get eliminated as the ring closes in, last ball wins",
        "color_infection": "a color-infection game — balls 'infect' each other on contact, one team color takes over the whole arena",
        "gravity_race": "a gravity pachinko drop-race — balls fall through pegs and race to a finish line",
    }[mode]

    target_words = int(target_dur * 2.6)

    system = (
        "You write short, genuinely funny sports-commentator voiceover scripts for "
        "physics-simulation TikTok videos (like satisfying marble-race / ball-battle content). "
        "Return ONLY valid JSON, no markdown: {title, hashtags: [string], text}. "
        "Tone: hyped, playful, a little absurd — like a caster who is WAY too invested in some "
        "bouncing balls. React to the specific events given, in order, but keep it snappy and "
        "punchy, never a dry play-by-play list. Add jokes, fake stakes, mock-dramatic reactions. "
        f"Target spoken length: about {target_words} words (~{target_dur:.0f}s). "
        "title: short catchy TikTok title. hashtags: 6-10 tags mixing #ballfight #satisfying "
        "#physics #fyp with a couple specific to the game mode."
    )
    prompt = (
        f"Write commentary for {mode_desc}.\n"
        f"Real events from this exact simulation, in order:\n{event_summary}\n"
        "Open with a big hook line, react to eliminations/conversions/finishes as they happen, "
        "and close with a punchy line about the winner."
    )

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=1.0, max_tokens=1200,
            )
            raw = resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
            data = json.loads(raw)
            if data.get("text"):
                return data
        except Exception as e:
            log.warning("Ballfight commentary attempt %d: %s", attempt + 1, e)
        time.sleep(2 ** attempt)
    # Fallback if Groq fails entirely — still produces a postable video
    return {"title": f"Ball {mode.replace('_',' ').title()}!",
            "hashtags": ["ballfight", "satisfying", "physics", "fyp", "foryou"],
            "text": f"Watch this {mode.replace('_',' ')} play out to the very end. Who wins?"}


# ─────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────

def run_ballfight_pipeline() -> None:
    mode = random.choice(BF_MODES)
    target_dur = random.uniform(24, 36)
    rng = random.Random()
    log.info("═══ VaultMind Ballfight — mode: %s (~%.0fs) ═══", mode, target_dur)

    ensure_font()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Simulate + render silent physics footage
        raw_video = str(tmp / "raw.mp4")
        events, meta = BF_SIM_FUNCS[mode](target_dur, raw_video, rng)
        real_dur = meta.get("duration", target_dur)
        log.info("Sim done: %d events, winner=%s", len(events), meta.get("winner_color"))

        # 2. Funny commentary script from the real event log
        script  = generate_ballfight_commentary(mode, events, meta, real_dur)
        title   = script.get("title", "Ball Fight!")
        hashtags = script.get("hashtags", ["ballfight", "satisfying", "fyp"])
        text    = script["text"]

        # 3. Voiceover (reuses existing ElevenLabs → OpenAI → edge-tts → gTTS chain)
        voice_path = str(tmp / "voice.mp3")
        generate_voiceover(text, voice_path)

        # 4. Transcription for captions
        aai_data = transcribe_with_assemblyai(voice_path)
        srt_path = str(tmp / "captions.srt")
        export_srt(aai_data, srt_path)

        # 5. Dynamic music mood + collision/elimination SFX, mixed with voice
        music_path  = str(tmp / "music.mp3")
        ensure_ballfight_music(real_dur, music_path, rng)
        audio_path  = mix_ballfight_audio(events, real_dur, voice_path, music_path, tmp)

        # 6. Mux silent physics video + mixed audio, padding with a frozen
        #    last frame if the commentary runs longer than the sim.
        voice_dur = len(AudioSegment.from_file(audio_path)) / 1000
        muxed = str(tmp / "muxed.mp4")
        if voice_dur > real_dur + 0.5:
            padded = str(tmp / "padded.mp4")
            pad_s = voice_dur - real_dur + 0.5
            subprocess.run(["ffmpeg", "-y", "-i", raw_video, "-vf",
                            f"tpad=stop_mode=clone:stop_duration={pad_s:.2f}",
                            "-c:v", "libx264", "-preset", "veryfast", padded],
                           check=True, capture_output=True)
            raw_video = padded
        subprocess.run(["ffmpeg", "-y", "-i", raw_video, "-i", audio_path,
                        "-c:v", "copy", "-c:a", "aac", "-shortest", muxed],
                       check=True, capture_output=True)

        # 7. Animated captions + color grade (reuses existing helpers)
        captioned = apply_beautiful_captions(
            video_path=muxed, srt_path=srt_path,
            output_path=str(tmp / "captioned.mp4"),
            accent=NICHES["ballfight"]["color"], words_per_line=2,
        )
        final_video = apply_color_grade(captioned, str(tmp / "final.mp4"), niche="ballfight")

        # 8. Upload to TikTok
        tiktok_id = upload_tiktok(final_video, title, hashtags)

        # 9. Backup + dashboard
        backup_to_drive(final_video, srt_path, title)
        update_dashboard("ballfight", title,
                         f"tiktok publish_id: {tiktok_id}" if tiktok_id else None,
                         ok=bool(tiktok_id), platform="tiktok")

    log.info("═══ Ballfight complete: %s ═══", tiktok_id or "upload failed/skipped")


def run_pipeline(niche: Optional[str] = None) -> None:
    if niche == "ballfight":
        run_ballfight_pipeline()
        return
    niche = niche or random.choice(STOCKFOOTAGE_NICHES)
    log.info("═══ VaultMind v4 — niche: %s ═══", niche)

    ensure_font()
    music_path             = ensure_music(duration_s=120)
    sfx_whoosh, sfx_impact = ensure_sfx()

    # Analyze curated viral videos (free, skips if <3 days old)
    analyze_viral_videos()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Script
        script    = generate_script(niche)
        scenes    = script["scenes"]
        title     = script["title"]
        full_text = script.get("hook","") + " " + " ".join(s["text"] for s in scenes)
        log.info("Title: %s | %d scenes", title, len(scenes))

        # 2. Voiceover
        audio_path = str(tmp/"voice.mp3")
        generate_voiceover(full_text, audio_path)

        # 3. Transcription
        aai_data = transcribe_with_assemblyai(audio_path)

        # 4. Cut points
        cuts = detect_cut_points(audio_path, aai_data, n_scenes=len(scenes))

        # 5. SRT
        srt_path = str(tmp/"captions.srt")
        export_srt(aai_data, srt_path)

        # 6. RENDER — Priority chain (first success wins):
        #    JSON2Video (free) → Shotstack (free sandbox) → Creatomate (paid) → MoviePy (local)
        raw_video      = str(tmp/"raw.mp4")
        render_args    = dict(scenes=scenes, audio_path=audio_path,
                              niche=niche, output_path=raw_video,
                              music_path=music_path)
        creatomate_used = False

        if json2video_render(**render_args):
            log.info("✅ Rendered via JSON2Video")
        elif shotstack_render(**render_args):
            log.info("✅ Rendered via Shotstack")
        elif creatomate_render(scenes, audio_path, niche, raw_video, music_path):
            log.info("✅ Rendered via Creatomate")
            creatomate_used = True   # captions already burned in
        else:
            log.info("⚠️ All cloud renders failed — using local MoviePy")
            build_video_moviepy(
                scenes=scenes, niche=niche,
                audio_path=audio_path, aai_data=aai_data,
                cut_points_s=cuts, output_path=raw_video,
                music_path=music_path,
                sfx_whoosh_path=sfx_whoosh,
                sfx_impact_path=sfx_impact,
            )

        # 7. Animated captions — SKIP if Creatomate already burned in
        #    transcript-highlight captions during its own render.
        if creatomate_used:
            log.info("Creatomate already applied captions — skipping caption step")
            captioned_video = raw_video
        else:
            captioned_video = str(tmp/"captioned.mp4")
            captioned_video = apply_beautiful_captions(
                video_path     = raw_video,
                srt_path       = srt_path,
                output_path    = captioned_video,
                accent         = NICHES[niche]["color"],
                words_per_line = 2,
            )

        # 8. Cinematic color grade (teal-orange + unsharp)
        final_video = str(tmp/"final.mp4")
        final_video = apply_color_grade(captioned_video, final_video, niche=niche)

        # 9. Upload
        hashtags  = script.get("hashtags", [])
        tag_line  = " ".join(h if h.startswith("#") else f"#{h}" for h in hashtags) \
                    or f"#shorts #{niche} #viral #fyp"
        desc      = f"{title}\n\n{tag_line}"
        video_url = upload_youtube(final_video, title, desc, srt_path, tags=hashtags)

        # 10. Google Drive backup
        backup_to_drive(final_video, srt_path, title)

        # 11. Dashboard
        update_dashboard(niche, title, video_url, ok=True)

    log.info("═══ v4 complete: %s ═══", video_url or "no URL")


if __name__ == "__main__":
    import sys
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else None)
