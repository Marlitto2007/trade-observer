#!/usr/bin/env python3
"""
observe.py — Infinity Algo Signal Observation Tool (Railway edition)

Runs ENTIRELY on the phone inside Termux. No laptop needed.
Uses yt-dlp + ffmpeg + numpy + scipy + PIL — NO opencv required.

USAGE:
  python observe.py

CALIBRATE:
  python observe_termux.py --calibrate
"""

import os
import sys
import time
import subprocess
import json
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

# ── env from ~/.trader_env if present ────────────────────────────
_env_file = Path.home() / ".trader_env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    import pytesseract
    # On Termux, tesseract is on PATH — no custom path needed
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("MISSING: pip install pytesseract")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SCREENSHOT_INTERVAL  = 30           # seconds between scans
TARGET_SIGNALS       = 40           # how many signals to collect
SIGNAL_COOLDOWN_SEC  = 300          # min 5 min between logging two signals
DISCORD_WEBHOOK      = os.getenv("DISCORD_WEBHOOK_OBSERVE", "")
SIGNAL_DIR           = Path("observe_signals")
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# SCREEN REGIONS  (2400x1080 landscape — confirmed by calibration)
# ═══════════════════════════════════════════════════════════════

SCREEN_W        = 1920
SCREEN_H        = 1080
CHART_DIVIDER_X = 728

REGIONS = {
    "chart_full":  (0,    0,   1460,  820),
    "chart_left":  (0,    0,    660,  820),
    "chart_right": (728,  0,   1390,  820),
    "mtf_left":    (8,   878,   215, 1080),
    "mtf_right":   (965, 878,  1150, 1080),
    "asset_left":  (0,    0,    560,   35),
    "asset_right": (728,  0,   1300,   35),
    "price_left":  (10,   38,   210,   68),
    "price_right": (740,  38,   935,   68),
}

# ═══════════════════════════════════════════════════════════════
# SIGNAL DETECTION — HSV color ranges
# BUY  = bright green,  SELL = bright red/magenta
# ═══════════════════════════════════════════════════════════════

BUY_HSV_LOW    = np.array([40,  120, 120])
BUY_HSV_HIGH   = np.array([90,  255, 255])
SELL_HSV_LOW1  = np.array([0,   120, 120])
SELL_HSV_HIGH1 = np.array([12,  255, 255])
SELL_HSV_LOW2  = np.array([165, 120, 120])
SELL_HSV_HIGH2 = np.array([179, 255, 255])

MIN_SIGNAL_PIXELS    = 40
LABEL_SEARCH_RADIUS  = 150

MTF_TIMEFRAMES = ["1s", "5s", "15s", "30s", "1m", "5m", "15m", "30m", "1h", "4h"]

# ═══════════════════════════════════════════════════════════════
# MIDNIGHT / STREAM RESTART
# ═══════════════════════════════════════════════════════════════

MIDNIGHT_START_IST    = 23
MIDNIGHT_END_IST      = 1
STREAM_OFFLINE_WAIT   = 600
DARK_SCREEN_THRESHOLD = 0.72

# ═══════════════════════════════════════════════════════════════
# RECENT-ZONE — only look for NEW signals near right edge of chart
# ═══════════════════════════════════════════════════════════════

BTC_RECENT_X_MIN        = 580
ETH_RECENT_X_MIN        = 1300
HEARTBEAT_EVERY_N_SCANS = 20

# ═══════════════════════════════════════════════════════════════
# SCREENSHOT — grab frame directly from YouTube live stream
# No phone screen capture needed. No ADB. No permissions.
# ═══════════════════════════════════════════════════════════════

TMP_SCREENSHOT = "obs_tmp.png"

def _ffmpeg_bin():
    """Find ffmpeg: system PATH or imageio-ffmpeg bundled binary."""
    import shutil as _shutil
    f = _shutil.which("ffmpeg")
    if f: return f
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise FileNotFoundError("ffmpeg not found — pip install imageio-ffmpeg")

YOUTUBE_URL    = os.getenv("YOUTUBE_STREAM_URL", "")

def _get_stream_url(yt_url: str) -> str | None:
    r = None
    for fmt in ["best[height<=1080]/best", "best", "worst"]:
        try:
            r = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "-f", fmt, "-g",
                 "--no-playlist", "--no-warnings", "--no-check-certificates",
                 "--extractor-args", "youtube:player_client=tv_embedded,ios,android",
                 yt_url],
                capture_output=True, timeout=60
            )
            if r.returncode == 0:
                out = r.stdout.decode().strip().splitlines()
                if out: return out[0]
        except Exception as e:
            print(f"  [Stream] attempt failed: {e}")
    err = r.stderr.decode(errors="replace")[-300:] if r else "none"
    print(f"  [Stream] yt-dlp failed: {err}")
    return None
    except FileNotFoundError:
        print("  [Stream] yt-dlp not found — run: pip install yt-dlp")
        return None
    except Exception as e:
        print(f"  [Stream] yt-dlp error: {e}")
        return None

_stream_url_cache: dict = {"url": None, "ts": 0.0}
STREAM_URL_TTL = 3600  # re-resolve every hour (live stream URLs expire)

def _cached_stream_url() -> str | None:
    now = time.time()
    if not _stream_url_cache["url"] or (now - _stream_url_cache["ts"]) > STREAM_URL_TTL:
        url = _get_stream_url(YOUTUBE_URL)
        _stream_url_cache["url"] = url
        _stream_url_cache["ts"]  = now
    return _stream_url_cache["url"]

def take_screenshot() -> "Image.Image | None":
    """Grab one frame from the YouTube live stream using ffmpeg."""
    if not YOUTUBE_URL:
        print("  [Stream] YOUTUBE_STREAM_URL not set in ~/.trader_env")
        return None
    stream = _cached_stream_url()
    if not stream:
        return None
    try:
        r = subprocess.run(
            [_ffmpeg_bin(), "-y",
             "-allowed_extensions", "ALL",
             "-reconnect", "1", "-reconnect_streamed", "1",
             "-i", stream,
             "-t", "2", "-vframes", "1", "-q:v", "2", TMP_SCREENSHOT],
            capture_output=True, timeout=90
        )
        if r.returncode != 0 or not Path(TMP_SCREENSHOT).exists():
            # stream URL may have expired — force refresh next time
            _stream_url_cache["url"] = None
            err = r.stderr.decode(errors="replace").strip()
            print(f"  [Stream] ffmpeg rc={r.returncode} err={err[:600]}")
            print(f"  [Stream] ffmpeg failed: {err}")
            return None
        return Image.open(TMP_SCREENSHOT).convert("RGB")
    except FileNotFoundError:
        print("  [Stream] ffmpeg not found")
        return None
    except Exception as e:
        print(f"  [Stream] error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# SCREEN ANALYSIS
# ═══════════════════════════════════════════════════════════════

def is_dark_screen(img: "Image.Image") -> bool:
    crop  = img.crop(REGIONS["chart_full"])
    arr   = np.array(crop)
    dark  = np.sum(arr.mean(axis=2) < 20)
    total = arr.shape[0] * arr.shape[1]
    return (dark / total) > DARK_SCREEN_THRESHOLD


def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """Pure numpy RGB→HSV. arr shape (H,W,3) float32 0-255. Returns HSV 0-179/0-255/0-255."""
    arr = arr.astype(np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc + 1e-9

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc.copy()

    mask_r = (maxc == r)
    mask_g = (maxc == g) & ~mask_r
    mask_b = ~mask_r & ~mask_g

    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r])) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120)
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240)

    nonzero = maxc > 0
    s[nonzero] = diff[nonzero] / maxc[nonzero]

    hsv = np.stack([h / 2, s * 255, v * 255], axis=2).astype(np.uint8)
    return hsv


def _hsv_in_range(hsv: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return ((hsv[:,:,0] >= lo[0]) & (hsv[:,:,0] <= hi[0]) &
            (hsv[:,:,1] >= lo[1]) & (hsv[:,:,1] <= hi[1]) &
            (hsv[:,:,2] >= lo[2]) & (hsv[:,:,2] <= hi[2]))


def detect_signals(img: "Image.Image",
                   seen_positions: "set | None" = None) -> list[dict]:
    """
    Find BUY/SELL arrows via HSV (pure numpy+scipy, no opencv).
    Only scans recent zone (rightmost candles).
    Returns [{direction, x, y, pixels, side}]
    """
    crop = img.crop(REGIONS["chart_full"])
    arr  = np.array(crop)
    hsv  = _rgb_to_hsv(arr)
    ox, oy = REGIONS["chart_full"][0], REGIONS["chart_full"][1]
    results = []
    struct = ndimage.generate_binary_structure(2, 2)  # 3x3 connectivity

    def _find(mask, direction):
        # Morphological close to fill gaps
        mask = ndimage.binary_closing(mask, structure=struct, iterations=2)
        labeled, num = ndimage.label(mask)
        if num == 0:
            return
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        centroids = ndimage.center_of_mass(mask, labeled, range(1, num + 1))
        for i, (area, (cy_f, cx_f)) in enumerate(zip(sizes, centroids)):
            if area < MIN_SIGNAL_PIXELS:
                continue
            cx = int(cx_f) + ox
            cy = int(cy_f) + oy
            if cx < CHART_DIVIDER_X:
                if cx < BTC_RECENT_X_MIN:
                    continue
                side = "left"
            else:
                if cx < ETH_RECENT_X_MIN:
                    continue
                side = "right"
            pos_key = (cx // 30, cy // 30)
            if seen_positions is not None and pos_key in seen_positions:
                continue
            results.append({"direction": direction, "x": cx, "y": cy,
                             "pixels": int(area), "side": side})

    buy_mask  = _hsv_in_range(hsv, BUY_HSV_LOW, BUY_HSV_HIGH)
    sell_mask = (_hsv_in_range(hsv, SELL_HSV_LOW1, SELL_HSV_HIGH1) |
                 _hsv_in_range(hsv, SELL_HSV_LOW2, SELL_HSV_HIGH2))
    _find(buy_mask,  "BUY")
    _find(sell_mask, "SELL")
    return results


def _ocr_region(img: "Image.Image", box: tuple, scale: int = 3) -> str:
    if not OCR_AVAILABLE:
        return ""
    crop = img.crop(box)
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    crop = crop.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(crop, config="--psm 6 --oem 3").strip()


def read_label_near(img: "Image.Image", x: int, y: int) -> str:
    if not OCR_AVAILABLE:
        return "UNKNOWN"
    w, h = img.size
    r   = LABEL_SEARCH_RADIUS
    box = (max(0, x - r), max(0, y - 2 * r), min(w, x + r), min(h, y + r // 2))
    text = _ocr_region(img, box, scale=4).upper()
    if "SMART"  in text: return "SMART"
    if "NORMAL" in text: return "NORMAL"
    return f"?({text[:30].strip()})"


def read_mtf_table_ocr(img: "Image.Image", side: str) -> dict:
    box = REGIONS.get(f"mtf_{side}")
    if box is None:
        return {"raw_text": "", "rows": []}
    raw  = _ocr_region(img, box, scale=4)
    rows = []
    for line in [l.strip() for l in raw.splitlines() if l.strip()]:
        parts = line.split()
        if not parts:
            continue
        tf = parts[0].lower()
        if re.match(r"^\d+[smhd]$", tf) or tf in MTF_TIMEFRAMES:
            rows.append({
                "timeframe":  tf,
                "trend":      parts[1] if len(parts) > 1 else "?",
                "volatility": parts[2] if len(parts) > 2 else "?",
            })
    return {"raw_text": raw, "rows": rows}


def read_full_screen_ocr(img: "Image.Image") -> str:
    if not OCR_AVAILABLE:
        return ""
    resized = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    return pytesseract.image_to_string(resized, config="--psm 3 --oem 3").strip()


def read_asset_label(img: "Image.Image", side: str) -> str:
    if not OCR_AVAILABLE:
        return "BTC/USDT" if side == "left" else "ETH/USDT"
    box  = REGIONS.get(f"asset_{side}", REGIONS["asset_left"])
    text = _ocr_region(img, box, scale=3).upper()
    for asset in ["BTC", "ETH", "SOL", "BNB", "XRP"]:
        if asset in text:
            return f"{asset}/USDT"
    return "BTC/USDT" if side == "left" else "ETH/USDT"


def read_price(img: "Image.Image", side: str) -> str:
    if not OCR_AVAILABLE:
        return "?"
    box  = REGIONS.get(f"price_{side}", REGIONS["price_left"])
    text = _ocr_region(img, box, scale=3)
    nums = re.findall(r"[\d,]+\.?\d*", text)
    return nums[0] if nums else text[:20]

# ═══════════════════════════════════════════════════════════════
# MARKET DATA — Binance public API
# ═══════════════════════════════════════════════════════════════

def fetch_market_data(asset: str = "BTC/USDT") -> dict:
    sym = asset.replace("/", "").replace(":USDT", "").upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    result = {}
    for interval in ["5m", "15m", "1h", "4h"]:
        try:
            url  = (f"https://api.binance.com/api/v3/klines"
                    f"?symbol={sym}&interval={interval}&limit=60")
            klines = requests.get(url, timeout=8).json()
            closes = np.array([float(k[4]) for k in klines])
            highs  = np.array([float(k[2]) for k in klines])
            lows   = np.array([float(k[3]) for k in klines])
            result[interval] = {
                "rsi":   round(float(_calc_rsi(closes)[-1]),   2),
                "ema9":  round(float(_calc_ema(closes,  9)[-1]), 4),
                "ema21": round(float(_calc_ema(closes, 21)[-1]), 4),
                "atr":   round(float(_calc_atr(highs, lows, closes)[-1]), 4),
                "close": round(float(closes[-1]), 4),
            }
        except Exception as exc:
            result[interval] = {"error": str(exc)[:60]}
    return result


def _calc_rsi(closes, period=14):
    d = np.diff(closes)
    ag = np.convolve(np.where(d > 0, d,  0.0), np.ones(period)/period, "valid")
    al = np.convolve(np.where(d < 0, -d, 0.0), np.ones(period)/period, "valid")
    rs  = ag / np.where(al == 0, 1e-9, al)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([np.full(period, np.nan), rsi])


def _calc_ema(closes, period):
    k   = 2.0 / (period + 1)
    ema = np.full_like(closes, np.nan, dtype=float)
    ema[period - 1] = closes[:period].mean()
    for i in range(period, len(closes)):
        ema[i] = closes[i] * k + ema[i-1] * (1 - k)
    return ema


def _calc_atr(highs, lows, closes, period=14):
    tr  = np.maximum(highs[1:] - lows[1:],
          np.maximum(np.abs(highs[1:] - closes[:-1]),
                     np.abs(lows[1:]  - closes[:-1])))
    atr = np.full(len(closes), np.nan)
    atr[period] = tr[:period].mean()
    for i in range(period + 1, len(closes)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
    return atr

# ═══════════════════════════════════════════════════════════════
# DISCORD
# ═══════════════════════════════════════════════════════════════

def post_signal(record: dict, img: "Image.Image", signal_count: int):
    if not DISCORD_WEBHOOK:
        print("  [Discord] DISCORD_WEBHOOK_OBSERVE not set in ~/.trader_env")
        return

    direction   = record.get("direction", "?")
    label       = record.get("label",     "?")
    asset       = record.get("asset",     "?")
    side        = record.get("side",      "?")
    mtf_data    = record.get("mtf",       {})
    market      = record.get("market",    {})
    screen_text = record.get("screen_text", "")
    price_ocr   = record.get("price_ocr", "?")
    ts          = record.get("timestamp", _now_iso())

    dir_emoji   = "🟢" if direction == "BUY"  else "🔴"
    label_emoji = "⚡" if label    == "SMART" else "📍"
    progress    = f"{signal_count}/{TARGET_SIGNALS}"

    mtf_rows = mtf_data.get("rows", [])
    mtf_text = ("\n".join(
        f"`{r['timeframe']:>4}` | {r['trend']:<10} | {r['volatility']}"
        for r in mtf_rows
    ) or f"_(raw OCR)_\n```\n{mtf_data.get('raw_text','')[:200]}\n```")

    market_lines = []
    for tf in ["5m", "15m", "1h", "4h"]:
        d = market.get(tf, {})
        if "error" not in d:
            market_lines.append(
                f"`{tf:>3}` RSI={d.get('rsi','?'):5}  "
                f"EMA9={d.get('ema9','?')}  EMA21={d.get('ema21','?')}  "
                f"ATR={d.get('atr','?')}  Close={d.get('close','?')}"
            )
        else:
            market_lines.append(f"`{tf}` ERR: {d['error'][:40]}")

    embed = {
        "title":       f"{dir_emoji} {label_emoji} Signal #{signal_count} — {asset}",
        "description": f"**{progress}** signals collected",
        "color":       0x00ff88 if direction == "BUY" else 0xff3333,
        "fields": [
            {"name": "Direction",   "value": f"`{direction}`",     "inline": True},
            {"name": "Label",       "value": f"`{label}`",         "inline": True},
            {"name": "Asset/Chart", "value": f"`{asset}` ({side} chart)", "inline": True},
            {"name": "Price (OCR)", "value": f"`{price_ocr}`",     "inline": True},
            {"name": "Timestamp",   "value": f"`{ts}`",            "inline": False},
            {"name": "MTF Dashboard (Trend | Volatility)",
             "value": mtf_text[:1000] or "_none_",                 "inline": False},
            {"name": "Market Data (Binance)",
             "value": "\n".join(market_lines)[:1000] or "_none_",  "inline": False},
        ],
        "footer": {"text": "Infinity Algo Observer — Railway"},
    }

    # Save annotated screenshot to disk
    img_path = SIGNAL_DIR / f"signal_{signal_count:03d}.png"
    img.save(str(img_path))

    # Post embed
    try:
        requests.post(DISCORD_WEBHOOK, json={"embeds": [embed]}, timeout=10)
    except Exception as e:
        print(f"  [Discord] embed post failed: {e}")

    # Post screenshot as file
    try:
        with open(str(img_path), "rb") as f:
            requests.post(
                DISCORD_WEBHOOK,
                files={"file": (f"signal_{signal_count:03d}.png", f, "image/png")},
                data={"content": f"📸 Signal #{signal_count} screenshot (annotated)"},
                timeout=15,
            )
    except Exception as e:
        print(f"  [Discord] screenshot post failed: {e}")

    # Post full screen OCR text if not empty
    if screen_text and len(screen_text) > 20:
        ocr_chunk = screen_text[:1800]
        try:
            requests.post(
                DISCORD_WEBHOOK,
                json={"content": f"📋 **Full screen OCR dump** (signal #{signal_count}):\n```\n{ocr_chunk}\n```"},
                timeout=10,
            )
        except Exception as e:
            print(f"  [Discord] OCR post failed: {e}")


def post_status(msg: str):
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=8)
    except Exception:
        pass


def post_heartbeat(signal_count: int, scan_count: int):
    if not DISCORD_WEBHOOK:
        return
    msg = (f"👁 **Still watching** — scan #{scan_count}  •  "
           f"{signal_count}/{TARGET_SIGNALS} signals  •  "
           f"IST {_ist_hour():02d}h")
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=8)
    except Exception:
        pass


def post_progress_update(signal_count: int, scan_count: int):
    if not DISCORD_WEBHOOK:
        return
    pct = int(signal_count / TARGET_SIGNALS * 100)
    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
    msg = (f"📊 **Progress update** — {signal_count}/{TARGET_SIGNALS} signals\n"
           f"`{bar}` {pct}%  •  {scan_count} scans total")
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=8)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════
# ANNOTATION
# ═══════════════════════════════════════════════════════════════

def annotate_screenshot(img: "Image.Image", sig: dict,
                        label: str, side: str) -> "Image.Image":
    ann  = img.copy()
    draw = ImageDraw.Draw(ann)
    x, y  = sig["x"], sig["y"]
    color = (0, 255, 0) if sig["direction"] == "BUY" else (255, 50, 50)
    r = 30
    draw.ellipse([(x-r, y-r), (x+r, y+r)], outline=color, width=4)
    draw.text((x+r+5, y-10), f"{sig['direction']} [{label}]", fill=color)
    draw.rectangle(
        [max(0, x-LABEL_SEARCH_RADIUS), max(0, y-2*LABEL_SEARCH_RADIUS),
         min(img.width, x+LABEL_SEARCH_RADIUS), min(img.height, y+LABEL_SEARCH_RADIUS//2)],
        outline=(255, 200, 0), width=2
    )
    mtf_key = f"mtf_{side}"
    if mtf_key in REGIONS:
        draw.rectangle(REGIONS[mtf_key], outline=(255, 165, 0), width=3)
        draw.text((REGIONS[mtf_key][0]+3, REGIONS[mtf_key][1]+3), "MTF", fill=(255, 165, 0))
    if side == "left":
        draw.line([(BTC_RECENT_X_MIN, 0), (BTC_RECENT_X_MIN, img.height)],
                  fill=(255, 255, 0), width=2)
    else:
        draw.line([(ETH_RECENT_X_MIN, 0), (ETH_RECENT_X_MIN, img.height)],
                  fill=(255, 255, 0), width=2)
    return ann

# ═══════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════

def run_calibration(debug_colors: bool = False):
    print("[Calibrate] Taking screenshot...")
    img = take_screenshot()
    if img is None:
        print("[Calibrate] FAILED — check YOUTUBE_STREAM_URL env var")
        return
    w, h = img.size
    print(f"[Calibrate] Screen: {w}x{h}")
    draw = ImageDraw.Draw(img)
    for x in range(0, w, 100):
        draw.line([(x, 0), (x, h)], fill=(180, 0, 0), width=1)
        draw.text((x+2, 2), str(x), fill=(255, 60, 60))
    for y in range(0, h, 100):
        draw.line([(0, y), (w, y)], fill=(0, 0, 180), width=1)
        draw.text((2, y+2), str(y), fill=(60, 60, 255))
    colors = {
        "chart_full":  (0, 255, 0),   "chart_left":  (0, 200, 100),
        "chart_right": (0, 100, 200), "mtf_left":    (255, 165, 0),
        "mtf_right":   (255, 200, 0), "asset_left":  (255, 0, 255),
        "asset_right": (200, 0, 255), "price_left":  (0, 255, 255),
        "price_right": (0, 200, 255),
    }
    for name, box in REGIONS.items():
        c = colors.get(name, (255, 255, 255))
        draw.rectangle(box, outline=c, width=3)
        draw.text((box[0]+4, box[1]+4), name, fill=c)
    draw.line([(CHART_DIVIDER_X, 0), (CHART_DIVIDER_X, h)],
              fill=(255, 255, 0), width=2)
    out = Path("observe_calibrate.png")
    img.save(str(out))
    print(f"[Calibrate] Saved: {out}")
    print(f"[Calibrate] Pull with: adb pull /sdcard/... or open in Gallery")
    if debug_colors:
        arr = np.array(img.convert("RGB"))
        hsv = _rgb_to_hsv(arr)
        print("[Calibrate] HSV samples (saturated/bright pixels):")
        for x in range(100, min(w, 700), 100):
            for y in range(100, min(h, 575), 100):
                hv = hsv[y, x];  rv = arr[y, x]
                if hv[1] > 100 and hv[2] > 100:
                    print(f"  ({x:4},{y:4}) HSV=({hv[0]:3},{hv[1]:3},{hv[2]:3})  RGB={tuple(rv)}")

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _now_hms()  -> str: return datetime.now().strftime("%H:%M:%S")
def _now_iso()  -> str: return datetime.now(timezone.utc).isoformat()
def _ist_hour() -> int:
    utc = datetime.now(timezone.utc)
    return ((utc.hour * 60 + utc.minute + 330) % (24 * 60)) // 60

def _in_midnight_window() -> bool:
    h = _ist_hour()
    if MIDNIGHT_START_IST > MIDNIGHT_END_IST:
        return h >= MIDNIGHT_START_IST or h < MIDNIGHT_END_IST
    return MIDNIGHT_START_IST <= h < MIDNIGHT_END_IST

# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Infinity Algo Observation Tool (Termux)")
    parser.add_argument("--calibrate",    action="store_true")
    parser.add_argument("--debug-colors", action="store_true")
    parser.add_argument("--target", type=int, default=TARGET_SIGNALS)
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(debug_colors=args.debug_colors)
        return

    target = args.target

    print("=" * 64)
    print("  INFINITY ALGO OBSERVER  [Railway]")
    print(f"  Interval : {SCREENSHOT_INTERVAL}s")
    print(f"  Target   : {target} signals")
    print(f"  Saving to: {SIGNAL_DIR}")
    print(f"  Discord  : {'✓ ' + DISCORD_WEBHOOK[:50] + '...' if DISCORD_WEBHOOK else '✗ NOT SET — edit ~/.trader_env'}")
    print(f"  OCR      : {'✓' if OCR_AVAILABLE else '✗ (pip install pytesseract)'}")
    print("=" * 64)
    print("  Stream frames fetched from YouTube — no phone needed.")
    print("  Enable Stay Awake in Developer Options!")
    print("  Press Ctrl+C to stop.\n")

    post_status(
        f"🔍 **Observer started (Railway).**\n"
        f"Target: `{target}` signals  •  Scan interval: `{SCREENSHOT_INTERVAL}s`\n"
        f"Records: direction, Smart/Normal label, full MTF table, "
        f"market data, annotated screenshot.\n"
        f"Midnight restart (11PM–1AM IST): auto-handled."
    )

    signal_count     = 0
    scan_count       = 0
    last_signal_ts   = 0.0
    consecutive_dark = 0
    seen_positions   = set()

    while signal_count < target:
        try:
            img = take_screenshot()
            if img is None:
                print(f"[{_now_hms()}] Screenshot failed — retrying in 10s")
                time.sleep(10)
                continue

            scan_count += 1

            # Heartbeat
            if scan_count % HEARTBEAT_EVERY_N_SCANS == 0:
                post_heartbeat(signal_count, scan_count)

            # Midnight / stream offline
            if is_dark_screen(img):
                consecutive_dark += 1
                if consecutive_dark >= 3:
                    if _in_midnight_window():
                        if consecutive_dark == 3:
                            msg = (f"🌙 Stream offline (midnight IST {_ist_hour()}h). "
                                   f"Pausing {STREAM_OFFLINE_WAIT//60}min. "
                                   f"Progress: {signal_count}/{target}")
                            print(f"[{_now_hms()}] {msg}")
                            post_status(msg)
                        time.sleep(STREAM_OFFLINE_WAIT)
                    else:
                        print(f"[{_now_hms()}] Scan #{scan_count:4d} — dark screen")
                        time.sleep(SCREENSHOT_INTERVAL)
                    continue
            else:
                consecutive_dark = 0

            # Cooldown
            elapsed = time.time() - last_signal_ts
            if elapsed < SIGNAL_COOLDOWN_SEC:
                remaining = int(SIGNAL_COOLDOWN_SEC - elapsed)
                print(f"[{_now_hms()}] Scan #{scan_count:4d}  "
                      f"signals:{signal_count}/{target}  cooldown:{remaining:3d}s")
                time.sleep(SCREENSHOT_INTERVAL)
                continue

            # Signal detection
            signals = detect_signals(img, seen_positions=seen_positions)
            if not signals:
                print(f"[{_now_hms()}] Scan #{scan_count:4d}  "
                      f"signals:{signal_count}/{target}  — no new arrows in recent zone")
                time.sleep(SCREENSHOT_INTERVAL)
                continue

            sig  = max(signals, key=lambda s: s["pixels"])
            side = sig["side"]

            print(f"[{_now_hms()}] *** SIGNAL DETECTED: {sig['direction']} "
                  f"at ({sig['x']},{sig['y']}) side={side} "
                  f"area={sig['pixels']}px ***")

            label     = read_label_near(img, sig["x"], sig["y"])
            asset     = read_asset_label(img, side)
            price_ocr = read_price(img, side)
            mtf       = read_mtf_table_ocr(img, side)
            row_summary = ", ".join(
                f"{r['timeframe']}={r['trend']}/{r['volatility']}" for r in mtf["rows"]
            ) or "(no rows parsed)"
            print(f"[{_now_hms()}]   Label={label}  Asset={asset}  Price={price_ocr}")
            print(f"[{_now_hms()}]   MTF: {row_summary}")

            print(f"[{_now_hms()}]   Running full-screen OCR...")
            screen_text = read_full_screen_ocr(img)
            print(f"[{_now_hms()}]   OCR: {len(screen_text)} chars")

            print(f"[{_now_hms()}]   Fetching Binance data for {asset}...")
            market    = fetch_market_data(asset)
            api_price = market.get("5m", {}).get("close", "?")
            print(f"[{_now_hms()}]   API price: {api_price}")

            annotated = annotate_screenshot(img, sig, label, side)
            seen_positions.add((sig["x"] // 30, sig["y"] // 30))

            signal_count += 1
            record = {
                "signal_num":  signal_count,
                "timestamp":   _now_iso(),
                "direction":   sig["direction"],
                "label":       label,
                "asset":       asset,
                "side":        side,
                "arrow_x":     sig["x"],
                "arrow_y":     sig["y"],
                "price_ocr":   price_ocr,
                "mtf":         mtf,
                "market":      market,
                "screen_text": screen_text,
            }

            # Save JSON record
            rec_path = SIGNAL_DIR / f"signal_{signal_count:03d}.json"
            rec_path.write_text(json.dumps(record, indent=2))

            post_signal(record, annotated, signal_count)
            last_signal_ts = time.time()

            if signal_count % 5 == 0:
                post_progress_update(signal_count, scan_count)

            if signal_count >= target:
                final_msg = (f"✅ **Collection complete!** {signal_count}/{target} signals.\n"
                             f"Total scans: {scan_count}.\n"
                             f"Run `python analyze_signals.py` next.")
                print(f"\n[{_now_hms()}] {final_msg}")
                post_status(final_msg)
                break

            time.sleep(SCREENSHOT_INTERVAL)

        except KeyboardInterrupt:
            msg = (f"🛑 Stopped manually.\n"
                   f"Collected **{signal_count}/{target}** signals in {scan_count} scans.")
            print(f"\n[{_now_hms()}] {msg}")
            post_status(msg)
            break
        except Exception as exc:
            print(f"[{_now_hms()}] Error: {exc}")
            time.sleep(10)

    print(f"\nDone. {signal_count} signals saved in {SIGNAL_DIR}/")


if __name__ == "__main__":
    main()
