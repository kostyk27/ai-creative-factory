"""
AI Creative Factory — Flask backend.
Workflow: POST /generate-ideas (product → 10 creative ideas) → POST /generate-prompts (idea → prompts) → POST /generate-images (prompt → images).
Uses OpenAI for ideas, prompts, scoring. Uses Replicate (flux-2-pro) for image generation.
API keys are server-side only.
"""

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI

os.environ["REPLICATE_API_TOKEN"] = os.environ.get("REPLICATE_API_TOKEN", "")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, static_folder="static")
ROOT = Path(__file__).resolve().parent

_generation_jobs = {}
_jobs_lock = threading.Lock()
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")
STATIC_DIR = ROOT / "static"
IMAGES_DIR = STATIC_DIR / "images"
BANNERS_DIR = STATIC_DIR / "banners"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "").strip()

OPENAI_CHAT_MODEL = "gpt-4o-mini"
REPLICATE_MODEL = "black-forest-labs/flux-2-pro"

STYLE_GUIDES = {
    "ugc": "authentic smartphone testimonial style, social media ad",
    "luxury": "high-end product photography, premium luxury advertising",
    "medical": "doctor authority advertisement, clinical environment",
    "influencer": "social media influencer holding product, lifestyle photography",
    "studio": "professional studio lighting, clean product photography",
}

@app.errorhandler(429)
def _ratelimit_handler(e):
    return jsonify({"error": "Вы исчерпали лимит запросов. Пожалуйста, подождите немного"}), 429


def _get_client():
    key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)


def _generate_creative_ideas(idea):
    """Generate 10 creative ad concepts from product idea. Returns (ideas[:10], None) or ([], error)."""
    client = _get_client()
    if not client:
        return [], "OPENAI_API_KEY is not set"

    prompt = (
        "You are an expert creative director for affiliate marketing ads.\n\n"
        f"Product: {idea}\n\n"
        "Generate 10 high-conversion advertising creative ideas for social media ads.\n"
        "Ideas should describe a visual ad concept.\n"
        "GENERATE CONCEPTS FOR STATIC IMAGES ONLY. STRICTLY NO VIDEO, NO ANIMATION, NO GIFS, NO REELS.\n\n"
        "Examples:\n"
        "- influencer holding the product and smiling\n"
        "- before and after transformation\n"
        "- doctor authority recommendation\n"
        "- luxury product showcase\n\n"
        "Return 10 ideas, one per line."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        text = (resp.choices[0].message.content or "").strip()
        ideas = [i.strip("- ").strip() for i in text.splitlines() if i.strip()]
        return ideas[:10], None
    except Exception as e:
        err = str(e).strip() or "Unknown OpenAI error"
        if "api_key" in err.lower() or "invalid" in err.lower() or "authentication" in err.lower():
            err = "Invalid or expired API key. Check OPENAI_API_KEY."
        return [], err


def _build_prompts(idea, style_prompt=""):
    """Use OpenAI Chat Completions to generate 8–12 high-conversion advertising prompts."""
    idea = (idea or "").strip()
    if not idea:
        return [], None

    client = _get_client()
    if not client:
        return [], "OPENAI_API_KEY is not set"

    system = (
        "You are an expert in performance marketing and affiliate ad creatives. "
        "Input is an approved creative concept that has ALREADY been selected. "
        "Your job is to write exact text-to-image prompts for the Flux AI image generator "
        "to bring THIS EXACT concept to life. Do not invent new concepts. "
        "Describe the visual layout, subject, and lighting perfectly based on the provided idea. "
        "Every prompt must stay strictly within the boundaries of the given concept."
    )
    style_line = f"Style: {style_prompt}\n\n" if style_prompt else ""
    user = (
        "You will receive an approved creative concept for an advertising image.\n\n"
        f"Approved concept: {idea}\n\n"
        f"{style_line}"
        "Your task:\n"
        "- Write 10 detailed text-to-image prompts for the Flux image generator.\n"
        "- Each prompt must be a DIRECT visualization of the approved concept above.\n"
        "- DO NOT invent new storylines, angles, or products.\n"
        "- Focus on: subject, product, environment/scene, camera angle, framing, and lighting.\n"
        "- Optimize for static social media ads (no video, no animation).\n\n"
        "Important:\n"
        "- Keep the main idea, character role, and setting exactly as in the approved concept.\n"
        "- Vary small visual details only (angles, background details, mood), but never change the core idea.\n\n"
        "Return exactly 10 prompts, one per line. No numbering, no bullets, no extra commentary."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=3000,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        err = str(e).strip() or "Unknown OpenAI error"
        if "api_key" in err.lower() or "invalid" in err.lower() or "authentication" in err.lower():
            err = "Invalid or expired API key. Check OPENAI_API_KEY."
        elif "rate" in err.lower():
            err = "Rate limit exceeded. Try again in a moment."
        return [], err

    # Parse: one prompt per line, strip numbers/bullets; also split by " - " or ". " if single block
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines and text:
        lines = [s.strip() for s in re.split(r"\s*[\.\-]\s+", text) if len(s.strip()) > 15]
    prompts = []
    for ln in lines:
        cleaned = re.sub(r"^[\d\.\)\-\*\•]+\s*", "", ln).strip()
        if cleaned and len(cleaned) > 10 and cleaned.lower() not in ("product idea", "include styles"):
            prompts.append(cleaned)

    if not prompts:
        return [], "Could not parse prompts from OpenAI response. Try again."
    # Exactly 10 prompts
    while len(prompts) < 10:
        prompts.append(prompts[-1])
    return prompts[:10], None


def _score_prompts(client, prompts):
    """
    Send prompts to OpenAI for evaluation. Return list of { "prompt": str, "score": float } sorted by score desc.
    On failure return original prompts with score None (frontend can hide badge).
    """
    if not prompts:
        return []
    user = (
        "Evaluate these advertising image prompts for affiliate marketing creatives.\n\n"
        "Score each prompt from 1 to 10 based on:\n"
        "• marketing appeal\n"
        "• emotional hook\n"
        "• product visibility\n"
        "• realism for social media ads\n"
        "• likelihood to convert\n\n"
        "Return JSON format only, no other text:\n"
        "[\n"
        '  { "prompt": "...", "score": 9.2 },\n'
        '  { "prompt": "...", "score": 8.7 }\n'
        "]\n\n"
        "Prompts to evaluate:\n"
    )
    for i, p in enumerate(prompts, 1):
        user += f"{i}. {p}\n"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": user}],
            max_tokens=2000,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Remove markdown code block if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        if not isinstance(data, list):
            return [{"prompt": p, "score": None} for p in prompts]
        # Match by index first (same order), then by prompt text
        score_by_index = {}
        score_by_prompt = {}
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                pt = (item.get("prompt") or item.get("text") or "").strip()
                sc = item.get("score")
                if isinstance(sc, (int, float)):
                    score_by_index[idx] = float(sc)
                    if pt:
                        score_by_prompt[pt] = float(sc)
        scored = []
        for i, p in enumerate(prompts):
            s = score_by_index.get(i)
            if s is None:
                s = score_by_prompt.get(p) or score_by_prompt.get(p[:80])
            if s is None and score_by_prompt:
                for k, v in score_by_prompt.items():
                    if k in p or p in k or k[:50] == p[:50]:
                        s = v
                        break
            scored.append({"prompt": p, "score": s})
        scored.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0)))
        return scored
    except Exception:
        return [{"prompt": p, "score": None} for p in prompts]


def _get_replicate_token():
    return REPLICATE_API_TOKEN or os.environ.get("REPLICATE_API_TOKEN", "").strip()


def _generate_one_image(prompt, seed=None):
    """Generate one image via raw Replicate HTTP API; return (image_bytes, extension)."""
    token = _get_replicate_token()
    if not token:
        raise ValueError("REPLICATE_API_TOKEN is not set")

    base = (prompt or "").strip()
    if not base:
        raise ValueError("prompt is required")
    enhanced_prompt = f"{base}, highly detailed, professional ad photography"

    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }

    if seed is None:
        seed = int(time.time() * 1000) % 2147483647

    payload = {
        "version": REPLICATE_MODEL,
        "input": {
            "prompt": enhanced_prompt,
            "aspect_ratio": "1:1",
            "seed": seed,
            "output_format": "webp",
            "output_quality": 90,
        },
    }

    # Create prediction with retry on 429
    prediction = None
    for attempt in range(4):
        resp = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code == 429 and attempt < 3:
            time.sleep(3)
            continue
        resp.raise_for_status()
        prediction = resp.json()
        break
    if not prediction:
        raise ValueError("Failed to create prediction on Replicate after retries")
    status_url = prediction.get("urls", {}).get("get")
    if not status_url:
        raise ValueError("Replicate response missing status URL")

    while True:
        # Poll prediction status with retry on 429
        result = None
        for attempt in range(4):
            status_resp = requests.get(status_url, headers=headers, timeout=60)
            if status_resp.status_code == 429 and attempt < 3:
                time.sleep(3)
                continue
            status_resp.raise_for_status()
            result = status_resp.json()
            break
        if result is None:
            raise ValueError("Failed to poll prediction status after retries")
        status = result.get("status")
        if status == "succeeded":
            output = result.get("output")
            if isinstance(output, list) and output:
                image_url = output[0]
            else:
                image_url = output
            if not image_url:
                raise ValueError("Replicate returned empty output")
            img_resp = requests.get(image_url, timeout=60)
            img_resp.raise_for_status()
            return img_resp.content, "webp"
        if status == "failed":
            raise ValueError(result.get("error") or "Replicate generation failed")
        time.sleep(2)


def _generate_images(prompt, count=4):
    """Generate images in parallel without blocking the request worker."""
    if count <= 0:
        return []

    def one(i):
        seed = (int(time.time() * 1000) + i * 100003) % 2147483647
        return i, _generate_one_image(prompt, seed=seed)

    from concurrent.futures import ThreadPoolExecutor

    results = [None] * count
    with ThreadPoolExecutor(max_workers=min(count, 4)) as ex:
        futures = [ex.submit(one, i) for i in range(count)]
        for f in futures:
            i, val = f.result()
            results[i] = val
    return results


@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/generate-ideas", methods=["POST", "OPTIONS"])
@limiter.limit("5 per minute", methods=["POST"])
def generate_ideas():
    if request.method == "OPTIONS":
        response = jsonify()
        response.headers["Allow"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 204
    data = request.get_json(silent=True) or {}
    idea = (data.get("idea") or "").strip()

    if not idea:
        return jsonify({"error": "idea is required"}), 400

    ideas, err = _generate_creative_ideas(idea)

    if err:
        return jsonify({"error": err}), 500

    return jsonify({"ideas": ideas})


@app.route("/generate-prompts", methods=["POST"])
@limiter.limit("5 per minute", methods=["POST"])
def generate_prompts():
    data = request.get_json(silent=True) or {}
    idea = (data.get("idea") or "").strip()
    style = (data.get("style") or "ugc").strip().lower()
    if not idea:
        return jsonify({"error": "idea is required"}), 400

    style_prompt = STYLE_GUIDES.get(style, "")
    prompts, err = _build_prompts(idea, style_prompt)
    if err:
        return jsonify({"error": err}), 500
    if not prompts:
        return jsonify({"error": "Failed to generate prompts"}), 500

    client = _get_client()
    scored = _score_prompts(client, prompts)
    return jsonify({"prompts": scored})


def _run_generation_job(job_id, prompt, count):
    """Background task: generate images and save; update job status."""
    with _jobs_lock:
        _generation_jobs[job_id]["status"] = "processing"
    try:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        date_folder = datetime.now().strftime("%Y-%m-%d")
        out_dir = IMAGES_DIR / date_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        base_ts = int(time.time())
        images = _generate_images(prompt, count=count)
        paths = []
        for i, (content, ext) in enumerate(images, start=1):
            name = f"img_{base_ts}_{i}.{ext}"
            (out_dir / name).write_bytes(content)
            paths.append(f"/images/{date_folder}/{name}")
        with _jobs_lock:
            _generation_jobs[job_id]["status"] = "completed"
            _generation_jobs[job_id]["images"] = paths
    except Exception as e:
        with _jobs_lock:
            _generation_jobs[job_id]["status"] = "failed"
            _generation_jobs[job_id]["error"] = str(e)


def _prune_jobs(max_age_seconds=3600, max_jobs=200):
    now = time.time()
    with _jobs_lock:
        # remove old jobs
        to_delete = []
        for jid, job in _generation_jobs.items():
            created = job.get("created_at") or now
            if now - created > max_age_seconds:
                to_delete.append(jid)
        for jid in to_delete:
            _generation_jobs.pop(jid, None)
        # cap size
        if len(_generation_jobs) > max_jobs:
            items = sorted(_generation_jobs.items(), key=lambda kv: (kv[1].get("created_at") or 0))
            for jid, _ in items[: max(0, len(_generation_jobs) - max_jobs)]:
                _generation_jobs.pop(jid, None)


@app.route("/generate-images", methods=["POST"])
@limiter.limit("2 per minute", methods=["POST"])
def generate_images():
    token = _get_replicate_token()
    if not token:
        return jsonify({"error": "REPLICATE_API_TOKEN is not set"}), 500
    os.environ["REPLICATE_API_TOKEN"] = token

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    try:
        count = min(4, max(1, int(data.get("count", 4))))
    except (TypeError, ValueError):
        count = 4
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    job_id = str(uuid.uuid4())
    _prune_jobs()
    with _jobs_lock:
        _generation_jobs[job_id] = {
            "status": "processing",
            "images": [],
            "error": None,
            "created_at": time.time(),
        }
    thread = threading.Thread(target=_run_generation_job, args=(job_id, prompt, count))
    thread.daemon = True
    thread.start()
    return jsonify({"job_id": job_id, "status": "processing"}), 202


@app.route("/generate-images/result/<job_id>", methods=["GET"])
@limiter.limit("60 per minute")
def generate_images_result(job_id):
    with _jobs_lock:
        job = _generation_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "images": job.get("images", []),
        "error": job.get("error")
    })


@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(IMAGES_DIR, filename)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/<path:filename>")
def catch_all(filename):
    return "", 404


if __name__ == "__main__":
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    if debug:
        print("AI Creative Factory: http://127.0.0.1:{}".format(port))
    app.run(host="0.0.0.0", port=port, debug=debug)
