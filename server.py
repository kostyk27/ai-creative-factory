"""
AI Creative Factory — Flask backend.
Workflow: POST /generate-ideas (product → 10 creative ideas) → POST /generate-prompts (idea → prompts) → POST /generate-images (prompt → images).
Uses OpenAI for ideas, prompts, scoring. Uses Replicate (flux-2-pro) for image generation.
API keys are server-side only.
"""

import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

os.environ["REPLICATE_API_TOKEN"] = os.environ.get("REPLICATE_API_TOKEN", "")

# Загружаем .env только локально (на хостинге ключи задаются через переменные окружения)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, static_folder="static")
ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
IMAGES_DIR = STATIC_DIR / "images"
BANNERS_DIR = STATIC_DIR / "banners"
PROMPTS_DIR = ROOT / "prompts"

# Ключи только из переменных окружения (никогда не хранить в коде)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "").strip()

OPENAI_CHAT_MODEL = "gpt-4o-mini"
REPLICATE_MODEL = "black-forest-labs/flux-2-pro"

IMAGE_QUALITY_SUFFIX = (
    "ultra realistic, cinematic lighting, professional advertising photography, "
    "high-end product photography, shallow depth of field, dramatic lighting, "
    "studio lighting, commercial product shoot, 8k realism, "
    "professional color grading, hyper detailed, sharp focus"
)

STYLE_GUIDES = {
    "ugc": "authentic smartphone testimonial style, social media ad",
    "luxury": "high-end product photography, premium luxury advertising",
    "medical": "doctor authority advertisement, clinical environment",
    "influencer": "social media influencer holding product, lifestyle photography",
    "studio": "professional studio lighting, clean product photography",
}


def _get_client():
    """Return OpenAI client. Key must be set."""
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
        "Ideas should describe a visual ad concept.\n\n"
        "Examples:\n"
        "- influencer testimonial video\n"
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
        "Your task: generate image prompts that STRICTLY match the user's product/idea. "
        "Every single prompt must clearly feature or describe the user's exact product/idea — no generic or off-topic prompts. "
        "Reply with exactly 10 image prompts, one per line. No numbering, no bullets, no other text."
    )
    style_line = f"Style: {style_prompt}\n\n" if style_prompt else ""
    user = (
        "Generate 10 detailed advertising image prompts for performance marketing creatives.\n\n"
        f"Product/idea: {idea}\n\n"
        f"{style_line}"
        "Each prompt must include:\n"
        "- subject and product\n"
        "- environment or scene\n"
        "- camera angle\n"
        "- lighting\n"
        "- advertising style\n"
        "- emotional tone\n\n"
        "Prompts should be cinematic, realistic, and optimized for social media ads.\n\n"
        "Example structure:\n"
        "fitness influencer holding supplement bottle in gym, smartphone selfie camera angle, "
        "authentic testimonial vibe, natural lighting, shallow depth of field, "
        "high-conversion social media advertisement, ultra realistic photography\n\n"
        "Return 10 prompts. Each prompt should be detailed and around 40–80 words."
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


def _save_prompts(idea, prompt_items):
    """Append idea and prompts to prompts.txt. prompt_items: list of str or list of { 'prompt': str }."""
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROMPTS_DIR / "prompts.txt"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n--- idea: {idea} ---\n")
        for p in prompt_items:
            text = p.get("prompt", p) if isinstance(p, dict) else p
            f.write(text + "\n")


def _get_replicate_token():
    return REPLICATE_API_TOKEN or os.environ.get("REPLICATE_API_TOKEN", "").strip()


def _generate_one_image(prompt):
    import requests
    token = os.environ.get("REPLICATE_API_TOKEN")

    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }

    enhanced_prompt = f"""
{prompt},

professional advertising photography,
commercial product shoot,
high detail,
cinematic lighting,
hyper realistic,
sharp focus,
high-end marketing creative,
social media advertisement,
photorealistic
"""

    data = {
        "version": "black-forest-labs/flux-1.1-pro",
        "input": {
            "prompt": enhanced_prompt,
            "aspect_ratio": "1:1",
            "num_outputs": 4
            "guidance": 3
        }
    }

    resp = requests.post(
        "https://api.replicate.com/v1/predictions",
        json=data,
        headers=headers,
        timeout=60
    )

    resp.raise_for_status()
    prediction = resp.json()

    get_url = prediction["urls"]["get"]

    while True:
        result = requests.get(get_url, headers=headers).json()
        if result["status"] == "succeeded":
            output = result["output"]
            image_url = output[0] if isinstance(output, list) else output
            img = requests.get(image_url).content
            return img, "png"
        elif result["status"] == "failed":
            raise ValueError("Replicate generation failed")


@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/generate-ideas", methods=["POST", "OPTIONS"])
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
    _save_prompts(idea, scored)
    return jsonify({"prompts": scored})


@app.route("/generate-images", methods=["POST"])
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

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    date_folder = datetime.now().strftime("%Y-%m-%d")
    out_dir = IMAGES_DIR / date_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    base_ts = int(time.time())

    for i in range(count):
        try:
            content, ext = _generate_one_image(prompt)
        except Exception as e:
            return jsonify({"error": str(e), "images": paths}), 500
        name = f"img_{base_ts}_{i + 1}.{ext}"
        (out_dir / name).write_bytes(content)
        paths.append(f"/images/{date_folder}/{name}")

    return jsonify({"images": paths})


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
    BANNERS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    if debug:
        print("AI Creative Factory: http://127.0.0.1:{}".format(port))
    app.run(host="0.0.0.0", port=port, debug=debug)
