import io
import requests
import streamlit as st
from PIL import Image, ImageStat

st.set_page_config(page_title="AI 25", page_icon="🎬", layout="centered")

HF_URL = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-large"

COLOR_NAMES = {
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
    "red": (200, 30, 30), "orange": (240, 140, 30), "yellow": (240, 220, 50),
    "green": (40, 160, 60), "teal": (30, 150, 150), "blue": (40, 90, 200),
    "navy": (20, 30, 90), "purple": (130, 50, 170), "pink": (240, 130, 180),
    "brown": (110, 70, 40), "gold": (212, 175, 55), "beige": (225, 205, 170),
}


def color_name(rgb):
    return min(
        COLOR_NAMES,
        key=lambda n: sum((a - b) ** 2 for a, b in zip(rgb, COLOR_NAMES[n])),
    )


def analyze_image(img):
    rgb = img.convert("RGB")
    w, h = rgb.size
    small = rgb.resize((120, 120))
    stat = ImageStat.Stat(small.convert("L"))
    brightness = stat.mean[0]
    contrast = stat.stddev[0]
    hsv = small.convert("HSV")
    saturation = ImageStat.Stat(hsv).mean[1]

    q = small.quantize(colors=5)
    pal = q.getpalette()[:15]
    counts = sorted(q.getcolors(), reverse=True)
    colors = []
    for _, idx in counts:
        name = color_name(tuple(pal[idx * 3: idx * 3 + 3]))
        if name not in colors:
            colors.append(name)

    ratio = w / h
    if ratio > 1.2:
        orient, aspect = "landscape", "16:9"
    elif ratio < 0.85:
        orient, aspect = "portrait", "9:16"
    else:
        orient, aspect = "square", "1:1"

    if brightness < 80:
        light = "low-key, moody, dramatic shadows"
    elif brightness > 170:
        light = "bright, high-key, airy and luminous"
    else:
        light = "balanced natural lighting with soft shadows"

    if saturation > 130:
        tone = "vivid, rich and bold colors"
    elif saturation < 60:
        tone = "muted, subtle, cinematic desaturated tones"
    else:
        tone = "natural, harmonious colors"

    contrast_txt = "high contrast" if contrast > 65 else "soft, gentle contrast"
    return {
        "size": (w, h), "orient": orient, "aspect": aspect, "light": light,
        "tone": tone, "contrast": contrast_txt, "colors": colors[:4],
        "brightness": brightness, "saturation": saturation,
    }


def caption_image(img_bytes, token):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(HF_URL, headers=headers, data=img_bytes, timeout=90)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        return data[0].get("generated_text", "").strip()
    return ""


def build_prompt(caption, info, style, camera, mood, duration, extra):
    colors = ", ".join(info["colors"])
    parts = [
        f"A {style.lower()} video of {caption}.",
        f"Scene: bring the uploaded image to life, keeping the exact subject, "
        f"composition and identity of the original photo, in {info['orient']} framing.",
        f"Camera: {camera.lower()}, smooth stabilized motion, professional "
        f"cinematography, shallow depth of field, natural lens focus shifts.",
        f"Lighting: {info['light']}, {info['contrast']}, volumetric light rays, "
        f"realistic reflections and global illumination.",
        f"Color palette: dominant {colors}; {info['tone']}, professional color grading.",
        f"Mood: {mood.lower()}.",
        "Motion: subtle natural movement in the subject, flowing air, drifting "
        "particles, gentle parallax between foreground and background, "
        "realistic physics, fluid temporal consistency, no flicker.",
        f"Duration: {duration} seconds, aspect ratio {info['aspect']}, 24 fps, "
        "4K resolution, ultra detailed, sharp focus, photorealistic textures, "
        "cinematic film grain, masterpiece, award-winning quality.",
    ]
    if extra.strip():
        parts.append(f"Extra direction: {extra.strip()}.")
    positive = " ".join(parts)
    negative = (
        "blurry, low quality, low resolution, distorted face, extra limbs, "
        "deformed hands, watermark, text, logo, jitter, flicker, morphing, "
        "overexposed, underexposed, artifacts, cropped, duplicate subject"
    )
    return positive, negative


st.markdown(
    "<h1 style='text-align:center'>🎬 AI 25</h1>"
    "<p style='text-align:center'>Upload a picture. Get a powerful video prompt.</p>",
    unsafe_allow_html=True,
)

token = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""
if not token:
    token = st.text_input("Hugging Face token (for picture analysis)", type="password")

up = st.file_uploader("Upload picture", type=["png", "jpg", "jpeg", "webp"])

c1, c2 = st.columns(2)
style = c1.selectbox(
    "Style",
    ["Cinematic", "Photorealistic", "Anime", "3D animation", "Documentary",
     "Fantasy epic", "Commercial advert", "Vintage film"],
)
camera = c2.selectbox(
    "Camera motion",
    ["Slow push-in", "Smooth orbit around subject", "Drone aerial reveal",
     "Handheld tracking shot", "Slow pan left to right", "Dolly zoom"],
)
mood = c1.selectbox(
    "Mood",
    ["Epic and inspiring", "Calm and peaceful", "Dark and mysterious",
     "Joyful and energetic", "Romantic and warm", "Luxury and elegant"],
)
duration = c2.slider("Duration (seconds)", 3, 30, 10)
extra = st.text_area("Extra ideas (optional)", placeholder="e.g. add rain, golden hour sunset")

if up and st.button("🚀 Generate Massive Prompt", use_container_width=True):
    raw = up.getvalue()
    img = Image.open(io.BytesIO(raw))
    st.image(img, use_container_width=True)

    with st.spinner("Analyzing picture..."):
        info = analyze_image(img)
        caption = ""
        if token:
            try:
                caption = caption_image(raw, token)
            except Exception as e:
                st.warning(f"Caption failed ({e}). Using visual analysis only.")
        if not caption:
            caption = "the scene shown in the uploaded image"

    st.subheader("🔍 Analysis")
    st.write(f"**What I see:** {caption}")
    st.write(
        f"**Size:** {info['size'][0]}x{info['size'][1]} ({info['orient']}, "
        f"{info['aspect']})  \n**Colors:** {', '.join(info['colors'])}  \n"
        f"**Lighting:** {info['light']}"
    )

    pos, neg = build_prompt(caption, info, style, camera, mood, duration, extra)

    st.subheader("✅ Video Prompt")
    st.text_area("Copy this", pos, height=300)
    st.subheader("🚫 Negative Prompt")
    st.text_area("Copy this too", neg, height=100)

    st.download_button(
        "⬇️ Download prompt",
        f"PROMPT:\n{pos}\n\nNEGATIVE PROMPT:\n{neg}",
        file_name="ai25_prompt.txt",
        use_container_width=True,
    )
elif not up:
    st.info("Upload a picture to begin.")
