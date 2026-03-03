import io
import numpy as np
from PIL import Image, ImageFilter
import streamlit as st

st.set_page_config(page_title="Random Wolken Generator", layout="centered")
st.title("☁️ Random Wolken Generator")
st.write("Genereer realistisch ogende wolken met blauwe lucht en witte wolken. Pas de parameters aan naar smaak.")

# -------------------------------
# Helper: fractal noise (octaves)
# -------------------------------
def fractal_noise(width, height, octaves=5, persistence=0.5, seed=None):
    """
    Maakt 'zachte' ruis door meerdere lagen (octaves) van geblurde ruis te stapelen.
    Geen externe libs nodig; gebruikt upscaling en blur om low-freq patronen te verkrijgen.
    Geeft een float array [0..1] met wolk-achtige patronen.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0

    # Start met een zeer lage resolutie en schaal telkens omhoog
    # voor smooth, large-scale structuren
    for o in range(octaves):
        # resolutie voor deze octave (lager bij vroege octaves)
        scale = 2 ** (octaves - o - 1)
        h_small = max(1, height // scale)
        w_small = max(1, width // scale)

        # random field en upscale met bicubic (zachte overgangen)
        small = rng.random((h_small, w_small)).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8), mode="L")
        img = img.resize((width, height), resample=Image.BICUBIC)

        # zachte blur per octave om blokvorming verder te reduceren
        # hogere octaves → minder blur
        blur_radius = max(0, int(10 / (o + 1)))
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        layer = np.asarray(img).astype(np.float32) / 255.0

        base += layer * amplitude
        total_amplitude += amplitude
        amplitude *= persistence  # lagere bijdrage van hogere frequenties

    if total_amplitude > 0:
        base /= total_amplitude

    # Normaliseer voorzichtig naar [0,1]
    base -= base.min()
    maxv = base.max()
    if maxv > 1e-8:
        base /= maxv
    return base


def clouds_rgb(noise, threshold=0.5, softness=0.25,
               sky_top=(70, 130, 180), sky_bottom=(135, 206, 235)):

    # Zorg dat noise 2D is
    if noise.ndim != 2:
        raise ValueError(f"noise moet shape (h,w) zijn, maar is {noise.shape}")

    h, w = noise.shape

    # --- SKY GRADIENT ---
    y = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
    sky_top = np.array(sky_top, dtype=np.float32).reshape(1, 1, 3)
    sky_bottom = np.array(sky_bottom, dtype=np.float32).reshape(1, 1, 3)

    # gradient over hoogte (h,1,3)
    sky = sky_top * (1 - y) + sky_bottom * y

    # broadcast automatisch naar (h,w,3)
    sky = np.repeat(sky, w, axis=1)

    # --- CLOUD MASK ---
    eps = 1e-6
    s = max(softness, eps)
    t0 = threshold - s
    t1 = threshold + s

    m = (noise - t0) / (t1 - t0)
    m = np.clip(m, 0.0, 1.0) ** 1.5  # puffiness

    # reshape m naar (h,w,1)
    m = m[..., None]

    white = np.array([255, 255, 255], dtype=np.float32).reshape(1, 1, 3)

    # --- MIX SKY + CLOUDS ---
    rgb = sky * (1 - m) + white * m
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb, mode="RGB")

# ----------------------------------
# UI Controls
# ----------------------------------
col1, col2 = st.columns(2)
with col1:
    width = st.slider("Breedte (px)", 400, 2048, 960, step=16)
    octaves = st.slider("Octaves (detailniveau)", 1, 8, 5)
    blur_after = st.slider("Extra blur na kleur (px)", 0, 20, 2)
with col2:
    height = st.slider("Hoogte (px)", 300, 2048, 640, step=16)
    persistence = st.slider("Persistentie (impact hogere frequenties)", 0.2, 0.95, 0.55)
    seed = st.number_input("Seed (voor reproduceerbaarheid)", min_value=0, max_value=10_000_000, value=0)

st.markdown("### Wolk‑/lucht verhouding")
threshold = st.slider("Drempel voor wolkvorming", 0.2, 0.8, 0.52)
softness = st.slider("Zachtheid overgang (softness)", 0.05, 0.5, 0.22)

st.markdown("### Luchtkleur (gradient)")
c1, c2 = st.columns(2)
with c1:
    sky_top = st.color_picker("Boven", "#4682B4")   # SteelBlue
with c2:
    sky_bottom = st.color_picker("Onder", "#87CEEB")  # SkyBlue

def hex_to_rgb(hx: str):
    hx = hx.lstrip("#")
    return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))

# ----------------------------------
# Generate
# ----------------------------------
if st.button("Genereer wolkenfoto"):
    # 1) fractal noise
    noise = fractal_noise(width, height, octaves=octaves, persistence=persistence, seed=seed or None)

    # 2) kleurmapping naar lucht + wolken
    img = clouds_rgb(
        noise,
        threshold=threshold,
        softness=softness,
        sky_top=hex_to_rgb(sky_top),
        sky_bottom=hex_to_rgb(sky_bottom),
    )

    # 3) optionele nabewerking (zachte blur)
    if blur_after > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_after))

    st.image(img, caption="Random wolken met blauwe lucht en witte wolken", use_container_width=True)

    # Download-knop
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.download_button("Download als PNG", data=buf.getvalue(), file_name="wolken.png", mime="image/png")

else:
    st.info("Stel je parameters in en klik op **Genereer wolkenfoto**.")
