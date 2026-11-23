import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from streamlit_paste_button import paste_image_button as pbutton

IMG_SIZE = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
prep = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

# Put the "Suggested MIN_GREEN" number from calibrate.ipynb here
MIN_GREEN = 0.088


@st.cache_resource
def load_model():
    ckpt = torch.load("models/mobilenetv2.pth", map_location="cpu")
    classes = ckpt["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes


def pretty(name):
    plant, disease = name.split("___")
    return plant.replace("_", " "), disease.replace("_", " ")


def green_ratio(img):
    """Share of pixels that look like green plant tissue (0 to 1)."""
    hsv = np.asarray(img.resize((128, 128)).convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= 28) & (h <= 125) & (s >= 40) & (v >= 40)).mean()


st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detector")
st.write("Upload or paste a clear photo of a single leaf and the model will predict "
         "the plant and its disease (38 classes, 14 plants).")

model, classes = load_model()

with st.expander("Supported plants"):
    st.write(", ".join(sorted({pretty(c)[0] for c in classes})))

# 1. Choose how to add the image
mode = st.radio("How do you want to add a leaf image?",
                ["Upload a file", "Paste from clipboard"], horizontal=True)

img = None
if mode == "Upload a file":
    file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
else:
    st.write("Copy a leaf image first (take a screenshot with Win + Shift + S, or "
             "right-click an image and choose **Copy image**), then click the button below.")
    pasted = pbutton("📋 Paste image from clipboard", errors="raise")
    if pasted.image_data is not None:
        img = pasted.image_data.convert("RGB")

# 2. Check it looks like a leaf, then predict
if img is not None:
    st.image(img, caption="Your image", use_container_width=True)

    if green_ratio(img) < MIN_GREEN:
        st.warning("This doesn't look like a leaf (very little green plant tissue found). "
                   "Please use a clear photo of a single leaf.")
        st.stop()

    x = prep(img.resize((IMG_SIZE, IMG_SIZE))).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 3)

    plant, disease = pretty(classes[top.indices[0].item()])
    conf = top.values[0].item()

    if disease.lower() == "healthy":
        st.success(f"**{plant}: healthy** ({conf:.1%} confident)")
    else:
        st.error(f"**{plant}: {disease}** ({conf:.1%} confident)")
    if conf < 0.70:
        st.warning("The model is not very sure. This may not be one of the supported plants, "
                   "or the photo may be unclear. Try a clear, close photo of a single leaf.")

    st.subheader("Top 3 predictions")
    for p, i in zip(top.values, top.indices):
        pl, dis = pretty(classes[i.item()])
        st.write(f"{pl}: {dis}")
        st.progress(float(p))

st.caption("Trained on the PlantVillage dataset (lab photos on plain backgrounds), "
           "so accuracy on real field photos may be lower. The app uses a simple "
           "leaf-color check and only knows the supported plants.")