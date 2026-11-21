import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

IMG_SIZE = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
prep = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


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


st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detector")
st.write("Upload a clear photo of a single leaf and the model will predict "
         "the plant and its disease (38 classes, 14 plants).")

model, classes = load_model()
file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Your image", use_container_width=True)

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
        st.warning("The model is not very sure. Try a clearer, closer photo of one leaf.")

    st.subheader("Top 3 predictions")
    for p, i in zip(top.values, top.indices):
        pl, dis = pretty(classes[i.item()])
        st.write(f"{pl}: {dis}")
        st.progress(float(p))

st.caption("Trained on the PlantVillage dataset (lab photos on plain backgrounds), "
           "so accuracy on real field photos may be lower.")