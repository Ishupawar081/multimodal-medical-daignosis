import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image

# ---------------- LABELS ----------------
SYSTEM_LABELS = [
    'cardiovascular','dermatological','endocrine','ent','gastrointestinal',
    'genetic','genitourinary','hematological','hepatobiliary','immunological',
    'lymphatic','multisystemic','musculoskeletal','neurological','ophthalmic',
    'renal','respiratory'
]

TYPE_LABELS = [
    'autoimmune','degenerative','infection','metabolic','trauma','tumor','vascular'
]

# ---------------- MODEL ----------------
class MultiModalModel(nn.Module):
    def __init__(self, num_system, num_type):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size

        resnet = models.resnet18(weights="DEFAULT")
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(text_dim + 512, 256)

        self.system_head = nn.Linear(256, num_system)
        self.type_head = nn.Linear(256, num_type)

    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_out.last_hidden_state[:, 0]

        img_feat = self.image_encoder(image)
        img_feat = img_feat.view(img_feat.size(0), -1)

        x = torch.cat([text_emb, img_feat], dim=1)
        x = self.fc(x)

        return self.system_head(x), self.type_head(x)

# ---------------- LOAD ----------------
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = MultiModalModel(len(SYSTEM_LABELS), len(TYPE_LABELS))
model.load_state_dict(torch.load("multimodal_model.pt", map_location=device))
model.to(device)
model.eval()

# ---------------- IMAGE ----------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- POST-CORRECTION ----------------
def post_correct(system, dtype, text):
    text = text.lower()

    if any(k in text for k in ["hypertrophy", "stones", "enlargement"]):
        dtype = "metabolic"

    if "masseter" in text or "mandible" in text:
        system = "ent"

    if "sepsis" in text:
        system = "multisystemic"

    if "lupus" in text:
        system = "immunological"

    if "retinal detachment" in text:
        dtype = "degenerative"

    if "kidney stones" in text:
        dtype = "metabolic"

    if "pulmonary embolism" in text:
        system = "cardiovascular"

    return system, dtype

# ---------------- PREDICT ----------------
def predict(text, image):
    enc = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    if image:
        img = image_transform(image).unsqueeze(0).to(device)
    else:
        img = torch.zeros(1, 3, 224, 224).to(device)

    with torch.no_grad():
        sys_logits, type_logits = model(input_ids, attention_mask, img)

    sys_pred = torch.argmax(sys_logits, dim=1).item()
    type_pred = torch.argmax(type_logits, dim=1).item()

    system = SYSTEM_LABELS[sys_pred]
    dtype = TYPE_LABELS[type_pred]

    system, dtype = post_correct(system, dtype, text)

    return system, dtype

# ---------------- UI ----------------
st.title("🩺 Medical Multimodal Classifier")

text = st.text_area("Enter Case Description")

image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

image = None
if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("Predict"):
    if text:
        system, dtype = predict(text, image)
        st.success(f"System: {system}")
        st.success(f"Type: {dtype}")
    else:
        st.warning("Please enter text")