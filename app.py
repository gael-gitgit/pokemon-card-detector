import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import functions
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trouve la Poké-pétite", layout="wide")
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🎴 Pokemon Card Detector")

# --- Charger les modèles une seule fois ---
@st.cache_resource(show_spinner=False)
def load_models():
    collection, meta = functions.load_faiss_index()
    embedding_model, preprocess, device = functions.load_embbedings_model()
    yolo_model = 'models/my-modelv4.pt'
    model = YOLO(yolo_model)
    return model, collection, meta, embedding_model, preprocess, device

model, collection, meta, embedding_model, preprocess, device = load_models()

# --- Input : Upload ou caméra ---
img_file_buffer = st.camera_input("📸 Prends une photo ou sélectionne-en une", key="camera_input")
uploaded_file = st.file_uploader("Ou charge une image existante", type=["jpg", "jpeg", "png"])

image = None
if img_file_buffer:
    image = Image.open(img_file_buffer).convert("RGB")
elif uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

if image is not None:
    st.info("🔍 Analyse en cours…")
    
    # Convert PIL image to numpy
    img = np.array(image)

    # YOLO prediction
    results = model.predict(source=img, conf=0.8, device='cpu')

    if len(results) == 0:
        st.warning("Aucun objet détecté.")
    else:
        for r in results:
            masks = r.masks
            boxes = r.boxes

            if masks is not None:
                for i, m in enumerate(masks.data):
                    # Masque binaire
                    mask = (m.cpu().numpy() > 0.5).astype(np.uint8) * 255
                    if mask.shape != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    
                    segmented = cv2.bitwise_and(img, img, mask=mask)
                    box = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    crop_pre_improved = segmented[y1:y2, x1:x2]

                    crop = functions.preprocess_img(crop_pre_improved)
                    crop = functions.improve_img(crop)

                    # Recherche FAISS
                    search_results = functions.search_card_correspondance(
                        collection, meta, crop, embedding_model, preprocess, device
                    )

                    # Récupère l'image de référence
                    img_reference = functions.get_image_from_url(search_results['img'])

                    # --- Affichage ---
                    st.subheader(f"Carte détectée : {search_results['name']}")
                    col1, col2, col3 = st.columns(3)
                    col1.image(crop_pre_improved, caption="Crop original", use_container_width =True)
                    col2.image(crop, caption="Crop amélioré", use_container_width =True)
                    col3.image(img_reference, caption="Carte correspondante", use_container_width =True)

                    # Liens et prix
                    st.markdown(f"""
                    **Prix EUR** : {search_results['price_eur']} €  
                    [TCGPlayer]({search_results['tcgplayer_link']}) | 
                    [CardMarket]({search_results['cardmarket_link']}) | 
                    [Historique prix]({search_results['price_evolution_url']})
                    """)

    st.success("✅ Analyse terminée !")
