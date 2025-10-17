import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pickle
import time
import requests
import functions
from io import BytesIO

# -----------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------
st.set_page_config(
    page_title="Card Detector App",
    page_icon="🃏",
    layout="wide"
)

st.title("🃏 Carte Detector & Matching")
st.markdown("Télécharge une image pour détecter et identifier les cartes grâce à YOLO et la recherche de similarité FAISS.")

# -----------------------------
# CHARGEMENT DES MODÈLES
# -----------------------------
@st.cache_resource
def load_all_models():
    collection, meta = functions.load_faiss_index()
    emb_model, preprocess, device = functions.load_embbedings_model()
    yolo_model = YOLO("models/my-modelv6.pt")
    return collection, meta, emb_model, preprocess, device, yolo_model

with st.spinner("Chargement des modèles..."):
    collection, meta, emb_model, preprocess, device, yolo_model = load_all_models()


# -----------------------------
# UPLOAD D'IMAGE
# -----------------------------
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = None #st.file_uploader("Téléverser une image", type=['png', 'jpg', 'jpeg'])
    camera_img = st.camera_input("Ou prendre une photo avec la webcam")

    # déterminer la source d'image : priorité caméra si présente
    image_bytes = None
if camera_img is not None:
    image_bytes = camera_img.getvalue()
elif uploaded_file is not None:
    image_bytes = uploaded_file.read()


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    arr = np.array(pil_image.convert('RGB'))
    return arr




if image_bytes is not None:
# on lance immédiatement l'analyse
    try:
        pil_img = Image.open(BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        st.error(f"Impossible d'ouvrir l'image : {e}")
        st.stop()


    image = pil_to_cv2(pil_img)
    st.image(image, caption="Image d'entrée", width='stretch')

# -----------------------------
# DÉTECTION YOLO
# -----------------------------
if uploaded_file or camera_img:
    with st.spinner("Détection des cartes en cours..."):
        # Convertir PIL → array pour YOLO
        #image = np.array(image, dtype=np.uint8) 
        results = yolo_model.predict(source=image, conf=0.8, device="cpu")
        
    # -----------------------------
    # TRAITEMENT DES RÉSULTATS
    # -----------------------------
    if not results:
        st.warning("Aucune carte détectée.")
    else:
        st.success(f"{len(results[0].boxes)} carte(s) détectée(s).")
        col1, col2,col3 = st.columns(3)
        for r in results:
            masks = r.masks
            boxes = r.boxes

            if masks is None:
                st.warning("Aucun masque détecté.")
                continue

            # -----------------------------
            # AFFICHAGE RÉSULTATS
            # -----------------------------
            

            # Affichage des résultats
            for i, m in enumerate(masks.data):
                # Conversion du masque en numpy (0 ou 1)
                mask = m.cpu().numpy()
                # Si masque est float (0.0–1.0) → binarisation
                mask = (mask > 0.5).astype(np.uint8) * 255
                # Vérifier la taille : doit être HxW comme img
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                # Appliquer le masque sur l'image
                segmented = cv2.bitwise_and(image, image, mask=mask)
                # Extraire la bbox correspondante
                box = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                crop = segmented[y1:y2, x1:x2]

                #redécoupage de l'image
                crop = functions.preprocess_img(crop)
                crop = functions.improve_img(crop)

                # Recherche FAISS
                with st.spinner("Recherche de correspondances..."):
                    search_result = functions.search_card_correspondance(
                        collection, meta, crop, emb_model, preprocess, device
                    )

                with col1:
                    st.image(crop, caption=f"Carte détectée #{i+1}",  width='stretch')
                with col2:
                    st.image(search_result['img'], caption=f"{search_result['name']} -  {search_result['distance']} -  {search_result['price_eur']}",  width='stretch')
        
        st.success("Analyse terminée ✅")

