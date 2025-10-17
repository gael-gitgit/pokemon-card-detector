import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import functions
import time

st.set_page_config(page_title="Trouve la Poké-pétite", layout="wide")

# Masquer toolbar et footer
hide_streamlit_style = """
<style>
div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"], #MainMenu, header, footer {
    visibility: hidden;
    height: 0%;
    position: fixed;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🎴 Pokemon Card Detector")

# Charger modèles
@st.cache_resource(show_spinner=False)
def load_models():
    collection, meta = functions.load_faiss_index()
    embedding_model, preprocess, device = functions.load_embbedings_model()
    yolo_model = 'models/my-modelv6.pt'
    model = YOLO(yolo_model)
    return model, collection, meta, embedding_model, preprocess, device

model, collection, meta, embedding_model, preprocess, device = load_models()

# Inputs
img_file_buffer = st.camera_input("📸 Prends une photo ou sélectionne-en une", key="camera_input")
#uploaded_file = st.file_uploader("Ou charge une image existante", type=["jpg", "jpeg", "png"])

st.session_state.process_step = "start"

image = None
if img_file_buffer:
    image = Image.open(img_file_buffer).convert("RGB")
#elif uploaded_file:
    #image = Image.open(uploaded_file).convert("RGB")

# Session state pour collection
if "detected_cards" not in st.session_state:
    st.session_state.detected_cards = []

# Container pour affichage progressif
collection_container = st.container()

# --- Traitement de l'image ---
if image is not None:
    st.session_state.process_step = "inprogress"  
    img = np.array(image, dtype=np.uint8)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) => fonctionne moins bien si je le laisse, vérifications à faire sur les types
    results = model.predict(source=img, conf=0.8, device='cpu')

    #Check si on a des 
    if 1 > 2 :
        st.warning("Aucun objet détecté.")
    else:
        for r in results:
            masks = r.masks
            boxes = r.boxes

            if masks is not None:
                for i, m in enumerate(masks.data):
                    mask = (m.cpu().numpy() > 0.5).astype(np.uint8) * 255
                    if mask.shape != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

                    segmented = cv2.bitwise_and(img, img, mask=mask)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    crop_pre = segmented[y1:y2, x1:x2]

                    crop = functions.preprocess_img(crop_pre)
                    crop = functions.improve_img(crop)

                    search_results = functions.search_card_correspondance(
                        collection, meta, crop, embedding_model, preprocess, device
                    )

                    #img_reference = functions.get_image_from_url(search_results['img'])

                    # --- Ajouter carte à session_state ---
                    card_data = {
                        "crop": crop,
                        "reference": search_results['img'],
                        "name": search_results['name'],
                        "price": search_results['price_eur'],
                        "tcgplayer": search_results['tcgplayer_link'],
                        "cardmarket": search_results['cardmarket_link'],
                        "history": search_results['price_evolution_url']
                    }
                    st.session_state.detected_cards.append(card_data)

                    # --- Affichage immédiat dans le container ---
                    with collection_container:
                        st.markdown(f"### {card_data['name']} — 💰 <span style='color:red;'>{card_data['price']} €</span>", unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        col1.image(card_data['crop'], caption="Crop", width="stretch")
                        col2.image(card_data['reference'], caption="Correspondance", width="stretch")
                        st.markdown(f"""
                        **Liens :** [TCGPlayer]({card_data['tcgplayer']}) | [CardMarket]({card_data['cardmarket']}) | [Historique prix]({card_data['history']})
                        """)
    st.session_state.process_step = "end"                


# --- Valeur totale ---
if st.session_state.detected_cards:
    total_value = sum([c['price'] for c in st.session_state.detected_cards])
    st.markdown(f"## 💰 Valeur totale de la collection : {total_value:.2f} €")

# --- Valeur totale ---
if st.session_state.process_step  == "start":
    st.info("Analyse à lancer")
elif st.session_state.process_step  == "inprogress":
    st.info("Analyse en cours")
elif st.session_state.process_step  == "end" and len(st.session_state.detected_cards) ==0:
    st.warning("Analyse terminée - Aucun objet détecté")
elif st.session_state.process_step  == "end":
    st.sucess("Analyse terminée")
