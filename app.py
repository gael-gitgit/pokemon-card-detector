import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import functions
import time
import requests

import time

from io import BytesIO

import base64
import cv2

import cv2
import numpy as np

def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def draw_boxes_on_image(image_np, cards):
    img_annotated = image_np.copy()
    for card in cards:
        x1, y1, x2, y2 = card['box']
        name = card['name']
        price = card['price']

        # Rectangle rouge
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Texte (nom + prix)
        text = f"{name} - {price:.2f}EUR"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        # Fond noir
        cv2.rectangle(img_annotated, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 0, 0), -1)
        cv2.putText(img_annotated, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_annotated


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



st.markdown("## Pokemon scanner")

# Charger modèles
@st.cache_resource(show_spinner=False)
def load_models():
    collection, meta = functions.load_faiss_index()
    embedding_model, preprocess, device = functions.load_custom_model("models/dino_small_lora_merged")
    yolo_model = 'models/my-model12.pt'
    model = YOLO(yolo_model)
    return model, collection, meta, embedding_model, preprocess, device

model, collection, meta, embedding_model, preprocess, device = load_models()

print(collection)

image = None

# Inputs
img_file_buffer = st.camera_input("📸 Prends une photo ou sélectionne-en une", key="camera_input")
uploaded_file = None #st.file_uploader("Ou charge une image existante", type=["jpg", "jpeg", "png"])

# Nouvelle image sélectionnée
if img_file_buffer:
    image = Image.open(img_file_buffer).convert("RGB")
    # Réinitialiser les états
    st.session_state.detected_cards = []
    st.session_state.detected_collections = []

elif uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    # Réinitialiser les états
    st.session_state.detected_cards = []
    st.session_state.detected_collections = []

# Container pour affichage progressif

# --- Initialisation session_state pour collections ---
if "detected_collections" not in st.session_state:
    st.session_state.detected_collections = []


# Session state pour collection
if "detected_cards" not in st.session_state:
    st.session_state.detected_cards = []

# --- Affichage de l'image principale ---
#image_placeholder = st.empty()  # Placeholder pour l'image source annotée


# Placeholder unique pour toutes les collections
collections_placeholder = st.empty()

# --- Initialisation session_state pour collections ---
if "detected_collections" not in st.session_state:
    st.session_state.detected_collections = []

# --- Traitement de l'image ---
if image is not None:
    img = np.array(image, dtype=np.uint8)
    results = model.predict(source=img, conf=0.6, device='cpu')

    for r in results:
        masks = r.masks
        boxes = r.boxes

        if masks is not None:
            for i, m in enumerate(masks.data):

                #si la conf es trop faible on pass
                conf = float(boxes.conf[i])
                if conf < 0.6 :
                    continue

                mask = (m.cpu().numpy() > 0.5).astype(np.uint8) * 255
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

                crop = cv2.bitwise_and(img, img, mask=mask)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                crop = crop[y1:y2, x1:x2]

                crop = functions.preprocess_img(crop)
                crop = functions.improve_img(crop)
                crop_pil = Image.fromarray(crop)

                candidate_order_ids = [int(card["order"]) for card in meta]
                distances, indices = functions.search_card_correspondance(
                    collection, crop_pil, embedding_model, preprocess , 10, candidate_order_ids
                )

                search_results = []
                for i, idx in enumerate(indices):
                    search_results.append(meta[idx])
                    search_results[i]['distance_faiss'] = float(distances[i])

                #Un gap est anormal si il est au moins N fois plus grand que la moyenne des 2–3 gaps suivants.
                search_results = sorted(search_results, key=lambda x: x["distance_faiss"], reverse=False)
                gaps = [search_results[i+1]['distance_faiss'] - search_results[i]['distance_faiss'] for i in range(len(search_results)-1)]
                #Pour chaque on regarde les deux suivants
                for j in range(len(gaps)-1):
                    # On regarde les 2 gaps suivants (si disponibles)
                    next_gaps = gaps[j+1:j+3]
                    if not next_gaps:
                        continue

                    mean_next = sum(next_gaps) / len(next_gaps)

                    # Gap anormal si il est >> des gaps suivants
                    if gaps[j] > 3 * mean_next:
                        search_results = search_results[0:j+1]
                        break


                # Reranking
                search_results = sorted(search_results, key=lambda x: x["distance_faiss"])
                reranked_indices = functions.rerank_hash(crop, search_results, indices)
                search_results = []
                for idx, score in reranked_indices:
                    meta[idx]['distance_phash'] = score
                    search_results.append(meta[idx])



                # --- Créer la collection ---
                card_data = {
                    "crop": crop,
                    "box": (x1, y1, x2, y2), 
                    "reference": search_results[0]['img'],
                    "set" : search_results[0]['set'],
                    "number" : search_results[0]['number'],
                    "name": search_results[0]['name'],
                    "price": 0 if search_results[0]['price_eur'] ==None else search_results[0]['price_eur'],
                    "tcgplayer": search_results[0]['tcgplayer_link'],
                    "cardmarket": search_results[0]['cardmarket_link'],
                    "history": search_results[0]['price_evolution_url'],
                    "distance_faiss": search_results[0]['distance_faiss'],
                    "img": search_results[0]['img'],
                    "distance_phash": 0
                }

                st.session_state.detected_collections.append({
                    "main_card": card_data,
                    "search_results": search_results
                })

                # --- Trier toutes les collections par prix décroissant ---
                st.session_state.detected_collections.sort(
                    key=lambda c: c["main_card"]["price"], reverse=True
                )

                # --- Affichage dynamique ---
                # Image annotée avec toutes les cartes détectées
                img_annotated = draw_boxes_on_image(np.array(image), 
                                                    [c["main_card"] for c in st.session_state.detected_collections])
                #image_placeholder.image(img_annotated, use_container_width=True)

                # Mettre à jour l'affichage dans le placeholder
                with collections_placeholder.container():
                    for c in st.session_state.detected_collections:
                        card_data = c["main_card"]
                        search_results = c["search_results"]


                        #img_b64 = pil_to_base64(card_data["crop"])
                        modal_id = f"modal_img_{i}"  # ID unique pour chaque carte

                        md = f"""
                        <table style="margin-bottom:12px; border-radius:8px; padding:6px; width:100%;">
                            <tr>
                                <td style="width:130px;">
                                    <img src="{card_data['img']}" width=150% style="border-radius:4px;"/>
                                </td>
                                <td style="vertical-align:top; padding-left:10px;">
                                    <strong style="font-size:30px;">{card_data['name']}</strong><br/>
                                    <span style="font-size:16px;">{card_data['set']}/{card_data['number']} </span><br/>
                                    <span style="color:#e63946;font-size:20px; font-weight:bold;">💰 {card_data['price']} EUR</span><br/>
                                    <span style="font-size:14px;">
                                        <a href="{card_data['tcgplayer']}" target="_blank">TCGPlayer</a> | 
                                        <a href="{card_data['cardmarket']}" target="_blank">CardMarket</a> | 
                                        <a href="{card_data['history']}" target="_blank">Historique</a>
                                    </span>
                                </td>
                            </tr>
                        </table>
                        """

                        st.markdown(md, unsafe_allow_html=True)






                        if None:

                            max_cols = 1
                            cols = st.columns(max_cols )

                            #cols[0].image(card_data['crop'], caption="Crop", width="stretch")

                            for i, result in enumerate(search_results):
                                if i >= max_cols: #mettre i+1 si on affiche aussi le crop
                                    break
                                caption = f"{result['price_eur']} EUR "#- {result['distance_faiss']}"
                                cols[i ].image(result['img'], caption=caption, width="stretch") #mettre i+1 si on affiche aussi le crop

    st.markdown("## Analyse terminée")

# --- Valeur totale de la collection ---
total_value = sum([c["main_card"]['price'] for c in st.session_state.detected_collections])
st.markdown(f"##### 💰 Valeur totale : {total_value:.2f} €")




