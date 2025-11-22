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




st.set_page_config(page_title="Trouve la Poké-pétite", layout="wide")

# Masquer toolbar et footer
#hide_streamlit_style = """
#<style>
#div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"], #MainMenu, header, footer {
#    visibility: hidden;
#    height: 0%;
#    position: fixed;
#}
#</style>
#"""
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🎴 Pokemon Card Detector")

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

# Inputs
img_file_buffer = st.camera_input("📸 Prends une photo ou sélectionne-en une", key="camera_input")
uploaded_file = st.file_uploader("Ou charge une image existante", type=["jpg", "jpeg", "png"])

st.session_state.process_step = "start"
# Container pour affichage progressif
collection_container = st.container()

image = None
if img_file_buffer:
    image = Image.open(img_file_buffer).convert("RGB")
    st.session_state.detected_cards = []

elif uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.detected_cards = []

# Session state pour collection
if "detected_cards" not in st.session_state:
    st.session_state.detected_cards = []
    

# --- Traitement de l'image ---
if image is not None:
    st.session_state.detected_cards = []
    st.session_state.process_step = "inprogress"  
    img = np.array(image, dtype=np.uint8)
    #img = functions.improve_img(img)
    start = time.time()
    results = model.predict(source=img, conf=0.6, device='cpu')
    print("yolo img processed in : ",(time.time() - start) * 1e3, "ms")

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

                    crop = cv2.bitwise_and(img, img, mask=mask)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    crop = crop[y1:y2, x1:x2]

                    start = time.time()
                    crop = functions.preprocess_img(crop)
                    print(" imgpreprocess in : ",(time.time() - start) * 1e3, "ms")
                    start = time.time()
                    crop = functions.improve_img(crop)
                    print(" img improved in : ",(time.time() - start) * 1e3, "ms")
                    start = time.time()


                    candidate_order_ids = [int(card["order"]) for card in meta]

                    #crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_pil = Image.fromarray(crop)

                    #recherche FAISS
                    distances,indices = functions.search_card_correspondance(
                        collection, crop_pil, embedding_model, preprocess , 10, candidate_order_ids
                    )

                    print("search card correspondance : ",(time.time() - start) * 1e3, "ms")
                    start = time.time()
                    #get metadata
                    search_results = []
                    for i,idx in enumerate(indices):
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

                    #On rerank avec phash
                    reranked_indices = functions.rerank_hash(crop, search_results, indices)
                    search_results = []
                    for idx,score in reranked_indices:
                        meta[idx]['distance_phash'] = score
                        search_results.append(meta[idx])


                    print("réorganisations : ",(time.time() - start) * 1e3, "ms")
                    start = time.time()
                    
                    
                    # --- Ajouter carte à session_state ---
                    card_data = {
                        "crop": crop,
                        "reference": search_results[0]['img'],
                        "name": search_results[0]['name'],
                        "set" : search_results[0]['set'],
                        "number" : search_results[0]['number'],
                        "price": search_results[0]['price_eur'],
                        "tcgplayer": search_results[0]['tcgplayer_link'],
                        "cardmarket": search_results[0]['cardmarket_link'],
                        "history": search_results[0]['price_evolution_url'],
                        "distance_faiss": search_results[0]['distance_faiss'],
                        "distance_phash": 0#search_results[0]['distance_phash']

                    }
                    print(f"le premier pokemon était  : {search_results[0]['name']}")
                    st.session_state.detected_cards.append(card_data)

                    with collection_container:
                        st.markdown(
                            f"### {card_data['name']} — ###{card_data['set']}{card_data['number']}###💰 <span style='color:red;'>{card_data['price']} €</span>", 
                            unsafe_allow_html=True
                        )
                        
                        # Crée jusqu'à 11 colonnes
                        max_cols = 12
                        cols = st.columns(max_cols)
                        
                        # Affiche l'image "Crop" dans la première colonne
                        cols[0].image(crop, caption="Crop", width='stretch')
                        
                        # Parcours des résultats de recherche existants
                        for i, result in enumerate(search_results):
                            if i + 1 >= max_cols:
                                break  # Évite d'aller au-delà du nombre de colonnes
                            caption = f"{result['price_eur']} EUR - {result['distance_faiss']} "#- {result['distance_phash']}"
                            cols[i + 1].image(result['img'], caption=caption, width='stretch')
                        
                        st.markdown(f"""
                        **Liens :** [TCGPlayer]({card_data['tcgplayer']}) | 
                        [CardMarket]({card_data['cardmarket']}) | 
                        [Historique prix]({card_data['history']})
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
    st.info("Amélioration de la qualité des images (oui c'est long...)")
elif st.session_state.process_step  == "end" and len(st.session_state.detected_cards) ==0:
    st.warning("Analyse terminée - Aucun objet détecté")
elif st.session_state.process_step  == "end":
    st.success("Analyse terminée")
