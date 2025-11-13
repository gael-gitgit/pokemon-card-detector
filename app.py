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
    yolo_model = 'models/my-modelv10.pt'
    model = YOLO(yolo_model)
    return model, collection, meta, embedding_model, preprocess, device

model, collection, meta, embedding_model, preprocess, device = load_models()

print(collection)

# Inputs
img_file_buffer = st.camera_input("📸 Prends une photo ou sélectionne-en une", key="camera_input")
uploaded_file = st.file_uploader("Ou charge une image existante", type=["jpg", "jpeg", "png"])

st.session_state.process_step = "start"

image = None
if img_file_buffer:
    image = Image.open(img_file_buffer).convert("RGB")

elif uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

# Session state pour collection
if "detected_cards" not in st.session_state:
    st.session_state.detected_cards = []
    

# Container pour affichage progressif
collection_container = st.container()

# --- Traitement de l'image ---
if image is not None:
    st.session_state.detected_cards = []
    st.session_state.process_step = "inprogress"  
    img = np.array(image, dtype=np.uint8)
    #img = functions.improve_img(img)
    start = time.time()
    results = model.predict(source=img, conf=0.6, device='cpu')
    print("yolo img processed in : ",(time.time() - start) * 1e3, "ms")
    session = requests.Session()

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

                    start = time.time()
                    crop = functions.preprocess_img(crop_pre)
                    print("yolo imgpreprocess in : ",(time.time() - start) * 1e3, "ms")
                    start = time.time()
                    crop = functions.improve_img(crop)
                    print("yolo img improved in : ",(time.time() - start) * 1e3, "ms")
                    start = time.time()


                    candidate_order_ids = [int(card["order"]) for card in meta]

                    #recherche FAISS
                    distances,indices = functions.search_card_correspondance(
                        collection, crop, embedding_model, preprocess, device , 200, candidate_order_ids
                    )

                    print("search card correspondance : ",(time.time() - start) * 1e3, "ms")
                    start = time.time()
                    #get metadata
                    search_results = []
                    for i,idx in enumerate(indices):
                        search_results.append(meta[idx])
                        search_results[i]['distance_faiss'] = float(distances[i])

                    # on supprimer du tableau les éléents dont la distance est trop différente de la première (quand on a winner)
                    search_results = sorted(search_results, key=lambda x: x["distance_faiss"], reverse=False)
                    first_distance = search_results[0]['distance_faiss']
                    if search_results[1]['distance_faiss'] - search_results[0]['distance_faiss'] > 0.10:
                        search_results = [search_results[0]]

                    print("Garder le premier élément quand nécéssaire: ",(time.time() - start) * 1e3, "ms")
                    start = time.time()


                    # Si pas assez de différence, on rerank avec le hash et on cherche la 
                    reranked_indices = functions.rerank_hash(crop, search_results, indices)
                    search_results = []
                    for idx,score in reranked_indices:
                        meta[idx]['distance_phash'] = score
                        search_results.append(meta[idx])

                    #On selectionne les 10 premiers
                    search_results = search_results[0:5]

                    print("rerank hash: ",(time.time() - start) * 1e3, "ms")
                    start = time.time()

                    # On rerank les 10 finalist via la distance faiss
                    temp_search_results = sorted(search_results, key=lambda x: x["distance_faiss"], reverse=False)
                    corresp = [idx  for idx in range(0,len(temp_search_results)-1) if abs(temp_search_results[idx]['distance_faiss'] - temp_search_results[idx+1]['distance_faiss']) >= 0.10]
                    search_results = temp_search_results[0:max(corresp)+1] if len(corresp)>0 else search_results
                    
                    # On recherche encore une cassure mais sur le phash
                    temp_search_results = sorted(search_results, key=lambda x: x["distance_phash"], reverse=True)
                    corresp = [idx  for idx in range(0,len(temp_search_results)-1) if abs(temp_search_results[idx]['distance_phash'] - temp_search_results[idx+1]['distance_phash']) >= 0.0005]
                    search_results = temp_search_results[0:max(corresp)+1] if len(corresp)>0 else search_results

                    # On rerank les 10 finalist via la distance faiss
                    temp_search_results = sorted(search_results, key=lambda x: x["distance_faiss"], reverse=False)
                    corresp = [idx  for idx in range(0,len(temp_search_results)-1) if abs(temp_search_results[idx]['distance_faiss'] - temp_search_results[idx+1]['distance_faiss']) >= 0.10]
                    search_results = temp_search_results[0:max(corresp)+1] if len(corresp)>0 else search_results

                    print("réorganisations : ",(time.time() - start) * 1e3, "ms")
                    start = time.time()

                    
                    # --- Ajouter carte à session_state ---
                    card_data = {
                        "crop": crop_pre,
                        "reference": search_results[0]['img'],
                        "name": search_results[0]['name'],
                        "price": search_results[0]['price_eur'],
                        "tcgplayer": search_results[0]['tcgplayer_link'],
                        "cardmarket": search_results[0]['cardmarket_link'],
                        "history": search_results[0]['price_evolution_url'],
                        "distance_faiss": search_results[0]['distance_faiss'],
                        "distance_phash": search_results[0]['distance_phash']

                    }
                    print(f"le premier pokemon était  : {search_results[0]['name']}")
                    st.session_state.detected_cards.append(card_data)

                    with collection_container:
                        st.markdown(
                            f"### {card_data['name']} — 💰 <span style='color:red;'>{card_data['price']} €</span>", 
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
                            caption = f"{result['price_eur']} EUR - {result['distance_faiss']} - {result['distance_phash']}"
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
