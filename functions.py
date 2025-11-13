import os
import torch
import numpy as np
from PIL import Image
import open_clip

import requests
import pickle
from io import BytesIO
from transformers import AutoModel, AutoImageProcessor

import faiss
import json
import time

import imagehash

import cv2

def get_image_from_url(session,url):
    response = session.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")

    return img



def improve_img(img):


    # === Étape 1 : analyse de base ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)

    # --- Étape 1 : correction du contraste (CLAHE dans l'espace LAB) ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced_lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- Étape 2 : boost de saturation et luminosité (espace HSV) ---
    hsv = cv2.cvtColor(enhanced_lab, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.1, 0, 255).astype(np.uint8)

    if brightness < 115:
        v = np.clip(v * 1.1, 0, 255).astype(np.uint8)
    elif brightness > 140:
        v = np.clip(v * 0.9, 0, 255).astype(np.uint8)
    else :
        v = np.clip(v * 1, 0, 255).astype(np.uint8)

    hsv_boosted = cv2.merge((h, s, v))
    enhanced_hsv = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    # --- Étape 3 : débruitage et renforcement de la netteté ---
    #denoised = cv2.fastNlMeansDenoisingColored(enhanced_hsv, None, 7, 7, 7, 21)
    #sharp = cv2.addWeighted(denoised, 1.2, cv2.GaussianBlur(denoised, (0, 0), 2), -0.2, 0)

    #Plus rapide mais un peu de perte
    denoised = cv2.bilateralFilter(enhanced_hsv, d=5, sigmaColor=75, sigmaSpace=75)
    sharp = cv2.addWeighted(denoised, 1.3, cv2.GaussianBlur(denoised, (0,0), 2), -0.3, 0)

    return sharp

def preprocess_img(img):

    border_size = int(min(img.shape[:2]) * 0.10)  # marge = 5% de la taille
    img = cv2.copyMakeBorder(
        img, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)  # fond noir
        )
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # since findContours alters the image
    contours, hierarchy = cv2.findContours(gray.copy(),
        cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # plus grand contour = carte
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        pts = np.int0(box)

    # Réordonne les points
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (w, h) = (460, 640)
    #(w, h) = (230, 320)
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (w, h))


    return warp


def load_faiss_index():
    # Load your FAISS index
    collection = faiss.read_index("./faiss_db/pokemon-cards.index")
    #with open("./faiss_db/pokemon-cards_schema-with-hashes.json") as f:
        #meta = json.load(f)
    with open("./faiss_db/test-hash-plus-colorsig.json") as f:
        meta = json.load(f)

    
    return collection,meta

def load_embbedings_model(model_name="facebook/dinov2-base"):
    """
    Chargement du modèle DINOv2 depuis Hugging Face
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Chargement du processeur et du modèle
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return model, processor, device


def get_embedding(img, model, processor, device):
    
    #convert CV2 img to PIL for encoding
    # Convert BGR → RGB
    #cv_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv_img_rgb = img
    pil_img = Image.fromarray(cv_img_rgb).convert("RGB")
        
    # Préprocessing automatique
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        # DINOv2 renvoie un dictionnaire avec 'last_hidden_state' et 'pooler_output'
        if "pooler_output" in outputs:
            emb = outputs.pooler_output  # [batch_size, hidden_dim]
        else:
            # fallback: moyenne spatiale sur les patchs
            emb = outputs.last_hidden_state.mean(dim=1)

        emb = emb.cpu().numpy().flatten()

    # Normalisation L2
    emb = emb / np.linalg.norm(emb)
    emb = emb.reshape(1, -1)
    return emb


def search_card_correspondance(collection, img, embbedings_model, processor, device,k, filter_ids=[]):
    """
    Recherche la correspondance dans un index FAISS à partir d'une image
    """
    
    start = time.time()              
    query_emb = get_embedding(img, embbedings_model, processor, device)
    query_emb = np.vstack(query_emb).astype("float32")
    print("getting emmbeddings :  ",(time.time() - start) * 1e3, "ms")
    start = time.time() 

    #recherche de la correspondance sur un subset
    filter_ids = np.array(filter_ids, dtype=np.int64)
    id_selector = faiss.IDSelectorArray(len(filter_ids), faiss.swig_ptr(filter_ids))
    params = faiss.SearchParameters()
    params.sel = id_selector

    #recherche
    distances, indices = collection.search(query_emb, k=k, params=params)
    print("getting search results :  ",(time.time() - start) * 1e3, "ms")
    return distances[0], indices[0]




def get_phash(img):
    pil_img = Image.fromarray(img).convert("RGB")
    hash = imagehash.phash(img,hash_size=30)
    return hash

def read_crop_hash(hash_as_str):
    return imagehash.hex_to_hash(hash_as_str)

def store_hash(hash):
    return str(hash)




def compute_phash(img):
    pil_img = Image.fromarray(img).convert("RGB")
    return imagehash.phash(pil_img,hash_size=30)

def rerank_hash(query_img, candidates, indices):
    """
    Rerank FAISS results using perceptual hash similarity (pHash).
    """
    query_hash = compute_phash(query_img)
    scores = []

    for cand, idx in zip(candidates, indices):
        cand_hash = imagehash.hex_to_hash(cand["phash"])
        # Distance de Hamming (0 = identique)
        hamming_dist = query_hash - cand_hash
        # Score inverse (plus haut = plus similaire)
        score = 1 / (1 + hamming_dist)
        scores.append((idx, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    reranked_indices = [idx for idx, _ in scores]
    return scores

