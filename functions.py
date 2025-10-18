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

import imagehash

import cv2
from skimage.metrics import structural_similarity as ssim

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
    s = np.clip(s * 1, 0, 255).astype(np.uint8)

    if brightness < 115:
        v = np.clip(v * 1.1, 0, 255).astype(np.uint8)
    elif brightness > 140:
        v = np.clip(v * 0.9, 0, 255).astype(np.uint8)
    else :
        v = np.clip(v * 1, 0, 255).astype(np.uint8)

    hsv_boosted = cv2.merge((h, s, v))
    enhanced_hsv = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    # --- Étape 3 : débruitage et renforcement de la netteté ---
    denoised = cv2.fastNlMeansDenoisingColored(enhanced_hsv, None, 7, 7, 7, 21)
    sharp = cv2.addWeighted(denoised, 1.2, cv2.GaussianBlur(denoised, (0, 0), 2), -0.2, 0)

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
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (w, h))

    return warp



def load_faiss_index():
    # Load your FAISS index
    collection = faiss.read_index("./faiss_db/pokemon-cards.index")
    with open("./faiss_db/pokemon-cards_schema.json") as f:
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


def search_card_correspondance(collection, img, embbedings_model, processor, device):
    """
    Recherche la correspondance dans un index FAISS à partir d'une image
    """
    query_emb = get_embedding(img, embbedings_model, processor, device)
    distances, indices = collection.search(query_emb, k=20)

    return distances[0], indices[0]




def rerank_ssim(query_img, candidate_imgs, indices, distances):
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
    scores = []

    for cand, idx, dist in zip(candidate_imgs, indices, distances):
        cand_gray = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY)

        # 🔧 redimensionne pour que les tailles matchent
        if cand_gray.shape != query_gray.shape:
            cand_gray = cv2.resize(cand_gray, (query_gray.shape[1], query_gray.shape[0]))

        s, _ = ssim(query_gray, cand_gray, full=True)
        scores.append((idx, s))

    # Trie par score SSIM décroissant
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    reranked_indices = [idx for idx, _ in scores]
    return reranked_indices



def rerank_orb(query_img, candidate_imgs, indices, distances, max_features=1000):
    """
    Rerank FAISS results using ORB feature matching.
    
    Args:
        query_img: numpy array (RGB)
        candidate_imgs: list of numpy arrays (RGB)
        indices: indices des résultats FAISS
        distances: distances FAISS associées
        max_features: nombre max de keypoints ORB
    
    Returns:
        reranked_indices: indices triés par similarité ORB décroissante
    """

    # Convertir la requête en niveaux de gris
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)

    # Initialiser ORB et le matcher Hamming
    orb = cv2.ORB_create(nfeatures=max_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Calculer les keypoints et descripteurs de la requête
    kp1, des1 = orb.detectAndCompute(query_gray, None)
    if des1 is None:
        print("⚠️ Aucun descripteur trouvé pour l'image requête.")
        return indices  # retourne le classement FAISS par défaut

    scores = []

    for cand, idx, dist in zip(candidate_imgs, indices, distances):
        cand_gray = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY)

        # Détecter keypoints et descripteurs pour la candidate
        kp2, des2 = orb.detectAndCompute(cand_gray, None)
        if des2 is None:
            scores.append((idx, 0))
            continue

        # Matcher les descripteurs
        matches = bf.match(des1, des2)
        if not matches:
            scores.append((idx, 0))
            continue

        # Calculer un score basé sur la qualité des matches
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 60]  # seuil empirique
        score = len(good_matches) / len(matches)  # ratio de bons matches
        scores.append((idx, score))

    # Trier par score décroissant
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    reranked_indices = [idx for idx, _ in scores]

    return reranked_indices


def compute_phash(img):
    pil_img = Image.fromarray(img).convert("RGB")
    return imagehash.phash(pil_img)

def rerank_hash(query_img, candidate_imgs, indices, distances):
    """
    Rerank FAISS results using perceptual hash similarity (pHash).
    """
    query_hash = compute_phash(query_img)
    scores = []

    for cand, idx, dist in zip(candidate_imgs, indices, distances):
        cand_hash = compute_phash(cand)
        # Distance de Hamming (0 = identique)
        hamming_dist = query_hash - cand_hash
        # Score inverse (plus haut = plus similaire)
        score = 1 / (1 + hamming_dist)
        scores.append((idx, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    reranked_indices = [idx for idx, _ in scores]
    return reranked_indices