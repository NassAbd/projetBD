import os
import zipfile
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from deep_translator import GoogleTranslator
import re

# Initialiser le traducteur
translator = GoogleTranslator(source='fr', target='en')

# Initialiser EasyOCR avec support GPU
reader = easyocr.Reader(['fr'], gpu=True)

def extraire_et_traduire_textes_repertoire(repertoire_bd):
    if os.path.isfile(repertoire_bd):
        if repertoire_bd.endswith('.cbz'):
            extraire_et_traduire_texte_cbz(repertoire_bd)
        else:
            extraire_et_traduire_texte_fichier(repertoire_bd)
    elif os.path.isdir(repertoire_bd):
        for nom_fichier in os.listdir(repertoire_bd):
            if nom_fichier.endswith(('.png', '.jpg', '.jpeg', '.cbz')):
                chemin_image = os.path.join(repertoire_bd, nom_fichier)
                if nom_fichier.endswith('.cbz'):
                    extraire_et_traduire_texte_cbz(chemin_image)
                else:
                    extraire_et_traduire_texte_fichier(chemin_image)
    else:
        print("Le chemin fourni n'est ni un fichier ni un répertoire valide.")

def extraire_et_traduire_texte_cbz(chemin_cbz):
    # Décompresser le fichier CBZ (qui est un fichier ZIP)
    with zipfile.ZipFile(chemin_cbz, 'r') as zip_ref:
        fichiers_images = [f for f in zip_ref.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]
        for i, fichier_image in enumerate(fichiers_images):
            if i >= 10:
                break
            with zip_ref.open(fichier_image) as image_file:
                image = Image.open(image_file).convert("RGB")
                texte_bulles, image_annotee = extraire_texte_image(image, page_num=i+1)
                texte_traduit = traduire_texte(texte_bulles)

                # Affichage du texte extrait et traduit
                print(f"Texte extrait de {fichier_image} dans {chemin_cbz} :\n{texte_bulles}\n")
                print(f"Texte traduit :\n{texte_traduit}\n")

                # Afficher l'image annotée
                image_annotee.show()

def extraire_et_traduire_texte_fichier(chemin_image):
    image = Image.open(chemin_image).convert("RGB")
    texte_bulles, image_annotee = extraire_texte_image(image, page_num=1)
    texte_traduit = traduire_texte(texte_bulles)

    # Affichage du texte extrait et traduit
    print(f"Texte extrait de {chemin_image} :\n{texte_bulles}\n")
    print(f"Texte traduit :\n{texte_traduit}\n")

    # Afficher l'image annotée
    image_annotee.show()

def extraire_texte_image(image, page_num):
    # Agrandir l'image pour améliorer la détection des petits textes
    facteur_zoom = 2  # Augmenter la résolution de 2 fois
    largeur, hauteur = image.size
    image_zoom = image.resize((int(largeur * facteur_zoom), int(hauteur * facteur_zoom)), Image.LANCZOS)

    # Améliorer le contraste et la netteté de l'image pour mieux détecter le texte
    enhancer_contrast = ImageEnhance.Contrast(image_zoom)
    image_zoom = enhancer_contrast.enhance(2.0)  # Augmenter le contraste
    enhancer_sharpness = ImageEnhance.Sharpness(image_zoom)
    image_zoom = enhancer_sharpness.enhance(2.0)  # Augmenter la netteté

    # Convertir l'image PIL en format OpenCV
    image_cv = cv2.cvtColor(np.array(image_zoom), cv2.COLOR_RGB2BGR)

    # Convertir en niveaux de gris
    image_gris = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuillage adaptatif pour mieux détecter les bordures des bulles (supposées foncées)
    image_seuillee = cv2.adaptiveThreshold(image_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Détection des contours pour identifier les bordures des bulles
    contours, _ = cv2.findContours(image_seuillee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Trier les contours pour qu'ils soient traités de haut en bas, puis de gauche à droite
    contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1] // 50, cv2.boundingRect(ctr)[0]))

    textes_bulles = []
    image_annotee = image_zoom.copy()
    draw = ImageDraw.Draw(image_annotee)

    # Charger une police de caractères pour rendre le texte plus lisible
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Traiter chaque bulle détectée
    bulle_num = 1
    for contour in contours:
        # Obtenir un rectangle englobant pour chaque bulle de texte
        x, y, w, h = cv2.boundingRect(contour)
        # Filtrer les contours en fonction de leur forme et de leur taille
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 0.3 < aspect_ratio < 3.0 and w > 30 and h > 30 and area > 500:  # Assouplir les filtres pour capter plus de bulles
            # Ajouter une marge intérieure pour éviter les bordures des bulles
            marge = 5
            x_min = max(0, x + marge)
            y_min = max(0, y + marge)
            x_max = min(image_cv.shape[1], x + w - marge)
            y_max = min(image_cv.shape[0], y + h - marge)

            # Extraire la région de la bulle de texte
            bulle_image = image_zoom.crop((x_min, y_min, x_max, y_max))

            # Créer un masque pour détecter la région blanche à l'intérieur de la bulle
            bulle_cv = cv2.cvtColor(np.array(bulle_image), cv2.COLOR_RGB2BGR)
            bulle_hsv = cv2.cvtColor(bulle_cv, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])  # Borne inférieure pour la couleur blanche
            upper_white = np.array([180, 30, 255])  # Borne supérieure pour la couleur blanche
            masque_blanc = cv2.inRange(bulle_hsv, lower_white, upper_white)

            # Réduction du bruit et détection des zones de texte dans la bulle
            masque_blanc = cv2.medianBlur(masque_blanc, 5)

            # Utiliser EasyOCR pour extraire le texte de la bulle entière
            resultats_bulle = reader.readtext(masque_blanc, detail=1)
            texte_bulle = ""
            for res in resultats_bulle:
                if len(res[1].strip()) > 1:  # Éliminer les faux positifs (lettres ou chiffres isolés)
                    texte_bulle += res[1] + " "
                    # Annoter le texte détecté en rouge
                    (tx_min, ty_min), (tx_max, ty_max) = res[0][0], res[0][2]
                    # Vérifier les coordonnées pour s'assurer qu'elles sont valides
                    if tx_min < tx_max and ty_min < ty_max:
                        draw.rectangle([x_min + tx_min, y_min + ty_min, x_min + tx_max, y_min + ty_max], outline="red", width=2)
                        # Mettre la zone en blanc
                        draw.rectangle([x_min + tx_min, y_min + ty_min, x_min + tx_max, y_min + ty_max], fill="white")

            texte_traduit = traduire_texte(texte_bulle.strip())
            if texte_traduit:
                # Ajouter des retours à la ligne tous les 8 mots pour une meilleure lisibilité
                texte_traduit_formate = " ".join([texte_traduit.split()[i] + ("\n" if (i + 1) % 8 == 0 else "") for i in range(len(texte_traduit.split()))])
                # Ajouter le texte traduit dans la première zone rouge détectée
                if resultats_bulle:
                    (tx_min, ty_min), (tx_max, ty_max) = resultats_bulle[0][0][0], resultats_bulle[0][0][2]
                    if tx_min < tx_max and ty_min < ty_max:
                        draw.text((x_min + tx_min, y_min + ty_min), texte_traduit_formate, fill="black", font=font)

            texte_bulle = re.sub(r'\b(\w+)-\s*\+?\b(\w+)', r'\1\2', texte_bulle)  # Gérer les mots coupés
            if texte_bulle.strip():  # Ajouter seulement si du texte a été détecté
                textes_bulles.append(f"Page {page_num} => Bulle {bulle_num}: {texte_bulle.strip()}")
                bulle_num += 1

            # Annoter la bulle en vert
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

    texte_final = "\n\n".join(textes_bulles)  # Joindre les bulles avec des sauts de ligne double pour séparer les bulles distinctes

    return texte_final, image_annotee

def traduire_texte(texte):
    # Utiliser deep_translator pour traduire le texte en anglais
    if texte:
        traduction = translator.translate(texte)
        return traduction
    return ""

if __name__ == "__main__":
    # Indiquer le répertoire contenant les images de BD ou un fichier unique
    repertoire_bd = "tintin/Les Aventures de Tintin Le Sceptre d'Ottokar.cbz"  # Remplacer par le chemin de votre répertoire ou fichier
    extraire_et_traduire_textes_repertoire(repertoire_bd)
