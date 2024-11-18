import cv2
import pytesseract
from tkinter import Tk, filedialog
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Chemin vers Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Modifiez selon votre système

# Variables globales pour gérer la sélection
start_point = None
end_point = None
selecting = False
image = None

# Initialiser le traducteur
translator = Translator()

def select_region(event, x, y, flags, param):
    """
    Fonction de callback pour gérer les clics de souris et les mouvements.
    """
    global start_point, end_point, selecting, image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Début de la sélection
        start_point = (x, y)
        selecting = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            # Dessiner un rectangle temporaire
            temp_image = image.copy()
            cv2.rectangle(temp_image, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Comic Image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        # Fin de la sélection
        end_point = (x, y)
        selecting = False

        # Appliquer l'OCR à la zone sélectionnée
        extract_text_from_region(image, start_point, end_point)


def extract_text_from_region(image, start_point, end_point):
    """
    Extraire du texte d'une région spécifique définie par deux points et le traduire en français.
    Remplacer ensuite le texte extrait par le texte traduit sur l'image, ligne par ligne.
    """
    x1, y1 = start_point
    x2, y2 = end_point

    # Gérer les cas où les points sont inversés
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Extraire la région d'intérêt (ROI)
    roi = image[y1:y2, x1:x2]
    text = pytesseract.image_to_string(roi, config='--psm 6')

    # Nettoyer le texte pour qu'il soit sur une seule ligne pour une meilleure traduction
    cleaned_text = " ".join(text.split())
    print(f"Texte extrait de la région sélectionnée (nettoyé) :\n{cleaned_text}\n")

    # Traduire le texte en français
    translated_text = translator.translate(cleaned_text, src='auto', dest='fr').text
    print(f"Texte traduit en français :\n{translated_text}\n")

    # Convertir l'image OpenCV en image PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Définir une taille de police de base et adapter pour que tout le texte tienne dans la zone
    font_size = 16
    font = ImageFont.truetype("arial.ttf", font_size)

    max_width = x2 - x1
    max_height = y2 - y1

    # Adapter la taille de la police pour que tout le texte tienne dans la largeur et la hauteur de la zone
    while True:
        lines = []
        words = translated_text.split()
        current_line = ""

        # Diviser le texte en lignes pour voir si chaque ligne tient dans la zone
        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line = test_line
            else:
                # Ajouter la ligne complète aux lignes et démarrer une nouvelle ligne
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Calculer la hauteur totale des lignes de texte
        line_height = draw.textbbox((0, 0), "Test", font=font)[3] - draw.textbbox((0, 0), "Test", font=font)[1]
        total_text_height = line_height * len(lines) * 1.5  # Augmenter l'espacement entre les lignes

        # Vérifier si le texte tient dans la zone sélectionnée
        if total_text_height <= max_height and all(draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] <= max_width for line in lines):
            break
        else:
            # Réduire la taille de la police si le texte ne tient pas
            font_size -= 1
            font = ImageFont.truetype("arial.ttf", font_size)
            if font_size < 8:  # Fixer une taille minimale pour garder le texte lisible
                break

    # Effacer l'ancien texte en remplissant la zone avec une couleur blanche dans l'image PIL
    draw.rectangle([x1, y1, x2, y2], fill="white")

    # Dessiner chaque ligne de texte avec un espacement vertical
    current_y = y1
    for line in lines:
        # Centrer chaque ligne horizontalement
        text_width = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        text_x = x1 + (max_width - text_width) // 2

        # Dessiner la ligne de texte
        draw.text((text_x, current_y), line, fill="black", font=font)
        current_y += int(line_height * 1.5)  # Espacement vertical augmenté (50% supplémentaire)

    # Convertir l'image PIL en image OpenCV
    image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Mettre à jour l'image affichée
    cv2.imshow("Comic Image", image)



def select_image():
    """
    Fonction pour demander à l'utilisateur de sélectionner une image.
    """
    Tk().withdraw()  # Masquer la fenêtre principale Tkinter
    file_path = filedialog.askopenfilename(
        title="Sélectionner une image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
    )
    return file_path


# Demander à l'utilisateur de sélectionner une image
image_path = select_image()

if not image_path:
    print("Aucune image sélectionnée. Fin du programme.")
    exit(0)

# Charger l'image
image = cv2.imread(image_path)

if image is None:
    print(f"Erreur : Impossible de charger l'image. Vérifiez le fichier : {image_path}")
    exit(1)

# Créer une fenêtre interactive
cv2.namedWindow("Comic Image")
cv2.setMouseCallback("Comic Image", select_region)

print("Instructions :\n- Cliquez et glissez pour sélectionner une zone.\n- Relâchez pour extraire et remplacer le texte.\n- Appuyez sur 'q' pour quitter.")

# Afficher l'image et attendre les interactions
while True:
    cv2.imshow("Comic Image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quitter avec la touche 'q'
        break

cv2.destroyAllWindows()
