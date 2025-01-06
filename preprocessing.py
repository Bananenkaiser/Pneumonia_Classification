# Preprocessing mit Multiprocessing

import time
#import cupy as cp # CuPy ist eine GPU-basierte Bibliothek, die NumPy-ähnliche Operationen unterstützt
import os
import cv2
import numpy as np
from multiprocessing import Pool

# Hyperparameter
input_folder = "./chest_xray/chest_xray/"  # Hauptordner
output_folder = "./chest_xray/preprocessed_images/" # Gemeinsamer Ausgabeordner für die bearbeiteten Bilder

def resize_with_padding(image, target_size=(224, 224)):
    old_size = image.shape[:2]  # Alte Größe (Höhe, Breite)
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))  # Neue Größe (Breite, Höhe)

    # Bild skalieren
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Padding hinzufügen
    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]  # Padding-Farbe (hier schwarz)
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def crop_percentage(image, percentage=0.1):
    """
    Schneidet einen bestimmten Prozentsatz von allen Seiten des Bildes ab.

    Parameter:
        - image: Eingabebild (NumPy-Array).
        - percentage: Prozentsatz der Breite und Höhe, der abgeschnitten werden soll (zwischen 0 und 1).

    Rückgabe:
        - Beschnittenes Bild.
    """
    # Bilddimensionen
    height, width = image.shape[:2]

    # Berechnung der zu entfernenden Pixel
    crop_h = int(height * percentage)  # 10% der Höhe
    crop_w = int(width * percentage)   # 10% der Breite

    # Bild zuschneiden
    cropped_image = image[crop_h:height-crop_h, crop_w:width-crop_w]

    return cropped_image



def process_single_image(input_path, output_path):
    """
    Bearbeitet ein einzelnes Bild und speichert es am Zielort.
    """
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Das Bild konnte nicht geladen werden")

    # 1. Kontrasterhöhung mit CLAHE (optional, verbessert Lungenregion)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    cropped_image = crop_percentage(image, percentage=0.1)

    """
    Wahrscheinlich brauche ich für Virus_Bilder einen anderen Filter als für die restlichen beiden Bilder
    """

    lower_bound = 0
    upper_bound = 150

    # cv2.inRange erstellt eine Maske für Pixel im Bereich [lower_bound, upper_bound]
    filtered_image = cv2.inRange(cropped_image, lower_bound, upper_bound)

    # 3. Konturen finden
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Größte Konturen auswählen (angenommen: Lungenflügel sind die größten Bereiche)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # Wähle die zwei größten Konturen

    # 5. Maske erstellen
    lung_mask = np.zeros_like(cropped_image)
    cv2.drawContours(lung_mask, contours, -1, (255), thickness=cv2.FILLED)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened_image = cv2.morphologyEx(lung_mask, cv2.MORPH_OPEN, kernel)

    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated_image = cv2.dilate(opened_image, kernel_2, iterations=1)

    # Final segmented image
    segmented_image = cv2.bitwise_and(cropped_image, dilated_image)

    # Set pixels >240 to 0
    segmented_image[segmented_image > 240] = 0

    resized_image = resize_with_padding(segmented_image, target_size=(224, 224))
    #resized_image = resize_with_padding(segmented_image, target_size=(299, 299))


    # Speichere das Ergebnis
    cv2.imwrite(output_path, resized_image)


def process_images_in_folder(folder_path, output_folder, num_workers=4):
    """
    Bearbeitet alle Bilder in der Ordnerstruktur von folder_path und repliziert die Struktur im output_folder.
    """
    tasks = []
    
    for root, dirs, files in os.walk(folder_path):
        rel_path = os.path.relpath(root, folder_path)
        target_folder = os.path.join(output_folder, rel_path)
        os.makedirs(target_folder, exist_ok=True)

        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_folder, file)
                tasks.append((input_path, output_path))

    # Parallelisierte Verarbeitung
    with Pool(num_workers) as pool:
        pool.starmap(process_single_image, tasks)

if __name__ == '__main__':

    start_time = time.time()
    process_images_in_folder(input_folder, output_folder, num_workers=4)
    end_time = time.time()
    print(f"Dauer: {round(end_time - start_time, 2)} Sekunden") 