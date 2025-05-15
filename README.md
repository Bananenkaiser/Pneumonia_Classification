# Röntgenbild-Klassifikation mit ResNet50V2

## Inhaltsverzeichnis
1. [Projektbeschreibung](#projektbeschreibung)
2. [Motivation](#motivation)
3. [Features](#features)
4. [Technologie-Stack](#technologie-stack)
5. [Daten & Preprocessing](#daten--preprocessing)
6. [Modellarchitektur](#modellarchitektur)
7. [Usage / Beispiele](#usage--beispiele)
8. [Evaluation & Ergebnisse](#evaluation--ergebnisse)
9. [Screenshots / Visualisierungen](#screenshots--visualisierungen)
10. [Kontakt](#kontakt)

## Projektbeschreibung
Dieses Projekt untersucht die automatisierte Klassifikation von Röntgenbildern zur Erkennung von Lungenentzündungen. Dazu wurden Bilder vorverarbeitet und mit einem vortrainierten ResNet50V2-Netzwerk feinabgestimmt.

## Motivation
Die schnelle und zuverlässige Diagnose von Lungenentzündungen kann Leben retten. Machine-Learning-Modelle bieten hier großes Potenzial, um Radiologen zu unterstützen.

## Features
- Vorverarbeitung der Röntgenbilder (CLAHE Kontrasterhöhung, Cropping, Konturerkennung, Maskenerstellung, morphologische Operationen)
- Resizing auf 224×224 Pixel für ResNet50V2
- Feintuning eines vortrainierten ResNet50V2-Netzwerks
- Fokus auf hohen Recall (Sensitivität) zur Minimierung von False Negatives in der medizinischen Diagnose
- Evaluierung mittels Metriken wie Accuracy, Precision, Recall und F1-Score
- Visualisierung von Trainings- und Validierungsverläufen

## Technologie-Stack
- Python 3.12+
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Git, GitHub

## Daten & Preprocessing

Die Röntgenbilder befinden sich im Ordner data/raw. Das Preprocessing-Skript preprocess.py führt folgende Schritte durch:

1. Kontrasterhöhung: Anwendung von CLAHE (Contrast Limited Adaptive Histogram Equalization)

2. Cropping & Konturenerkennung mit OpenCV:
- Finden aller Konturen im Bild
- Auswahl der größten zusammenhängenden Kontur als Lungenflügel

3. Maskenerstellung:
- Erzeugen einer Binärmaske der ausgewählten Kontur
- Durchführen morphologischer Operationen (Erosion, Dilation) zur Verfeinerung der Maske
- Überlagern der Maske mit dem Originalbild und bitweises Verknüpfen, so dass nur der Lungenbereich erhalten bleibt

4. Resizing: Skalierung auf 224×224 Pixel für das ResNet50V2-Modell

## Modellarchitektur
Folgender Code zeigt den Modellaufbau:

```python
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Basis-Modell laden und einfrieren
base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Vortrainierte Schichten einfrieren

# Klassifikations-Head
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binärklassifikation
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Recall', 'Precision']
)
```  

## Evaluation & Ergebnisse


- **Accuracy:** 0.84  
- **Recall:** 0.98  
- **Precision:** 0.81  
- **F1-Score:** 0.89
- **Specificity:** 0.62

<p align="center">
  <img src="https://github.com/user-attachments/assets/db5e27d9-76d5-48d8-b0d8-4b1a9ed908ab" alt="Confusion Matrix" width="600"/>
</p>


- Der hohe Recall (~97,9 %) bedeutet, dass fast alle tatsächlichen Pneumonie-Fälle erkannt werden.
- Die Specificity (~62,4 %) zeigt hingegen, dass ein Teil der Gesunden fälschlich als krank eingestuft wird – ein bewusster Trade-off zugunsten des hohen Recall in deinem medizinischen Anwendungsfall.

## Screenshots / Visualisierungen

### Keine Lungenentzündung
<p align="center">
  <img src="https://github.com/user-attachments/assets/027c2d0b-58dc-4644-9221-af9430a8a15f" alt="Keine Lungenentzündung" width="400"/>
</p>

### Lungenentzündung vorhanden
<p align="center">
  <img src="https://github.com/user-attachments/assets/86fdd941-2020-4368-9ca1-8535e57388da" alt="Lungenentzündung vorhanden" width="400"/>
</p>

### Nach Preprocessing
<p align="center">
  <img src="https://github.com/user-attachments/assets/b3975fb8-bcdd-4afe-87b3-986ec35c468a" alt="Nach Preprocessing" width="400"/>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/2d83d46f-7a8b-4c59-b17e-dacc60c68985" alt="Trainingsverlauf" width="800"/>
</p>


## Kontakt

E-Mail: dominik@tisleric.de

GitHub: https://github.com/Bananenkaiser
