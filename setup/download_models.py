import os
import gdown

os.makedirs("models", exist_ok=True)

# Replace these with your actual Google Drive file IDs
model_links = {
    "detection.pt": "https://drive.google.com/uc?export=download&id=1DeGoaR-jpGiWKd_e_WEOM1nalhFIrxhG",
    "tray_classifier.pt": "https://drive.google.com/uc?export=download&id=1dvETWI2nqB0hrM_Xxc_3yIql1z25CUQY",
    "dish_classifier.pt": "https://drive.google.com/uc?export=download&id=1JxJUEph6RSv_ZzPHfNwmTOIK6-mh5Zq3"
}

for name, url in model_links.items():
    output_path = os.path.join("models", name)
    if not os.path.exists(output_path):
        print(f"📥 Downloading {name}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"✅ {name} already exists. Skipping.")
