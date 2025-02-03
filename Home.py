import streamlit as st
import pandas as pd
from PIL import Image
import io
from models.EnsambleModel import EnsambleModel
from scripts.ImageDataset import ImageDataset
from pipeline_degraded.elimianzione.delete_noise import is_image_noisy

ltf = pd.read_csv('ground_truth/foods_names.csv').to_dict()['Food']
models_names = ['resnet18', 'efficientnet', 'vgg16']
models_weights = [0.32075472, 0.33692722, 0.34231806]

em = EnsambleModel(models_name=models_names, models_weights=models_weights)

st.title("🍔📸 Food Classifier")


uploaded_file = st.file_uploader("Carica un'immagine JPG", type=["jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Immagine caricata", use_container_width=True)
    temp_image_path = "temp_uploaded_image.jpg"
    image.save(temp_image_path)
    df = pd.DataFrame({"image_id": [temp_image_path]})  # Il path è già completo
    image_dataset = ImageDataset(df, image_path=None, train=False)  # image_path=None
    is_noisy = is_image_noisy(image)
    if is_noisy:
        st.warning("⚠️ L'immagine potrebbe essere **rumorosa** e non adatta alla classificazione.")
    images_idx, image_labels, confidence = em.predict(image_dataset, lc=None)
    df_labels = pd.read_csv("ground_truth/foods_names.csv", index_col=0)

    # Converti in dizionario
    label_dict = df_labels["Food"].to_dict()
    image_label = int(image_labels[0])  # Use the first element from image_labels
    labels = label_dict[image_label]
    formatted_labels = labels.replace("_", " ").title()
    st.write("### 📊 Risultati della Classificazione:")
    st.write(f"**🍽️ Etichetta Predetta:** {formatted_labels}")
    st.write("**📉 Confidence:**", ", ".join([f"{c:.2f}" for c in confidence]))