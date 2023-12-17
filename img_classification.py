import keras
from PIL import Image, ImageOps
import numpy as np

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Image sizing and resizing
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.LANCZOS)

    # Convert grayscale to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Run the inference
    prediction = model.predict(data)
    return np.argmax(prediction)  # return position of the highest probability