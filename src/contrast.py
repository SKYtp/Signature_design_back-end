import base64
import io
from PIL import Image, ImageEnhance

def increase_contrast(base64_string, factor=1.5):
    # Step 1: Decode base64 to image
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))

    # Step 2: Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(factor)  # Increase contrast by the given factor

    # Step 3: Convert back to base64
    buffered = io.BytesIO()
    enhanced_image.save(buffered, format="PNG")  # Change format if needed
    return base64.b64encode(buffered.getvalue()).decode()
