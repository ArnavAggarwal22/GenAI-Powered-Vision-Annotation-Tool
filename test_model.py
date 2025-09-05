from transformers import pipeline
from PIL import Image

def test_detection():
    model_id = "google/owlvit-base-patch32"
    detector = pipeline(model=model_id, task="zero-shot-object-detection")
    
    try:
        image = Image.open("test_image.jpeg")
    except FileNotFoundError:
        print("Error: 'test_image.jpg' not found. Please add a test image to your project folder.")
        return

    text_prompts = ["a red car", "a person on a bicycle"]
    predictions = detector(image, candidate_labels=text_prompts)
    
    print("Detections Found:")
    for pred in predictions:
        print(f"- Label: {pred['label']}, Score: {pred['score']:.4f}, Box: {pred['box']}")

if __name__ == "__main__":
    test_detection()