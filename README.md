# GenAI-Powered Vision Annotation Tool

An interactive web application that leverages a Vision Transformer model (Google's OWL-ViT) to perform real-time, zero-shot object detection from natural language prompts. This project demonstrates a modern, conversational approach to data annotation, significantly accelerating the process of creating computer vision datasets.

*(Feel free to record a short GIF of your application in action and replace the placeholder link above.)*

---

## 🎯 The Problem

Traditional data labeling for computer vision is a slow, manual, and restrictive process. Annotators must manually draw bounding boxes and are limited to a predefined list of object classes. This creates a significant bottleneck in the AI development lifecycle.

This tool reimagines that workflow by allowing users to simply "talk" to their images, requesting labels for any object they can describe.

## ✨ Features

-   **Zero-Shot Object Detection:** Identify and label objects that the model was not explicitly trained on.
-   **Natural Language Prompts:** Use simple, comma-separated text descriptions to find objects (e.g., "a red car, a person walking a dog").
-   **Interactive Web Interface:** A user-friendly UI built with Streamlit for easy image upload and real-time results.
-   **Dynamic Annotation:** Bounding boxes and confidence scores are drawn directly onto the image for immediate visual feedback.

## 🛠️ Tech Stack

-   **Language:** Python
-   **AI Model:** Google's OWL-ViT (Vision Transformer)
-   **Core Libraries:**
    -   `streamlit` - For building the interactive web application.
    -   `transformers` (Hugging Face) - For seamless interaction with the pre-trained model.
    -   `Pillow` - For image manipulation and drawing annotations.
    -   `torch` - As the backend deep learning framework.

## 🚀 Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

-   Python 3.8+
-   A [Hugging Face](https://huggingface.co/join) account.

## usage

1.  Open the application in your browser.
2.  Click "Browse files" to upload a `.jpg`, `.jpeg`, or `.png` image.
3.  In the text input box, type a comma-separated list of objects you want to find.
4.  The application will automatically process the image and display the results with bounding boxes in the "Labeled Image" column.
