import sys
import re
import requests
from label_list import labels
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def get_model_inputs():
    while True:
        title = input('\n✨ 🛸   Enter the title of the image and press enter\n').strip()
        url = input('\n✨ 🛸   Paste the URL of the image and press enter\n').strip()
        if not title:
            print('\n ❌ Title cannot be empty. Please try again.')
            continue
        if not url:
            print('\n ❌ URL cannot be empty. Please try again.')
            continue
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code != 200:
                print(f'\n ❌ URL could not be reached (status code: {response.status_code}). Please double-check the URL and try again.')
                continue
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                print('\n ❌ URL does not point to an image. Please provide a direct image URL.')
                continue
            # Try to open with PIL to confirm it's a valid image
            try:
                img = Image.open(response.raw)
                img.verify()  # Verify image integrity
            except Exception:
                print('\n ❌ The content at the URL is not a valid image. Please try again.')
                continue
        except Exception as e:
            print(f'\n ❌ Error fetching the URL: {e}. Please try again.')
            continue
        print(f'\n✨ 🛸   Photo Submitted: "{title}"\n')
        print('\n🤠 📝   Checking your image...\n')
        inputUrls = [
            {
                'Title': title,
                'link': url,
            }
        ]
        return [inputUrls, labels]

def call_model(urls, labels):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
    images = [''] * 1
    for idx, url in enumerate(urls):
        images[idx] = Image.open(requests.get(url['link'], stream=True).raw)
    inputs = processor(text=labels, images=images, return_tensors="pt", padding=True)
    return model(**inputs)

def format_outputs(outputs):
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    # Detach the grad function from tensor object
    detached_probs = probs.detach()
    # Remove the tensor object and just return the matrix
    # The 1st element is our percentages
    return detached_probs.numpy()

def print_outputs(outputs):
    print('Accuracy of provided classification labels:')
    for resultRowIndex, row in enumerate(outputs):
        for probability_idx, prob in enumerate(row):
            print(str(probability_idx+1) + ")", labels[probability_idx], "-", format(prob, ".2%"))

if __name__ == '__main__':
    [urls, labels] = get_model_inputs()
    tensor_outputs = call_model(urls, labels)
    non_tensor_output = format_outputs(tensor_outputs)
    print_outputs(non_tensor_output)
