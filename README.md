# UFO Classifier Using Pre-trained CLIP Models 🛸

A small Python application that uses AI to classify images and detect any UFO-related phenomena. Upload an image URL and get predictions on what the image might contain!

This app uses a pre-trained AI model (CLIP) to analyze images and classify them into different categories. Simply provide an image title and URL, and the app will tell you how likely the image contains various objects or scenes.

## Technical Details

This app uses:

- **CLIP Model**: A pre-trained AI model from OpenAI that can understand both images and text
- **PIL (Python Imaging Library)**: For image processing
- **Transformers**: For loading and running the AI model
- **Requests**: For downloading images from URLs

## Prerequisites

Before you start, make sure you have:

- **Python 3.8 or higher** installed on your computer
- **Git** (to clone the repository)

## Installation Steps

### Step 1: Clone the Repository

Open your command prompt/terminal and navigate to where you want to store the project. Then run:

```bash
git clone https://github.com/Thenvir/UFO-Classification-Model.git
cd UFO-Classification-Model
```

### Step 2: Install Python Dependencies

The app needs several Python packages to work. Install them by running:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

Once the installation is complete, start the app with:

```bash
python ufo-app.py
```

## How to Use the App

1. **Enter Image Title**: When prompted, type a descriptive title for your image (e.g., "Mysterious light in the sky")

2. **Enter Image URL**: Paste the direct URL to an image you want to analyze

   - The URL should end with an image extension like `.jpg`, `.png`, `.gif`, etc.
   - Make sure the URL is publicly accessible

3. **Wait for Analysis**: The app will download and analyze your image using AI

4. **View Results**: The app will show you percentages for different categories that might describe your image

## Example Usage

```
✨ 🛸   Enter the title of the image and press enter
Mysterious light in the sky

✨ 🛸   Paste the URL of the image and press enter
https://example.com/my-ufo-image.jpg

🤠 📝   Checking your image...

Accuracy of provided classification labels:
1) airplane - 15.23%
2) bird - 8.45%
3) car - 2.12%
...
```

## Customizing Labels

You can easily add your own custom labels to make the app classify images into categories that interest you!

### How to Add Custom Labels

1. **Open the label file**: Open `label_list.py` in any text editor

2. **Add your labels**: Add new labels to the `labels` list. Each label should be a descriptive phrase that starts with "a photo of" or "an image of"

3. **Save the file**: Save your changes

4. **Restart the app**: Run `python ufo-app.py` again to use your new labels

### Example: Adding UFO-Related Labels

```python
labels = [
    ...default list here
    "a photo of a triangular UFO",
    "a photo of a mettalic sphere UFO"
]
```

### Tips for Good Labels

- **Be specific**: "a photo of a flying saucer" is better than "a photo of a UFO"
- **Use descriptive phrases**: "a photo of a mysterious light in the sky"
- **Include variations**: Add both "a photo of an alien" and "a photo of an extraterrestrial being"
- **Keep it simple**: Avoid overly complex descriptions
- **Test your labels**: Try different images to see how well your custom labels work

## Troubleshooting

### Slow Performance

- The first run might be slow as it downloads the AI model
- Subsequent runs will be faster
- Make sure you have a stable internet connection

### Image URL Issues

- Make sure the URL is a direct link to an image file
- The image should be publicly accessible (not behind a login)
- Try copying the image URL from your browser's address bar when viewing the image

### "pip not found" Error

If you get an error saying `pip` is not found, try:

```bash
python -m pip install -r requirements.txt
```

### "Module not found" Error

If you get errors about missing modules after installation, try:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
