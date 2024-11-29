import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
import numpy as np
import wikipedia
# Set the API key and CSE ID directly in the code
API_KEY = 'your_google_api_key'  # Replace with your actual API key
SEARCH_ENGINE_ID = 'your_custom_search_engine_id'  # Replace with your actual CSE ID

if not API_KEY:
    st.error("API Key not found. Please set the API_KEY variable.")
    st.stop()

# Load YOLOv8 model
model = YOLO("last (2).pt")

# Sidebar for confidence level adjustment
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

# Function to search the web using Google Custom Search API
def search_web(query, num_results=3):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={SEARCH_ENGINE_ID}&num={num_results}"
    response = requests.get(search_url)
    data = response.json()

    results = []
    if 'items' in data:
        for item in data['items']:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No description available')
            link = item.get('link', '')
            # Only keep results from Indian websites
            if 'site:.in' in query and 'amazon.in' in link:
                results.append({'title': title, 'snippet': snippet, 'link': link})
            elif 'site:.in' not in query:
                results.append({'title': title, 'snippet': snippet, 'link': link})
        return results
    else:
        st.error("No results found or an error occurred.")
        return []

# Function to get disease information from Wikipedia
def get_disease_info_from_wikipedia(disease_name):
    try:
        # Search for the disease in Wikipedia
        summary = wikipedia.summary(disease_name, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        st.error(f"DisambiguationError: {e}")
        return "Information is ambiguous, please refine your search."
    except wikipedia.exceptions.PageError:
        st.error(f"PageError: No Wikipedia page found for {disease_name}.")
        return "No Wikipedia page found for this disease."
    except Exception as e:
        st.error(f"An error occurred while fetching Wikipedia data: {e}")
        return "An error occurred while fetching Wikipedia data."

# Function to display disease information
def display_disease_info(disease_name):
    # Get and display disease description from Wikipedia
    disease_info = get_disease_info_from_wikipedia(disease_name)
    st.write(f"### Disease Information: {disease_name}")
    st.write(f"**Description**: {disease_info}")

    # Search for prevention and cure information (only from Indian websites)
    prevention_query = f"{disease_name} prevention and cure site:.in"
    prevention_results = search_web(prevention_query)

    # Search for product recommendations (e.g., pesticides, fertilizers) specifically from Amazon India
    product_query = f"{disease_name} pesticides fertilizers site:amazon.in"
    product_results = search_web(product_query)

    # Display prevention and cure information
    if prevention_results:
        st.write("**Prevention and Cure Information:**")
        for result in prevention_results:
            st.write(f"- **Title**: {result['title']}")
            st.write(f"  **Description**: {result['snippet']}")
            st.write(f"  **Reference**: [Read more]({result['link']})")
    else:
        st.write("**Prevention and Cure Information:** No information found.")

    # Display product recommendations
    if product_results:
        st.write("### Recommended Products from Amazon India:")
        for result in product_results:
            st.write(f"- **Title**: {result['title']}")
            st.write(f"  **Description**: {result['snippet']}")
            st.write(f"  **Reference**: [Product Link]({result['link']})")
    else:
        st.write("**Product Recommendations:** No products found.")

# Main Streamlit app
st.title("Plant Disease Detection")

# Image upload
uploaded_file = st.file_uploader("Upload an image of the plant", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform detection
    results = model.predict(np.array(image), conf=confidence_threshold)

    # Display results
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
    labels = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else []
    scores = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else []

    if len(boxes) > 0:
        for i, label in enumerate(labels):
            disease_name = model.names[int(label)]  # Assuming `model.names` contains class names
            st.write(f"### Detected Disease: {disease_name}")

            # Display disease information and product links
            display_disease_info(disease_name)

            st.write("---")
    else:
        st.write("No diseases detected. Please try another image.")
else:
    st.write("Please upload an image to detect plant diseases.")