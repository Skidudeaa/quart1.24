import json
import nltk
import numpy as np
import os
import re
import requests
import shutil
import uuid
import warnings
import syncedlyrics
import io
import mimetypes
import torch
import aiohttp
import aiofiles  # If you're doing asynchronous file operations
import asyncio
import tempfile

from quart import Quart, request, redirect, url_for, render_template, flash, jsonify, Response
#from flask import Flask, request, send_file, Response, jsonify 
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz, process
from m3u8 import M3U8
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
#from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from syncedlyrics import search
from urllib.parse import urljoin
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
#from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import string

# Ensure to adjust the import for 'Summarizer' based on its actual source
#from summarizer import Summarizer  # Uncomment and adjust if needed

# Ensure the required NLTK data is downloaded
#nltk.download("vader_lexicon")
#nltk.download("punkt")
# Check if the 'vader_lexicon' package is downloaded
#if not nltk.data.find("sentiment/vader_lexicon.zip"):
#    nltk.download("vader_lexicon")
# Check if the 'punkt' package is downloaded
#if not nltk.data.find("tokenizers/punkt"):
#    nltk.download("punkt")

# Initialization
GENIUS_API_KEY = "6IJtS4Xta8IPcEPwmC-8YVOXf5Eoc4RHwbhWINDbzomMcFVXQVxbVQapsFxzKewr"
APPLE_MUSIC_API_KEY = "eyJhbGciOiJFUzI1NiIsImtpZCI6IjYyMlcyTVVVV1EiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJVNEdMUUdGTlQzIiwiaWF0IjoxNjk3MjQ4NDQ4LCJleHAiOjE3MTAyMDg0NDh9.XMe-WEuuAJS_LOirXG6yU8CZW1RL6Lw4cwxhc405rvZm_LesEsaLoqNnZ9l_n3SQ0eOqUQEsWXEPNZYJ5wdZXw"

headers = {"Authorization": "Bearer " + APPLE_MUSIC_API_KEY}
warnings.filterwarnings("ignore", category=FutureWarning)





def normalize_text(text):
    """Converts text to lowercase."""
    return text.lower()

def clean_text(text):
    """Removes URLs and text in brackets."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    return text

def process_sentence(sentence):
    """Capitalizes sentences and maintains periods and commas."""
    if '"' in sentence or "“" in sentence or "”" in sentence:
        return sentence.capitalize()
    else:
        # Keep periods and commas
        allowed_chars = string.ascii_letters + string.digits + ' .,'
        return ''.join(char for char in sentence if char in allowed_chars).capitalize()

def preprocess_text(text):
    """Preprocesses the text by normalizing, cleaning, and processing each sentence."""
    text = normalize_text(text)
    text = clean_text(text)
    sentences = sent_tokenize(text)
    processed_sentences = [process_sentence(sentence) for sentence in sentences]
    return ' '.join(processed_sentences)



# Sample text
sample_text = "Rainy days when you’re together but sunny days when you’re apart, you love each other so much that you would rather spend the rainy days together than to have the sunny days alone, even though the sunny weather is preferred by most. So it’s like we will stay together all these rainy days and I’m with you until the weather changes, because the love is that strong. Rainy days when you’re together but sunny days when you’re apart, you love each other so much that you would rather spend the rainy days together than to have the sunny days alone, even though the sunny weather is preferred by most. So it’s like we will stay together all these rainy days and I’m with you until the weather changes, because the love is that strong."
#preprocessed_text = preprocess_text(sample_text)
#print(preprocessed_text)







tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')


def summarize_text(text, min_len=100, max_len=400, length_penalty=2.0, num_beams=6):
    # Ensure the model and tokenizer are on the same device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5ForConditionalGeneration.from_pretrained('t5-large').to(device)
    
    # Encode the text
    input_ids = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=800, truncation=True)

    # Generate the summary
    summary_ids = model.generate(input_ids, 
                                 max_length=max_len, 
                                 min_length=min_len, 
                                 length_penalty=length_penalty, 
                                 num_beams=num_beams, 
                                 no_repeat_ngram_size=4,
                                 early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)





# Existing model loaded at the start of your script
model = SentenceTransformer('all-MiniLM-L6-v2')

# Existing function for individual text embedding
def get_embedding(text):
    # Generate the sentence embedding
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding

# New function for batch processing
def get_embeddings(texts, batch_size=64):
    # Generate embeddings for a batch of texts
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True)
    return embeddings

# Existing function to calculate semantic similarity
def get_semantic_similarity(embedding1, embedding2):
    # Ensure both embeddings are 1-D before calculating cosine similarity
    if embedding1.ndim > 1:
        embedding1 = embedding1.squeeze()
    if embedding2.ndim > 1:
        embedding2 = embedding2.squeeze()
    return 1 - cosine(embedding1, embedding2)


def compute_tfidf_vectorizer(documents):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
        min_df=0.01,            # Minimum document frequency
        max_df=0.85,         # Exclude terms that are too frequent
        stop_words=None,     # Do not automatically remove stop words
        max_features=5000    # Limit the number of features
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer


# Function to calculate cosine similarity
def calculate_cosine_similarity(tfidf_matrix, index, vectorizer, document):
    doc_vector = vectorizer.transform([document])
    similarity = cosine_similarity(tfidf_matrix[index : index + 1], doc_vector)
    return similarity[0][0]



# Genius API Interactions for Song ID
# Genius API Interactions for Song ID
async def get_song_id(search_term, artist_name, api_key):
    url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": f"{search_term} {artist_name}"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            # Check if the request was successful
            if response.status != 200:
                print(f"Failed to get song ID. HTTP status code: {response.status}")
                return None

            response_json = await response.json()

            # Check if the response contains the expected keys
            if "response" not in response_json or "hits" not in response_json["response"] or not response_json["response"]["hits"]:
                print(f"No results found for '{search_term} by {artist_name}'. Please try again with a different search term.")
                return None

            # Return the first song's ID
            song_id = response_json["response"]["hits"][0]["result"]["id"]
            return song_id



async def get_song_details(song_id, api_key):
    url = f"https://api.genius.com/songs/{song_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            # Check if the request was successful
            if response.status != 200:
                print(f"Failed to get song details. HTTP status code: {response.status}")
                return None

            response_json = await response.json()

            # Check if the response contains the expected keys
            if "response" not in response_json or "song" not in response_json["response"]:
                print("Unexpected response format from Genius API.")
                return None

            song_details = response_json["response"]["song"]

            # Check if the song details contain the expected keys
            if "description" not in song_details or "dom" not in song_details["description"] or "children" not in song_details["description"]["dom"]:
                print("Unexpected song details format from Genius API.")
                return None

            # Use parse_description to handle the description parsing
            song_description_dom = song_details["description"]["dom"]["children"]
            song_description = parse_description(song_description_dom)

            # Summarize the song description asynchronously
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                song_description = await loop.run_in_executor(pool, summarize_text, song_description.strip())

            song_details["description"] = song_description
            return song_details


def parse_description(description):
    readable_description = []
    for item in description:
        if "children" in item:
            for child in item["children"]:
                if isinstance(child, str):
                    readable_description.append(child)
                elif isinstance(child, dict) and "children" in child:
                    readable_description.extend(parse_description(child["children"]))
    return " ".join(readable_description)



async def get_referents_and_annotations(song_id, api_key, lyrics_data, limit=20):
    url = f"https://api.genius.com/referents?song_id={song_id}&text_format=plain"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            referents = (await response.json())["response"]["referents"]

    processed_annotations = []

    for referent in referents:
        fragment = referent["fragment"]
        annotations = referent["annotations"]

        for annotation in annotations:
            if "body" in annotation and "plain" in annotation["body"]:
                annotation_text = annotation["body"]["plain"]
            else:
                continue  # Skip if plain text is not available

            # Assuming find_matching_lyric_timestamp is a synchronous function
            timestamp = find_matching_lyric_timestamp(fragment, lyrics_data, threshold=60)

            processed_annotation = {
                "fragment": fragment,
                "annotation": annotation_text,
                "timestamp": timestamp
            }
            processed_annotations.append(processed_annotation)

    # Limit the number of annotations if necessary
    processed_annotations = processed_annotations[:min(limit, len(processed_annotations))]
    return processed_annotations



def find_matching_lyric_timestamp(fragment, lyrics_data, threshold=60):
    # Extract the list of lyrics from the lyrics_data dictionary
    lyrics_list = [lyric_entry['lyric'] for lyric_entry in lyrics_data['lyrics'] if 'lyric' in lyric_entry]

    # Create a TfidfVectorizer and fit it on the lyrics list
    tfidf = TfidfVectorizer(max_df=0.8, min_df=0.01).fit_transform([fragment] + lyrics_list)  # Adjust vectorizer parameters


    # Compute the cosine similarity between the fragment and each lyric
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()

    # Find the index of the best match
    best_match_index = np.argmax(cosine_similarities[1:]) + 1  # Skip the first element, which is the fragment itself

    # Check if the best match score is above the threshold
    if cosine_similarities[best_match_index] >= threshold:
        # Find the corresponding entry in lyrics_data['lyrics']
        for lyric_entry in lyrics_data['lyrics']:
            if lyric_entry['lyric'] == lyrics_list[best_match_index - 1]:  # Subtract 1 because we added the fragment at the beginning
                return lyric_entry['timestamp']

    return None  # Return None if no suitable match is found or if match is None

def calculate_scores(params):
    tfidf_matrix, lyric_index, vectorizer, annotation = params
    return calculate_cosine_similarity(tfidf_matrix, lyric_index, vectorizer, annotation)


def adjust_annotation_clusters(annotations, min_distance=10, max_shift=20, cluster_time=7):
    if not annotations:
        return []

    annotations.sort(key=lambda x: x["timestamp"])
    clusters = []
    current_cluster = [annotations[0]]

    for i in range(1, len(annotations)):
        current_cluster.append(annotations[i])
        if len(current_cluster) >= 3:
            average_time = (current_cluster[-1]["timestamp"] - current_cluster[0]["timestamp"]) / (len(current_cluster) - 1)
            if average_time > cluster_time:
                current_cluster.pop()
                clusters.append(current_cluster)
                current_cluster = [annotations[i]]

    if len(current_cluster) >= 3:
        clusters.append(current_cluster)

    for cluster in clusters:
        if is_cluster_adjustable(cluster, max_shift):
            start_time = cluster[0]["timestamp"]
            time_increment = min((cluster[-1]["timestamp"] - start_time) / (len(cluster) - 1), max_shift)
            for i, ann in enumerate(cluster):
                new_timestamp = start_time + i * time_increment
                if not is_timestamp_valid(annotations, new_timestamp, min_distance):
                    new_timestamp = find_new_timestamp(annotations, min_distance)
                if new_timestamp is None:
                    print(
                        f"No suitable gap found for annotation {annotation['id']}. Disregarding."
                    )
                else:
                    ann["timestamp"] = new_timestamp
        else:
            for annotation in cluster:
                new_timestamp = find_new_timestamp(annotations, min_distance)
                if new_timestamp is None:
                    print(
                        f"No suitable gap found for annotation {annotation['id']}. Disregarding."
                    )
                else:
                    annotation["timestamp"] = new_timestamp

    return annotations

def is_timestamp_valid(annotations, timestamp, min_distance):
    for ann in annotations:
        if abs(ann["timestamp"] - timestamp) < min_distance:
            return False
    return True


def is_cluster_adjustable(cluster, max_shift):
    return cluster[-1]["timestamp"] - cluster[0]["timestamp"] <= max_shift * (
        len(cluster) - 1
    )


def find_new_timestamp(sorted_annotations, min_distance):
    largest_gap = 0
    best_position = None

    # Iterate through annotations to find the largest gap
    for i in range(len(sorted_annotations) - 1):
        gap = sorted_annotations[i + 1]["timestamp"] - sorted_annotations[i]["timestamp"]
        if gap > largest_gap:
            largest_gap = gap
            # Place new timestamp in the middle of the largest gap
            best_position = sorted_annotations[i]["timestamp"] + gap / 2

    # Return the best position if it respects the min_distance from both ends of the gap
    if best_position is not None:
        prev_timestamp = sorted_annotations[i]["timestamp"]
        next_timestamp = sorted_annotations[i + 1]["timestamp"]

        if (
            best_position - prev_timestamp >= min_distance
            and next_timestamp - best_position >= min_distance
        ):
            return best_position

    # If no suitable position is found, return None
    return None



def handle_single_annotation_case(lyrics_data, annotations_data):
    # If there's only one annotation, process it accordingly
    if len(annotations_data) == 1:
        annotation = annotations_data[0]
        # Here you can either return the annotation as is, or process it as needed
        # For example, returning the annotation with its existing timestamp
        return [lyrics_data['lyrics']] if lyrics_data and 'lyrics' in lyrics_data else [], [annotation]
    else:
        # If there are no annotations, return an appropriate response
        placeholder_annotation = {
            "id": str(uuid.uuid4()),
            "annotation": "No annotations available.",
            "lyric": "N/A",
            "timestamp": 0.0
        }
        return [], [placeholder_annotation]



def merge_similar_annotations(lyrics_data, annotations_data, max_clusters=10, window_size=3, song_duration=3000, time_threshold=8):
    # Check if annotations_data is empty or insufficient
    if not annotations_data or len(annotations_data) < 2:
        print("Insufficient annotations data. Continuing with the rest of the script.")
        placeholder_annotation = {
            "id": str(uuid.uuid4()),
            "annotation": "No annotations available.",
            "lyric": "N/A",
            "timestamp": 0.0
        }
        return [lyrics_data['lyrics']] if lyrics_data and 'lyrics' in lyrics_data else [], [placeholder_annotation]

    # Generate embeddings for all annotations
    annotation_texts = [ann["annotation"] for ann in annotations_data]
    annotation_embeddings = get_embeddings(annotation_texts)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(ann["annotation"])["compound"] for ann in annotations_data]
    sentiment_scores = MinMaxScaler().fit_transform(np.array(sentiment_scores).reshape(-1, 1))

    # Combine features for clustering
    combined_features = np.concatenate([annotation_embeddings, sentiment_scores], axis=1)

    # Determine the number of clusters
    actual_max_clusters = min(max_clusters, len(annotations_data) - 1)
    num_clusters = 4 if len(annotations_data) > 3 else len(annotations_data)
    best_score = -1
    for n_clusters in range(5, actual_max_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(combined_features)
        if len(clustering.labels_) > 1:
            score = silhouette_score(combined_features, clustering.labels_)
            if score > best_score:
                best_score = score
                num_clusters = n_clusters

    clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(combined_features)

    # Merging annotations
    merged_annotations = []
    assigned_timestamps = []  # Initialize assigned_timestamps
    for cluster_id in set(clustering.labels_):
        cluster_annotations = [annotations_data[i] for i in range(len(annotations_data)) if clustering.labels_[i] == cluster_id]
        combined_annotation_text = " ".join(ann["annotation"] for ann in cluster_annotations)

        # Print the combined annotation text before summarization
        #print(f"Combined Annotation Text for Cluster {cluster_id}: {combined_annotation_text}")

        summarized_annotation = summarize_text(combined_annotation_text, min_len=100, max_len=450, length_penalty=3.0, num_beams=6)
        
        best_timestamp = find_best_timestamp_based_on_content(lyrics_data, cluster_annotations, assigned_timestamps)
        if best_timestamp is None:
            best_timestamp = find_farthest_timestamp(merged_annotations, cluster_annotations, song_duration)
        best_timestamp = best_timestamp if best_timestamp is not None else 0.0

        chosen_lyric = find_lyric_for_timestamp(lyrics_data, best_timestamp)

        merged_annotation = {
            "id": str(uuid.uuid4()),
            "annotation": summarized_annotation,
            "lyric": chosen_lyric,
            "timestamp": best_timestamp
        }
        merged_annotations.append(merged_annotation)

    # Adjust the timestamp of a merged annotation if it's within the 'time_threshold' of another merged annotation
    for i, annotation in enumerate(merged_annotations):
        if annotation["timestamp"] is None:
            annotation["timestamp"] = find_farthest_timestamp(merged_annotations, cluster_annotations, song_duration)
            if annotation["timestamp"] is None:
                print(f"Warning: No suitable timestamp found for annotation {annotation['id']}.")
            else:
                annotation["lyric"] = find_lyric_for_timestamp(lyrics_data, annotation["timestamp"])

    # Check if any two annotations are too close together
    for i, annotation in enumerate(merged_annotations):
        for j in range(i + 1, len(merged_annotations)):
            if abs(annotation["timestamp"] - merged_annotations[j]["timestamp"]) <= time_threshold:
                # If two annotations are too close, move one of them to a new timestamp
                new_timestamp = find_farthest_timestamp(merged_annotations, [merged_annotations[j]], song_duration)
                if new_timestamp is not None:
                    merged_annotations[j]["timestamp"] = new_timestamp
                    merged_annotations[j]["lyric"] = find_lyric_for_timestamp(lyrics_data, new_timestamp)

    return [lyrics_data['lyrics']] if lyrics_data and 'lyrics' in lyrics_data else [], merged_annotations



def find_best_timestamp_based_on_content(lyrics_data, cluster_annotations, assigned_timestamps, semantic_weight=0.7, sentiment_weight=0.3, min_threshold=10):
    # Generate embeddings for the merged annotation text
    merged_annotation_text = " ".join(ann["annotation"] for ann in cluster_annotations)
    annotation_embedding = get_embedding(merged_annotation_text)

    # Prepare all lyrics for batch processing
    lyrics_texts = [lyric['lyric'] for lyric in lyrics_data['lyrics']]
    lyrics_embeddings = get_embeddings(lyrics_texts)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(lyric)["compound"] for lyric in lyrics_texts]

    best_score = -1
    best_timestamp = None

    # Iterate over each lyric embedding to find the best match
    for i, (lyric_embedding, sentiment_score) in enumerate(zip(lyrics_embeddings, sentiment_scores)):
        semantic_similarity = get_semantic_similarity(annotation_embedding, lyric_embedding)
        # Normalize sentiment score to be between 0 and 1
        normalized_sentiment_score = (sentiment_score + 1) / 2

        # Calculate the final score as a weighted sum of semantic similarity and sentiment score
        final_score = semantic_weight * semantic_similarity + sentiment_weight * normalized_sentiment_score

        current_timestamp = lyrics_data['lyrics'][i].get("timestamp")
        if current_timestamp is not None:
            # Check if the current timestamp is too close to any assigned timestamp
            if any(abs(current_timestamp - timestamp) < min_threshold for timestamp in assigned_timestamps):
                continue

            # If the current score is higher, update the best score and timestamp
            if final_score > best_score:
                best_score = final_score
                best_timestamp = current_timestamp

    # Add the best timestamp to the list of assigned timestamps
    if best_timestamp is not None:
        assigned_timestamps.append(best_timestamp)

    return best_timestamp




def find_farthest_timestamp(all_annotations, cluster_annotations, song_duration):
    timestamps = [ann['timestamp'] for ann in all_annotations if ann not in cluster_annotations]
    timestamps.append(song_duration)  # Include song duration as a possible timestamp
    cluster_timestamps = [ann['timestamp'] for ann in cluster_annotations]

    max_distance = -1
    farthest_timestamp = None
    for timestamp in cluster_timestamps:
        min_distance = min(abs(timestamp - other_timestamp) for other_timestamp in timestamps)
        if min_distance > max_distance:
            max_distance = min_distance
            farthest_timestamp = timestamp

    return farthest_timestamp


def find_lyric_for_timestamp(lyrics_data, timestamp):
    for lyric_entry in lyrics_data['lyrics']:
        if lyric_entry["timestamp"] == timestamp:
            return lyric_entry["lyric"]
    return "Lyric not found"

'''
def summarize_text(text, num_sentences):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)
'''





async def fetch_lyrics_with_syncedlyrics(artist_name, track_name):
    # Search for the synced lyrics
    lrc = syncedlyrics.search(
        f"{track_name} {artist_name}"
    )
    # Process the lyrics
    lyrics_data = None
    if lrc:
        parsed_lyrics = [
            {
                "id": str(uuid.uuid4()),
                "timestamp": round(
                    float(ts[1:].split(":")[0]) * 60 + float(ts[1:].split(":")[1]), 1
                ),
                "lyric": l,
            }
            for line in (
                line
                for line in lrc.split("\n")
                if line and "] " in line and len(line.split("] ")) == 2
            )
            for ts, l in [line.split("] ")]
        ]
        lyrics_data = {"lyrics": parsed_lyrics}

    # Asynchronous file operation (if applicable)
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        temp_name = temp.name
        async with aiofiles.open(temp_name, "w") as file:
            await file.write(lrc)

    return lyrics_data



# Utility Functions
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()




async def get_webpage_content(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()
    except aiohttp.ClientError as e:
        print(f"Error occurred while fetching page: {e}")
        return None

async def fetch_variant_playlist_url(playlist_url):
    content = await get_webpage_content(playlist_url)
    if content:
        playlists = M3U8(content).playlists
        if playlists:
            playlists.sort(key=lambda p: abs(p.stream_info.resolution[0] - 720))
            return urljoin(playlist_url, playlists[0].uri)
    print("No variant playlist found.")
    return None



async def fetch_playlist_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.text()
                match = re.search(r'src="h([^"]*)', content)
                if match:
                    return "h" + match.group(1)
            # print("No video URL found.")
            return None

async def fetch_segment_urls(variant_playlist_url):
    content = await get_webpage_content(variant_playlist_url)
    if content:
        return [
            urljoin(variant_playlist_url, segment.uri)
            for segment in M3U8(content).segments
        ]
    return None


# Functions for Apple Music interactions

async def download_image(image_url, image_path):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status == 200:
                # Create the finalOutput directory if it doesn't exist
                output_dir = os.path.join(os.getcwd(), "finalOutput")
                os.makedirs(output_dir, exist_ok=True)

                # Save the image in the finalOutput directory asynchronously
                async with aiofiles.open(os.path.join(output_dir, image_path), mode='wb') as out_file:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        await out_file.write(chunk)




async def search_song(song_title, artist_name, developer_token):
    headers = {"Authorization": "Bearer " + developer_token}
    params = {"term": song_title + " " + artist_name, "limit": "5", "types": "songs"}

    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.music.apple.com/v1/catalog/us/search", headers=headers, params=params) as response:
            json_response = await response.json()

    # Check if any song data is returned
    if "songs" not in json_response["results"]:
        print("No songs found.")
        return None, None, None

    song_data = json_response["results"]["songs"]["data"]
    bg_color, text_colors, song_duration = None, None, None

    # Assuming fetch_playlist_url, fetch_variant_playlist_url, fetch_segment_urls are also async
    for song in song_data:
        song_url = song["attributes"]["url"]
        playlist_url = await fetch_playlist_url(song_url)

        if playlist_url:
            variant_playlist_url = await fetch_variant_playlist_url(playlist_url)
            if variant_playlist_url:
                segment_urls = await fetch_segment_urls(variant_playlist_url)
                if segment_urls:
                    await download_video_segments(segment_urls, "video_segments")
                    break  # Stop once a video is downloaded

    # Assuming download_image is also async
    artwork_url = song_data[0]["attributes"]["artwork"]["url"].replace("{w}", "3000").replace("{h}", "3000")
    await download_image(artwork_url, "artwork.jpg")

    bg_color = song_data[0]["attributes"]["artwork"]["bgColor"]
    text_colors = {
        "textColor1": song_data[0]["attributes"]["artwork"]["textColor1"],
        "textColor2": song_data[0]["attributes"]["artwork"]["textColor2"],
        "textColor3": song_data[0]["attributes"]["artwork"]["textColor3"],
        "textColor4": song_data[0]["attributes"]["artwork"]["textColor4"],
    }

    song_duration = song_data[0]["attributes"]["durationInMillis"]
    return bg_color, text_colors, song_duration



async def download_video_segments(segment_urls, video_dir):
    output_dir = os.path.join(os.getcwd(), "finalOutput")
    os.makedirs(output_dir, exist_ok=True)

    segment_url = segment_urls[0]  # Get the first segment URL

    async with aiohttp.ClientSession() as session:
        async with session.get(segment_url) as response:
            if response.status == 200:
                content = await response.read()
                async with aiofiles.open(os.path.join(output_dir, f"AnimatedArt.mp4"), "wb") as file:
                    await file.write(content)
                print("AnimatedArt downloaded.")
            else:
                print(f"No AnimatedArt. Status code: {response.status}")



#app = Flask(__name__)
app = Quart(__name__)

# Configurations
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['USE_X_SENDFILE'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/process_song", methods=["POST"])
async def process_song():
    response_data = b""
    # Delete old files in the output directory at the start of each request
    output_dir = 'finalOutput'
    for filename in ['final_1.json', 'AnimatedArt.mp4', 'artwork.jpg']:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    data =await request.get_json()  # Get data sent to the route
    song_title =  data.get("song_title")
    artist_name =  data.get("artist_name")

    ## Async function calls with 'await'
    song_id = await get_song_id(song_title, artist_name, GENIUS_API_KEY)
    song_data = await get_song_details(song_id, GENIUS_API_KEY)
    lyrics_data = await fetch_lyrics_with_syncedlyrics(artist_name, song_title)
    annotations_data = await get_referents_and_annotations(song_id, GENIUS_API_KEY, lyrics_data, 20)
    
    print(f"song_id: {song_id}, song data recieved")
          
    # Synchronous function call
    try:
        lyrics_with_timestamps, annotations_with_timestamps = merge_similar_annotations(lyrics_data, annotations_data, 10) 
    except Exception as e:
        print(f"An error occurred while merging annotations: {e}")
        lyrics_with_timestamps, annotations_with_timestamps = lyrics_data, []

    # Flatten the list of lists into a single list
    flattened_lyrics_with_timestamps = [item for sublist in lyrics_with_timestamps for item in sublist]

    # Log the flattened data
    #print("flattened_lyrics_with_timestamps data:", flattened_lyrics_with_timestamps)

    # Proper async function call
    bg_color, text_colors, song_duration = await search_song(song_title, artist_name, APPLE_MUSIC_API_KEY)


    # Handling the possibility of 'album' being None
    album_name = song_data.get("album", {}).get("name", "") if song_data and song_data.get("album") else ""

    print("json has been created...")
    # Generate final JSON
    #print("lyrics_with_timestamps data:", lyrics_with_timestamps)
    # Generate final JSON
    final_1 = {
        "title": song_data.get("title", ""),
        "artist": song_data.get("primary_artist", {}).get("name", ""),
        "album": album_name,
        "release_date": song_data.get("release_date", "") if song_data and song_data.get("release_date") else "",
        "description": song_data.get("description", ""),   
        "bgColor": bg_color,
        "textColors": text_colors,
        "songDuration": song_duration,
        "lyrics_with_timestamps": flattened_lyrics_with_timestamps,  # Use the flattened list here
        "annotations_with_timestamps": annotations_with_timestamps,
    }
    # Save the JSON data to a file asynchronously
    json_file_path = os.path.join(output_dir, 'final_1.json')
    async with aiofiles.open(json_file_path, 'w') as json_file:
        await json_file.write(json.dumps(final_1))

    # Check if other files exist and read them into memory asynchronously
    files = {}
    for filename in ['final_1.json', 'AnimatedArt.mp4', 'artwork.jpg']:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            async with aiofiles.open(file_path, 'rb') as file:
                files[filename] = await file.read()
        else:
            print(f"Warning: File {filename} not found in {output_dir}")


    # Create multipart response
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    response = Response(mimetype='multipart/form-data; boundary=' + boundary)

    # Initialize response_data as a bytes object
    response_data = b""

    for filename, filedata in files.items():
        mimetype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    
        # Convert string parts to bytes and concatenate
        response_data += f"--{boundary}\r\n".encode('utf-8')
        response_data += f"Content-Disposition: form-data; name=\"{filename}\"; filename=\"{filename}\"\r\n".encode('utf-8')
        response_data += f"Content-Type: {mimetype}\r\n\r\n".encode('utf-8')
    
        # Append the binary data directly
        response_data += filedata
        response_data += b"\r\n"

    # Final boundary
    response_data += f"--{boundary}--\r\n".encode('utf-8')

    # Set the response data
    response.set_data(response_data)

    return response

@app.route('/')
async def index():
    return 'Welcome to the Quart Server!'


#if __name__ == '__main__':
app.run(host='0.0.0.0')

   # config = Config()
    #bind = ["0.0.0.0:8000"]
    #debug = False

    

#hypercorn application:app --bind 0.0.0.0:5000 --reload


    
    
'''
def preprocess_text(text):
    sentences = sent_tokenize(text)
    return '. '.join(sentences)

def summarize_text(text, num_sentences):
    # Preprocess the text. Might not need this but could be helping break up text into formated sentences?
    text = preprocess_text(text)
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)
'''    


'''
def parse_description(description):
    readable_description = []
    for item in description:
        if "children" in item:
            for child in item["children"]:
                if isinstance(child, str):
                    readable_description.append(child)
                elif isinstance(child, dict) and "children" in child:
                    readable_description.extend(parse_description(child["children"]))
    return " ".join(readable_description)
'''


    
'''
This application is a music analysis tool that enhances your song listening experience. It works behind the scenes to gather and process a wealth of information about your favorite songs.

When you input a song title and artist name, the application springs into action. It reaches out to music databases like Genius and Apple Music to gather data about the song. It fetches the song's lyrics and even the annotations or explanations associated with different parts of the song.

But it doesn't stop there. The application also analyzes the lyrics and annotations, grouping similar ones together. This helps to provide a more coherent and meaningful understanding of the song.

In addition to the textual data, the application also fetches visual elements related to the song. It downloads the album artwork and even fetches an animated version if available. It also gathers color data related to the artwork, which can be used to create a visually appealing and consistent user interface on the GUI side of the app.

Finally, the application packages all this data into a neat package (a JSON file) and prepares it for delivery. It also checks for any additional files (like the animated artwork) and includes them in the package.

In essence, this application works like a backstage crew at a concert, doing all the heavy lifting to ensure that you, the user, have a seamless and enriched music experience.
'''