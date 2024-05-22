import requests
import re
import html
import os
import subprocess
import whisper
from tqdm import tqdm

# Function to read URLs from a file
def read_urls_from_file(filename):
    with open(filename, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    return urls

# Read URLs from wdis_urls.txt
urls = read_urls_from_file('wdis_urls.txt')

# User agent string
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
}

# Function to get the content of a URL
def get_url_content(url):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

# Function to extract and decode master.m3u8 links from the content
def extract_master_m3u8_links(content):
    matches = re.findall(r'\"hlsUrl\":\"(https:\\u002F\\u002F[^\"]+?master\.m3u8[^\"]*?)\"', content)
    decoded_links = [html.unescape(link.replace('\\u002F', '/')) for link in matches]
    return decoded_links

# Function to get the content of a master.m3u8 file
def get_master_m3u8_content(url):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

# Function to extract URL with the lowest resolution
def extract_lowest_resolution_url(content):
    # Find all resolution lines and corresponding URLs
    streams = re.findall(r'#EXT-X-STREAM-INF:.*RESOLUTION=(\d+x\d+).*?\n(https?://[^\s]+)', content)
    if not streams:
        return None

    # Find the stream with the lowest resolution
    streams.sort(key=lambda x: tuple(map(int, x[0].split('x'))))
    return streams[0][1]  # Return the URL with the lowest resolution

# Function to extract the episode name from the URL
def extract_episode_name(url):
    match = re.search(r'episodes/([^/]+)', url)
    if match:
        return match.group(1).replace('-', ' ').title()
    return "Unknown Episode"

# Ensure the audio and transcript directories exist
audio_dir = "audio"
transcript_dir = "transcript"
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(transcript_dir, exist_ok=True)

# Load Whisper model
model = whisper.load_model("base")

# Main script
master_m3u8_links = []
episodes = {}

print("Step 1: Extracting master.m3u8 links from each URL...")
# Step 1: Get master.m3u8 links from each URL
for idx, url in enumerate(tqdm(urls, desc="Extracting links")):
    content = get_url_content(url)
    episode_name = f"{idx + 1:03d} - {extract_episode_name(url)}.mp3"
    episodes[episode_name] = extract_master_m3u8_links(content)

print("Step 2: Extracting URL with the lowest resolution from each master.m3u8 link...")
# Step 2: Get URL with the lowest resolution from each master.m3u8 link
lowest_resolution_urls = []

for episode, links in tqdm(episodes.items(), desc="Extracting resolutions"):
    for link in links:
        m3u8_content = get_master_m3u8_content(link)
        lowest_resolution_url = extract_lowest_resolution_url(m3u8_content)
        if lowest_resolution_url:
            lowest_resolution_urls.append((episode, lowest_resolution_url))

print("Step 3: Downloading each URL as an MP3...")
# Step 3: Download each URL as an MP3 using ffmpeg if the file does not already exist
for episode, url in tqdm(lowest_resolution_urls, desc="Downloading MP3s"):
    episode_path = os.path.join(audio_dir, episode)
    if not os.path.exists(episode_path):
        print(f"Downloading {episode} from {url}")
        subprocess.run(['ffmpeg', '-i', url, '-q:a', '0', '-map', 'a', episode_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"{episode} already exists. Skipping download.")

print("Step 4: Generating transcripts for each MP3 file using Whisper...")
# Step 4: Generate transcripts for each MP3 file using Whisper
all_transcripts = []

for episode in tqdm(episodes.keys(), desc="Transcribing MP3s"):
    episode_path = os.path.join(audio_dir, episode)
    transcript_path = os.path.join(transcript_dir, episode.replace('.mp3', '.txt'))
    if not os.path.exists(transcript_path):
        print(f"Transcribing {episode_path}")
        result = model.transcribe(episode_path)
        transcript_text = result["text"]
        with open(transcript_path, 'w') as f:
            f.write(transcript_text)
    else:
        print(f"Transcript for {episode} already exists. Skipping transcription.")
        with open(transcript_path, 'r') as f:
            transcript_text = f.read()
    
    all_transcripts.append(f"Video: {episode}\nTranscript: {transcript_text}\n")

print("Step 5: Generating consolidated transcript file...")
# Step 5: Save all transcripts to a single file
consolidated_transcript_path = os.path.join(transcript_dir, 'consolidated_transcripts.txt')
with open(consolidated_transcript_path, 'w') as f:
    f.writelines(all_transcripts)

print("Process complete.")
