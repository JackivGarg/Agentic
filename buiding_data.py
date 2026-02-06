import requests
from bs4 import BeautifulSoup
from groq import Groq
import os
import time
from dotenv import load_dotenv  # Added this

# 1. Load the .env file (Same as your working script)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- CONFIGURATION ---
INPUT_FILE = 'urls.txt'
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Your specific categories
CATEGORY_DESC = {
    "admissions": "admission process, eligibility, fees, scholarships",
    "programs": "academic programs, courses, degrees",
    "hostel": "hostel, accommodation, campus life",
    "placements": "placements, recruiters, internships",
    "policies": "refund policy, rules, regulations",
    "general": "about university, overview, misc"
}

# Initialize Groq using the key from .env
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file.")
    exit()
client = Groq(api_key=api_key)

def clean_html(html_content):
    """
    Removes headers, footers, navigation, and returns the main text.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    for tag in soup.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style', 'noscript', 'form', 'iframe']):
        tag.decompose()

    noise_keywords = ['menu', 'sidebar', 'copyright', 'social', 'widget', 'breadcrumb', 'navbar', 'top-bar']
    for k in noise_keywords:
        for tag in soup.find_all(attrs={'class': lambda x: x and k in x.lower()}):
            tag.decompose()
        for tag in soup.find_all(attrs={'id': lambda x: x and k in x.lower()}):
            tag.decompose()

    text = soup.get_text(separator=' ')
    return ' '.join(text.split())

def get_category_from_url(url):
    """
    Sends the URL to Groq with strict instructions to stop the 'general' bias.
    """
    cat_list_str = "\n".join([f"- {key}: {desc}" for key, desc in CATEGORY_DESC.items()])
    
    # Using an aggressive prompt to ensure it doesn't default to general
    prompt = f"""
    SYSTEM: You are a strict URL classifier for a university. 
    TASK: Look at the URL and map it to the most specific category. 
    
    CATEGORIES:
    {cat_list_str}

    URL: "{url}"

    INSTRUCTIONS:
    - If the URL has 'apply', 'fee', or 'scholarship', output 'admissions'.
    - If the URL has 'course', 'curriculum', or 'dept', output 'programs'.
    - If the URL has 'placement' or 'career', output 'placements'.
    - ONLY output 'general' if absolutely no other category fits.

    OUTPUT ONLY THE CATEGORY KEY. NO PUNCTUATION.
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You output only the category key."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=15
        )
        
        category = completion.choices[0].message.content.strip().lower().replace('"', '').replace("'", "")
        
        if category in CATEGORY_DESC:
            return category
        
        for key in CATEGORY_DESC:
            if key in category:
                return key
                
        return "general"
        
    except Exception as e:
        print(f" -> Groq Error: {e}")
        return "general"

def append_to_file(category, url, content):
    filename = f"{category}.txt"
    entry = f"\n{'='*40}\nURL: {url}\n{'='*40}\n{content}\n\n"
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(entry)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"✅ Key loaded. Processing {len(urls)} URLs...")

    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Analyzing: {url}")
        category = get_category_from_url(url)
        print(f"   -> Category: {category}")

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                cleaned_text = clean_html(response.text)
                append_to_file(category, url, cleaned_text)
                print(f"   -> Content saved to {category}.txt")
            else:
                print(f"   -> Failed to fetch (Status: {response.status_code})")

        except Exception as e:
            print(f"   -> Error scraping: {e}")

        time.sleep(1.5)

if __name__ == "__main__":
    main()