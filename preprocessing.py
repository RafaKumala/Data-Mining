import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

# Download resources if not already done
nltk.download('punkt')

# Function to load stopwords from a file
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())
    return stopwords

# Load stopwords from file
stop_words_path = 'D:/Perkuliahan/Semester 4/Machine Learning/Text Maining/Preprocecing/KataBaku.txt' 
# Ganti dengan path file stopwords Anda
stop_words = load_stopwords(stop_words_path)

# Stemming Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Load CSV file
file_path = 'D:/Perkuliahan/Semester 5/Data Mining/Preprocecing/Contoh.csv' 
# Ganti dengan path file CSV Anda
df = pd.read_csv(file_path)

# Assuming the text column is named 'comment'
df['processed_text'] = df['comment'].apply(preprocess_text)

# Save the processed dataset to a new CSV file
output_path = 'D:/Perkuliahan/Semester 5/Data Mining/Preprocecing/SaveData/Databaru.csv'  
# Ganti dengan path dan nama file untuk menyimpan file CSV yang telah diproses
df.to_csv(output_path, index=False)

print("Preprocessing selesai. Dataset yang telah diproses disimpan di:", output_path)
