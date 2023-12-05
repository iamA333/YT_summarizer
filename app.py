import nltk
nltk.download('stopwords')
nltk.download('punkt')
import gradio as gr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
def summarize_paragraph(paragraph, num_sentences=3):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    # Tokenize and preprocess each sentence
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Remove stop words and perform stemming
    preprocessed_sentences = [
        [stemmer.stem(word) for word in sentence if word not in stop_words]
        for sentence in tokenized_sentences
    ]

    # Calculate sentence scores based on word frequency
    word_frequencies = {}
    for sentence in preprocessed_sentences:
        for word in sentence:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Calculate sentence scores
    sentence_scores = {}
    for i, sentence in enumerate(preprocessed_sentences):
        for word in sentence:
            if word in word_frequencies:
                if i not in sentence_scores:
                    sentence_scores[i] = word_frequencies[word]
                else:
                    sentence_scores[i] += word_frequencies[word]

    # Sort sentences based on scores and select the top ones
    sorted_sentences = sorted(sentence_scores, key=lambda x: sentence_scores[x], reverse=True)
    top_sentences = sorted_sentences[:num_sentences]

    # Build the summarized paragraph
    summarized_paragraph = ' '.join([sentences[i] for i in top_sentences])
    return summarized_paragraph

def greet(name):

    from pytube import YouTube
    link=name
    try:
        yt = YouTube(link)
    except:
        print("Connection Error")

    from pytube import YouTube as YT
    video   = YT(link, use_oauth=False, allow_oauth_cache=False)
    stream  = video.streams.get_by_itag(140)
    stream.download('',"GoogleImagen.mp4")



    import whisper

    model = whisper.load_model("base")
    result = model.transcribe("GoogleImagen.mp4")
    print(result['text'])

#Text Summarizer




    paragraph =result['text']

    summary = summarize_paragraph(paragraph, num_sentences=2)
    print(summary)

    return summary

#JUTUBE





with gr.Blocks(theme=gr.themes.Soft()) as demo:
    name = gr.Textbox(label="Link")
    output = gr.Textbox(label="Video Summary")
    greet_btn = gr.Button("Submit")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")
demo.launch()

