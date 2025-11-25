# NLP Master - Question Answering & Text Summarization

[🌐 Live Demo](https://question-answering-summarization.streamlit.app/)
[Kaggle Notebook](https://www.kaggle.com/code/yousefabdelsalam11/question-answering-summarization-using-transform)

## Overview
**NLP Master** is an interactive web application that leverages state-of-the-art **Natural Language Processing (NLP)** models to provide:

- **Question Answering (QA):** Extract precise answers from any given text.
- **Text Summarization:** Condense long texts into concise and meaningful summaries.  

Built using **Streamlit** for the front-end and **Hugging Face Transformers** for the NLP models, this tool provides a modern, user-friendly interface with advanced customization options.

---

## Features

### 1️⃣ Question Answering
- Input any text as context.
- Ask specific questions about the text.
- Get answers with confidence scores.
- Adjustable confidence threshold to filter low-confidence answers.
- Tips and examples for better results.

### 2️⃣ Text Summarization
- Input long texts and generate concise summaries.
- Control summary length with **min/max word sliders**.
- Adjustable **beam search** for better quality.
- Summary statistics: original words, summary words, and compression rate.

### 3️⃣ Examples & Tutorial
- Preloaded QA and Summarization examples.
- Copy example text directly to the input field for testing.
- Step-by-step guidance for using the app efficiently.

### 4️⃣ Modern UI & UX
- Responsive and visually appealing design with **custom CSS**.
- Animated progress bars, feature cards, info boxes, and metric cards.
- Fully functional sidebar for settings and model information.

---

## Tech Stack
- **Python 3.10+**
- **Streamlit** - Interactive web interface
- **Hugging Face Transformers**  
  - QA Model: `deepset/roberta-base-squad2`  
  - Summarization Model: `facebook/bart-large-cnn`
- **HTML & CSS** - Custom styling for modern UI

---

## How to Use
1. Open the [live app](https://question-answering-summarization.streamlit.app/).  
2. **Question Answering Tab:**  
   - Paste context text.  
   - Enter your question.  
   - Adjust confidence threshold and get the answer.  
3. **Text Summarization Tab:**  
   - Paste your long text.  
   - Adjust minimum and maximum summary length, and beam search.  
   - Click **Generate Summary**.  
4. **Examples Tab:**  
   - Explore preloaded examples and try them instantly in QA or Summarization.

---

## Installation & Local Deployment
To run the app locally:  

```bash
# Clone the repo
git clone <your-repo-link>
cd <repo-folder>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
