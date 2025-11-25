import streamlit as st
from transformers import pipeline
import time

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="NLP Master - QA & Summarization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS with Modern Theme
# ----------------------------
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        color: #2d3436;
        text-align: center;
        margin-bottom: 3rem;
        font-size: 1.3rem;
        font-weight: 300;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.8rem;
        border-radius: 20px;
        border-left: 6px solid #ff6b6b;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        color: #2d3436;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border: none;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0, 184, 148, 0.3);
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border: none;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(253, 203, 110, 0.3);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border: none;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(116, 185, 255, 0.3);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%);
    }
    
    /* Text Input */
    .stTextInput input, .stTextArea textarea {
        background: rgba(255, 255, 255, 0.9);
        color: #2d3436;
        border: 2px solid #dfe6e9;
        border-radius: 12px;
        padding: 0.8rem;
        font-size: 1rem;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #74b9ff;
        box-shadow: 0 0 0 3px rgba(116, 185, 255, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background: transparent;
        border-radius: 10px;
        padding: 0px 25px;
        color: #636e72;
        font-weight: 500;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.95);
        border: none;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Slider Styling */
    .stSlider {
        color: #ff6b6b;
    }
    
    /* Custom section headers */
    .section-header {
        font-size: 1.8rem;
        color: #2d3436;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .section-subheader {
        font-size: 1.1rem;
        color: #636e72;
        margin-bottom: 2rem;
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: white; margin-bottom: 0;'>⚙️ Settings</h1>
        <p style='color: rgba(255,255,255,0.8);'>Customize your NLP experience</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 15px;'>
    <h3 style='color: #2d3436; margin-bottom: 1rem;'>ℹ️ About This App</h3>
    <p style='color: #636e72;'>This app uses state-of-the-art NLP models for:</p>
    <ul style='color: #636e72;'>
    <li><strong>Question Answering:</strong> Find answers in your text</li>
    <li><strong>Text Summarization:</strong> Create concise summaries</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 15px;'>
    <h3 style='color: #2d3436; margin-bottom: 1rem;'>📊 Model Information</h3>
    <ul style='color: #636e72;'>
    <li><strong>QA Model:</strong> RoBERTa-base</li>
    <li><strong>Summarization:</strong> BART-large-CNN</li>
    <li><strong>Powered by:</strong> 🤗 Transformers</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Main UI
# ----------------------------
st.markdown('<h1 class="main-header">🤖 NLP Master</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Question Answering & Text Summarization Tool</p>', unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["❓ Question Answering", "📝 Text Summarization", "🎯 Examples"])

# ==============================
# TAB 1 — Question Answering
# ==============================
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="section-header">🔍 Question Answering</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Enter your context and question to extract precise answers.</p>', unsafe_allow_html=True)
        
        # Context Input
        st.markdown("**📖 Context Text:**")
        context = st.text_area(
            label="context_input",
            height=250,
            placeholder="Paste your text here... The model will search for answers within this context.",
            label_visibility="collapsed"
        )
        
        # Question Input
        st.markdown("**❓ Your Question:**")
        question = st.text_input(
            label="question_input",
            placeholder="What would you like to know from the text above?",
            label_visibility="collapsed"
        )
        
        # Controls
        col1_1, col1_2 = st.columns([1, 1])
        with col1_1:
            st.markdown("**🎯 Confidence Threshold:**")
            confidence_threshold = st.slider(
                "confidence_threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                label_visibility="collapsed"
            )
        
        with col1_2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Get Answer", use_container_width=True):
                if context.strip() == "" or question.strip() == "":
                    st.markdown('<div class="warning-box">⚠️ Please enter both context and question.</div>', unsafe_allow_html=True)
                else:
                    with st.spinner("🔎 Searching for answer..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        try:
                            qa_model = load_qa_model()
                            result = qa_model(question=question, context=context)
                            
                            if result["score"] > confidence_threshold:
                                st.markdown(f"""
                                <div class="success-box">
                                    <h4>✅ Answer Found (Confidence: {result["score"]:.2%})</h4>
                                    <p style="font-size: 1.4rem; font-weight: bold; margin: 1rem 0;">{result["answer"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="warning-box">
                                    <h4>⚠️ Low Confidence Answer ({result["score"]:.2%})</h4>
                                    <p>Try rephrasing your question for better results.</p>
                                    <p style="font-size: 1.1rem; margin-top: 0.5rem;">Best match: <strong>{result['answer']}</strong></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.markdown(f'<div class="warning-box">❌ Processing Error: {str(e)}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
        <h4>💡 Tips for Better Answers</h4>
        <ul>
        <li>Provide detailed and comprehensive context</li>
        <li>Ask specific and clear questions</li>
        <li>Ensure the answer is present in the text</li>
        <li>Use clear and direct language</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📚 Quick Example</h4>
        <p><strong>Context:</strong> The Eiffel Tower is located in Paris, France. It was built in 1889 and is one of the most famous landmarks in the world.</p>
        <p><strong>Question:</strong> Where is the Eiffel Tower located?</p>
        <p><strong>Answer:</strong> Paris, France</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# TAB 2 — Summarization
# ==============================
with tabs[1]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="section-header">📄 Text Summarization</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Transform long texts into concise, meaningful summaries.</p>', unsafe_allow_html=True)
        
        # Text Input
        st.markdown("**📝 Text to Summarize:**")
        text = st.text_area(
            label="text_to_summarize",
            height=300,
            placeholder="Paste your long text here to generate a concise summary...",
            label_visibility="collapsed"
        )
        
        # Controls
        col2_1, col2_2, col2_3 = st.columns([1, 1, 1])
        with col2_1:
            st.markdown("**📏 Min Length:**")
            min_len = st.slider("min_length", 10, 200, 50, label_visibility="collapsed")
        with col2_2:
            st.markdown("**📐 Max Length:**")
            max_len = st.slider("max_length", 50, 500, 150, label_visibility="collapsed")
        with col2_3:
            st.markdown("**⚡ Beam Search:**")
            num_beams = st.slider("beam_search", 1, 8, 4, label_visibility="collapsed")
        
        if st.button("✨ Generate Summary", use_container_width=True):
            if text.strip() == "":
                st.markdown('<div class="warning-box">⚠️ Please enter some text to summarize.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("📝 Creating summary..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    try:
                        summarizer = load_summarizer()
                        summary = summarizer(
                            text, 
                            min_length=min_len, 
                            max_length=max_len,
                            num_beams=num_beams,
                            early_stopping=True
                        )
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>📋 Generated Summary</h4>
                            <p style="font-size: 1.1rem; line-height: 1.6; margin: 1rem 0;">{summary[0]['summary_text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show stats
                        orig_words = len(text.split())
                        summ_words = len(summary[0]['summary_text'].split())
                        compression = ((orig_words - summ_words) / orig_words) * 100 if orig_words > 0 else 0
        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Original Words", orig_words)
                        with col_stat2:
                            st.metric("Summary Words", summ_words)
                        with col_stat3:
                            st.metric("Compression", f"{compression:.1f}%")
                            
                    except Exception as e:
                        st.markdown(f'<div class="warning-box">❌ Summary Generation Error: {str(e)}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
        <h4>🎯 Best Practices</h4>
        <ul>
        <li>Input text should be >100 words for best results</li>
        <li>Adjust length sliders based on your needs</li>
        <li>Higher beam search = better quality (slower)</li>
        <li>Ideal for articles, documents, and reports</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# TAB 3 — Examples
# ==============================
with tabs[2]:
    st.markdown('<p class="section-header">🎯 Examples & Tutorial</p>', unsafe_allow_html=True)
    
    example_tab1, example_tab2 = st.tabs(["QA Examples", "Summarization Examples"])
    
    with example_tab1:
        st.markdown("#### Question Answering Examples")
        
        examples = [
            {
                "context": "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers, of which 5,500,000 square kilometers are covered by the rainforest.",
                "questions": [
                    "What is another name for the Amazon rainforest?",
                    "How large is the Amazon basin?",
                    "Where is the Amazon rainforest located?"
                ]
            },
            {
                "context": "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
                "questions": [
                    "Who created Python?",
                    "When was Python first released?",
                    "What is Python's design philosophy?"
                ]
            }
        ]
        
        for i, example in enumerate(examples):
            with st.expander(f"Example {i+1}", expanded=True):
                st.markdown("**📖 Context:**")
                st.markdown(f'<div class="info-box">{example["context"]}</div>', unsafe_allow_html=True)
                st.markdown("**🔍 Try these questions:**")
                for j, question in enumerate(example["questions"]):
                    st.write(f"{j+1}. {question}")

    with example_tab2:
        st.markdown("#### Summarization Examples")
        
        sample_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of intelligent agents: any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term artificial intelligence to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving, however this definition is rejected by major AI researchers.
        
        AI applications include advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Tesla), automated decision-making and competing at the highest level in strategic game systems (such as chess and Go). As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect.
        """
        
        st.markdown("**📝 Sample Input Text:**")
        st.markdown(f'<div class="info-box">{sample_text}</div>', unsafe_allow_html=True)
        
        if st.button("Try this Example in Summarizer"):
            st.session_state.summarization_text = sample_text
            st.markdown('<div class="success-box">✅ Text copied to summarization tab! Switch to the Text Summarization tab to see it in action.</div>', unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #636e72; padding: 2rem;'>"
    "Built with ❤️ using Streamlit and Hugging Face Transformers | "
    "© 2024 NLP Master - All rights reserved"
    "</div>",
    unsafe_allow_html=True
)