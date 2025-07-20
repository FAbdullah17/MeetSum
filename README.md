# 📘 MeetSum – AI-Powered Meeting Transcript Summarizer

**MeetSum** is a smart, AI-powered web application that simplifies the process of extracting meaningful summaries from long, dense meeting transcripts. This tool uses state-of-the-art Natural Language Processing techniques, particularly a fine-tuned BART (Bidirectional and Auto-Regressive Transformers) model, to distill large amounts of text into clear, digestible summaries. MeetSum aims to enhance productivity by saving time spent reading lengthy transcripts, ensuring no important detail is overlooked.

The application supports various text input formats such as `.txt`, `.pdf`, and `.docx`, and also allows for raw text to be directly entered via a user-friendly interface. Designed with scalability and practicality in mind, MeetSum incorporates a backend powered by FastAPI and a clean frontend interface, making it accessible for both technical and non-technical users. In this README, we will explore the complete details of the architecture, workflow, model training, deployment pipeline, and potential enhancements.

---

## 📂 Project Structure

A modular structure is followed to separate responsibilities, aid readability, and promote ease of maintenance and extension.

```
MEETSUM/
├── app/
│   ├── __init__.py                # Initialization for app module
│   ├── inference_engine.py        # Inference logic for summarization
│   └── utils.py                   # Utility functions (file reading, chunking)
├── data/
│   ├── raw/                       # Unprocessed input files (PDF, DOCX, etc.)
│   ├── cleaned/                   # Cleaned and preprocessed transcripts
│   └── tokenized/                 # Tokenized inputs for model training/inference
├── models/
│   ├── config.json                # Model config
│   ├── generation_config.json     # Text generation-specific config
│   ├── merges.txt                 # Tokenizer merge rules
│   ├── model.safetensors          # Trained model weights
│   ├── special_tokens_map.json    # Token special character mapping
│   ├── tokenizer_config.json      # Tokenizer config
│   ├── tokenizer.json             # Raw tokenizer definition
│   └── vocab.json                 # Tokenizer vocabulary
├── notebooks/
│   ├── data.ipynb                 # Data overview and EDA
│   ├── data_loading.ipynb         # Load and inspect datasets
│   ├── model.ipynb                # Model architecture and fine-tuning
│   └── test.ipynb                 # Post-training model validation
├── samples/
│   ├── sample.txt                 # Sample transcript for testing
│   ├── sample2.txt                # Another example
│   └── test.txt                   # Testing chunk inference
├── src/
│   ├── data_preprocessing_n_tokenization/
│   │   ├── __init__.py
│   │   ├── data_loading.py        # Load and parse transcripts
│   │   ├── data_preprocessing.py  # Clean text, remove noise
│   │   ├── data_tokenization.py   # Tokenize for model input
│   │   └── eda.py                 # Exploratory data analysis
│   ├── testing/
│   │   ├── __init__.py
│   │   └── test.py                # Test inference performance and output
│   └── training_n_evaluation/
│       ├── __init__.py
│       ├── evaluate.py            # Evaluate on validation/test data
│       └── train.py               # Fine-tune BART model
├── static/
│   ├── index.html                 # Frontend HTML
│   └── style.css                  # CSS styling
├── main.py                        # FastAPI application entrypoint
├── LICENSE                        # MIT License file
├── .gitignore
└── README.md                      # Project documentation
```

---

## 🎯 Objective

The objective of MeetSum is to address the growing need for efficient information processing in organizational and academic settings. Meetings often involve extended discussions, and the resulting transcripts can span thousands of words. Manually reviewing these transcripts is time-consuming and error-prone.

MeetSum aims to:

- Automatically extract concise, high-quality summaries from long-form transcripts.
- Support multiple input formats, including text, PDF, DOCX, and direct text entry.
- Perform accurate summarization using a fine-tuned BART-based transformer model.
- Ensure that summaries are generated chunk-wise and contextually merged.
- Deliver a seamless user experience with an interactive, responsive web interface.

---

## 📄 Dataset Info

The training dataset for MeetSum was compiled using a blend of publicly available and synthetic sources. These include:

- ✅ **MeetingBank Dataset**: A comprehensive and curated dataset of real-world meeting transcripts and summaries, widely used for academic summarization benchmarks.
- 🏛️ **Council Meeting Transcripts**: Public domain transcripts from city council and organizational meetings.
- 🎓 **TEDx & Lecture Transcripts**: Extracted from academic platforms and TED repositories.
- 🎙️ **YouTube/Podcast Transcript Dumps**: Transcripts fetched and preprocessed from media and online platforms.
- 🧠 **Synthetic AI-generated Meeting Scripts**: Programmatically generated using GPT-3.5/4 to simulate structured meeting environments for data augmentation.


All datasets were curated to reflect real-world meeting styles: multi-speaker, domain-specific vocabulary, and unstructured dialogue.

### 🔄 Preprocessing Pipeline

Each transcript goes through a rigorous pipeline to make it suitable for summarization:

1. **Cleaning**
   - Timestamps (e.g., `00:10:12`) and speaker labels (e.g., `Speaker 1:`) are removed.
   - Non-verbal noise (`[Laughter]`, `[Applause]`) and repeated fillers are stripped.
   - Extra whitespace and irrelevant line breaks are normalized.

2. **Splitting**
   - Transcripts are segmented by semantic units or speaker turns.
   - If a transcript exceeds 1024 tokens (BART limit), it is divided into logical chunks.

3. **Tokenization**
   - Uses HuggingFace’s `facebook/bart-base` tokenizer.
   - The text is converted to token IDs, and segment boundaries are preserved.

4. **Storage**
   - Cleaned texts are stored in `/data/cleaned/`.
   - Tokenized forms are saved in `/data/tokenized/`.

---

## 🧠 Model Info

| Parameter           | Details                         |
|---------------------|----------------------------------|
| Model Architecture  | BART (Bidirectional + Auto Regressive Transformer) |
| Pretrained Base     | `facebook/bart-base`            |
| Fine-Tuned Task     | Text Summarization              |
| Max Input Tokens    | 1024                            |
| Chunk Strategy      | Sequential overlapping chunks   |
| Output              | Abstract summary per chunk, merged |
| Training Framework  | PyTorch + HuggingFace Transformers |
| Weight Format       | `.safetensors`                  |

### 🔧 Fine-Tuning Details

- Used teacher forcing with label smoothing
- ROUGE-1 and ROUGE-L as evaluation metrics
- Batch size: 4
- Epochs: 3 (early stopping on val loss)
- Optimizer: AdamW with linear decay
- Gradient accumulation to handle long context windows

---

## ⚙️ FastAPI Backend

The backend is developed using FastAPI for speed and simplicity. It handles routing, file handling, model inference, and response formatting.

### 🔌 API Endpoints

- `/`: Returns the UI interface (HTML).
- `/summarize/text`: Accepts raw text via form and returns summary.
- `/summarize/file`: Accepts file uploads (`.txt`, `.pdf`, `.docx`), extracts text, chunks, summarizes, and responds.

### 🧱 Core Backend Modules

- `inference_engine.py`  
  Loads model and tokenizer. Handles chunk-wise summarization.

- `utils.py`  
  Includes:
  - `read_txt_file`, `read_docx_file`, `read_pdf_file`
  - `chunk_text()`: Splits text into logical chunks without mid-sentence cuts.
  - `merge_chunks()`: Assembles chunked summaries.

---

## 🖥️ Frontend UI

### 📄 HTML (`static/index.html`)

A sleek frontend built with standard HTML5. Features include:

- Drag-and-drop or select file upload
- Paste raw transcript text
- Responsive design with real-time feedback
- Displays the summary output in styled text box

### 🎨 CSS (`static/style.css`)

- Color theme: Blue & Gray
- Button styling, hover effects, mobile responsiveness
- Smooth layout adaptation from desktop to mobile

---

## 🚀 Run Locally

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 1.12
- HuggingFace Transformers ≥ 4.36
- Uvicorn for ASGI

### 🔧 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/FAbdullah17/MeetSum.git
cd MEETSUM

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn main:app --reload
```

Then visit:  
`http://127.0.0.1:8000`

---

## 🧪 Testing

The project includes testing logic to validate the inference flow:

```bash
python src/testing/test.py
```

Sample test includes:

- Summary length check
- Semantic coherence check
- Chunk merge consistency
- Output text format and encoding test

---

## 🔧 Development Notes

- **Chunking**: BART supports 1024 tokens max input. Chunking is mandatory for longer inputs and uses sentence-aware splitting to avoid semantic breaks.
- **Parallel Inference**: Chunks can be inferred in parallel (currently sequential to preserve order).
- **Merge Strategy**: Merged summaries are re-ranked based on keyword overlap and semantic similarity.
- **Error Handling**: Ensures non-text files are rejected, and corrupt files trigger custom messages.
- **Tokenizer Config**: Using a pre-saved tokenizer to reduce cold start latency.

---

## 📈 Future Work

- 🎙️ **Live Audio Summarization**: Integrate Whisper to convert real-time speech into text, followed by summarization.
- 📡 **Zoom/Slack Integration**: Plug into calendar events or chat logs to auto-fetch meeting notes.
- 🗃️ **Named Entity Extraction**: Extract and highlight names, organizations, decisions made.
- 🔖 **Summary Types**: Let users choose between action-item summary, decision-only, or full summary.
- 👥 **User Sessions**: Enable login and historical summary downloads.
- 🧠 **LLM Backend Option**: Option to switch to Mixtral, GPT-4, Claude or LLaMA3 on Groq backend.
- ☁️ **Deploy on Cloud**: Docker + Vultr/GCP-based API scaling.

---

## 📜 License

MeetSum is released under the MIT License.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

Refer to `LICENSE` file for full legal text.

---

## 👤 Author

**Fahad Abdullah**  

## 👤 Contributors

**Asma Zubair**  



---
