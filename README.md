# 🩺 Bacteremia Clinical Guideline Assistant 

[![Live App](https://img.shields.io/badge/🚀_Live_App-Streamlit-FF4B4B?style=for-the-badge)](https://bacteremia-clinical-guideline-assistant.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge)](https://www.python.org/)


Ask a clinical question about bacteremia and get an answer pulled straight 
from antibiotic stewardship guidelines, with the source document 
and page number attached to every claim.

**[👉 Try it live here](https://bacteremia-clinical-guideline-assistant.streamlit.app/)**

Built to stop the back-and-forth of flipping through 20-page PDF guidelines for a quick answer. This is a research and prototyping tool, not a replacement for clinical judgment or an ID consult.

**[ This is a DOMAIN agnostic pipeline (isn't tied to bacteremia specifically, point it at a different set of clinical guideline PDFs and it works the same way for another condition.) ]**

---

## Intended Purpose

This is a research prototype demonstrating grounded retrieval-augmented generation for clinical guideline lookup. It is intended to help clinicians, pharmacists, or students quickly locate guideline-based information during research or education - not to autonomously diagnose, recommend treatment, 
or replace an infectious disease (ID) consult. All outputs must be verified against the cited source before any clinical use.

---

## How It Works

PDFs → chunked by meaning → embedded into a FAISS index
│
Question comes in → rewritten into 3 clinical phrasings
(catches synonym mismatches like "complicated" vs "source control")
│
Best-matching chunks retrieved → reranked for relevance
│
LLM answers only from those chunks, citing filename + page

---

## Regulatory Context (EU AI Act)

This prototype is a retrieval tool grounded in static guideline documents. It is not currently a safety component of a medical device and is not marketed for clinical deployment, so under the AI Act's Article 6 classification logic it would not currently fall under Annex I. If adapted for real clinical use - e.g., directly influencing treatment decisions in a hospital workflow - its intended purpose and safety role would need to be reassessed, likely triggering medical-device (MDR) and Annex I considerations. This project was built with that distinction in mind: strict grounding, mandatory citations, and explicit non-diagnostic framing 
are deliberate design choices to keep the current version outside high-risk territory while demonstrating the safety principles that would apply if it were scaled toward real use.

---

## Human Oversight

Every answer requires the user to verify the cited source and page before acting on it. The system is designed to support review, not replace it - it does not make autonomous recommendations and will not answer questions outside the loaded guideline documents.

---

## 📊 Evaluation

Answer quality was measured with [RAGAS](https://github.com/explodinggym/ragas) across multiple evaluation runs on a held-out test set of clinical questions:

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | ~0.73 | How well the answer sticks to the retrieved sources, without adding unsupported claims |
| Context Precision | ~0.87 | How relevant the retrieved chunks are to the question |
| Context Recall | ~0.97 | How well the retrieval step captures all the relevant information available |
| Answer Relevancy | ~0.75 | How directly the final answer addresses the question asked |

High context recall combined with lower faithfulness suggests the retriever is doing its job well, and most remaining improvement is in how tightly the generation step sticks to the retrieved text = a known trade-off worth tuning further. (Faithfulness (~0.73) is the metric prioritised for improvement, since strict grounding is the core safety requirement of this system - a lower score here matters more than a lower score on relevancy.)

---

## Getting Started

```bash
git clone https://github.com/sunilkumar-nallani/Bacteremia_Clinical_Guideline_Assistant.git
cd Bacteremia_Clinical_Guideline_Assistant
pip install -r requirements.txt
```

Add a `.env` file (see `.env.example`):

OPENAI_API_KEY=your_key_here
OPENAI_API_BASE=your_base_url_here   #optional


## Usage

```bash
python build_index.py     # run once, builds the FAISS index
streamlit run app.py       # launches the chat interface
```

## Tech Stack

LangChain · FAISS · FlashRank · Streamlit · OpenAI-compatible LLM/embeddings API

## Limitations

- Prototyping tool, not for clinical decision-making
- Only answers from the PDFs you load in, nothing outside them
- Page numbers come from PDF metadata and may not match the printed page exactly

## Contributing

PRs and issues welcome, whether it's a code fix or a suggestion on how a 
clinical question should be handled. Domain feedback from clinicians and ID 
pharmacists is just as valuable as code.

## License

MIT

## Contact

**Sunil Kumar Nallani** - sunilkumar.nallani@stud.uni-goettingen.de - [LinkedIn](https://www.linkedin.com/in/nallani-sunil-kumar-67227a243/)
