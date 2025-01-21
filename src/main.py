import gradio as gr
import os
import pickle
import logging
import arxiv
import requests
import pdfplumber
import re
from io import BytesIO
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "Enter your token")  # Replace with actual API key

# ✅ Initialize LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama-3.1-70b-Versatile")

# ✅ Initialize Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Fetch Papers from Arxiv
def fetch_papers(topic, num_papers):
    search = arxiv.Search(
        query=topic,
        max_results=num_papers,
        # sort_by=arxiv.SortCriterion.Relevance
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = [{"title": r.title, "summary": r.summary, "pdf_url": r.pdf_url} for r in search.results()]

    if not papers:
        return "No papers found. Try a different query.", gr.update(choices=[], value=[]), {}

    paper_titles = [paper["title"] for paper in papers]
    paper_dict = {paper["title"]: paper for paper in papers}

    # ✅ Ensure no preselected papers
    return "✅ Papers found! Select papers to process.", gr.update(choices=paper_titles, value=[]), paper_dict

# ✅ Extract Text from PDF
def extract_pdf_text(url):
    response = requests.get(url)
    response.raise_for_status()
    pdf_file = BytesIO(response.content)
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n\n"
    return re.sub(r'\s+', ' ', text).strip()

# ✅ Process Papers & Store in FAISS
def process_papers(selected_paper_titles, paper_dict, processed_papers):
    if not selected_paper_titles:
        return "⚠️ Please select at least one paper!", processed_papers, gr.update(choices=[], value=[]), gr.update(choices=[], value=[])

    for title in selected_paper_titles:
        if title not in paper_dict:
            continue

        paper = paper_dict[title]
        text = extract_pdf_text(paper["pdf_url"])

        if not text:
            continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = text_splitter.create_documents([text])

        for chunk in chunks:
            chunk.metadata = {"source": paper["pdf_url"], "title": title}

        vectorstore = FAISS.from_documents(chunks, embedding_model)
        processed_papers[title] = {"vectorstore": vectorstore, "title": title, "url": paper["pdf_url"]}

    # ✅ Auto-select processed papers for discussion
    return (
        f"✅ Selected papers processed successfully! You can now ask questions.",
        processed_papers,
        gr.update(choices=list(processed_papers.keys()), value=list(processed_papers.keys())),  # ✅ Auto-select
        gr.update(choices=list(processed_papers.keys()), value=list(processed_papers.keys()))   # ✅ Auto-select
    )

# ✅ Ask Questions (Multi-Paper Support)
def chat(history, selected_papers, question, processed_papers):
    if not selected_papers or len(selected_papers) == 0:
        return history + [{"role": "user", "content": question}, {"role": "assistant", "content": "⚠️ Please select at least one paper!"}]

    retrieved_text = ""
    for paper in selected_papers:
        if paper in processed_papers:
            retriever = processed_papers[paper]["vectorstore"].as_retriever()
            retrieved_docs = retriever.invoke(question)
            retrieved_text += f"\n\n**Paper: {paper}**\n" + "\n".join([doc.page_content for doc in retrieved_docs[:3]]) + "\n\n"

    if not retrieved_text.strip():
        return history + [{"role": "user", "content": question}, {"role": "assistant", "content": "⚠️ No relevant information found for your question."}]

    prompt_text = f"""
    You are an AI research assistant. Answer the question using the retrieved document.

    Research Papers: {', '.join(selected_papers)}

    Context:
    {retrieved_text}

    Question: {question}
    """

    try:
        ai_message = llm.invoke(prompt_text)
        answer = ai_message.content

        if not answer or not answer.strip():
            answer = "⚠️ The AI could not generate a response for this query."
    except Exception as e:
        logging.error(f"❌ LLM Error: {e}")
        answer = f"⚠️ Error while generating response: {str(e)}"

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    return history

# ✅ Clear Chat Function
def clear_chat():
    return []

# ✅ Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📚 AI Research Paper Chatbot (Multi-Paper Selection & Discussion)")

    with gr.Row():
        topic = gr.Textbox(label="Enter Research Topic")
        # num_papers = gr.Slider(1, 10, value=5, label="Number of Papers")
        num_papers = gr.Number(value=5, label="📄 Number of Papers you want to fetch", precision=0)
        fetch_btn = gr.Button("Fetch Papers")

    paper_status = gr.Textbox(label="Status", interactive=False)
    paper_list = gr.CheckboxGroup(choices=[], label="Select Papers you want to discuss about", value=[])  # ✅ No preselected papers
    paper_dict_state = gr.State({})
    processed_papers = gr.State({})
    selected_papers = gr.CheckboxGroup(choices=[], label="Discuss Papers", value=[])  # ✅ No preselected papers

    fetch_btn.click(fetch_papers, inputs=[topic, num_papers], outputs=[paper_status, paper_list, paper_dict_state])

    with gr.Row():
        process_btn = gr.Button("Process Selected Papers")

    process_status = gr.Textbox(label="Processing Status", interactive=False)
    process_btn.click(process_papers, inputs=[paper_list, paper_dict_state, processed_papers], outputs=[process_status, processed_papers, selected_papers, paper_list])

    chatbot = gr.Chatbot(type="messages")
    message = gr.Textbox(label="Ask a question about the selected papers")

    send_btn = gr.Button("Ask")
    clear_btn = gr.Button("Clear Chat")

    send_btn.click(chat, inputs=[chatbot, selected_papers, message, processed_papers], outputs=[chatbot])
    clear_btn.click(clear_chat, outputs=[chatbot])

demo.launch(share=True)
