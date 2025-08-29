import gradio as gr
from dotenv import load_dotenv
from pipeline import read_pdf, save_summary_to_pdf
from groq import Groq
from fpdf import FPDF
load_dotenv()
client = Groq()

def summarize_case(file_obj):
    case_text = read_pdf(file_obj.name)

    # Step 1: Summarize the case using LLM
    client2 = Groq()
    completion = client2.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a highly experienced senior advocate tasked with reviewing legal case files. Your role is to read and analyze the provided case, and then summarize it clearly and objectively. The summary should include key details such as the parties involved, case background, relevant facts, legal issues, applicable laws or sections, and the current status. Do not provide legal opinions or decisions—your job is to extract and organize essential information. Use formal legal language and keep the summary accurate and concise."
            },
            {
                "role": "user",
                "content": case_text
            },
        ],
        stream=True
    )

    summary_text = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            summary_text += chunk.choices[0].delta.content

    # Save summary to PDF
    save_summary_to_pdf(summary_text, "summaries/summary_output.pdf")

    return summary_text

def analyze_case(file_obj, query_text=None, voice_file=None):
    summary = summarize_case(file_obj)

    # Use voice if query not given
    if not query_text and voice_file:
        with open(voice_file.name, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="verbose_json"
            )
            query_text = transcript.text

    # Step 2: Use LLM to analyze and give legal opinion
    client3 = Groq()
    completion = client3.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "system",
                "content": "You are a highly respected and experienced senior advocate. A legal case has been provided to you in PDF format. Your task is to analyze it deeply and search for 2–3 similar or relevant cases online. For each similar case, include a brief context — such as the facts, outcome, and why it is relevant to the current case. Then, based on your expertise and the comparison, provide your own legal suggestion or recommendation for how to approach or resolve the case. Use clear legal language and ensure the response is concise, well-organized, and actionable."
            },
            {
                "role": "user",
                "content": summary + "\n\nUser Query: " + (query_text or "No query provided.")
            }
        ],
        stream=True
    )

    response = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content

    return summary, query_text or "No query", response

import gradio as gr

def smart_case_analyzer(file_obj, text_query, voice_query):
    if not text_query and not voice_query:
        return "Please provide either a text query or a voice query.", "", ""

    # Run core analysis
    return analyze_case(file_obj, text_query, voice_query)

iface = gr.Interface(
    fn=smart_case_analyzer,
    inputs=[
        gr.File(label="📄 Upload Legal Case PDF", type="filepath"),
        gr.Textbox(label="📝 Enter Legal Query (optional)", placeholder="e.g., Can the defendant appeal?"),
        gr.Audio(label="🎙️ Or Upload Voice Query (optional)", type="filepath")
    ],
    outputs=[
        gr.Textbox(label="📝 Case Summary"),
        gr.Textbox(label="❓ Your Query"),
        gr.Textbox(label="⚖️ Legal Recommendation")
    ],
    title="🧠 AI Legal Assistant",
    description="Upload a legal case PDF. Then either type your legal question OR upload a voice query (MP3/WAV). Get a case summary and legal suggestion.",
)

iface.launch()
