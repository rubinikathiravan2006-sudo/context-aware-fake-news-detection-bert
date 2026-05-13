import os
import platform
from collections import namedtuple

uname_result = namedtuple("uname_result", ["system", "node", "release", "version", "machine", "processor"])
platform.uname = lambda: uname_result("Windows", "localhost", "10", "10.0.0", "AMD64", "AMD64")
platform.machine = lambda: "AMD64"
platform.win32_ver = lambda: ("10", "10.0.0", "SP0", "Multiprocessor Free")

import gradio as gr
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.inference import InferencePipeline
from src.news_analyzer import NewsAnalyzer

# Initialize globally
pipeline = InferencePipeline(model_path="best_model.pth" if os.path.exists("best_model.pth") else None)
news_analyzer = NewsAnalyzer()

def analyze_news(raw_text, url, image, video):
    text_extracted = ""
    source = "Unknown"

    # Priority: Text > URL > Image > Video
    if raw_text and raw_text.strip() and len(raw_text.strip()) > 10:
        text_extracted = raw_text.strip()
        source = "Direct Text"
    elif url and url.strip():
        text_extracted = pipeline.extract_from_url(url)
        source = f"URL ({url})"
    elif image is not None:
        text_extracted = pipeline.extract_from_image(image)
        source = "Uploaded Image (OCR)"
    elif video is not None:
        text_extracted = pipeline.extract_from_video(video)
        source = "Uploaded Video (Transcript)"
    else:
        return "Please provide an input (Text, URL, Image, or Video).", "N/A"

    if text_extracted.startswith("Error"):
        return text_extracted, "Extraction Error"

    # 1. Local AI inference (BERT + GNN)
    pred, conf = pipeline.predict(text_extracted)

    # 2. Web verification using NewsAnalyzer
    # Use first 150 chars as headline for search
    headline = text_extracted[:150].split('\n')[0].strip()
    web_analysis = news_analyzer.analyze_news(headline)

    print("\n========== WEB ANALYSIS ==========")
    print(web_analysis)
    print("==================================\n")

    # 3. Build result string
    result_parts = []

    # --- AI Analysis Section ---
    result_parts.append("## 🤖 AI Analysis")
    if conf is not None:
        result_parts.append(f"**Prediction:** {pred}")
        result_parts.append(f"**Confidence:** {conf:.2f}%")
    else:
        result_parts.append(f"{pred}")

    # --- Web Verification Section ---
    result_parts.append("\n## 🌐 Web Verification")

    if "error" not in web_analysis:
        result_parts.append(f"**Assessment:** {web_analysis['assessment']}")
        result_parts.append(f"**Sources Found:** {web_analysis['source_count']}")
        result_parts.append(f"**Presence Score:** {web_analysis['presence_score']}/100")

        if web_analysis.get('peak_day') and web_analysis['peak_day'] != "Unknown":
            result_parts.append(f"**Peak Reporting Day:** {web_analysis['peak_day']}")

        if web_analysis.get('days_to_peak') and web_analysis['days_to_peak'] > 0:
            result_parts.append(f"**Days to Peak:** {web_analysis['days_to_peak']}")

        if web_analysis.get('sources'):
            result_parts.append("\n## 📰 Websites Reporting This News")
            for idx, s in enumerate(web_analysis['sources'][:5], start=1):
                title = s.get('title', 'No title')
                source_name = s.get('source_name', 'Unknown')
                domain = s.get('domain', '')
                url = s.get('url', '')
                result_parts.append(
                    f"""
                    ### {idx}. {source_name}
                    **Title:** {title}
                    **Website:** {domain}
                    🔗 {url}
"""
            )
    else:
        result_parts.append(f"⚠️ {web_analysis['error']}")

    # --- Footer ---
    result_parts.append(f"\n---\n*Input Source: {source}*")

    result_str = "\n".join(result_parts)
    return text_extracted, result_str


# --- Gradio UI ---
with gr.Blocks(title="Fake News Detector") as demo:
    gr.Markdown("# 🕵️ Fake News Detector")
    gr.Markdown("Identify fake news using AI analysis + real-time web verification.")

    with gr.Row():
        # Input Column
        with gr.Column():
            gr.Markdown("### 📥 Input Options")
            text_input = gr.Textbox(
                label="News Text",
                placeholder="Paste article text or headline here...",
                lines=5
            )
            url_input = gr.Textbox(
                label="News URL",
                placeholder="https://example.com/article"
            )
            image_input = gr.Image(
                label="Upload Image (OCR)",
                type="filepath"
            )
            video_input = gr.Video(
                label="Upload Video (Transcript)"
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")

        # Output Column
        with gr.Column():
            gr.Markdown("### 📊 Analysis Result")
            output_result = gr.Markdown("*Waiting for input...*")
            output_text = gr.Textbox(
                label="Extracted Content",
                interactive=False,
                lines=10,
                placeholder="Extracted text will appear here after analysis..."
            )

    analyze_btn.click(
        fn=analyze_news,
        inputs=[text_input, url_input, image_input, video_input],
        outputs=[output_text, output_result]
    )

if __name__ == "__main__":
    demo.launch()
