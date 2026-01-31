# AI-Powered-Market-Research-Assistant

🤖 MarketMind AI: Your Intelligence Assistant
Stop manually researching. Start strategizing. MarketMind AI is an end-to-end research agent designed to automate secondary market research. It combines real-time web scraping with document intelligence to turn massive amounts of data into structured, executive-level insights.

✨ Key Features

💬 Conversational Research UI: A ChatGPT-style interface where you can ask complex questions like "What are the top 5 fintech innovations in Canada?"

🌐 Web-Aware Intelligence: Integrates Firecrawl to scrape the live web, ensuring your insights are based on today’s news, not last year's training data.

📄 PDF/Document Analysis: Upload market reports, case studies, or whitepapers to include internal data in the AI's reasoning.

📊 Multi-Format Outputs: Generates structured Executive Summaries, SWOT analyses, and strategic recommendations.

📁 Auto-Export: Automatically saves every session to a memory.json file and allows you to download your final report as a PDF.

🏠 Private & Local: Uses Ollama to run models locally, keeping your sensitive research data off external servers.

🛠️ Tech Stack

Orchestration: LangChain

Brain: Ollama (Mistral / Llama 3)

Web Scraping: Firecrawl

Interface: Streamlit

Persistence: JSON-based memory & PDF Export

🚀 Getting Started (Quick Run)

Prerequisites: Install Ollama and pull your model (ollama pull mistral).

Setup:

cd market-research-assistant
python3 -m venv venv
source venv/bin/activate  # venv\Scripts\activate for Windows
pip install -r requirements.txt
Launch:
streamlit run app.py
