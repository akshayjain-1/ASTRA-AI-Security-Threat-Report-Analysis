import datetime
import html
import ipaddress
import re
import sqlite3
from typing import Any
import feedparser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

FEEDS = ['https://cybersecuritynews.com/feed/']
HASH_PATTERN = re.compile(r'\b([a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b')
MODEL = "qwen2.5-coder:7b"

# ==========================================
# 1. SETUP LANGCHAIN COMPONENTS
# ==========================================
# Initialize the local LLM and local embeddings

# Add parameter base_url="http://<ip>:11434" to both the llm and embeddings if your Ollama server is running inside a container and you are connecting from the host machine. Use "http://localhost:11434" if you are running the script inside the same container as Ollama.
llm = ChatOllama(model=MODEL, temperature=0.0)  
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize Chroma (creates a local folder named 'chroma_db')
vector_store = Chroma(
    collection_name="rss_iocs",
    embedding_function=embeddings,
    persist_directory="./threat_intel_rag"
)

# Define the structured output schema using Pydantic
class IOCSchema(BaseModel):
    summary: str = Field(description="A concise summary of the threat campaign or threat actor")
    ips: list[str] = Field(default=[], description="IPv4 or IPv6 addresses found in the text.")
    domains: list[str] = Field(default=[], description="Domain names, excluding protocols like http.")
    urls: list[str] = Field(default=[], description="URLs found in the text.")
    hashes: list[str] = Field(default=[], description="MD5, SHA-1, or SHA-256 cryptographic hashes.")

    @field_validator("ips", mode="after")
    @classmethod
    def validate_ips(cls, values: list[str]) -> list[str]:
        """Filters out strings that are not valid IPv4 addresses and cleans whitespace."""
        valid_ips = []
        for ip in values:
            try:
                cleaned_ip = ip.strip().strip(".,()[]{}'\"")
                ipaddress.ip_address(cleaned_ip)
                valid_ips.append(cleaned_ip)
            except ValueError:
                continue  # Discard the incorrect IP value
        return valid_ips
    
    @field_validator("hashes", mode="after")
    @classmethod
    def validate_hashes(cls, values: list[str]) -> list[str]:
        """Filters out strings that are not valid hashes."""
        valid_hashes = []
        for hash in values:
            cleaned_hash = hash.strip()
            if HASH_PATTERN.match(cleaned_hash):
                valid_hashes.append(cleaned_hash)
        return valid_hashes

# Bind the schema to the LLM so it forces structured JSON output
structured_llm = llm.with_structured_output(IOCSchema)

# LLM based extraction of IOC from the articles
def process_and_store_rag(article_link: str, title: str, decoded_text: str):
    '''
    Uses LangChain to extract IOCs and saves everything as a document in Chroma
    '''
    print(f"[*] Extracting IOCs with {MODEL} via LangChain for: {title}")

    # Prompt the structured LLM directly
    prompt = f"""You are an advanced cybersecurity analyst. Inspect the following threat intelligence report and extract all unique Indicators of Compromise (IOCs) matching the requested schema.

    CRITICAL INSTRUCTIONS:
    1. CLEAN DEFANGED DATA: Threat reports often mask malicious links. You MUST normalize defanged indicators before saving them (e.g., convert 'hxxp://malicious[.]com' to 'http://malicious.com', and '192[.]168[.]1[.]1' to '192.168.1.1').
    2. DOMAINS VS URLS: Ensure any value placed in the 'domains' list contains ONLY the host/domain name (e.g., 'badsite.com'). Do NOT include 'http://' or trailing paths in the domain list. Full paths must go into the 'urls' list.
    3. EXCLUDE BENIGN INFRASTRUCTURE: Do not extract legitimate, trusted entities mentioned in the text (such as 'google.com', 'microsoft.com', 'adobe.com') unless they are explicitly flagged as hijacked or acting as a direct malicious C2 endpoint.
    4. NO INVENTED DATA: If a specific IOC category (e.g., hashes) is not present in the text, leave that array empty. Do not guess or hallucinate indicators.

    Threat Report Text:
    \"\"\"
    {decoded_text}
    \"\"\"
    """
    try:
        extracted_iocs = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"[!] Exctraction failed. Reason: {e}")
        return ""
    
    page_content = "".join(
        (
            f"title: {title}\n",
            f"summary: {extracted_iocs.summary}\n", # pyright: ignore[reportAttributeAccessIssue]
            f"extracted_ips: {', '.join(extracted_iocs.ips) if extracted_iocs.ips else 'None'}\n", # pyright: ignore[reportAttributeAccessIssue]
            f"extracted_domains: {', '.join(extracted_iocs.domains) if extracted_iocs.domains else 'None'}\n", # pyright: ignore[reportAttributeAccessIssue]
            f"extracted_urls: {', '.join(extracted_iocs.urls) if extracted_iocs.urls else 'None'}\n", # pyright: ignore[reportAttributeAccessIssue]
            f"extracted_hashes: {', '.join(extracted_iocs.hashes) if extracted_iocs.hashes else 'None'}\n", # pyright: ignore[reportAttributeAccessIssue]
            "\n--- Full Article Content ---\n",
            f"{decoded_text}\n"
        )
    )

    # Create a LangChain Document object
    doc = Document(
        page_content=page_content, # pyright: ignore[reportArgumentType]
        metadata={
            "source": article_link,
            "title": title,
            "has_ips": len(extracted_iocs.ips) > 0, # pyright: ignore[reportAttributeAccessIssue]
            "has_domains": len(extracted_iocs.domains) > 0, # pyright: ignore[reportAttributeAccessIssue]
            "has_urls": len(extracted_iocs.urls) > 0, # pyright: ignore[reportAttributeAccessIssue]
            "has_hashes": len(extracted_iocs.hashes) > 0 # pyright: ignore[reportAttributeAccessIssue]
        }
    )

    # Save the document to Chroma (it automatically handles text embedding generation)
    vector_store.add_documents([doc])
    print("[+] Saved to Chroma RAG database.")

# Query the RAG to extract information based on the user input
def query_threat_intel_rag(vector_store: Chroma, query: str, k: int = 5, require_ips: bool = False, require_domains: bool = False, require_urls: bool = False, require_hashes: bool = False) -> list[Document]:
    '''
    Queries Chroma using semantic similarity combined with hard metadata filters.
    '''

    # Only apply filters if the user query is clearly about IOCs, otherwise search all documents
    conditions = []
    if require_ips:
        conditions.append({"has_ips": True})
    if require_domains:
        conditions.append({"has_domains": True})
    if require_urls:
        conditions.append({"has_urls": True})
    if require_hashes:
        conditions.append({"has_hashes": True})

    search_kwargs: dict[str, Any] = {"k": k}
    if len(conditions) > 0:
        # Only filter if at least one IOC type is requested
        if len(conditions) == 1:
            search_kwargs["filter"] = conditions[0]
        else:
            search_kwargs["filter"] = {"$or": conditions}

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    prompt = ChatPromptTemplate.from_template(
        """
You are an advanced Threat Intelligence Analyst. Use only the provided context to answer the user's question. If the user asks for affected versions, patches, or specific details, extract and list them if present in the context. If the context does not contain explicit threat intelligence matching the question, reply with: 'No matches or indicators found inside local threat intelligence database.'

Context:
{context}

Question: {input}
Analyst answer:
        """
    )

    doc_chain = create_stuff_documents_chain (llm, prompt)  # Summarizes the text from all the list of documents into 1 string which is passed as context to the prompt
    rag_chain = create_retrieval_chain(retriever, doc_chain)  # Take incoming user questions, give them to the retriever to find matching documents and then automatically hand those discovered documents directly over to the doc_chain

    response = rag_chain.invoke({"input": query})
    print(f"\n Query Results for {query}")
    print("-"*60)
    print(response["answer"])
    print("-"*60)

    # Fallback: If answer is generic, print the top-matching document
    if response["answer"].strip().lower().startswith("no matches"):
        # Try to print the most relevant document
        context_docs = response.get("context", [])
        if context_docs:
            print("\n[Top-matching document snippet]:\n")
            print(context_docs[0].page_content[:1000])  # Print up to 1000 chars
            print("\n--- End of snippet ---\n")

    return response.get("context", [])

# SQLite deduplication engine - Stores the article link, title and a timestamp. If the article is already processed, skip
def init_db() -> sqlite3.Connection:
    '''
    Setup the DB and table
    '''
    conn = sqlite3.connect('rss_feeds.db')  # Create/Open the DB
    cursor = conn.cursor()  # Execute/Fetch results of commands

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            link TEXT PRIMARY KEY,
            title TEXT,
            published TEXT
        )
    ''')
    conn.commit()
    return conn

def is_already_saved(conn: sqlite3.Connection, article_link: str) -> bool:
    '''
    Check if an article link has already been parsed
    '''
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM articles WHERE link = ?", (article_link,)
    )
    return cursor.fetchone() is not None

def save_article(conn: sqlite3.Connection, article_link: str, title: str, published_date: str) -> None:
    '''
    Insert the given article link, title and published date in the DB
    '''
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO articles (link, title, published) VALUES (?, ?, ?)",
            (article_link, title, published_date)
        )
        conn.commit()
        print(f"[+] New article saved: {title}")
    except sqlite3.IntegrityError:
        print(f"[-] Skipping duplicate: {title}")
    return None

# Cleanup the text - Fang the IOCs, html decode the text
def clean_and_normalize(raw_content: str) -> str | None:
    '''
    HTML decode the text and Fang the IOCs
    '''
    if not raw_content:
        return None
    
    html_decoded_text = html.unescape(raw_content)  # Decode HTML entities (e.g., &lt; to <, &amp; to &)
    html_decoded_text = re.sub(r'<[^>]+>', ' ', html_decoded_text)  # Strip HTML tags (e.g., <p>, <a>) to get pure plain text

    # Normalize Defanged URLs (hXXp://, hxxps://, hxxp[:]//, fxp://)
    html_decoded_text = re.compile(r'\b[hf]xxp(s)?:\/\/', re.IGNORECASE).sub(r'http\1://', html_decoded_text)
    html_decoded_text = re.compile(r'\b[hf]xxp(s)?\[:\]\/\/', re.IGNORECASE).sub(r'http\1://', html_decoded_text)
    html_decoded_text = re.compile(r'\b[hf]xxp(s)?\(:\)\/\/', re.IGNORECASE).sub(r'http\1://', html_decoded_text)

    # Fixes [.] [d] [t] (.) (d) (t)
    html_decoded_text = re.compile(r'\[\.\]|\[d\]|\[t\]|\(\.\)|\(d\)|\(t\)', re.IGNORECASE).sub('.', html_decoded_text)
    
    # Fixes [:] or (:) for ports/IPv6
    html_decoded_text = re.compile(r'\[:\]|\(: \)', re.IGNORECASE).sub(':', html_decoded_text)

    # 5. Clean up multi-spaces or deep newlines caused by stripping tags
    html_decoded_text = re.sub(r'\s+', ' ', html_decoded_text).strip()

    return html_decoded_text

# List of RSS feeds from which you want to fetch articles
def ingest_articles(conn: sqlite3.Connection) -> None:
    '''
    Given the list of RSS feeds, fetch the articles from each URL
    '''
    for url in FEEDS:
        print(f"[*] Processing Feed: {url}")
        feed = feedparser.parse(url)
    
        for entry in feed.entries:
            article_link = entry.get('link')  # Use the article link as the unique identifier to see if the article is already parsed or not

            if not article_link:
                print("[!] Skipping entry: No link found in feed data.")
                continue

            article_link = str(article_link)

            # Check if the link has already been processes
            if is_already_saved(conn, article_link):
                print(f"[-] Skipping duplicate (Already in DB): {entry.get('title')}")
                continue

            title = str(entry.get('title', 'Not Found/Untitled'))
            published_date = str(entry.get('published', entry.get('updated', datetime.datetime.now())))
            content_list = entry.get('content', [])
            raw_content = content_list[0]['value'] if content_list else ""

            decoded_text = clean_and_normalize(str(raw_content))

            if not decoded_text:  # skip in case the raw content was not extracted correctly
                continue
            
            ##  Note: If you have insufficient context window, truncate the decoded_text instead and pass for embedding. 
            process_and_store_rag(article_link, title, decoded_text)

            save_article(conn, article_link, title, published_date)
        return None

if __name__=="__main__":
    db_conn = init_db()
    try:
        ingest_articles(db_conn)
    finally:
        db_conn.close()
    
    # Integrated query after ingestion
    while True:
         # Reset all filters to False at the start of each new turn
        require_ips = False
        require_domains = False
        require_urls = False
        require_hashes = False

        query = str(input("\nASTRA at your service! Please let me know how I can help you? (Type 'bye' to exit): ").strip())

        if query.lower() == "bye":
            print("Bye! Have a great day.")
            break
        if not query:
            continue

        if "ip" in query.lower():
            require_ips = True
        if "hash" in query.lower():
            require_hashes = True
        if "domains" in query.lower():
            require_domains = True
        if "urls" in query.lower():
            require_urls = True

        print("\n[+] Fetching answer from the model, please wait...\n")
        results = query_threat_intel_rag(
            vector_store=vector_store,
            query=query,
            k=5,
            require_ips=require_ips,
            require_domains=require_domains,
            require_urls=require_urls,
            require_hashes=require_hashes,
        )
