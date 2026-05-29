#usr/bin/env python3

"""
ASTRA: AI - Security and Threat Report Analysis
Privacy-first, air-gapped, threat intelligence analysis and RAG system built with LangChain, ChromaDB, and Ollama. 
ASTRA fetches cybersecurity news from RSS feeds, extracts Indicators of Compromise (IOCs) using a local LLM, and stores them in a local vector database for fast retrieval. 
Users can query the system for specific IOCs or threat details, and ASTRA will provide structured insights based on the ingested data.

Architecture:
1. Data Ingestion: Fetches articles from specified RSS feeds and processes them to extract raw content.
2. IOC Extraction: Uses a local LLM (via Ollama) with a structured output schema to extract IOCs such as IPs, domains, URLs, and hashes from the articles.
3. Storage: Saves the extracted IOCs and article summary in a local Chroma DB for semantic search. Parsed articles and full article content are also stored in a SQLite database.
4. Query Interface: Users can input natural language queries, and ASTRA will retrieve relevant documents from Chroma DB based on semantic similarity and metadata filters, then generate a concise answer using the LLM.

Usage:
python app.py --mode ingest  # To fetch and process articles from RSS feeds
python app.py --mode query   # To start the interactive query interface after ingestion
"""
import argparse
import datetime
import html
import ipaddress
import re
import sqlite3
from typing import Optional

import feedparser
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

# ==========================================
# 0. CONFIGURATION
# ==========================================
LLM_MODEL = "granite4.1:8b"
EMBEDDING_MODEL = "qwen3-embedding:latest"
CHROMA_COLLECTION_NAME = "rss_iocs"
CHROMA_PERSIST_DIRECTORY = "./threat_intel_rag"
SQLITE_DB_PATH = "rss_feeds.db"
TOP_K = 5  # Number of top similar documents to retrieve for each query

FEEDS = ['https://cybersecuritynews.com/feed/']

# ===========================================
# 1. PROMPT TEMPLATE CONFIGURATION
# ==========================================
IOC_EXTRACTION_PROMPT = ChatPromptTemplate.from_template('''
You are a senior threat hunter that is an expert in extracting IOCs from unstructured news articles. You are good at identifying and extracting IOCs such as IP addresses, domains, URLs, and file hashes, compromised packages, affected software versions from cybersecurity news articles.
Your task is to extract Indicators of Compromise (IOCs) from the provided threat report text.

STRICT RULES:
1. Extract ONLY IOCs that are explicitly mentioned in the text. Do NOT infer, guess, or hallucinate any indicators that are not clearly present.
2. Normalize defanged indicators (e.g., convert 'hxxp://malicious[.]com' to 'http://malicious.com', and '192[.]168[.]1[.]1' to 192.168.1.1, [.] or [dot] to . etc.)
3. For domains, extract only the host/domain name (e.g., 'badsite.com') without any protocols (http://) or paths. Full URLs with paths should go into the URLs list.
4. Exclude private ip ranges: 10.x.x.x, 172.16.x.x - 172.31.x.x, 192.168.x.x unless they are explicitly mentioned as C2 or malicious in the text.
5. File hashes should be valid MD5, SHA-1, or SHA-256 hashes. Do not include any strings that do not match these formats.
6. If a specific IOC category (e.g., hashes) is not present in the text, leave that array empty. Do not guess or hallucinate indicators.
7. Do not include any legitimate, well-known entities (e.g., google.com, microsoft.com, adobe.com) as IOCs unless they are explicitly flagged as hijacked or acting as direct malicious C2 endpoints in the text.
8. **Pay special attention to any section, table, or list labeled 'Indicators of Compromise', 'IoCs', or similar.**
9. **If the text contains a table or structured list of IOCs, extract all valid indicators from it, even if they are not mentioned elsewhere in the narrative.**
10. Extract IOCs from both narrative and structured (table/list) formats.

Threat Report Text:
\"\"\"{content}\"\"\"                                                         
''')

SUMMARY_GENERATION_PROMPT = ChatPromptTemplate.from_template(''' 
You are a senior cybersecurity threat intelligence analyst. Read the following threat intelligence report and generate a comprehensive summary of the article below.

This summary will be stored as the semantic index for this article in a vector database. When an analyst queries the threat intelligence database, this summary must ensure that the article is retrieved for ANY relevant questions about its content.
                                                             
INCLUDE ALL OF THE FOLLOWING IF PRESENT:
1. Threat Actor or Campaign Names: Extract any specific names of threat actors, groups, or campaigns mentioned in the article (e.g., "APT28", "Lazarus Group", "Operation Aurora").
2. Attack Techniques and TTPs: include the MITRE ATT&CK techniques, tactics, and procedures described in the article along with the IDs.
3. Vulnerabilities and CVEs: List any specific vulnerabilities (with CVE identifiers) that are exploited or discussed in the article.
4. Affected Software and Versions: Mention any software, platforms, or versions that are identified as being targeted or affected by the threat.
5. Indicators of Compromise (IOCs) summary: e.g. "contains 3 IP addresses, 2 domains, 5 URLs, and 4 file hashes" without listing the actual IOCs since they are stored separately in the vector database.
6. Attack narrative: A concise description of the attack flow, techniques used, and any unique aspects of the campaign that are highlighted in the article.
7. Malware Family Names: If specific malware families are mentioned, include those as well (e.g., "Emotet", "TrickBot", "Pegasus").
8. Defensive Measures and Mitigations: If the article discusses any recommended defenses, patches, or mitigations, summarize those as well.
9. Campaign Timeline: If the article provides a timeline of events (e.g., when the attack started, key milestones), include that in the summary.
10. Any other unique or critical information that would help an analyst understand the nature and impact of the threat campaign described in the article.
                                                             
Write in dense information-rich prose. Every sentence must add retrieval value. Do not include any generic filler sentences that do not contain specific threat intelligence details.

Article Text:
\"\"\"{content}\"\"\"

Detailed Summary:
''')

RAG_QUERY_PROMPT = ChatPromptTemplate.from_template('''
You are a Threat Intelligence Analyst assistant with access to a local threat intelligence database. Use ONLY the provided context to answer the user's question. 

RESPONSE RULES:
1. Base your answer solely on the information contained in the provided context. Do NOT use any external knowledge or make assumptions beyond what is explicitly stated in the context.
2. If the context is insufficient to answer the question, or if the context does not contain explicit threat intelligence matching the question, reply with: 'The available threat intelligence database does not contain sufficient information on this topic. Try broadening your question or ingesting additional data.'
3. For IOC queries: list indicators grouped by type (IPs, domains, URLs, hashes) and present them in a clear, organized manner. If no indicators of a specific type are found, explicitly state 'No [IOC type] found in the database' for that category.
4. Attribute findings to their source article title when possible (e.g., "According to the article 'Title of Article', the following IPs were identified: ...").
5. If multiple articles in the context mention similar information, synthesize that information into a cohesive answer rather than listing each article separately, while still ensuring that all relevant details from the context are included in the response.
6. Use precise security terminology appropriate for a SOC analyst.
7. Do not speculate or extrapolate beyond the provided context. If the context does not explicitly mention a specific detail, do not include it in the answer.

THREAT INTELLIGENCE DATABASE CONTEXT:
{context}
                                                    
USER QUESTION: {input}

ANALYSIS:
''')

# ==========================================
# 2. IOC SCHEMA CONFIGURATION
# ==========================================
HASH_PATTERN = re.compile(r'^[a-fA-F0-9]{32}$|^[a-fA-F0-9]{40}$|^[a-fA-F0-9]{64}$')

class IOCSchema(BaseModel):
    ips: list[str] = Field(default=[], description="IPv4 or IPv6 addresses found in the text.")
    domains: list[str] = Field(default=[], description="Domain names, excluding protocols like http.")
    urls: list[str] = Field(default=[], description="URLs found in the text.")
    hashes: list[str] = Field(default=[], description="MD5, SHA-1, or SHA-256 cryptographic hashes.")

    @field_validator("ips", mode="after")
    @classmethod
    def validate_ips(cls, values: list[str]) -> list[str]:
        '''Accepts both IP addresses and subnets (CIDR notation), cleans whitespace.'''
        valid_ips = []
        for ip in values:
            cleaned_ip = ip.strip().strip(".,()[]{}'\"")
            try:
                # Accept both IP addresses and subnets
                if "/" in cleaned_ip:
                    ipaddress.ip_network(cleaned_ip, strict=False)
                else:
                    ipaddress.ip_address(cleaned_ip)
                valid_ips.append(cleaned_ip)
            except ValueError:
                print(f"[!] Invalid IP address or subnet found and skipped: {cleaned_ip}")
                continue  # Skip invalid entries but continue processing others
        return valid_ips

    @field_validator("hashes", mode="after")
    @classmethod
    def validate_hashes(cls, values: list[str]) -> list[str]:
        '''Filters out strings that are not valid hashes.'''
        valid_hashes = []
        for hash in values:
            cleaned_hash = hash.strip()
            if HASH_PATTERN.match(cleaned_hash):
                valid_hashes.append(cleaned_hash)
            else:
                print(f"[!] Invalid hash found and skipped: {cleaned_hash}")
        return valid_hashes

    def summary_line(self) -> str:
        '''Generates a one-line summary of the extracted IOCs for quick reference.'''
        summary_parts = []
        if self.ips:
            summary_parts.append(f"{len(self.ips)} IPs")
        if self.domains:
            summary_parts.append(f"{len(self.domains)} domains")
        if self.urls:
            summary_parts.append(f"{len(self.urls)} URLs")
        if self.hashes:
            summary_parts.append(f"{len(self.hashes)} hashes")

        return ", ".join(summary_parts) if summary_parts else "No IOCs extracted"

# ==========================================
# 3. DATABASE CONFIGURATION
# ==========================================

def init_db(db_path: str = SQLITE_DB_PATH) -> sqlite3.Connection:
    '''
    Setup the SQLite database and articles table
    '''
    conn = sqlite3.connect(db_path)  # Create/Open the DB
    cursor = conn.cursor()  # Execute/Fetch results of commands

    # Deduplication table to track processed articles by their unique link
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            link TEXT PRIMARY KEY,
            title TEXT,
            processed_at TIMESTAMP
        )
    ''')

    # Additional table to store full article content for reference
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS article_contents (
            link TEXT PRIMARY KEY,
            title TEXT,
            content TEXT
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

def save_article(conn: sqlite3.Connection, article_link: str, title: str) -> None:
    '''
    Insert the given article link and title in the DB with a timestamp
    '''
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO articles (link, title, processed_at) VALUES (?, ?, ?)",
        (article_link, title, datetime.datetime.now().isoformat())
    )
    conn.commit()
    print(f"[+] New article saved: {title}")
    return None

def save_article_content(conn: sqlite3.Connection, article_link: str, title: str, content: str) -> None:
    '''
    Insert the full article content in the DB for reference
    '''
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO article_contents (link, title, content) VALUES (?, ?, ?)",
        (article_link, title, content)
    )
    conn.commit()
    print(f"[+] Article content saved for: {title}")
    return None

def fetch_article_content(conn: sqlite3.Connection, article_link: list[str | None]) -> dict[str, str | None]:
    '''
    Fetch the full article content from the DB based on the article link
    '''
    if not article_link:
        print("[!] No article link provided for content retrieval.")
        return {}
    cursor = conn.cursor()
    placeholders = ','.join('?' for _ in article_link)
    cursor.execute(
        f"SELECT title, content FROM article_contents WHERE link IN ({placeholders})", article_link
    )
    results = cursor.fetchall()
    content_dict = {title: content for title, content in results}
    return content_dict

# ==========================================
# 4. CONTENT CLEANUP
# =========================================
def clean_and_normalize(raw_content: str) -> str:
    '''
    HTML decode the text and Fang the IOCs
    '''
    if not raw_content:
        return "None"
    
    html_decoded_text = html.unescape(raw_content)  # Decode HTML entities (e.g., &lt; to <, &amp; to &)
    html_decoded_text = re.sub(r'<[^>]+>', ' ', html_decoded_text)  # Strip HTML tags (e.g., <p>, <a>) to get pure plain text

    # Preserve <table>, <tr>, <th>, <td>, <thead>, <tbody>, <figure> tags as markers for LLM
    # Replace table tags with clear text markers
    html_decoded_text = re.compile(r'\b[hf]xxp(s)?\[:\]\/\/', re.IGNORECASE).sub(r'http\1://', html_decoded_text)
    html_decoded_text = re.compile(r'\b[hf]xxp(s)?\(:\)\/\/', re.IGNORECASE).sub(r'http\1://', html_decoded_text)

    html_decoded_text = re.sub(r'<figure[^>]*class="wp-block-table"[^>]*>', '\n[IOC TABLE]\n', html_decoded_text, flags=re.IGNORECASE)
    html_decoded_text = re.sub(r'</figure>', '\n[/IOC TABLE]\n', html_decoded_text, flags=re.IGNORECASE)
    
    html_decoded_text = re.sub(r'<table[^>]*>', '\n[TABLE]\n', html_decoded_text, flags=re.IGNORECASE)
    html_decoded_text = re.sub(r'</table>', '\n[/TABLE]\n', html_decoded_text, flags=re.IGNORECASE)

    # Remove all other HTML tags
    html_decoded_text = re.sub(r'<[^>]+>', ' ', html_decoded_text)

    # Clean up multi-spaces or deep newlines caused by stripping tags, but preserve newlines for table/section clarity
    html_decoded_text = re.sub(r'\n+', '\n', html_decoded_text)

    return html_decoded_text.strip()

# ==========================================
# 5. UTILITIES
# ==========================================
def chroma_metadata(ioc_result: IOCSchema) -> dict[str, bool]:
    '''
    Generates metadata for Chroma based on the presence of different IOC types. This allows for efficient filtering during retrieval.
    '''
    return {
        "has_ips": len(ioc_result.ips) > 0,
        "has_domains": len(ioc_result.domains) > 0,
        "has_urls": len(ioc_result.urls) > 0,
        "has_hashes": len(ioc_result.hashes) > 0
    }

def add_document_to_vector_store(vector_store: Chroma, page_content: str, metadata: dict[str, str | bool]) -> None:
    '''
    Adds a document to the Chroma vector store with the given content and metadata. This function abstracts the document creation and addition process.
    '''
    doc = Document(
        page_content=page_content,  # pyright: ignore[reportArgumentType]
        metadata=metadata
    )
    vector_store.add_documents([doc])
    print("[+] Document added to Chroma RAG database.")
    return None

def build_metadata_filter(query:str) -> Optional[dict]:
    '''
    Analyzes the user query to determine if specific IOC types are being requested, and builds a metadata filter for Chroma accordingly.
    '''
    q = query.lower()
    conditions = []

    if any(kw in q for kw in ["ip", "ip address", "ipv4", "ipv6"]):
        conditions.append({"has_ips": True})
    if any(kw in q for kw in ["domain", "hostname", "fqdn", "subdomain"]):
        conditions.append({"has_domains": True})
    if any(kw in q for kw in ["url", "link", "endpoint"]):
        conditions.append({"has_urls": True})
    if any(kw in q for kw in ["hash", "md5", "sha1", "sha256"]):
        conditions.append({"has_hashes": True})

    if not conditions:
        return None  # No specific IOC type requested, so no filter needed
    if len(conditions) == 1:
        return conditions[0]  # Single condition can be returned directly
    return {"$or": conditions}



# ==========================================
# 6. INGESTION AND PROCESSING
# ==========================================
def process_article_and_store_in_rag(entry, llm: ChatOllama, vector_store: Chroma, conn: sqlite3.Connection) -> bool:
    '''
    For a given RSS feed entry, extract the required content, clean it up, extract IOCs using the LLM, and save everything as a document in Chroma. Also saves article metadata and content in SQLite for reference.
    '''
    article_link = entry.get('link')
    title = str(entry.get('title', 'Not Found/Untitled').strip())
    content_list = entry.get('content', [])
    raw_content = content_list[0]['value'] if content_list else ""

    if not article_link:
        print("[!] Article skipped due to missing link.")
        return False
    if is_already_saved(conn, article_link):
        print(f"[-] Skipping already processed article: {title}")
        return False
    
    cleaned_content = clean_and_normalize(raw_content)

    # IOC extraction and RAG storage
    try:
        structured_llm = llm.with_structured_output(IOCSchema)
        ioc_result = structured_llm.invoke(IOC_EXTRACTION_PROMPT.format(content=cleaned_content))
        ioc_result = IOCSchema.model_validate(ioc_result)  # Validate and clean the extracted IOCs using Pydantic
        print("\n" + "-"*50)
        print(f"[*] Extracted IOCs for: {title}\nSummary of extracted IOCs: {ioc_result.summary_line()}")

    except Exception as e:
        print(f"[!] IOC extraction failed for {title}. Reason: {e}")
        ioc_result = IOCSchema()  # Use an empty IOC result to still save the article content and metadata
    
    # Summary generation for semantic indexing in Chroma
    try: 
        chain = SUMMARY_GENERATION_PROMPT | llm  | StrOutputParser()  # Create a simple chain to generate the summary
        summary = chain.invoke({"content": cleaned_content})
        print(f"[*] Generated summary for: {title}\nSummary snippet: {summary[:200]}...")
    
    except Exception as e:
        print(f"[!] Summary generation failed for {title}. Reason: {e}")
        summary = "Summary generation failed."  # Fallback summary in case of failure
    
    # Construct the page content for Chroma, combining the summary and the extracted IOCs in a structured format
    flags = chroma_metadata(ioc_result) if ioc_result else {"has_ips": False, "has_domains": False, "has_urls": False, "has_hashes": False}
    metadata = {
        "article_link": article_link,
        "title": title,
        **flags
    }

    add_document_to_vector_store(vector_store, page_content=summary, metadata=metadata)

    # Save the article metadata and full content in SQLite for reference
    save_article(conn, article_link, title)
    save_article_content(conn, article_link, title, cleaned_content)

    return True

def ingest_articles(llm: ChatOllama, vector_store: Chroma, conn: sqlite3.Connection):
    '''
    Fetches articles from the specified RSS feeds, processes them, and stores the extracted information in Chroma and SQLite.
    '''
    count = 0
    for feed_url in FEEDS:
        print(f"\n[+] Fetching articles from feed: {feed_url}")
        feed = feedparser.parse(feed_url)
        entries = feed.entries
        print(f"[*] Found {len(entries)} articles in the feed {feed_url}.")

        for entry in entries:
            if process_article_and_store_in_rag(entry, llm, vector_store, conn):
                count += 1
    print(f"\n[+] Ingestion complete. Total new articles processed and stored: {count}\n")

# ==========================================
# 7. QUERYING THE RAG
# ==========================================
def query_threat_intel_rag(query:str, conn: sqlite3.Connection, vector_store: Chroma, llm: ChatOllama) -> str:
    '''
    Queries the Chroma RAG database based on the user input and returns a concise answer generated by the LLM.
    '''
    search_kwargs: dict = {"k": TOP_K}
    metadata_filter = build_metadata_filter(query)
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)  # Convert the Chroma collection into a retriever with the specified search parameters

    relevant_docs = retriever.invoke(query)  # Retrieve relevant documents based on the query and metadata filter

    if not relevant_docs:
        return("The available threat intelligence database does not contain sufficient information on this topic. Try broadening your question or ingesting additional data.")
    
    # For the retrieved documents, we can fetch the full article content from SQLite using the article links stored in the metadata. This allows the LLM to have more context when generating the answer.
    article_links: list[str | None] = [doc.metadata.get("article_link") for doc in relevant_docs if doc.metadata.get("article_link")]
    article_content_dict = fetch_article_content(conn, article_links)

    # Build the context for LLM
    separator = "\n---\n"
    context = separator.join(
        f"Title: {title}\n\n{content}"
        for title, content in article_content_dict.items())
    
    try:
        rag_chain = RAG_QUERY_PROMPT | llm | StrOutputParser()  # Create a simple chain to generate the answer based on the retrieved context
        return rag_chain.invoke({"context": context, "input": query})
    except Exception as e:
        print(f"[!] RAG query failed. Reason: {e}")
        return "An error occurred while processing your query. Please try again."
    
# ==========================================
# 8. INTERACTIVE QUERY INTERFACE
# ==========================================
def interactive_query_interface(conn: sqlite3.Connection, vector_store: Chroma, llm: ChatOllama):
    '''
    Provides an interactive command-line interface for users to input queries and receive answers based on the ingested threat intelligence data.
    '''
    print("\n" + "="*50)
    print("\n[+] Entering ASTRA - interactive query mode.")
    print("\n" + "="*50)
    print("\n[Instructions]")
    print(" - Type your threat intelligence query and press Enter to get an answer based on the ingested data.")
    print(" - Commands: Type 'exit' or 'quit' to leave the interactive mode.")

    while True:
        try:
            query = input("\nQuery > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[+] Exiting interactive query mode. Goodbye!")
            break

        if query.lower() in ['exit', 'quit']:
            print("[+] Exiting interactive query mode. Goodbye!")
            break
        if not query:
            print("[!] Please enter a valid query.")
            continue
        
        print("\n[+] Processing your query, please wait...\n")
        print(" [*] This may take a moment as the system retrieves relevant information and generates a response based on the ingested threat intelligence data.\n")
        answer = query_threat_intel_rag(query=query, conn=conn, vector_store=vector_store, llm=llm)
        print("\n[+] Query processed. Here's the answer based on the available threat intelligence data:\n")
        print("\n" + "="*50)
        print(f"{answer}")
        print("\n" + "="*50)

# ==========================================
# 9. MAIN FUNCTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="ASTRA: AI - Security and Threat Report Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
            Usage:
            python app.py --mode ingest  # To fetch and process articles from RSS feeds
            python app.py --mode query   # To start the interactive query interface after ingestion
        ''')
    parser.add_argument('--mode', choices=['ingest', 'query'], required=True, help="ingest: process RSS Feed | query: interactive query interface.")
    args = parser.parse_args()

    print("\n[+] Initializing ASTRA components...")
    print(f"[*] LLM Model: {LLM_MODEL}")
    print(f"[*] Embedding Model: {EMBEDDING_MODEL}")

    # Add parameter base_url="http://<ip>:11434" to both the llm and embeddings if your Ollama server is running inside a container and you are connecting from the host machine. Use "http://localhost:11434" if you are running the script inside the same container as Ollama.
    llm = ChatOllama(base_url="http://172.23.96.1:11434", model=LLM_MODEL, temperature=0.0)  # Initialize the local LLM via Ollama
    embeddings = OllamaEmbeddings(base_url="http://172.23.96.1:11434", model=EMBEDDING_MODEL)  # Initialize the embeddings model via Ollama
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME, 
        embedding_function=embeddings, 
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"})  # Initialize Chroma vector store
    conn = init_db()  # Initialize SQLite database

    try:
        if args.mode == 'ingest':
            ingest_articles(llm, vector_store, conn)
        elif args.mode == 'query':
            interactive_query_interface(conn, vector_store, llm)
    finally:
        conn.close()  # Ensure the database connection is closed when the program exits
        print("\n[+] ASTRA execution completed. Database connection closed.")

if __name__ == "__main__":
    main()


