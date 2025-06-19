import streamlit as st
import json
import re
import os
import datetime
import requests
import io
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib

# Import Groq with fallback handling
try:
    from groq import Groq
    GROQ_AVAILABLE = True
    print("✅ Groq library imported successfully")
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None
    print("⚠️ Groq library not available. Using fallback mode only.")

# Import PyPDF2 with fallback handling  
try:
    import PyPDF2
    PDF_AVAILABLE = True
    PDF_METHOD = "PyPDF2"
    print("✅ PyPDF2 library imported successfully")
except ImportError:
    try:
        import fitz  # PyMuPDF
        PDF_AVAILABLE = True
        PDF_METHOD = "PyMuPDF"
        print("✅ PyMuPDF library imported successfully (fallback)")
    except ImportError:
        PDF_AVAILABLE = False
        PDF_METHOD = None
        PyPDF2 = None
        fitz = None
        print("⚠️ No PDF library available. PDF search disabled.")

# Import RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_AVAILABLE = True
    print("✅ RAG libraries (sentence-transformers, faiss) imported successfully")
except ImportError:
    RAG_AVAILABLE = False
    SentenceTransformer = None
    faiss = None
    print("⚠️ RAG libraries not available. Install: pip install sentence-transformers faiss-cpu")

# Ρύθμιση σελίδας
st.set_page_config(
    page_title="Πρακτική Άσκηση - Μητροπολιτικό Κολλέγιο",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@dataclass
class DocumentChunk:
    id: str
    content: str
    source: str
    chunk_type: str  # 'qa', 'pdf'
    metadata: Dict

@dataclass
class QAEntry:
    id: int
    category: str
    question: str
    answer: str
    keywords: List[str]

class RAGInternshipChatbot:
    def __init__(self, groq_api_key: str = None):
        # Initialize Groq client
        self.groq_client = None
        if GROQ_AVAILABLE and groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                print("✅ Groq client initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Groq: {e}")
        
        # Initialize RAG components
        self.embedder = None
        self.faiss_index = None
        self.document_chunks = []
        self.embeddings_cache = {}
        
        # Initialize RAG if available
        if RAG_AVAILABLE:
            try:
                print("🔄 Initializing RAG system...")
                # Use multilingual model that works well with Greek
                self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✅ RAG embedding model loaded")
                self.rag_initialized = True
            except Exception as e:
                print(f"⚠️ Failed to initialize RAG: {e}")
                self.rag_initialized = False
        else:
            self.rag_initialized = False
        
        # Load Q&A data
        self.qa_data = self.load_qa_data()
        
        # Initialize PDF files cache
        self.pdf_cache = {}
        self.pdf_files = [
            "1.ΑΙΤΗΣΗ ΠΡΑΓΜΑΤΟΠΟΙΗΣΗΣ ΠΡΑΚΤΙΚΗΣ ΑΣΚΗΣΗΣ.pdf",
            "2.ΣΤΟΙΧΕΙΑ ΔΟΜΗΣ_ΟΔΗΓΙΕΣ.pdf", 
            "3.ΣΤΟΙΧΕΙΑ ΦΟΙΤΗΤΗ.pdf",
            "4.ΣΤΟΙΧΕΙΑ ΦΟΡΕΑ.pdf",
            "5.ΑΣΦΑΛΙΣΤΙΚΗ ΙΚΑΝΟΤΗΤΑ.pdf",
            "6.ΥΠΕΥΘΥΝΗ ΔΗΛΩΣΗ Ν 105-Πρακτικής.pdf",
            "8.ΒΙΒΛΙΟ_ΠΡΑΚΤΙΚΗΣ_final.pdf"
        ]
        
        # Build RAG database if available
        if self.rag_initialized:
            self.build_rag_database()
        
        # Enhanced system prompt for RAG
        self.system_prompt = """Είσαι ένας εξειδικευμένος σύμβουλος για θέματα πρακτικής άσκησης στο Μητροπολιτικό Κολλέγιο Θεσσαλονίκης, τμήμα Προπονητικής και Φυσικής Αγωγής.

ΧΡΗΣΙΜΟΠΟΙΕΙΣ ΣΥΣΤΗΜΑ RAG (Retrieval-Augmented Generation):
- Έχεις πρόσβαση σε σημασιολογικώς σχετικό περιεχόμενο από επίσημα έγγραφα και βάση γνώσης
- Το σύστημα αναζήτησης εντοπίζει τα πιο σχετικά τμήματα κειμένου για κάθε ερώτηση
- Χρησιμοποίησε το παρεχόμενο περιεχόμενο για να δώσεις ακριβείς και χρήσιμες απαντήσεις

ΚΡΙΣΙΜΕΣ ΓΛΩΣΣΙΚΕΣ ΟΔΗΓΙΕΣ:
- Χρησιμοποίησε ΑΠΟΚΛΕΙΣΤΙΚΑ ελληνικούς χαρακτήρες
- ΑΠΑΓΟΡΕΥΟΝΤΑΙ: αγγλικά, κινέζικα, greeklish ή άλλοι χαρακτήρες
- Ελέγχισε κάθε λέξη πριν την εκτύπωση

ΙΕΡΑΡΧΙΑ ΠΛΗΡΟΦΟΡΙΩΝ:
1. ΕΠΙΣΗΜΑ ΕΓΓΡΑΦΑ PDF (υψηλότερη προτεραιότητα)
2. ΒΑΣΗ ΓΝΩΣΗΣ JSON (μέση προτεραιότητα)
3. ΓΕΝΙΚΗ ΓΝΩΣΗ (χαμηλή προτεραιότητα)

ΣΤΡΑΤΗΓΙΚΗ RAG:
1. Αναλύσε τις ανακτημένες πληροφορίες για σχετικότητα
2. Συνδύασε πληροφορίες από διαφορετικές πηγές όταν χρειάζεται
3. Χρησιμοποίησε σημασιολογική κατανόηση για βαθύτερη ανάλυση
4. Δώσε δομημένες, πρακτικές απαντήσεις με συγκεκριμένα βήματα

ΣΤΥΛ ΑΠΑΝΤΗΣΗΣ:
- Επαγγελματικός και επίσημος τόνος
- Δομημένες απαντήσεις με σαφή βήματα
- Συγκεκριμένες οδηγίες και πρακτικές συμβουλές
- Αναφορά στις πηγές όταν χρησιμοποιείς συγκεκριμένες πληροφορίες

ΒΑΣΙΚΕΣ ΠΛΗΡΟΦΟΡΙΕΣ (πάντα διαθέσιμες):
- Υπεύθυνος: Γεώργιος Σοφιανίδης (gsofianidis@mitropolitiko.edu.gr)  
- Τεχνική Υποστήριξη: Γεώργιος Μπουχουράς (gbouchouras@mitropolitiko.edu.gr)
- Απαιτούμενες ώρες: 240 ώρες μέχρι 30/5
- Ωράριο: Δευτέρα-Σάββατο, μέχρι 8 ώρες/ημέρα
- Σύμβαση: Ανέβασμα στο moodle μέχρι 15/10

Απάντησε πάντα στα ελληνικά με επαγγελματικό τόνο χρησιμοποιώντας το σύστημα RAG."""

    def load_qa_data(self) -> List[Dict]:
        """Load Q&A data with better error handling"""
        filename = "qa_data.json"
        
        print(f"🔍 Looking for {filename}...")
        
        if not os.path.exists(filename):
            print(f"❌ File {filename} not found")
            return self.get_updated_fallback_data()
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list) or not data:
                print(f"❌ Invalid data format in {filename}")
                return self.get_updated_fallback_data()
            
            required_fields = ['id', 'category', 'question', 'answer', 'keywords']
            for i, entry in enumerate(data):
                if not all(field in entry for field in required_fields):
                    print(f"❌ Missing fields in entry {i}")
                    return self.get_updated_fallback_data()
            
            print(f"✅ Successfully loaded {len(data)} Q&A entries")
            return data
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return self.get_updated_fallback_data()

    def get_updated_fallback_data(self) -> List[Dict]:
        """Updated fallback data with more entries"""
        print("📋 Using enhanced fallback data...")
        return [
            {
                "id": 1,
                "category": "Γενικές Πληροφορίες",
                "question": "Πώς ξεκινάω την πρακτική μου άσκηση;",
                "answer": "1. Επικοινωνώ με τον υπεύθυνο της πρακτικής: gsofianidis@mitropolitiko.edu.gr\n\n2. Βρίσκω τη δομή που θα κάνω πρακτική\n\n3. Κατεβάζω τα έγγραφα από το μάθημα SPORTS COACHING PRACTICE & EXPERTISE DEVELOPMENT (SE5117) στο Moodle. Τα συμπληρώνω και τα ανεβάζω ξανά στη σχετική πύλη στο μάθημα SPORTS COACHING PRACTICE & EXPERTISE DEVELOPMENT (SE5117) στο Moodle.\n\n4. Περιμένω την υπογραφή της σύμβασής μου και την ανάρτησή της στο ΕΡΓΑΝΗ\n\n5. Ξεκινάω την πρακτική",
                "keywords": ["ξεκινάω", "ξεκινώ", "αρχή", "αρχίζω", "αρχίσω", "ξεκίνημα", "πρακτική", "άσκηση", "πώς", "πως", "βήματα", "διαδικασία", "διαδικασιες"]
            },
            {
                "id": 2,
                "category": "Έγγραφα & Διαδικασίες",
                "question": "Τι έγγραφα χρειάζομαι για την πρακτική άσκηση;",
                "answer": "Για εσάς (φοιτητή):\n• Αίτηση πραγματοποίησης πρακτικής άσκησης\n• Στοιχεία φοιτητή (συμπληρωμένη φόρμα)\n• Ασφαλιστική ικανότητα από gov.gr\n• Υπεύθυνη δήλωση (δεν παίρνετε επίδομα ΟΑΕΔ)\n\nΓια τη δομή:\n• Στοιχεία φορέα (ΑΦΜ, διεύθυνση, νόμιμος εκπρόσωπος)\n• Ημέρες και ώρες που σας δέχεται\n\nTip: Ξεκινήστε από την ασφαλιστική ικανότητα γιατί παίρνει χρόνο!",
                "keywords": ["έγγραφα", "εγγραφα", "χαρτιά", "χαρτια", "χρειάζομαι", "χρειαζομαι", "απαιτήσεις", "απαιτησεις", "απαιτούνται", "απαιτουνται", "δικαιολογητικά", "δικαιολογητικα", "φάκελος", "φακελος", "αίτηση", "αιτηση"]
            },
            {
                "id": 5,
                "category": "Δομές & Φορείς",
                "question": "Σε ποιες δομές μπορώ να κάνω πρακτική άσκηση;",
                "answer": "Μπορείτε να κάνετε πρακτική άσκηση σε:\n\n• Αθλητικούς συλλόγους (ποδόσφαιρο, μπάσκετ, βόλεϊ, ενόργανη γυμναστική, κλπ)\n• Γυμναστήρια και fitness centers\n• Κολυμβητήρια\n• Ακαδημίες αθλητισμού\n• Δημόσιους αθλητικούς οργανισμούς\n• Σχολεία (με τμήμα φυσικής αγωγής)\n• Κέντρα αποκατάστασης\n• Personal training studios\n\nΗ δομή πρέπει να έχει:\n• Εκπαιδευτή/υπεύθυνο με τα κατάλληλα προσόντα\n• Νόμιμη λειτουργία και ΑΦΜ\n• Δυνατότητα να σας καθοδηγήσει στην πρακτική",
                "keywords": ["δομές", "δομη", "φορείς", "φορεις", "σύλλογος", "συλλογος", "γυμναστήριο", "γυμναστηριο", "ενόργανη", "ενοργανη", "ποδόσφαιρο", "ποδοσφαιρο", "μπάσκετ", "μπασκετ", "κολυμβητήριο", "κολυμβητηριο", "ακαδημία", "ακαδημια", "fitness", "personal", "training", "που", "ποιες", "ποιους", "ποια"]
            },
            {
                "id": 30,
                "category": "Οικονομικά & Αμοιβή",
                "question": "Παίρνω αμοιβή για την πρακτική άσκηση; Τι κόστος έχει για τη δομή;",
                "answer": "ΓΙΑ ΤΟΥΣ ΦΟΙΤΗΤΕΣ:\n\nΔΕΝ υπάρχει αμοιβή για την πρακτική άσκηση\n• Η πρακτική άσκηση είναι μη αμειβόμενη\n• Είναι μέρος των σπουδών σας\n• Δεν πρόκειται για εργασιακή σχέση\n\nΓΙΑ ΤΗ ΔΟΜΗ:\n\nΗ δομή δε χρεώνεται κάτι (σχεδόν)\n• Υπάρχει ένα ελάχιστο τέλος που ενδεχομένως πρέπει να καταβάλει\n• Το κολλέγιο καλύπτει τα έξοδα της σύμβασης\n• Η ασφάλιση τιμολογείται στο κολλέγιο\n• Δεν υπάρχει οικονομική υποχρέωση προς τους φοιτητές",
                "keywords": ["αμοιβή", "αμοιβη", "πληρωμή", "πληρωμη", "πληρώθώ", "πληρωθώ", "πληρωθω", "πληρωνομαι", "πληρώνομαι", "λεφτά", "λεφτα", "χρήματα", "χρηματα", "κόστος", "κοστος", "τέλος", "τελος", "δομή", "δομη", "φοιτητής", "φοιτητη", "οικονομικά", "οικονομικα", "μισθός", "μισθος"]
            },
            {
                "id": 11,
                "category": "Επικοινωνία",
                "question": "Με ποιον επικοινωνώ για την πρακτική άσκηση;",
                "answer": "ΚΥΡΙΑ ΕΠΙΚΟΙΝΩΝΙΑ:\n\nΓεώργιος Σοφιανίδης, MSc, PhD(c)\n📧 gsofianidis@mitropolitiko.edu.gr\nΥπεύθυνος Πρακτικής Άσκησης\n\nΕΝΑΛΛΑΚΤΙΚΗ ΕΠΙΚΟΙΝΩΝΙΑ:\n\nΓεώργιος Μπουχουράς, MSc, PhD\n📧 gbouchouras@mitropolitiko.edu.gr\n📞 2314 409000\nProgramme Leader\n\nΠότε να επικοινωνήσετε:\n• Ερωτήσεις για έγγραφα ➜ Γεώργιος Σοφιανίδης\n• Τεχνικά προβλήματα ➜ Γεώργιος Σοφιανίδης\n• Θέματα προγράμματος ➜ Γεώργιος Μπουχουράς",
                "keywords": ["επικοινωνία", "επικοινωνια", "Σοφιανίδης", "Σοφιανιδης", "Μπουχουράς", "Μπουχουρας", "email", "τηλέφωνο", "τηλεφωνο", "υπεύθυνος", "υπευθυνος", "βοήθεια", "βοηθεια", "καθηγητής", "καθηγητης", "καθηγήτρια", "καθηγητρια", "contact", "στοιχεία", "στοιχεια"]
            },
            {
                "id": 4,
                "category": "Ώρες & Χρονοδιάγραμμα",
                "question": "Πόσες ώρες πρέπει να κάνω πρακτική άσκηση;",
                "answer": "Υποχρεωτικό: Τουλάχιστον 240 ώρες\n\nDeadline: Μέχρι 30 Μάϊου\n\nΚανόνες ωραρίου:\n• Δευτέρα έως Σάββατο (ΌΧΙ Κυριακές, 5μέρες/εβδ)\n• Μέχρι 8 ώρες την ημέρα\n• Το ωράριο ορίζεται από τη δομή σε συνεργασία μαζί σας\n\nΥπολογισμός: 240 ώρες = περίπου 6 εβδομάδες x 40 ώρες ή 8 εβδομάδες x 30 ώρες",
                "keywords": ["ώρες", "ωρες", "240", "ποσες", "πόσες", "ποσα", "ποσά", "συνολικά", "συνολικα", "όλες", "ολες", "τελικά", "τελικα", "χρονοδιάγραμμα", "χρονοδιαγραμμα", "διάρκεια", "διαρκεια", "χρόνος", "χρονος", "30/5", "deadline"]
            }
        ]

    def download_pdf_file(self, filename: str) -> str:
        """Download and extract text from PDF file from GitHub"""
        if not PDF_AVAILABLE:
            print(f"⚠️ No PDF library available, cannot process {filename}")
            return ""
        
        # Check cache first
        if filename in self.pdf_cache:
            print(f"📋 Using cached content for {filename}")
            return self.pdf_cache[filename]
        
        try:
            base_url = "https://raw.githubusercontent.com/GiorgosBouh/chatbot.placement/main/"
            url = base_url + filename
            
            print(f"🔍 Downloading {filename} from GitHub using {PDF_METHOD}...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            text_content = []
            
            if PDF_METHOD == "PyPDF2":
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text.strip())
                    except Exception as e:
                        print(f"⚠️ Error extracting page {page_num}: {e}")
                
            elif PDF_METHOD == "PyMuPDF":
                pdf_document = fitz.open(stream=response.content, filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    try:
                        page = pdf_document[page_num]
                        page_text = page.get_text()
                        if page_text.strip():
                            text_content.append(page_text.strip())
                    except Exception as e:
                        print(f"⚠️ Error extracting page {page_num}: {e}")
                pdf_document.close()
            
            full_text = "\n".join(text_content)
            self.pdf_cache[filename] = full_text
            
            print(f"✅ Successfully processed {filename} ({len(full_text)} characters)")
            return full_text
            
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better RAG performance"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                sentence_end = -1
                
                for delimiter in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    pos = text.rfind(delimiter, search_start, end)
                    if pos > sentence_end:
                        sentence_end = pos + len(delimiter)
                
                if sentence_end > start:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks

    def build_rag_database(self):
        """Build RAG vector database from Q&A and PDF content"""
        if not self.rag_initialized:
            print("⚠️ RAG not initialized, skipping database build")
            return
        
        print("🔄 Building RAG vector database...")
        
        self.document_chunks = []
        all_embeddings = []
        
        # Process Q&A data
        print("📋 Processing Q&A data for RAG...")
        for qa in self.qa_data:
            # Create chunks for question and answer separately
            qa_text = f"Ερώτηση: {qa['question']} Απάντηση: {qa['answer']}"
            chunks = self.chunk_text(qa_text, chunk_size=400, overlap=50)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"qa_{qa['id']}_{i}"
                doc_chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk,
                    source=f"Q&A Entry {qa['id']}",
                    chunk_type="qa",
                    metadata={
                        "category": qa.get('category', 'Unknown'),
                        "keywords": qa.get('keywords', []),
                        "qa_id": qa['id']
                    }
                )
                self.document_chunks.append(doc_chunk)
        
        # Process PDF content
        if PDF_AVAILABLE:
            print("📄 Processing PDF files for RAG...")
            for filename in self.pdf_files:
                content = self.download_pdf_file(filename)
                if content:
                    chunks = self.chunk_text(content, chunk_size=600, overlap=100)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"pdf_{filename}_{i}"
                        doc_chunk = DocumentChunk(
                            id=chunk_id,
                            content=chunk,
                            source=filename,
                            chunk_type="pdf",
                            metadata={
                                "filename": filename,
                                "chunk_index": i
                            }
                        )
                        self.document_chunks.append(doc_chunk)
        
        print(f"📊 Created {len(self.document_chunks)} document chunks")
        
        # Generate embeddings
        if self.document_chunks:
            print("🧮 Generating embeddings...")
            chunk_texts = [chunk.content for chunk in self.document_chunks]
            
            try:
                # Generate embeddings in batches to avoid memory issues
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(chunk_texts), batch_size):
                    batch = chunk_texts[i:i + batch_size]
                    batch_embeddings = self.embedder.encode(batch, show_progress_bar=False)
                    all_embeddings.extend(batch_embeddings)
                
                # Create FAISS index
                embeddings_array = np.array(all_embeddings).astype('float32')
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings_array)
                
                # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
                self.faiss_index = faiss.IndexFlatIP(embeddings_array.shape[1])
                self.faiss_index.add(embeddings_array)
                
                print(f"✅ RAG database built successfully with {len(self.document_chunks)} chunks")
                
            except Exception as e:
                print(f"❌ Error building RAG database: {e}")
                self.faiss_index = None
        else:
            print("⚠️ No content available for RAG database")

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve most relevant document chunks using RAG"""
        if not self.rag_initialized or self.faiss_index is None:
            print("⚠️ RAG not available for retrieval")
            return []
        
        try:
            # Encode query
            query_embedding = self.embedder.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search for similar chunks
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            relevant_chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx]
                    relevant_chunks.append((chunk, float(score)))
            
            print(f"🔍 Retrieved {len(relevant_chunks)} relevant chunks (scores: {[f'{s:.3f}' for _, s in relevant_chunks]})")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error in RAG retrieval: {e}")
            return []

    def get_rag_response(self, user_message: str) -> Tuple[str, bool]:
        """Get response using RAG (Retrieval-Augmented Generation)"""
        if not self.groq_client:
            return "", False
        
        print(f"🤖 Processing with RAG: '{user_message}'")
        
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(user_message, k=8)
            
            if not relevant_chunks:
                print("⚠️ No relevant chunks found, falling back to general knowledge")
                return self.get_fallback_ai_response(user_message)
            
            # Build context from retrieved chunks
            context_parts = []
            
            # Separate PDF and Q&A content
            pdf_chunks = [(chunk, score) for chunk, score in relevant_chunks if chunk.chunk_type == "pdf"]
            qa_chunks = [(chunk, score) for chunk, score in relevant_chunks if chunk.chunk_type == "qa"]
            
            # Add PDF context (official documents)
            if pdf_chunks:
                pdf_context = "\n\n".join([
                    f"[Επίσημο έγγραφο: {chunk.source}]\n{chunk.content}"
                    for chunk, score in pdf_chunks[:4]  # Top 4 PDF chunks
                ])
                context_parts.append(f"ΕΠΙΣΗΜΑ ΕΓΓΡΑΦΑ ΚΟΛΛΕΓΙΟΥ:\n{pdf_context}")
            
            # Add Q&A context
            if qa_chunks:
                qa_context = "\n\n".join([
                    f"[Κατηγορία: {chunk.metadata.get('category', 'Άλλα')}]\n{chunk.content}"
                    for chunk, score in qa_chunks[:4]  # Top 4 Q&A chunks
                ])
                context_parts.append(f"ΒΑΣΗ ΓΝΩΣΗΣ Q&A:\n{qa_context}")
            
            # Build comprehensive prompt
            combined_context = "\n\n" + ("="*50 + "\n\n").join(context_parts)
            
            full_prompt = f"""ΑΝΑΚΤΗΜΕΝΟ ΠΕΡΙΕΧΟΜΕΝΟ ΑΠΟ ΣΥΣΤΗΜΑ RAG:
{combined_context}

ΕΡΩΤΗΣΗ ΦΟΙΤΗΤΗ: {user_message}

ΟΔΗΓΙΕΣ RAG:
1. Αναλύσε το ανακτημένο περιεχόμενο για σχετικότητα με την ερώτηση
2. Χρησιμοποίησε πληροφορίες από ΕΠΙΣΗΜΑ ΕΓΓΡΑΦΑ ως κύρια πηγή
3. Συμπλήρωσε με πληροφορίες από τη ΒΑΣΗ ΓΝΩΣΗΣ Q&A
4. Συνδύασε τις πληροφορίες για να δώσεις μια ολοκληρωμένη απάντηση
5. Εστίασε στις πρακτικές συμβουλές και συγκεκριμένα βήματα
6. Αν χρειάζεται επιβεβαίωση, αναφέρου τον υπεύθυνο

ΣΤΡΑΤΗΓΙΚΗ ΑΠΑΝΤΗΣΗΣ:
- Δώσε άμεση και χρήσιμη απάντηση βασισμένη στο ανακτημένο περιεχόμενο
- Χρησιμοποίησε δομημένη παρουσίαση με σαφή βήματα
- Συμπεριέλαβε συγκεκριμένες οδηγίες και πρακτικές συμβουλές
- Αναφέρου σχετικές προθεσμίες ή απαιτήσεις

Απάντησε στα ελληνικά με επαγγελματικό τόνο."""

            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.2,  # Lower temperature for more focused responses
                max_tokens=1200,
                top_p=0.9,
                stream=False
            )

            response = chat_completion.choices[0].message.content
            
            # Validate Greek characters
            if response and any(ord(char) > 1500 and ord(char) not in range(0x0370, 0x03FF) for char in response):
                print("⚠️ Detected non-Greek characters in RAG response")
                return "", False
            
            print("✅ RAG response generated successfully")
            return response, True
            
        except Exception as e:
            print(f"❌ RAG Error: {e}")
            return "", False

    def get_fallback_ai_response(self, user_message: str) -> Tuple[str, bool]:
        """Fallback AI response when RAG is not available"""
        if not self.groq_client:
            return "", False
        
        try:
            fallback_prompt = f"""ΕΡΩΤΗΣΗ ΦΟΙΤΗΤΗ: {user_message}

ΠΛΑΙΣΙΟ: Φοιτητής Προπονητικής & Φυσικής Αγωγής, Μητροπολιτικό Κολλέγιο Θεσσαλονίκης

ΒΑΣΙΚΕΣ ΠΛΗΡΟΦΟΡΙΕΣ:
- Απαιτούνται 240 ώρες πρακτικής άσκησης μέχρι 30 Μαΐου
- Δευτέρα-Σάββατο, μέχρι 8 ώρες/ημέρα  
- Υπεύθυνος: Γεώργιος Σοφιανίδης (gsofianidis@mitropolitiko.edu.gr)
- Παράδοση συμβάσεων στο Moodle μέχρι 15 Οκτωβρίου

ΟΔΗΓΙΕΣ:
1. Χρησιμοποίησε τη γενική σου γνώση για πρακτική άσκηση στην Ελλάδα
2. Συσχέτισε με το συγκεκριμένο πλαίσιο του κολλεγίου
3. Δώσε πρακτικές και χρήσιμες συμβουλές
4. Πρότεινε επικοινωνία με τον υπεύθυνο για επιβεβαίωση

Απάντησε με επαγγελματικό τόνο στα ελληνικά."""

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": fallback_prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=800,
                top_p=0.9,
                stream=False
            )

            response = chat_completion.choices[0].message.content
            
            if response and any(ord(char) > 1500 and ord(char) not in range(0x0370, 0x03FF) for char in response):
                print("⚠️ Detected non-Greek characters in fallback response")
                return "", False
            
            return response, True
            
        except Exception as e:
            print(f"❌ Fallback AI Error: {e}")
            return "", False

    def get_smart_fallback_response(self, question: str) -> str:
        """Smart fallback response when AI is not available"""
        question_lower = question.lower()
        
        # Enhanced concept-based responses
        if any(keyword in question_lower for keyword in ['σύλλογο', 'σύλλογος', 'γυμναστήριο', 'δομή', 'φορέα']):
            return """ΔΟΜΕΣ ΠΡΑΚΤΙΚΗΣ ΑΣΚΗΣΗΣ:

• Αθλητικούς συλλόγους όλων των αθλημάτων
• Γυμναστήρια και fitness centers
• Κολυμβητήρια  
• Ακαδημίες αθλητισμού
• Personal training studios
• Κέντρα αποκατάστασης
• Σχολεία με τμήμα φυσικής αγωγής

ΠΡΟΫΠΟΘΕΣΕΙΣ:
• Νόμιμη λειτουργία και ΑΦΜ
• Εκπαιδευτής με κατάλληλα προσόντα  
• Δυνατότητα καθοδήγησης

Για έγκριση δομής: gsofianidis@mitropolitiko.edu.gr"""

        elif any(keyword in question_lower for keyword in ['έγγραφα', 'χαρτιά', 'διαδικασία', 'αίτηση']):
            return """ΑΠΑΙΤΟΥΜΕΝΑ ΕΓΓΡΑΦΑ:

ΓΙΑ ΤΟΝ ΦΟΙΤΗΤΗ:
• Αίτηση πραγματοποίησης πρακτικής άσκησης
• Στοιχεία φοιτητή (συμπληρωμένη φόρμα)
• Ασφαλιστική ικανότητα από gov.gr
• Υπεύθυνη δήλωση (μη λήψη επιδόματος)

ΓΙΑ ΤΗ ΔΟΜΗ:
• Στοιχεία φορέα (ΑΦΜ, διεύθυνση, εκπρόσωπος)
• Ημέρες και ώρες δεκτότητας

⚠️ ΣΗΜΑΝΤΙΚΟ: Ξεκινήστε από την ασφαλιστική ικανότητα!

Επικοινωνία: gsofianidis@mitropolitiko.edu.gr"""

        elif any(keyword in question_lower for keyword in ['ώρες', 'χρόνος', 'προθεσμία', '240']):
            return """ΧΡΟΝΟΔΙΑΓΡΑΜΜΑ:

Απαιτούμενες ώρες: 240 ώρες
Προθεσμία: 30 Μαΐου

ΚΑΝΟΝΕΣ ΩΡΑΡΙΟΥ:
• Δευτέρα-Σάββατο (όχι Κυριακές)
• Μέχρι 8 ώρες/ημέρα
• 5 ημέρες/εβδομάδα

ΠΑΡΑΔΕΙΓΜΑΤΑ:
• 6 εβδομάδες × 40 ώρες
• 8 εβδομάδες × 30 ώρες  

Για προσαρμογή: gsofianidis@mitropolitiko.edu.gr"""

        else:
            return f"""Δεν βρέθηκε συγκεκριμένη απάντηση.

ΠΡΟΤΕΙΝΟΜΕΝΕΣ ΕΝΕΡΓΕΙΕΣ:
• Διατυπώστε την ερώτηση πιο συγκεκριμένα
• Επιλέξτε από τις συχνές ερωτήσεις
• Επικοινωνήστε με τον υπεύθυνο

ΕΠΙΚΟΙΝΩΝΙΑ:
📧 gsofianidis@mitropolitiko.edu.gr
📞 2314 409000

Για άμεση βοήθεια, περιγράψτε τη συγκεκριμένη απορία."""

    def get_response(self, question: str) -> str:
        """Main response method using RAG-first approach"""
        if not self.qa_data:
            return "Δεν υπάρχουν διαθέσιμα δεδομένα γνώσης."
        
        print(f"\n🤖 Processing question with RAG: '{question}'")
        
        # RAG-first approach
        if self.rag_initialized and self.faiss_index is not None:
            print("🧠 Step 1: RAG semantic search...")
            response, success = self.get_rag_response(question)
            if success and response.strip():
                print("✅ RAG response successful")
                return response
            else:
                print("⚠️ RAG failed, trying fallback AI...")
        else:
            print("⚠️ RAG not available, using fallback AI...")
        
        # Fallback to AI without RAG
        if self.groq_client:
            response, success = self.get_fallback_ai_response(question)
            if success and response.strip():
                print("✅ Fallback AI response successful")
                return response
        
        # Final fallback to smart responses
        print("📋 Using smart fallback response...")
        return self.get_smart_fallback_response(question)

def main():
    """Main Streamlit application with RAG-powered intelligence"""
    
    # Enhanced Responsive CSS Styling
    st.markdown("""
    <style>
    /* Base responsive styles */
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
    }
    
    .header-content {
        flex: 1;
        min-width: 300px;
    }
    
    .header-logo {
        max-height: 80px;
        max-width: 120px;
        object-fit: contain;
        flex-shrink: 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-left: 4px solid #007bff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #333;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .ai-message a {
        color: #ffeb3b !important;
        text-decoration: underline !important;
        font-weight: bold !important;
    }
    
    .ai-message a:hover {
        color: #fff9c4 !important;
        text-decoration: underline !important;
    }
    
    .info-card {
        background: white;
        border: 1px solid #e8f4f8;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    
    .api-status {
        position: fixed;
        top: 20px;
        right: 20px;
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        z-index: 1000;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .rag-status {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        animation: gradientShift 3s ease-in-out infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background: linear-gradient(45deg, #ff6b6b, #4ecdc4); }
        50% { background: linear-gradient(45deg, #4ecdc4, #45b7d1); }
    }
    
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e8f4f8;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e8f4f8;
        padding: 0.8rem 1.2rem;
        font-size: 16px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        min-height: 44px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 78, 121, 0.3);
    }
    
    /* Responsive breakpoints */
    @media screen and (max-width: 768px) {
        .main-header {
            padding: 1rem;
            gap: 1rem;
            flex-direction: column;
            text-align: center;
        }
        
        .header-content h1 {
            font-size: 1.5rem !important;
            margin-bottom: 0.5rem;
        }
        
        .header-content h3 {
            font-size: 1rem !important;
            margin-bottom: 0.5rem;
        }
        
        .header-content p {
            font-size: 0.9rem !important;
        }
        
        .header-logo {
            max-height: 60px;
            max-width: 100px;
        }
        
        .info-card {
            margin: 0.25rem;
            padding: 1rem;
            min-height: auto;
        }
        
        .info-card h4 {
            font-size: 1rem !important;
        }
        
        .info-card p {
            font-size: 1rem !important;
        }
        
        .info-card small {
            font-size: 0.8rem !important;
        }
        
        .chat-container {
            padding: 0.5rem;
            margin-left: -1rem;
            margin-right: -1rem;
            border-radius: 0;
        }
        
        .user-message, .ai-message {
            padding: 0.75rem;
            font-size: 0.9rem;
        }
        
        .api-status {
            position: relative;
            top: auto;
            right: auto;
            margin: 1rem 0;
            display: inline-block;
            font-size: 0.8rem;
        }
    }
    
    @media screen and (max-width: 480px) {
        .main-header {
            padding: 0.75rem;
        }
        
        .header-content h1 {
            font-size: 1.25rem !important;
        }
        
        .header-content h3 {
            font-size: 0.9rem !important;
        }
        
        .info-card {
            padding: 0.75rem;
        }
        
        .info-card h4 {
            font-size: 0.9rem !important;
            margin-bottom: 0.25rem !important;
        }
        
        .info-card p {
            font-size: 0.9rem !important;
            margin: 0.25rem 0 !important;
        }
        
        .user-message, .ai-message {
            padding: 0.5rem;
            font-size: 0.85rem;
            margin: 0.5rem 0;
        }
        
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    /* Hide scrollbars but keep functionality */
    .stApp {
        overflow-x: hidden;
    }
    
    /* Ensure content doesn't break on very small screens */
    * {
        max-width: 100%;
        box-sizing: border-box;
    }
    
    /* Better text scaling */
    html {
        -webkit-text-size-adjust: 100%;
        -ms-text-size-adjust: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with Logo
    logo_url = "https://raw.githubusercontent.com/GiorgosBouh/chatbot.placement/main/MK_LOGO_SEO_1200x630.png"
    
    st.markdown(f"""
    <div class="main-header">
        <img src="{logo_url}" alt="Μητροπολιτικό Κολλέγιο" class="header-logo">
        <div class="header-content">
            <h1>Πρακτική Άσκηση</h1>
            <h3>Μητροπολιτικό Κολλέγιο - Τμήμα Προπονητικής & Φυσικής Αγωγής</h3>
            <p><em>🧠 RAG-Powered AI Assistant με Σημασιολογική Αναζήτηση</em></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'chatbot' not in st.session_state:
        # Get Groq API key
        groq_api_key = None
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        except:
            pass
        
        with st.spinner("🔄 Initializing RAG system..."):
            st.session_state.chatbot = RAGInternshipChatbot(groq_api_key)
    else:
        # Refresh data if needed
        current_data_count = len(st.session_state.chatbot.qa_data)
        st.session_state.chatbot.qa_data = st.session_state.chatbot.load_qa_data()
        new_data_count = len(st.session_state.chatbot.qa_data)
        
        if new_data_count != current_data_count:
            st.toast(f"📊 Data updated: {new_data_count} entries")
            # Rebuild RAG database if needed
            if st.session_state.chatbot.rag_initialized:
                with st.spinner("🔄 Rebuilding RAG database..."):
                    st.session_state.chatbot.build_rag_database()

    # Quick info cards
    st.markdown("### 📊 Σημαντικές Πληροφορίες")
    
    quick_col1, quick_col2, quick_col3 = st.columns([1, 1, 1])
    
    with quick_col1:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h4 style="color: #1f4e79; margin-bottom: 0.5rem;">📅 Απαιτούμενες Ώρες</h4>
            <p style="font-size: 1.2rem; font-weight: 600; color: #28a745; margin: 0;">240 ώρες</p>
            <small style="color: #6c757d;">Προθεσμία: 30 Μαϊου</small>
        </div>
        """, unsafe_allow_html=True)

    with quick_col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h4 style="color: #1f4e79; margin-bottom: 0.5rem;">📋 Παράδοση Συμβάσεων</h4>
            <p style="font-size: 1.2rem; font-weight: 600; color: #ffc107; margin: 0;">Moodle Platform</p>
            <small style="color: #6c757d;">Προθεσμία: 15 Οκτωβρίου</small>
        </div>
        """, unsafe_allow_html=True)

    with quick_col3:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h4 style="color: #1f4e79; margin-bottom: 0.5rem;">⏰ Επιτρεπόμενο Ωράριο</h4>
            <p style="font-size: 1.2rem; font-weight: 600; color: #17a2b8; margin: 0;">Δευτέρα-Σάββατο</p>
            <small style="color: #6c757d;">Μέχρι 8 ώρες ανά ημέρα</small>
        </div>
        """, unsafe_allow_html=True)

    # RAG Status Indicator
    if st.session_state.chatbot.rag_initialized and st.session_state.chatbot.faiss_index is not None:
        chunks_count = len(st.session_state.chatbot.document_chunks)
        st.markdown(f'<div class="api-status rag-status">🧠 RAG Active ({chunks_count} chunks)</div>', unsafe_allow_html=True)
    elif st.session_state.chatbot.groq_client:
        st.markdown('<div class="api-status">🤖 AI Mode</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status" style="background: #ffc107;">📋 Basic Mode</div>', unsafe_allow_html=True)

    # Enhanced status information
    if st.session_state.chatbot.rag_initialized:
        status_text = f"RAG Semantic Search → AI Generation → Smart Fallback ({len(st.session_state.chatbot.document_chunks)} chunks)"
    elif st.session_state.chatbot.groq_client:
        status_text = "AI Generation → Smart Fallback"
    else:
        status_text = "Smart Concept-Based Responses"
    
    st.markdown(f"""
    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 0.6rem; margin-bottom: 1.5rem; text-align: center; font-size: 0.9rem;">
        <strong>🧠 RAG-Powered Assistant:</strong> Χρησιμοποιεί σημασιολογική αναζήτηση για βαθύτερη κατανόηση<br>
        <small>🔄 Architecture: {status_text}</small>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🗣️ Επικοινωνία")
        
        st.markdown("""
        **Υπεύθυνος Πρακτικής Άσκησης**  
        **Γεώργιος Σοφιανίδης**  
        **📞 2314409000**
        **📧 gsofianidis@mitropolitiko.edu.gr**
        
        **Σχεδιασμός/Ανάπτυξη/Τεχνική Υποστήριξη**  
        **Γεώργιος Μπουχουράς**  
        📧 gbouchouras@mitropolitiko.edu.gr

        ⚠️ Κανένα επίσημο έγγραφο δεν υπάρχει ή κατατίθεται στην παρούσα εφαρμογή
        """)

        st.markdown("---")

        # Frequent questions
        st.markdown("## 🔄 Συχνές Ερωτήσεις")
        
        categories = {}
        for qa in st.session_state.chatbot.qa_data:
            cat = qa.get('category', 'Άλλα')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(qa)

        for category, questions in categories.items():
            with st.expander(f"📂 {category}"):
                for qa in questions:
                    if st.button(qa['question'], key=f"faq_{qa['id']}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": qa['question']})
                        st.session_state.messages.append({"role": "assistant", "content": qa['answer']})
                        st.rerun()

        st.markdown("---")

        # Enhanced RAG Status
        if st.session_state.chatbot.rag_initialized:
            if st.session_state.chatbot.faiss_index is not None:
                st.success("🧠 RAG System Active")
                chunks = len(st.session_state.chatbot.document_chunks)
                st.info(f"Semantic search across {chunks} document chunks")
                
                # RAG Statistics
                qa_chunks = sum(1 for chunk in st.session_state.chatbot.document_chunks if chunk.chunk_type == "qa")
                pdf_chunks = sum(1 for chunk in st.session_state.chatbot.document_chunks if chunk.chunk_type == "pdf")
                st.write(f"📋 Q&A chunks: {qa_chunks}")
                st.write(f"📄 PDF chunks: {pdf_chunks}")
            else:
                st.warning("🧠 RAG Initialized but Database Missing")
        else:
            if RAG_AVAILABLE:
                st.warning("🧠 RAG Libraries Available but Not Initialized")
            else:
                st.error("⚠️ RAG Libraries Not Available")
                st.info("Install: pip install sentence-transformers faiss-cpu")

        st.markdown("---")

        if st.button("🗑️ Νέα Συνομιλία", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Enhanced Technical Information
        with st.expander("🔧 RAG System Details"):
            st.markdown("**For technical issues:**")
            st.markdown("📧 gbouchouras@mitropolitiko.edu.gr")
            
            st.write("**RAG System Status:**")
            st.write("• RAG Libraries:", RAG_AVAILABLE)
            st.write("• RAG Initialized:", st.session_state.chatbot.rag_initialized)
            st.write("• Vector Database:", st.session_state.chatbot.faiss_index is not None)
            st.write("• Embedding Model:", "paraphrase-multilingual-MiniLM-L12-v2" if st.session_state.chatbot.rag_initialized else "None")
            st.write("• Groq Available:", GROQ_AVAILABLE)
            st.write("• Groq Client:", st.session_state.chatbot.groq_client is not None)
            st.write("• PDF Available:", PDF_AVAILABLE)
            
            if st.session_state.chatbot.rag_initialized:
                st.write("**Document Chunks:**")
                st.write(f"• Total chunks: {len(st.session_state.chatbot.document_chunks)}")
                
                chunk_types = {}
                for chunk in st.session_state.chatbot.document_chunks:
                    chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
                
                for chunk_type, count in chunk_types.items():
                    st.write(f"• {chunk_type.upper()} chunks: {count}")
                
                # RAG Test
                st.subheader("🧠 RAG Retrieval Test")
                test_query = st.text_input("Test RAG query:", placeholder="Τι έγγραφα χρειάζομαι;")
                if test_query:
                    relevant_chunks = st.session_state.chatbot.retrieve_relevant_chunks(test_query, k=3)
                    if relevant_chunks:
                        st.write("**Retrieved chunks:**")
                        for i, (chunk, score) in enumerate(relevant_chunks):
                            st.write(f"**Chunk {i+1}** (score: {score:.3f}) - {chunk.source}")
                            st.write(f"Type: {chunk.chunk_type}")
                            st.write(f"Content preview: {chunk.content[:200]}...")
                            st.markdown("---")
                    else:
                        st.write("No relevant chunks found")
            
            # Enhanced file status
            qa_file_exists = os.path.exists("qa_data.json")
            st.write("**Data Sources:**")
            st.write("• qa_data.json exists:", qa_file_exists)
            st.write("• QA Data Count:", len(st.session_state.chatbot.qa_data))
            
            if PDF_AVAILABLE:
                st.write("• PDF Files:", len(st.session_state.chatbot.pdf_files))
                cached_pdfs = len(st.session_state.chatbot.pdf_cache)
                st.write(f"• Cached PDFs: {cached_pdfs}/{len(st.session_state.chatbot.pdf_files)}")

    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 💬 Κάντε την ερώτησή σας")

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>Εσείς:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            content = message["content"].replace('\n', '<br>')
            if st.session_state.chatbot.rag_initialized:
                assistant_name = "🧠 RAG Assistant"
            else:
                assistant_name = "🤖 Smart Assistant"
            st.markdown(f'<div class="ai-message"><strong>{assistant_name}:</strong><br><br>{content}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Γράψτε την ερώτησή σας εδώ...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        spinner_text = "Performing semantic search and generating response..." if st.session_state.chatbot.rag_initialized else "Generating intelligent response..."
        
        with st.spinner(spinner_text):
            try:
                response = st.session_state.chatbot.get_response(user_input)
            except Exception as e:
                response = f"Συγγνώμη, παρουσιάστηκε σφάλμα: {str(e)}"
                st.error(f"Error: {e}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Footer
    if st.session_state.chatbot.rag_initialized:
        footer_text = "RAG-Powered Semantic Search Assistant"
    elif st.session_state.chatbot.groq_client:
        footer_text = "AI-Enhanced Smart Assistant"
    else:
        footer_text = "Concept-Based Smart Assistant"
    
    st.markdown(f"""
    <div style="text-align: center; color: #6c757d; padding: 1rem; font-size: 0.9rem;">
        <small>
            🎓 <strong>Μητροπολιτικό Κολλέγιο Θεσσαλονίκης</strong> | 
            Τμήμα Προπονητικής & Φυσικής Αγωγής<br>
            <em>{footer_text}</em>
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()