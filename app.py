import streamlit as st
import json
import re
import os
import datetime
import requests
import io
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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

# Check for RAG libraries (optional - graceful degradation)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_AVAILABLE = True
    print("✅ RAG libraries available but not used (memory optimization)")
except ImportError:
    RAG_AVAILABLE = False
    print("ℹ️ RAG libraries not available (expected for lightweight deployment)")

# Ρύθμιση σελίδας
st.set_page_config(
    page_title="Πρακτική Άσκηση - Μητροπολιτικό Κολλέγιο",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@dataclass
class QAEntry:
    id: int
    category: str
    question: str
    answer: str
    keywords: List[str]

class OptimizedInternshipChatbot:
    def __init__(self, groq_api_key: str = None):
        # Initialize Groq client
        self.groq_client = None
        if GROQ_AVAILABLE and groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                print("✅ Groq client initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Groq: {e}")
        
        # Load Q&A data
        self.qa_data = self.load_qa_data()
        
        # Initialize PDF files cache with memory optimization
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
        
        # Enhanced concept patterns for smart matching
        self.concept_patterns = {
            'documents': {
                'keywords': ['έγγραφα', 'εγγραφα', 'χαρτιά', 'χαρτια', 'αίτηση', 'αιτηση', 'δικαιολογητικά', 'δικαιολογητικα', 'φόρμα', 'φορμα', 'στοιχεία', 'στοιχεια'],
                'weight': 1.0
            },
            'facilities': {
                'keywords': ['δομές', 'δομη', 'σύλλογος', 'συλλογος', 'γυμναστήριο', 'γυμναστηριο', 'φορείς', 'φορεις', 'εγκαταστάσεις', 'εγκαταστασεις'],
                'weight': 1.0
            },
            'sports': {
                'keywords': ['ενόργανη', 'ενοργανη', 'ποδόσφαιρο', 'ποδοσφαιρο', 'μπάσκετ', 'μπασκετ', 'βόλεϊ', 'βολει', 'fitness', 'γυμναστική', 'γυμναστικη'],
                'weight': 0.8
            },
            'time': {
                'keywords': ['ώρες', 'ωρες', '240', 'χρόνος', 'χρονος', 'διάρκεια', 'διαρκεια', 'deadline', 'προθεσμία', 'προθεσμια', 'χρονοδιάγραμμα', 'χρονοδιαγραμμα'],
                'weight': 1.0
            },
            'money': {
                'keywords': ['αμοιβή', 'αμοιβη', 'πληρωμή', 'πληρωμη', 'κόστος', 'κοστος', 'χρήματα', 'χρηματα', 'λεφτά', 'λεφτα', 'τέλος', 'τελος'],
                'weight': 0.9
            },
            'process': {
                'keywords': ['ξεκινάω', 'ξεκινω', 'βήματα', 'βηματα', 'διαδικασία', 'διαδικασια', 'πώς', 'πως', 'πως να', 'κάνω', 'κανω'],
                'weight': 1.0
            },
            'contact': {
                'keywords': ['επικοινωνία', 'επικοινωνια', 'υπεύθυνος', 'υπευθυνος', 'email', 'τηλέφωνο', 'τηλεφωνο', 'βοήθεια', 'βοηθεια'],
                'weight': 1.0
            }
        }
        
        # Enhanced system prompt for optimized AI
        self.system_prompt = """Είσαι ένας εξειδικευμένος σύμβουλος για θέματα πρακτικής άσκησης στο Μητροπολιτικό Κολλέγιο Θεσσαλονίκης, τμήμα Προπονητικής και Φυσικής Αγωγής.

ΣΥΣΤΗΜΑ ΕΞΥΠΝΗΣ ΑΝΑΛΥΣΗΣ:
- Χρησιμοποιείς προηγμένη ανάλυση εννοιών και συμπερασμού
- Έχεις πρόσβαση σε επίσημα έγγραφα και βάση γνώσης Q&A
- Συνδυάζεις πληροφορίες από πολλαπλές πηγές για πλήρεις απαντήσεις

ΚΡΙΣΙΜΕΣ ΓΛΩΣΣΙΚΕΣ ΟΔΗΓΙΕΣ:
- Χρησιμοποίησε ΑΠΟΚΛΕΙΣΤΙΚΑ ελληνικούς χαρακτήρες
- ΑΠΑΓΟΡΕΥΟΝΤΑΙ: αγγλικά, κινέζικα, greeklish ή άλλοι χαρακτήρες
- Ελέγχισε κάθε λέξη πριν την εκτύπωση

ΙΕΡΑΡΧΙΑ ΠΛΗΡΟΦΟΡΙΩΝ:
1. ΕΠΙΣΗΜΑ ΕΓΓΡΑΦΑ PDF (υψηλότερη προτεραιότητα)
2. ΒΑΣΗ ΓΝΩΣΗΣ JSON (μέση προτεραιότητα)
3. ΛΟΓΙΚΟΣ ΣΥΜΠΕΡΑΣΜΟΣ (χαμηλή προτεραιότητα)

ΣΤΡΑΤΗΓΙΚΗ ΕΞΥΠΝΗΣ ΑΝΑΛΥΣΗΣ:
1. Αναλύσε την ερώτηση για βασικές έννοιες και πρόθεση
2. Εντόπισε σχετικές πληροφορίες από διαθέσιμες πηγές
3. Συνδύασε δεδομένα με λογικό συμπερασμό
4. Δώσε δομημένες, πρακτικές απαντήσεις

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

Απάντησε πάντα στα ελληνικά με επαγγελματικό τόνο χρησιμοποιώντας εξυπνη ανάλυση."""

    def load_qa_data(self) -> List[Dict]:
        """Load Q&A data with memory optimization"""
        filename = "qa_data.json"
        
        print(f"🔍 Looking for {filename}...")
        
        if not os.path.exists(filename):
            print(f"❌ File {filename} not found")
            return self.get_enhanced_fallback_data()
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list) or not data:
                print(f"❌ Invalid data format in {filename}")
                return self.get_enhanced_fallback_data()
            
            required_fields = ['id', 'category', 'question', 'answer', 'keywords']
            for i, entry in enumerate(data):
                if not all(field in entry for field in required_fields):
                    print(f"❌ Missing fields in entry {i}")
                    return self.get_enhanced_fallback_data()
            
            print(f"✅ Successfully loaded {len(data)} Q&A entries")
            return data
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return self.get_enhanced_fallback_data()

    def get_enhanced_fallback_data(self) -> List[Dict]:
        """Enhanced fallback data with comprehensive coverage"""
        print("📋 Using enhanced fallback data...")
        return [
            {
                "id": 1,
                "category": "Γενικές Πληροφορίες",
                "question": "Πώς ξεκινάω την πρακτική μου άσκηση;",
                "answer": "ΒΗΜΑΤΑ ΕΝΑΡΞΗΣ ΠΡΑΚΤΙΚΗΣ ΑΣΚΗΣΗΣ:\n\n1. ΕΠΙΚΟΙΝΩΝΙΑ:\nΕπικοινωνώ με τον υπεύθυνο: gsofianidis@mitropolitiko.edu.gr\n\n2. ΕΠΙΛΟΓΗ ΔΟΜΗΣ:\nΒρίσκω κατάλληλη δομή (σύλλογος, γυμναστήριο, ακαδημία)\n\n3. ΕΓΓΡΑΦΑ:\nΚατεβάζω έγγραφα από το Moodle (SE5117)\nΣυμπληρώνω και ανεβάζω στην πλατφόρμα\n\n4. ΣΥΜΒΑΣΗ:\nΠεριμένω υπογραφή και ανάρτηση στο ΕΡΓΑΝΗ\n\n5. ΕΝΑΡΞΗ:\nΞεκινάω την πρακτική άσκηση\n\nΕΠΙΚΟΙΝΩΝΙΑ: gsofianidis@mitropolitiko.edu.gr",
                "keywords": ["ξεκινάω", "ξεκινώ", "αρχή", "αρχίζω", "αρχίσω", "ξεκίνημα", "πρακτική", "άσκηση", "πώς", "πως", "βήματα", "διαδικασία", "διαδικασιες", "κάνω", "κανω"]
            },
            {
                "id": 2,
                "category": "Έγγραφα & Διαδικασίες",
                "question": "Τι έγγραφα χρειάζομαι για την πρακτική άσκηση;",
                "answer": "ΑΠΑΙΤΟΥΜΕΝΑ ΕΓΓΡΑΦΑ:\n\nΓΙΑ ΤΟΝ ΦΟΙΤΗΤΗ:\n• Αίτηση πραγματοποίησης πρακτικής άσκησης\n• Στοιχεία φοιτητή (συμπληρωμένη φόρμα)\n• Ασφαλιστική ικανότητα από gov.gr\n• Υπεύθυνη δήλωση (μη λήψη επιδόματος ΟΑΕΔ)\n\nΓΙΑ ΤΗ ΔΟΜΗ:\n• Στοιχεία φορέα (ΑΦΜ, διεύθυνση, νόμιμος εκπρόσωπος)\n• Ημέρες και ώρες δεκτότητας\n\n⚠️ ΣΗΜΑΝΤΙΚΟ:\nΞεκινήστε από την ασφαλιστική ικανότητα - χρειάζεται χρόνο!\n\nΠΗΓΗ ΕΓΓΡΑΦΩΝ: Moodle SE5117\nΕΠΙΚΟΙΝΩΝΙΑ: gsofianidis@mitropolitiko.edu.gr",
                "keywords": ["έγγραφα", "εγγραφα", "χαρτιά", "χαρτια", "χρειάζομαι", "χρειαζομαι", "απαιτήσεις", "απαιτησεις", "δικαιολογητικά", "δικαιολογητικα", "φάκελος", "φακελος", "αίτηση", "αιτηση", "φόρμα", "φορμα"]
            },
            {
                "id": 5,
                "category": "Δομές & Φορείς",
                "question": "Σε ποιες δομές μπορώ να κάνω πρακτική άσκηση;",
                "answer": "ΕΓΚΕΚΡΙΜΕΝΕΣ ΔΟΜΕΣ ΠΡΑΚΤΙΚΗΣ:\n\n🏃‍♂️ ΑΘΛΗΤΙΚΕΣ ΔΟΜΕΣ:\n• Αθλητικούς συλλόγους (ποδόσφαιρο, μπάσκετ, βόλεϊ, ενόργανη γυμναστική)\n• Γυμναστήρια και fitness centers\n• Κολυμβητήρια\n• Ακαδημίες αθλητισμού\n• Personal training studios\n\n🏛️ ΔΗΜΟΣΙΟΙ ΦΟΡΕΙΣ:\n• Δημόσιους αθλητικούς οργανισμούς\n• Σχολεία με τμήμα φυσικής αγωγής\n• Κέντρα αποκατάστασης\n\nΠΡΟΥΠΟΘΕΣΕΙΣ ΔΟΜΗΣ:\n✅ Νόμιμη λειτουργία και ΑΦΜ\n✅ Εκπαιδευτής με κατάλληλα προσόντα\n✅ Δυνατότητα καθοδήγησης\n\nΕΓΚΡΙΣΗ ΔΟΜΗΣ: gsofianidis@mitropolitiko.edu.gr",
                "keywords": ["δομές", "δομη", "φορείς", "φορεις", "σύλλογος", "συλλογος", "γυμναστήριο", "γυμναστηριο", "ενόργανη", "ενοργανη", "ποδόσφαιρο", "ποδοσφαιρο", "μπάσκετ", "μπασκετ", "κολυμβητήριο", "κολυμβητηριο", "ακαδημία", "ακαδημια", "fitness", "personal", "training", "που", "ποιες", "ποιους", "ποια", "εγκαταστάσεις", "εγκαταστασεις"]
            },
            {
                "id": 30,
                "category": "Οικονομικά & Αμοιβή",
                "question": "Παίρνω αμοιβή για την πρακτική άσκηση; Τι κόστος έχει για τη δομή;",
                "answer": "ΟΙΚΟΝΟΜΙΚΑ ΘΕΜΑΤΑ ΠΡΑΚΤΙΚΗΣ:\n\n💰 ΓΙΑ ΤΟΥΣ ΦΟΙΤΗΤΕΣ:\n❌ ΔΕΝ υπάρχει αμοιβή\n• Η πρακτική άσκηση είναι μη αμειβόμενη\n• Αποτελεί μέρος των σπουδών\n• Δεν είναι εργασιακή σχέση\n\n🏢 ΓΙΑ ΤΗ ΔΟΜΗ:\n✅ Ελάχιστο ή μηδενικό κόστος\n• Ενδεχόμενο ελάχιστο διοικητικό τέλος\n• Το κολλέγιο καλύπτει έξοδα σύμβασης\n• Ασφάλιση τιμολογείται στο κολλέγιο\n• Χωρίς οικονομική υποχρέωση προς φοιτητές\n\nΠΛΗΡΟΦΟΡΙΕΣ: gsofianidis@mitropolitiko.edu.gr",
                "keywords": ["αμοιβή", "αμοιβη", "πληρωμή", "πληρωμη", "πληρώθώ", "πληρωθώ", "πληρωθω", "πληρωνομαι", "πληρώνομαι", "λεφτά", "λεφτα", "χρήματα", "χρηματα", "κόστος", "κοστος", "τέλος", "τελος", "δομή", "δομη", "φοιτητής", "φοιτητη", "οικονομικά", "οικονομικα", "μισθός", "μισθος"]
            },
            {
                "id": 11,
                "category": "Επικοινωνία",
                "question": "Με ποιον επικοινωνώ για την πρακτική άσκηση;",
                "answer": "ΣΤΟΙΧΕΙΑ ΕΠΙΚΟΙΝΩΝΙΑΣ:\n\n👨‍🏫 ΚΥΡΙΑ ΕΠΙΚΟΙΝΩΝΙΑ:\nΓεώργιος Σοφιανίδης, MSc, PhD(c)\n📧 gsofianidis@mitropolitiko.edu.gr\n🏷️ Υπεύθυνος Πρακτικής Άσκησης\n\n👨‍💼 ΕΝΑΛΛΑΚΤΙΚΗ ΕΠΙΚΟΙΝΩΝΙΑ:\nΓεώργιος Μπουχουράς, MSc, PhD\n📧 gbouchouras@mitropolitiko.edu.gr\n📞 2314 409000\n🏷️ Programme Leader\n\n📋 ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗ ΘΕΜΑΤΩΝ:\n• Ερωτήσεις για έγγραφα ➜ Γεώργιος Σοφιανίδης\n• Τεχνικά προβλήματα ➜ Γεώργιος Σοφιανίδης\n• Θέματα προγράμματος ➜ Γεώργιος Μπουχουράς\n\n⏰ ΩΡΑΡΙΟ: Δευτέρα-Παρασκευή, 9:00-17:00",
                "keywords": ["επικοινωνία", "επικοινωνια", "Σοφιανίδης", "Σοφιανιδης", "Μπουχουράς", "Μπουχουρας", "email", "τηλέφωνο", "τηλεφωνο", "υπεύθυνος", "υπευθυνος", "βοήθεια", "βοηθεια", "καθηγητής", "καθηγητης", "καθηγήτρια", "καθηγητρια", "contact", "στοιχεία", "στοιχεια"]
            },
            {
                "id": 4,
                "category": "Ώρες & Χρονοδιάγραμμα",
                "question": "Πόσες ώρες πρέπει να κάνω πρακτική άσκηση;",
                "answer": "ΧΡΟΝΙΚΕΣ ΑΠΑΙΤΗΣΕΙΣ:\n\n⏱️ ΣΥΝΟΛΙΚΕΣ ΩΡΕΣ:\n240 ώρες (υποχρεωτικό ελάχιστο)\n\n📅 ΠΡΟΘΕΣΜΙΑ:\nΜέχρι 30 Μαΐου\n\n📆 ΚΑΝΟΝΕΣ ΩΡΑΡΙΟΥ:\n• Δευτέρα έως Σάββατο\n• ΌΧΙ Κυριακές\n• Μέχρι 8 ώρες/ημέρα\n• 5 ημέρες/εβδομάδα\n\n📊 ΠΑΡΑΔΕΙΓΜΑΤΑ ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΥ:\n• 6 εβδομάδες × 40 ώρες\n• 8 εβδομάδες × 30 ώρες\n• 10 εβδομάδες × 24 ώρες\n\nΣΥΜΦΩΝΙΑ: Το ωράριο ορίζεται από τη δομή σε συνεργασία μαζί σας\n\nΠΛΗΡΟΦΟΡΙΕΣ: gsofianidis@mitropolitiko.edu.gr",
                "keywords": ["ώρες", "ωρες", "240", "ποσες", "πόσες", "ποσα", "ποσά", "συνολικά", "συνολικα", "όλες", "ολες", "τελικά", "τελικα", "χρονοδιάγραμμα", "χρονοδιαγραμμα", "διάρκεια", "διαρκεια", "χρόνος", "χρονος", "30/5", "deadline", "προθεσμία", "προθεσμια"]
            }
        ]

    def download_pdf_file(self, filename: str) -> str:
        """Memory-optimized PDF download and processing"""
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
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            text_content = []
            
            if PDF_METHOD == "PyPDF2":
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            # Memory optimization: limit content length
                            text_content.append(page_text.strip()[:2000])  # Limit per page
                    except Exception as e:
                        print(f"⚠️ Error extracting page {page_num}: {e}")
                
            elif PDF_METHOD == "PyMuPDF":
                pdf_document = fitz.open(stream=response.content, filetype="pdf")
                for page_num in range(min(pdf_document.page_count, 10)):  # Limit pages for memory
                    try:
                        page = pdf_document[page_num]
                        page_text = page.get_text()
                        if page_text.strip():
                            text_content.append(page_text.strip()[:2000])  # Limit per page
                    except Exception as e:
                        print(f"⚠️ Error extracting page {page_num}: {e}")
                pdf_document.close()
            
            full_text = "\n".join(text_content)
            
            # Memory optimization: Cache only essential content
            if len(full_text) > 5000:
                full_text = full_text[:5000] + "...\n[Περιεχόμενο περιορίστηκε για βελτιστοποίηση μνήμης]"
            
            self.pdf_cache[filename] = full_text
            
            print(f"✅ Successfully processed {filename} ({len(full_text)} characters)")
            return full_text
            
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
            return ""

    def extract_concepts(self, question: str) -> Dict[str, float]:
        """Enhanced concept extraction with scoring"""
        question_lower = question.lower()
        detected_concepts = {}
        
        for concept, data in self.concept_patterns.items():
            matches = sum(1 for keyword in data['keywords'] if keyword in question_lower)
            if matches > 0:
                # Calculate concept strength
                strength = (matches / len(data['keywords'])) * data['weight']
                detected_concepts[concept] = strength
        
        return detected_concepts

    def enhanced_similarity_calculation(self, question: str, qa_entry: Dict) -> float:
        """Enhanced similarity calculation with concept weighting"""
        question_lower = question.lower()
        
        # Extract concepts
        question_concepts = self.extract_concepts(question)
        
        # Base keyword matching
        keyword_matches = sum(1 for keyword in qa_entry.get('keywords', []) 
                            if keyword.lower() in question_lower)
        keyword_score = keyword_matches / max(len(qa_entry.get('keywords', [])), 1) * 0.4
        
        # Title similarity
        title_words = qa_entry['question'].lower().split()
        question_words = [w for w in question_lower.split() if len(w) > 2]
        
        title_matches = sum(1 for word in title_words if word in question_lower and len(word) > 2)
        reverse_matches = sum(1 for word in question_words if word in qa_entry['question'].lower())
        title_score = (title_matches + reverse_matches) / max(len(title_words) + len(question_words), 1) * 0.3
        
        # Concept-category matching
        qa_category = qa_entry.get('category', '').lower()
        concept_score = 0
        
        for concept, strength in question_concepts.items():
            if concept == 'documents' and ('έγγραφα' in qa_category or 'διαδικασίες' in qa_category):
                concept_score += strength * 0.3
            elif concept == 'facilities' and ('δομές' in qa_category or 'φορείς' in qa_category):
                concept_score += strength * 0.3
            elif concept == 'time' and ('ώρες' in qa_category or 'χρονοδιάγραμμα' in qa_category):
                concept_score += strength * 0.3
            elif concept == 'money' and 'οικονομικά' in qa_category:
                concept_score += strength * 0.3
            elif concept == 'contact' and 'επικοινωνία' in qa_category:
                concept_score += strength * 0.3
        
        total_score = keyword_score + title_score + concept_score
        return min(total_score, 1.0)

    def get_contextual_matches(self, question: str, max_matches: int = 3) -> List[Dict]:
        """Get contextually relevant Q&A matches"""
        if not self.qa_data:
            return []
        
        # Calculate similarities
        scored_matches = []
        for qa in self.qa_data:
            similarity = self.enhanced_similarity_calculation(question, qa)
            if similarity > 0.05:  # Threshold for relevance
                scored_matches.append((similarity, qa))
        
        # Sort and return top matches
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        return [qa for score, qa in scored_matches[:max_matches]]

    def search_pdfs_intelligently(self, question: str, concepts: Dict[str, float]) -> str:
        """Intelligent PDF search with concept-based filtering"""
        if not PDF_AVAILABLE:
            return ""
        
        print("📄 Searching PDFs with concept analysis...")
        
        question_lower = question.lower()
        question_words = [w for w in question_lower.split() if len(w) > 3]
        
        relevant_content = []
        
        for filename in self.pdf_files:
            content = self.download_pdf_file(filename)
            if content:
                content_lower = content.lower()
                
                # Calculate relevance score
                word_matches = sum(1 for word in question_words if word in content_lower)
                concept_matches = sum(strength for concept, strength in concepts.items() 
                                    if self._check_concept_in_pdf(concept, content_lower))
                
                relevance_score = word_matches * 0.4 + concept_matches * 0.6
                
                if relevance_score > 0.3:
                    # Extract relevant sections
                    sections = self._extract_relevant_sections(content, question_words, max_chars=800)
                    if sections:
                        relevant_content.append(f"[Από {filename}]\n{sections}")
                        print(f"✅ Found relevant content in {filename} (score: {relevance_score:.2f})")
        
        return "\n\n".join(relevant_content) if relevant_content else ""

    def _check_concept_in_pdf(self, concept: str, text: str) -> bool:
        """Check if concept keywords exist in PDF text"""
        keywords = self.concept_patterns.get(concept, {}).get('keywords', [])
        return any(keyword in text for keyword in keywords)

    def _extract_relevant_sections(self, content: str, keywords: List[str], max_chars: int) -> str:
        """Extract most relevant sections from PDF content"""
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        scored_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for keyword in keywords if keyword in sentence_lower)
            if matches > 0:
                scored_sentences.append((matches, sentence))
        
        # Sort by relevance and combine
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        result = []
        char_count = 0
        for score, sentence in scored_sentences[:3]:  # Top 3 sentences
            if char_count + len(sentence) > max_chars:
                break
            result.append(sentence.strip())
            char_count += len(sentence)
        
        return '. '.join(result) + ('.' if result else '')

    def get_smart_ai_response(self, user_message: str) -> Tuple[str, bool]:
        """Enhanced AI response with intelligent context building"""
        if not self.groq_client:
            return "", False
        
        try:
            # Extract concepts
            concepts = self.extract_concepts(user_message)
            print(f"🧠 Detected concepts: {list(concepts.keys())}")
            
            # Get relevant Q&A matches
            qa_matches = self.get_contextual_matches(user_message)
            
            # Get relevant PDF content
            pdf_content = self.search_pdfs_intelligently(user_message, concepts)
            
            # Build context
            context_parts = []
            
            if pdf_content:
                context_parts.append(f"ΕΠΙΣΗΜΑ ΕΓΓΡΑΦΑ:\n{pdf_content}")
            
            if qa_matches:
                qa_context = "\n\n".join([
                    f"ΕΡΩΤΗΣΗ: {qa['question']}\nΑΠΑΝΤΗΣΗ: {qa['answer']}"
                    for qa in qa_matches
                ])
                context_parts.append(f"ΒΑΣΗ ΓΝΩΣΗΣ:\n{qa_context}")
            
            # Enhanced prompt
            if context_parts:
                combined_context = "\n\n" + ("="*40 + "\n\n").join(context_parts)
                
                full_prompt = f"""ΔΙΑΘΕΣΙΜΕΣ ΠΛΗΡΟΦΟΡΙΕΣ:
{combined_context}

ΕΡΩΤΗΣΗ ΦΟΙΤΗΤΗ: {user_message}

ΕΝΤΟΠΙΣΜΕΝΕΣ ΕΝΝΟΙΕΣ: {', '.join(concepts.keys()) if concepts else 'Γενική ερώτηση'}

ΟΔΗΓΙΕΣ ΕΞΥΠΝΗΣ ΑΝΑΛΥΣΗΣ:
1. Αναλύσε την ερώτηση για το τι ζητάει συγκεκριμένα ο φοιτητής
2. Χρησιμοποίησε πληροφορίες από ΕΠΙΣΗΜΑ ΕΓΓΡΑΦΑ ως κύρια πηγή
3. Συμπλήρωσε με πληροφορίες από τη ΒΑΣΗ ΓΝΩΣΗΣ
4. Συνδύασε με λογικό συμπερασμό όπου χρειάζεται
5. Δώσε πρακτικές, δομημένες οδηγίες
6. Αναφέρου αν χρειάζεται επιβεβαίωση από τον υπεύθυνο

Απάντησε με δομημένο τρόπο και επαγγελματικό τόνο στα ελληνικά."""
            else:
                # Fallback prompt with enhanced reasoning
                full_prompt = f"""ΕΡΩΤΗΣΗ ΦΟΙΤΗΤΗ: {user_message}

ΠΛΑΙΣΙΟ: Φοιτητής Προπονητικής & Φυσικής Αγωγής, Μητροπολιτικό Κολλέγιο Θεσσαλονίκης

ΕΝΤΟΠΙΣΜΕΝΕΣ ΕΝΝΟΙΕΣ: {', '.join(concepts.keys()) if concepts else 'Γενική ερώτηση'}

ΒΑΣΙΚΕΣ ΠΛΗΡΟΦΟΡΙΕΣ:
- Απαιτούνται 240 ώρες πρακτικής άσκησης μέχρι 30 Μαΐου
- Δευτέρα-Σάββατο, μέχρι 8 ώρες/ημέρα
- Υπεύθυνος: Γεώργιος Σοφιανίδης (gsofianidis@mitropolitiko.edu.gr)
- Παράδοση συμβάσεων στο Moodle μέχρι 15 Οκτωβρίου

ΟΔΗΓΙΕΣ:
1. Χρησιμοποίησε γενική γνώση για πρακτική άσκηση στην Ελλάδα
2. Συσχέτισε με το πλαίσιο του κολλεγίου
3. Δώσε πρακτικές συμβουλές βασισμένες στις εντοπισμένες έννοιες
4. Πρότεινε επικοινωνία με υπεύθυνο για επιβεβαίωση

Απάντησε με επαγγελματικό τόνο στα ελληνικά."""

            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.2,  # Lower for consistency
                max_tokens=1000,
                top_p=0.9,
                stream=False
            )

            response = chat_completion.choices[0].message.content
            
            # Validate Greek characters
            if response and any(ord(char) > 1500 and ord(char) not in range(0x0370, 0x03FF) for char in response):
                print("⚠️ Detected non-Greek characters in response")
                return "", False
            
            print("✅ Smart AI response generated successfully")
            return response, True
            
        except Exception as e:
            print(f"❌ Smart AI Error: {e}")
            return "", False

    def get_concept_based_fallback(self, question: str) -> str:
        """Enhanced concept-based smart fallback"""
        concepts = self.extract_concepts(question)
        question_lower = question.lower()
        
        # Prioritize concepts by strength
        top_concept = max(concepts.items(), key=lambda x: x[1])[0] if concepts else None
        
        if top_concept == 'facilities' or any(keyword in question_lower for keyword in ['σύλλογο', 'γυμναστήριο', 'δομή', 'φορέα']):
            return """ΕΓΚΕΚΡΙΜΕΝΕΣ ΔΟΜΕΣ ΠΡΑΚΤΙΚΗΣ ΑΣΚΗΣΗΣ:

🏃‍♂️ ΑΘΛΗΤΙΚΕΣ ΕΓΚΑΤΑΣΤΑΣΕΙΣ:
• Αθλητικούς συλλόγους όλων των αθλημάτων
• Γυμναστήρια και fitness centers
• Κολυμβητήρια
• Ακαδημίες αθλητισμού
• Personal training studios
• Κέντρα αποκατάστασης

🏛️ ΔΗΜΟΣΙΟΙ ΦΟΡΕΙΣ:
• Δημόσιους αθλητικούς οργανισμούς
• Σχολεία με τμήμα φυσικής αγωγής

✅ ΠΡΟΫΠΟΘΕΣΕΙΣ:
• Νόμιμη λειτουργία και ΑΦΜ
• Εκπαιδευτής με κατάλληλα προσόντα
• Δυνατότητα καθοδήγησης

ΕΓΚΡΙΣΗ ΔΟΜΗΣ: gsofianidis@mitropolitiko.edu.gr"""

        elif top_concept == 'documents' or any(keyword in question_lower for keyword in ['έγγραφα', 'χαρτιά', 'διαδικασία']):
            return """ΑΠΑΙΤΟΥΜΕΝΑ ΕΓΓΡΑΦΑ ΠΡΑΚΤΙΚΗΣ:

📋 ΓΙΑ ΤΟΝ ΦΟΙΤΗΤΗ:
• Αίτηση πραγματοποίησης πρακτικής άσκησης
• Στοιχεία φοιτητή (συμπληρωμένη φόρμα)
• Ασφαλιστική ικανότητα από gov.gr
• Υπεύθυνη δήλωση (μη λήψη επιδόματος)

🏢 ΓΙΑ ΤΗ ΔΟΜΗ:
• Στοιχεία φορέα (ΑΦΜ, διεύθυνση, εκπρόσωπος)
• Ημέρες και ώρες δεκτότητας

⚠️ ΣΗΜΑΝΤΙΚΟ:
Ξεκινήστε από την ασφαλιστική ικανότητα - χρειάζεται χρόνο!

ΠΗΓΗ: Moodle SE5117
ΕΠΙΚΟΙΝΩΝΙΑ: gsofianidis@mitropolitiko.edu.gr"""

        elif top_concept == 'time' or any(keyword in question_lower for keyword in ['ώρες', 'χρόνος', 'προθεσμία']):
            return """ΧΡΟΝΙΚΕΣ ΑΠΑΙΤΗΣΕΙΣ ΠΡΑΚΤΙΚΗΣ:

⏱️ ΣΥΝΟΛΙΚΕΣ ΩΡΕΣ: 240 ώρες (υποχρεωτικό)
📅 ΠΡΟΘΕΣΜΙΑ: 30 Μαΐου

📆 ΚΑΝΟΝΕΣ ΩΡΑΡΙΟΥ:
• Δευτέρα-Σάββατο (όχι Κυριακές)
• Μέχρι 8 ώρες/ημέρα
• 5 ημέρες/εβδομάδα

📊 ΠΑΡΑΔΕΙΓΜΑΤΑ ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΥ:
• 6 εβδομάδες × 40 ώρες
• 8 εβδομάδες × 30 ώρες
• 10 εβδομάδες × 24 ώρες

ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΣ: gsofianidis@mitropolitiko.edu.gr"""

        elif top_concept == 'contact' or any(keyword in question_lower for keyword in ['επικοινωνία', 'υπεύθυνος']):
            return """ΣΤΟΙΧΕΙΑ ΕΠΙΚΟΙΝΩΝΙΑΣ:

👨‍🏫 ΚΥΡΙΑ ΕΠΙΚΟΙΝΩΝΙΑ:
Γεώργιος Σοφιανίδης, MSc, PhD(c)
📧 gsofianidis@mitropolitiko.edu.gr
🏷️ Υπεύθυνος Πρακτικής Άσκησης

👨‍💼 ΕΝΑΛΛΑΚΤΙΚΗ ΕΠΙΚΟΙΝΩΝΙΑ:
Γεώργιος Μπουχουράς, MSc, PhD
📧 gbouchouras@mitropolitiko.edu.gr
📞 2314 409000
🏷️ Programme Leader

📋 ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗ:
• Θέματα πρακτικής ➜ Γεώργιος Σοφιανίδης
• Τεχνικά προβλήματα ➜ Γεώργιος Μπουχουράς"""

        else:
            return f"""Δεν βρέθηκε συγκεκριμένη απάντηση για αυτή την ερώτηση.

ΠΡΟΤΕΙΝΟΜΕΝΕΣ ΕΝΕΡΓΕΙΕΣ:
• Διατυπώστε την ερώτηση πιο συγκεκριμένα
• Επιλέξτε από τις συχνές ερωτήσεις στο μενού
• Επικοινωνήστε απευθείας με τον υπεύθυνο

ΕΠΙΚΟΙΝΩΝΙΑ:
📧 gsofianidis@mitropolitiko.edu.gr
📞 2314 409000

Για άμεση βοήθεια, περιγράψτε τη συγκεκριμένη απορία σας."""

    def get_response(self, question: str) -> str:
        """Main response method - optimized for memory efficiency"""
        if not self.qa_data:
            return "Δεν υπάρχουν διαθέσιμα δεδομένα γνώσης."
        
        print(f"\n🤖 Processing question: '{question}'")
        
        # Step 1: Check for high-similarity direct matches
        print("📋 Step 1: Checking for direct matches...")
        best_match = max(self.qa_data, key=lambda x: self.enhanced_similarity_calculation(question, x))
        similarity = self.enhanced_similarity_calculation(question, best_match)
        
        if similarity > 0.4:  # High confidence threshold
            print(f"✅ High similarity match found (score: {similarity:.3f})")
            return best_match['answer']
        
        # Step 2: Enhanced AI processing with context
        print("🧠 Step 2: Enhanced AI processing...")
        if self.groq_client:
            response, success = self.get_smart_ai_response(question)
            if success and response.strip():
                print("✅ Smart AI response successful")
                return response
            else:
                print("⚠️ AI processing failed")
        else:
            print("⚠️ AI not available")
        
        # Step 3: Concept-based intelligent fallback
        print("📋 Step 3: Using intelligent fallback...")
        if similarity > 0.15:  # Medium confidence
            print(f"🟡 Medium similarity fallback (score: {similarity:.3f})")
            return best_match['answer']
        else:
            print("🔄 Using concept-based smart fallback")
            return self.get_concept_based_fallback(question)

def main():
    """Main Streamlit application - Optimized for Community Cloud"""
    
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
    
    .optimized-status {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
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
            <p><em>🧠 Memory-Optimized Smart Assistant για Community Cloud</em></p>
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
        
        st.session_state.chatbot = OptimizedInternshipChatbot(groq_api_key)
    else:
        # Refresh data if needed
        current_data_count = len(st.session_state.chatbot.qa_data)
        st.session_state.chatbot.qa_data = st.session_state.chatbot.load_qa_data()
        new_data_count = len(st.session_state.chatbot.qa_data)
        
        if new_data_count != current_data_count:
            st.toast(f"📊 Data updated: {new_data_count} entries")

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

    # Optimized Status Indicator
    if st.session_state.chatbot.groq_client:
        st.markdown('<div class="api-status optimized-status">🧠 Smart Mode (Optimized)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status" style="background: #ffc107; color: #000;">📋 Concept Mode</div>', unsafe_allow_html=True)

    # Enhanced status information
    if st.session_state.chatbot.groq_client:
        status_text = "Smart Matching → Enhanced AI → Concept Fallback"
    else:
        status_text = "Smart Matching → Concept-Based Responses"
    
    st.markdown(f"""
    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 0.6rem; margin-bottom: 1.5rem; text-align: center; font-size: 0.9rem;">
        <strong>🧠 Memory-Optimized Smart Assistant:</strong> Βελτιστοποιημένο για Streamlit Community Cloud<br>
        <small>🔄 Logic: {status_text}</small>
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

        # System Status
        if st.session_state.chatbot.groq_client:
            st.success("🧠 Smart AI Mode Active")
            st.info("Enhanced concept analysis + AI reasoning")
        else:
            st.warning("📋 Concept-Based Mode")
            if GROQ_AVAILABLE:
                st.info("For AI enhancement, add Groq API key")
            else:
                st.error("Groq library not available")

        # Memory optimization notice
        st.info("⚡ Optimized for Community Cloud memory limits")

        st.markdown("---")

        if st.button("🗑️ Νέα Συνομιλία", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Technical Information
        with st.expander("🔧 System Details"):
            st.markdown("**Technical Support:**")
            st.markdown("📧 gbouchouras@mitropolitiko.edu.gr")
            
            st.write("**Optimized System Status:**")
            st.write("• Memory Mode: Community Cloud Optimized ✅")
            st.write("• Enhanced Concept Analysis: Active ✅")
            st.write("• Smart Similarity Matching: Active ✅")
            st.write("• Groq Available:", GROQ_AVAILABLE)
            st.write("• Groq Client:", st.session_state.chatbot.groq_client is not None)
            st.write("• PDF Available:", PDF_AVAILABLE)
            st.write("• RAG Libraries:", RAG_AVAILABLE, "(Not used for memory optimization)")
            
            st.write("**Data Sources:**")
            st.write("• QA Data Count:", len(st.session_state.chatbot.qa_data))
            st.write("• PDF Files:", len(st.session_state.chatbot.pdf_files))
            cached_pdfs = len(st.session_state.chatbot.pdf_cache)
            st.write(f"• Cached PDFs: {cached_pdfs}/{len(st.session_state.chatbot.pdf_files)}")
            
            # Concept analysis test
            st.subheader("🧠 Concept Analysis Test")
            test_question = st.text_input("Test concept detection:", placeholder="Τι έγγραφα χρειάζομαι;")
            if test_question:
                concepts = st.session_state.chatbot.extract_concepts(test_question)
                if concepts:
                    st.write("**Detected Concepts:**")
                    for concept, strength in concepts.items():
                        st.write(f"• {concept}: {strength:.3f}")
                else:
                    st.write("No specific concepts detected")
                
                # Test similarity
                if st.session_state.chatbot.qa_data:
                    best_match = max(st.session_state.chatbot.qa_data, 
                                   key=lambda x: st.session_state.chatbot.enhanced_similarity_calculation(test_question, x))
                    similarity = st.session_state.chatbot.enhanced_similarity_calculation(test_question, best_match)
                    st.write(f"**Best match similarity:** {similarity:.3f}")
                    st.write(f"**Would use:** {'Direct match' if similarity > 0.4 else 'AI enhancement' if similarity > 0.15 else 'Concept fallback'}")

    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 💬 Κάντε την ερώτησή σας")

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>Εσείς:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            content = message["content"].replace('\n', '<br>')
            assistant_name = "🧠 Smart Assistant" if st.session_state.chatbot.groq_client else "📋 Concept Assistant"
            st.markdown(f'<div class="ai-message"><strong>{assistant_name}:</strong><br><br>{content}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Γράψτε την ερώτησή σας εδώ...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        spinner_text = "Αναλύω με έξυπνους αλγορίθμους..." if st.session_state.chatbot.groq_client else "Αναλύω με έννοιες..."
        
        with st.spinner(spinner_text):
            try:
                response = st.session_state.chatbot.get_response(user_input)
            except Exception as e:
                response = f"Συγγνώμη, παρουσιάστηκε σφάλμα: {str(e)}"
                st.error(f"Error: {e}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Footer
    footer_text = "Memory-Optimized Smart Assistant" if st.session_state.chatbot.groq_client else "Enhanced Concept-Based Assistant"
    st.markdown(f"""
    <div style="text-align: center; color: #6c757d; padding: 1rem; font-size: 0.9rem;">
        <small>
            🎓 <strong>Μητροπολιτικό Κολλέγιο Θεσσαλονίκης</strong> | 
            Τμήμα Προπονητικής & Φυσικής Αγωγής<br>
            <em>{footer_text}</em><br>
            <em>⚡ Optimized for Streamlit Community Cloud</em>
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()    main()