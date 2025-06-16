import streamlit as st
import json
import re
import os
import datetime
from typing import List, Dict, Tuple
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

class InternshipChatbot:
    def __init__(self, groq_api_key: str = None):
        # Initialize Groq client if available and API key provided
        self.groq_client = None
        if GROQ_AVAILABLE and groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                print("✅ Groq client initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Groq: {e}")
        
        # Load Q&A data
        self.qa_data = self.load_qa_data()
        
        # System prompt για το LM
        self.system_prompt = """Είσαι ένας εξειδικευμένος σύμβουλος για θέματα πρακτικής άσκησης στο Μητροπολιτικό Κολλέγιο Θεσσαλονίκης, τμήμα Προπονητικής και Φυσικής Αγωγής.

ΚΡΙΣΙΜΕΣ ΓΛΩΣΣΙΚΕΣ ΟΔΗΓΙΕΣ:
- Χρησιμοποίησε ΑΠΟΚΛΕΙΣΤΙΚΑ και ΜΟΝΟ ελληνικούς χαρακτήρες
- ΑΠΑΓΟΡΕΥΟΝΤΑΙ: αγγλικά, κινέζικα, greeklish ή οποιοιδήποτε άλλοι χαρακτήρες
- Ελέγχισε κάθε λέξη πριν την εκτύπωση - πρέπει να είναι ελληνική
- Αν δεν ξέρεις ελληνική λέξη, χρησιμοποίησε περιφραστικό τρόπο

ΚΡΙΤΙΚΕΣ ΟΔΗΓΙΕΣ:
- Μην προσθέτεις πληροφορίες που δεν υπάρχουν στο context
- Χρησιμοποίησε ΜΟΝΟ τις πληροφορίες που σου δίνονται
- Μην εφευρίσκεις ή μην υποθέτεις στοιχεία

ΣΤΥΛ ΑΠΑΝΤΗΣΗΣ:
- Αυστηρά επίσημος και επαγγελματικός τόνος
- Άμεσες και συγκεκριμένες οδηγίες
- Χωρίς χαιρετισμούς, φιλικές εκφράσεις ή περιττά λόγια
- Δομημένες απαντήσεις με σαφή βήματα
- Χωρίς emojis ή άτυπες εκφράσεις

ΒΑΣΙΚΕΣ ΠΛΗΡΟΦΟΡΙΕΣ (μόνο αυτές):
- Υπεύθυνος Πρακτικής Άσκησης: Γεώργιος Σοφιανίδης
- Email: gsofianidis@mitropolitiko.edu.gr
- Τεχνική Υποστήριξη: Γεώργιος Μπουχουράς (gbouchouras@mitropolitiko.edu.gr)
- Απαιτούμενες ώρες: 240 ώρες μέχρι 30/4
- Ωράριο: Δευτέρα-Σάββατο, μέχρι 8 ώρες/ημέρα
- Σύμβαση: Ανέβασμα στο moodle μέχρι 15/10

ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ:
- Κάθε απάντηση πρέπει να περιέχει ΜΟΝΟ ελληνικούς χαρακτήρες
- Καμία ξένη λέξη ή χαρακτήρας δεν επιτρέπεται
- Επαγγελματικό ύφος χωρίς φιλικότητες

Απάντησε στα ελληνικά με αυστηρά επαγγελματικό τόνο χρησιμοποιώντας μόνο τις δοσμένες πληροφορίες."""

    @st.cache_data
    def load_qa_data_from_file(_self, filename: str = "qa_data.json", _mtime: float = None) -> List[Dict]:
        """Load Q&A data from Git repository file with smart caching"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ Loaded {len(data)} Q&A entries from {filename}")
                    return data
            else:
                print(f"⚠️ File {filename} not found, using embedded data")
                return []
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return []

    def load_qa_data(self) -> List[Dict]:
        """Load Q&A data with auto-detection from Git repository"""
        # Get file modification time for cache invalidation
        filename = "qa_data.json"
        mtime = None
        if os.path.exists(filename):
            mtime = os.path.getmtime(filename)
        
        # First try to load from Git repository file
        qa_data = self.load_qa_data_from_file(filename, _mtime=mtime)
        
        if qa_data:
            return qa_data
        
        # Fallback to embedded data
        print("📋 Using embedded Q&A data as fallback")
        try:
            # Try to load from the embedded data
            qa_data_json = '''[
  {
    "id": 1,
    "category": "Γενικές Πληροφορίες",
    "question": "Πώς ξεκινάω την πρακτική μου άσκηση;",
    "answer": "**Βήμα 1:** Επικοινωνήστε με τον υπεύθυνο **Γεώργιο Σοφιανίδη** στο gsofianidis@mitropolitiko.edu.gr\\n\\n**Βήμα 2:** Βρείτε δομή (γυμναστήριο, σωματείο, σχολείο) που σας ενδιαφέρει\\n\\n**Βήμα 3:** Ετοιμάστε τα απαραίτητα έγγραφα (αίτηση, ασφαλιστική ικανότητα, υπεύθυνη δήλωση)\\n\\n**Σημαντικό:** Χρειάζεστε να ολοκληρώσετε **240 ώρες μέχρι 30/4**. Το κολλέγιο καλύπτει όλα τα έξοδα της σύμβασης.",
    "keywords": ["ξεκινάω", "ξεκινω", "αρχή", "αρχίζω", "πρακτική", "άσκηση", "πώς", "πως", "βήματα"]
  },
  {
    "id": 2,
    "category": "Έγγραφα & Διαδικασίες",
    "question": "Τι έγγραφα χρειάζομαι για την πρακτική άσκηση;",
    "answer": "**Για εσάς (φοιτητή):**\\n• Αίτηση πραγματοποίησης πρακτικής άσκησης\\n• Στοιχεία φοιτητή (συμπληρωμένη φόρμα)\\n• **Ασφαλιστική ικανότητα** από gov.gr\\n• **Υπεύθυνη δήλωση** (δεν παίρνετε επίδομα ΟΑΕΔ)\\n\\n**Για τη δομή:**\\n• Στοιχεία φορέα (ΑΦΜ, διεύθυνση, νόμιμος εκπρόσωπος, IBAN)\\n• Ημέρες και ώρες που σας δέχεται\\n\\n**Σημείωση:** Ξεκινήστε από την ασφαλιστική ικανότητα γιατί χρειάζεται χρόνο.",
    "keywords": ["έγγραφα", "εγγραφα", "χρειάζομαι", "χρειαζομαι", "απαιτήσεις", "απαιτησεις", "δικαιολογητικά", "δικαιολογητικα", "αίτηση", "αιτηση"]
  },
  {
    "id": 3,
    "category": "Τοποθέτηση",
    "question": "Πού μπορώ να κάνω την πρακτική μου άσκηση;",
    "answer": "**Δημόσιοι Φορείς:**\\n• Σχολεία (δημοτικά, γυμνάσια, λύκεια)\\n• Δημοτικά αθλητικά κέντρα\\n• ΔΑΚ (Δημοτικές Αθλητικές Κοινότητες)\\n\\n**Ιδιωτικοί Φορείς:**\\n• Γυμναστήρια & Fitness clubs\\n• Αθλητικά σωματεία\\n• Κολυμβητικά κέντρα\\n• Ιδιωτικά αθλητικά κέντρα\\n\\n**Προσοχή:** Η δομή πρέπει να είναι αναγνωρισμένη και να σχετίζεται με την προπονητική/φυσική αγωγή. Το ωράριο συμφωνείται μαζί τους.",
    "keywords": ["που", "πού", "τοποθέτηση", "τοποθετηση", "γυμναστήρια", "γυμναστηρια", "σωματεία", "σωματεια", "σχολεία", "σχολεια", "φορείς", "φορεις", "δομή", "δομη"]
  },
  {
    "id": 4,
    "category": "Ώρες & Χρονοδιάγραμμα",
    "question": "Πόσες ώρες πρέπει να κάνω πρακτική άσκηση;",
    "answer": "**Υποχρεωτικό:** Τουλάχιστον **240 ώρες**\\n\\n**Deadline:** Μέχρι **30 Απριλίου**\\n\\n**Κανόνες ωραρίου:**\\n• Δευτέρα έως Σάββατο (ΌΧΙ Κυριακές)\\n• Μέχρι **8 ώρες την ημέρα**\\n• Το ωράριο ορίζεται από τη δομή σε συνεργασία μαζί σας\\n\\n**Υπολογισμός:** 240 ώρες = περίπου 6 εβδομάδες x 40 ώρες ή 8 εβδομάδες x 30 ώρες\\n\\n**Αν τελειώσετε νωρίτερα** από την προβλεπόμενη ημερομηνία, ενημερώστε τον Γεώργιο Σοφιανίδη.",
    "keywords": ["ώρες", "ωρες", "240", "πόσες", "ποσες", "πόσα", "ποσα", "χρονοδιάγραμμα", "χρονοδιαγραμμα", "διάρκεια", "διαρκεια", "χρόνος", "χρονος", "30/4", "deadline"]
  },
  {
    "id": 5,
    "category": "Ασφαλιστική Ικανότητα",
    "question": "Πώς βγάζω ασφαλιστική ικανότητα;",
    "answer": "**Για να αρχίσετε τη διαδικασία, ακολουθήστε τα ακόλουθα βήματα:**\\n\\n**Βήμα 1:** Εισέλθετε στο ιστότοπο του Οργανισμού Ερευνών και Τεχνολογικής Ανάπτυξης (ΕΤΑΑΔ) στο gov.gr.\\n\\n**Βήμα 2:** Βρείτε την υπηρεσία \\"Ασφαλιστική ικανότητα\\" και κάντε κλικ πάνω σε αυτήν.\\n\\n**Βήμα 3:** Εισάγετε τα απαραίτητα στοιχεία σας, όπως το ΑΜΚΑ, το όνομα και το επώνυμό σας, και κάντε κλικ στο \\"Επιβεβαίωση\\".\\n\\n**Βήμα 4:** Αν σας απαιτείται, προσθέστε τα απαραίτητα έγγραφα, όπως το διαβατήριο ή το βεβαίωμα διαμονής.\\n\\n**Βήμα 5:** Κάντε κλικ στο \\"Απάντηση\\" για να δείτε την ασφαλιστική ικανότητά σας.\\n\\n**Σημαντικό:** Μπορεί να χρειαστείτε κάποιο χρόνο για να βγάλετε την ασφαλιστική ικανότητα, ενδεχομένως να ξεκινήσετε από αυτήν τη διαδικασία όσο το δυνατόν πιο σύντομα.\\n\\nΕάν αντιμετωπίσετε οποιαδήποτε δυσκολία ή εάν χρειάζεστε βοήθεια, μπορείτε να επικοινωνήσετε με τον υπεύθυνο Γεώργιο Σοφιανίδη στο gsofianidis@mitropolitiko.edu.gr.",
    "keywords": ["ασφαλιστική", "ασφαλιστικη", "ικανότητα", "ικανοτητα", "πιστοποιητικό", "πιστοποιητικο", "gov.gr", "taxisnet", "ασφάλιση", "ασφαλιση"]
  },
  {
    "id": 6,
    "category": "Επικοινωνία",
    "question": "Με ποιον επικοινωνώ για θέματα πρακτικής άσκησης;",
    "answer": "**Υπεύθυνος Πρακτικής Άσκησης:**\\n**Γεώργιος Σοφιανίδης**\\n📧 gsofianidis@mitropolitiko.edu.gr\\n\\n**Για τεχνικά προβλήματα (εφαρμογές, moodle, κλπ):**\\n**Γεώργιος Μπουχουράς**\\n📧 gbouchouras@mitropolitiko.edu.gr\\n\\n**Σημείωση:** Επικοινωνήστε με τον κ. Σοφιανίδη για όλα τα θέματα περιεχομένου της πρακτικής άσκησης. Για τεχνικά ζητήματα, απευθυνθείτε στον κ. Μπουχουρά.",
    "keywords": ["επικοινωνία", "επικοινωνια", "επικοινωνώ", "επικοινωνω", "email", "τηλέφωνο", "τηλεφωνο", "υπεύθυνος", "υπευθυνος", "Σοφιανίδης", "Σοφιανιδης"]
  },
  {
    "id": 7,
    "category": "Μηνιαίο Ημερολόγιο",
    "question": "Πώς συμπληρώνω το μηνιαίο ημερολόγιο;",
    "answer": "**Υποχρεωτικό:** Κάθε μήνα πρέπει να στείλετε ημερολόγιο με τις ώρες σας\\n\\n**Περιεχόμενο ημερολογίου:**\\n• **Ημερομηνία** κάθε μέρας\\n• **Ώρες άφιξης και αναχώρησης**\\n• **Σύνολο ωρών ανά ημέρα**\\n• **Περιγραφή δραστηριοτήτων** (προπόνηση, διοικητικά, κλπ)\\n• **Υπογραφή επόπτη** από τη δομή\\n\\n**Αποστολή:**\\n• Στείλτε το στον Γεώργιο Σοφιανίδη κάθε τέλος μήνα\\n• Μορφή: PDF ή φωτογραφία με καλή ανάγνωση\\n\\n**Προσοχή:** Χωρίς μηνιαίο ημερολόγιο δεν μπορεί να αναγνωριστεί η πρακτική σας.",
    "keywords": ["ημερολόγιο", "ημερολογιο", "μηνιαίο", "μηνιαιο", "ώρες", "ωρες", "καταγραφή", "καταγραφη", "στείλω", "στειλω"]
  },
  {
    "id": 8,
    "category": "Κλειδώματα & Καθυστερήσεις",
    "question": "Τι γίνεται αν καθυστερήσω;",
    "answer": "**Σημαντικές προθεσμίες:**\\n\\n**30 Απριλίου:** Τέλος πρακτικής άσκησης\\n• Αν δεν ολοκληρώσετε τις 240 ώρες, κλειδώνει το μάθημα\\n• Θα πρέπει να επαναλάβετε την επόμενη χρονιά\\n\\n**15 Οκτωβρίου:** Παράδοση συμβάσεων στο Moodle\\n• Αν δεν παραδώσετε, δεν θα περάσετε το μάθημα\\n• Ακόμη και αν έχετε κάνει τις ώρες\\n\\n**Συμβουλή:** Μην αφήνετε τίποτα για το τέλος. Ξεκινήστε νωρίς και ενημερώνετε τακτικά τον υπεύθυνο.",
    "keywords": ["καθυστερήσω", "καθυστερησω", "καθυστέρηση", "καθυστερηση", "προθεσμία", "προθεσμια", "κλείδωμα", "κλειδωμα", "deadline"]
  },
  {
    "id": 9,
    "category": "Ασφάλιση & Ατυχήματα",
    "question": "Τι γίνεται αν πάθω ατύχημα κατά την πρακτική;",
    "answer": "**Κάλυψη:** Είστε ασφαλισμένοι από το κολλέγιο κατά τη διάρκεια της πρακτικής\\n\\n**Σε περίπτωση ατυχήματος:**\\n1. **Άμεσα:** Ενημερώστε τον επόπτη της δομής\\n2. **Ιατρική βοήθεια:** Αν χρειάζεται, πηγαίνετε σε γιατρό/νοσοκομείο\\n3. **Καταγραφή:** Συμπληρώστε έντυπο ατυχήματος στη δομή\\n4. **Ενημέρωση:** Ειδοποιήστε ΑΜΕΣΑ τον Γεώργιο Σοφιανίδη\\n\\n**Παραστατικά που χρειάζεστε:**\\n• Αντίγραφο εντύπου ατυχήματος\\n• Ιατρικές εξετάσεις (αν υπάρχουν)\\n• Βεβαίωση από τη δομή\\n\\n**Σημαντικό:** Μην αγνοήσετε ακόμη και μικρά ατυχήματα.",
    "keywords": ["ατύχημα", "ατυχημα", "ασφάλιση", "ασφαλιση", "τραυματισμός", "τραυματισμος", "γιατρός", "γιατρος", "νοσοκομείο", "νοσοκομειο"]
  },
  {
    "id": 10,
    "category": "Αξιολόγηση",
    "question": "Πώς αξιολογούμαι στην πρακτική άσκηση;",
    "answer": "**Κριτήρια αξιολόγησης:**\\n\\n**Επόπτης δομής (70%):**\\n• Συνέπεια και παρουσία\\n• Συνεργασία και επαγγελματισμός\\n• Ικανότητες προπονητικής\\n• Συμμετοχή σε δραστηριότητες\\n\\n**Υπεύθυνος πρακτικής (30%):**\\n• Μηνιαία ημερολόγια\\n• Τελική αναφορά πρακτικής\\n• Συνολική εκτίμηση προόδου\\n\\n**Βαθμολογία:** 1-10 (πέρασμα από 5)\\n\\n**Τελική εξέταση:** ΔΕΝ υπάρχει γραπτή εξέταση. Η αξιολόγηση βασίζεται αποκλειστικά στην πρακτική εργασία και τα παραδοτέα.",
    "keywords": ["αξιολόγηση", "αξιολογηση", "βαθμός", "βαθμος", "βαθμολογία", "βαθμολογια", "εξέταση", "εξεταση", "πέρασα", "περασα"]
  }
]'''
            data = json.loads(qa_data_json)
            return data
        except Exception as e:
            print(f"❌ Error loading embedded data: {e}")
            return self.get_default_qa_data()

    def get_default_qa_data(self) -> List[Dict]:
        """Fallback Q&A data"""
        return [
            {
                "id": 1,
                "category": "Γενικές Πληροφορίες",
                "question": "Πώς ξεκινάω την πρακτική μου άσκηση;",
                "answer": "Για να ξεκινήσετε την πρακτική άσκηση, επικοινωνήστε με τον υπεύθυνο Γεώργιο Σοφιανίδη στο gsofianidis@mitropolitiko.edu.gr. Απαιτούνται 240 ώρες πρακτικής άσκησης σε δομή της επιλογής σας μέχρι 30/4.",
                "keywords": ["ξεκινάω", "πρακτική", "άσκηση", "αρχή", "πώς"]
            }
        ]

    def calculate_similarity(self, question: str, qa_entry: Dict) -> float:
        """Calculate similarity between question and QA entry"""
        question_lower = question.lower()
        
        # Check if any keyword matches
        keyword_matches = sum(1 for keyword in qa_entry.get('keywords', []) 
                            if keyword.lower() in question_lower)
        
        # Check title similarity
        title_words = qa_entry['question'].lower().split()
        title_matches = sum(1 for word in title_words if word in question_lower)
        
        # Combine scores
        total_score = (keyword_matches * 2 + title_matches) / max(len(qa_entry.get('keywords', [])) + len(title_words), 1)
        return min(total_score, 1.0)

    def get_ai_response(self, user_message: str, context: str) -> Tuple[str, bool]:
        """Get response from Groq AI with context"""
        if not self.groq_client:
            return "", False
        
        try:
            # Prepare the full prompt with context
            full_prompt = f"""Context πληροφοριών:
{context}

Ερώτηση φοιτητή: {user_message}

Χρησιμοποίησε ΜΟΝΟ τις πληροφορίες από το context για να απαντήσεις."""

            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,  # Χαμηλότερο για πιο συνεπείς απαντήσεις
                max_tokens=800,
                top_p=0.9,        # Πιο συντηρητικό για σταθερότητα
                stream=False
            )

            response = chat_completion.choices[0].message.content
            
            # Έλεγχος για μη-ελληνικούς χαρακτήρες
            if response and any(ord(char) > 1500 and ord(char) not in range(0x0370, 0x03FF) for char in response):
                print("⚠️ Detected non-Greek characters in response, using fallback")
                return "", False
            
            return response, True
            
        except Exception as e:
            print(f"❌ Groq API Error: {e}")
            return "", False

    def get_fallback_response(self, question: str) -> str:
        """Fallback response system"""
        if not self.qa_data:
            return "Δεν υπάρχουν διαθέσιμα δεδομένα. Επικοινωνήστε με τον Γεώργιο Σοφιανίδη: gsofianidis@mitropolitiko.edu.gr"

        # Find best match
        best_match = max(self.qa_data, key=lambda x: self.calculate_similarity(question, x))
        similarity = self.calculate_similarity(question, best_match)

        if similarity > 0.2:
            return best_match['answer']
        else:
            return f"""Δεν βρέθηκε συγκεκριμένη απάντηση για αυτή την ερώτηση.

**Προτεινόμενες ενέργειες:**
• Αναδιατυπώστε την ερώτηση
• Επιλέξτε από τις συχνές ερωτήσεις στο αριστερό μενού
• Επικοινωνήστε με τον Γεώργιο Σοφιανίδη: gsofianidis@mitropolitiko.edu.gr"""

    def get_response(self, question: str) -> str:
        """Get chatbot response with AI + fallback logic"""
        if not self.qa_data:
            return "Δεν υπάρχουν διαθέσιμα δεδομένα γνώσης."
        
        # Find relevant context
        matches = sorted(self.qa_data, 
                        key=lambda x: self.calculate_similarity(question, x), 
                        reverse=True)
        
        # Prepare context from top matches
        context_parts = []
        for match in matches[:3]:
            if self.calculate_similarity(question, match) > 0.1:
                context_parts.append(f"Q: {match['question']}\nA: {match['answer']}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        # Try AI response first
        if self.groq_client and context:
            ai_response, success = self.get_ai_response(question, context)
            if success and ai_response.strip():
                return ai_response
        
        # Fallback to rule-based response
        return self.get_fallback_response(question)

def initialize_qa_file():
    """Create initial qa_data.json if it doesn't exist (fallback for development)"""
    if not os.path.exists("qa_data.json"):
        print("📄 Creating initial qa_data.json file for development...")
        initial_data = [
            {
                "id": 1,
                "category": "Γενικές Πληροφορίες",
                "question": "Πώς ξεκινάω την πρακτική μου άσκηση;",
                "answer": "**Βήμα 1:** Επικοινωνήστε με τον υπεύθυνο **Γεώργιο Σοφιανίδη** στο gsofianidis@mitropolitiko.edu.gr\n\n**Βήμα 2:** Βρείτε δομή (γυμναστήριο, σωματείο, σχολείο) που σας ενδιαφέρει\n\n**Βήμα 3:** Ετοιμάστε τα απαραίτητα έγγραφα (αίτηση, ασφαλιστική ικανότητα, υπεύθυνη δήλωση)\n\n**Σημαντικό:** Χρειάζεστε να ολοκληρώσετε **240 ώρες μέχρι 30/4**. Το κολλέγιο καλύπτει όλα τα έξοδα της σύμβασης.",
                "keywords": ["ξεκινάω", "ξεκινω", "αρχή", "αρχίζω", "πρακτική", "άσκηση", "πώς", "πως", "βήματα"]
            }
        ]
        
        try:
            with open("qa_data.json", 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
            print("✅ Initial qa_data.json created for development")
        except Exception as e:
            print(f"❌ Error creating qa_data.json: {e}")

def main():
    """Main Streamlit application - Git-first content management"""
    # Initialize QA file if needed (development fallback)
    initialize_qa_file()
    
    # CSS Styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 1px solid #e8f4f8;
    }
    
    .user-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-left: 4px solid #007bff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #333;
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
    
    .quick-stats {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        gap: 1rem;
    }
    
    .stat-item {
        text-align: center;
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e8f4f8;
        flex: 1;
    }
    
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e8f4f8;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e8f4f8;
        padding: 0.8rem 1.2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 78, 121, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Πρακτική Άσκηση</h1>
        <h3>Μητροπολιτικό Κολλέγιο - Τμήμα Προπονητικής & Φυσικής Αγωγής</h3>
        <p><em>Εξειδικευμένος AI Assistant για υποστήριξη φοιτητών</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'chatbot' not in st.session_state:
        # Get Groq API key from secrets or environment
        groq_api_key = None
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        except:
            pass
        st.session_state.chatbot = InternshipChatbot(groq_api_key)
    else:
        # Refresh data if cache was cleared
        current_data_count = len(st.session_state.chatbot.qa_data)
        st.session_state.chatbot.qa_data = st.session_state.chatbot.load_qa_data()
        new_data_count = len(st.session_state.chatbot.qa_data)
        
        if new_data_count != current_data_count:
            st.toast(f"📊 Δεδομένα ενημερώθηκαν: {new_data_count} ερωτήσεις")

    # Quick info cards
    st.markdown("### 📊 Σημαντικές Πληροφορίες")
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h4 style="color: #1f4e79; margin-bottom: 0.5rem;">📅 Απαιτούμενες Ώρες</h4>
            <p style="font-size: 1.2rem; font-weight: 600; color: #28a745; margin: 0;">240 ώρες</p>
            <small style="color: #6c757d;">Προθεσμία: 30 Απριλίου</small>
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

    # API Status
    if st.session_state.chatbot.groq_client:
        st.markdown('<div class="api-status">🚀 AI Assistant Ενεργό</div>', unsafe_allow_html=True)
        
    # Επαγγελματική ενδειξη για sidebar
    st.markdown("""
    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 0.6rem; margin-bottom: 1.5rem; text-align: center; font-size: 0.9rem;">
        <strong>Πληροφορίες:</strong> Χρησιμοποιήστε το αριστερό μενού για συχνές ερωτήσεις και επικοινωνία 👈<br>
        <small>🔄 Τα δεδομένα ενημερώνονται αυτόματα από το Git repository</small>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 📞 Επικοινωνία")
        
        st.markdown("""
        **Υπεύθυνος Πρακτικής Άσκησης**  
        **Γεώργιος Σοφιανίδης**  
        📧 gsofianidis@mitropolitiko.edu.gr
        
        **Τεχνική Υποστήριξη**  
        **Γεώργιος Μπουχουράς**  
        📧 gbouchouras@mitropolitiko.edu.gr
        """)

        st.markdown("---")

        # Συχνές ερωτήσεις
        st.markdown("## 🔄 Συχνές Ερωτήσεις")
        
        # Group questions by category
        categories = {}
        for qa in st.session_state.chatbot.qa_data:
            cat = qa.get('category', 'Άλλα')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(qa)

        for category, questions in categories.items():
            if st.expander(f"📂 {category}"):
                for qa in questions:
                    if st.button(qa['question'], key=f"faq_{qa['id']}", use_container_width=True):
                        # Add to chat
                        st.session_state.messages.append({"role": "user", "content": qa['question']})
                        st.session_state.messages.append({"role": "assistant", "content": qa['answer']})
                        st.rerun()

        st.markdown("---")

        # AI Status
        if st.session_state.chatbot.groq_client:
            st.success("🤖 AI Assistant Ενεργό")
            st.info("Χρησιμοποιεί Llama 3.1 8B")
        else:
            st.warning("📚 Knowledge Base Mode")
            if GROQ_AVAILABLE:
                st.info("Για AI responses, χρειάζεται Groq API key")
            else:
                st.error("Groq library δεν είναι διαθέσιμη")

        st.markdown("---")

        if st.button("🗑️ Νέα Συνομιλία", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Τεχνικές πληροφορίες
        if st.checkbox("🔧 Τεχνικές Πληροφορίες"):
            st.markdown("**Για τεχνικά προβλήματα:**")
            st.markdown("📧 gbouchouras@mitropolitiko.edu.gr")
            st.write("Groq Available:", GROQ_AVAILABLE)
            st.write("Groq Client:", st.session_state.chatbot.groq_client is not None)
            st.write("QA Data Count:", len(st.session_state.chatbot.qa_data))
            
            # Check data source
            if os.path.exists("qa_data.json"):
                mtime = os.path.getmtime("qa_data.json")
                last_modified = datetime.datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M")
                st.success(f"📄 Data Source: qa_data.json (από Git)")
                st.info(f"🕒 Τελευταία ενημέρωση: {last_modified}")
            else:
                st.warning("📋 Data Source: Embedded (fallback)")
                st.info("💡 Για ενημέρωση: git pull + redeploy")

    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 💬 Κάντε την ερώτησή σας")

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>Εσείς:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Convert markdown to HTML for better display
            content = message["content"].replace('\n', '<br>')
            st.markdown(f'<div class="ai-message"><strong>🤖 Assistant:</strong><br><br>{content}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input - moved outside container for better functionality
    user_input = st.chat_input("Γράψτε την ερώτησή σας εδώ...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get chatbot response
        with st.spinner("Σκέφτομαι..."):
            try:
                response = st.session_state.chatbot.get_response(user_input)
            except Exception as e:
                response = f"Συγγνώμη, παρουσιάστηκε σφάλμα: {str(e)}"
                st.error(f"Σφάλμα: {e}")
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to display new messages
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <small>
            🎓 <strong>Μητροπολιτικό Κολλέγιο Θεσσαλονίκης</strong> | 
            Τμήμα Προπονητικής & Φυσικής Αγωγής<br>
            <em>AI-Powered Internship Assistant</em>
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()    main()