import streamlit as st
import json
import pandas as pd
from datetime import datetime
import difflib
import re
from typing import List, Dict, Tuple
import os

# Ρύθμιση σελίδας
st.set_page_config(
    page_title="Πρακτική Άσκηση - Μητροπολιτικό Κολλέγιο",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .main-header {
        color: #1f4e79;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #e8f4f8;
        padding-bottom: 1rem;
        margin-top: 1rem;
    }
    
    .sub-header {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 1px solid #e8f4f8;
    }
    
    .logo-container img {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: #f8f9fa;
        border-left: 4px solid #1f4e79;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .bot-message {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .confidence-high { border-left-color: #28a745 !important; }
    .confidence-medium { border-left-color: #ffc107 !important; }
    .confidence-low { border-left-color: #dc3545 !important; }
    
    .info-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Κρύψιμο Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Κρύψιμο τυχόν κατηγοριών από παλαιότερες εκδόσεις */
    .categories-section {display: none !important;}
    .element-container:has(.categories) {display: none !important;}
    div[data-testid="stSidebar"] div:contains("Κατηγορίες") {display: none !important;}
    
    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
        .sub-header { font-size: 1rem; }
        .logo-container img { max-width: 120px; }
    }
</style>
""", unsafe_allow_html=True)

class PracticeTrainingChatbot:
    def __init__(self):
        self.qa_data = self.load_qa_data()
        self.conversation_history = []
        
        # Συχνές ερωτήσεις για γρήγορη πρόσβαση
        self.frequent_questions = [
            "Πώς ξεκινάω την πρακτική άσκηση;",
            "Τι έγγραφα χρειάζομαι;",
            "Πόσες ώρες πρέπει να κάνω;",
            "Πώς βγάζω ασφαλιστική ικανότητα;",
            "Με ποιον επικοινωνώ;"
        ]
        
        # Επεκτεταμένο λεξικό συνώνυμων για καλύτερο matching
        self.synonyms = {
            # Βασικές λέξεις
            'πως': 'πώς', 'που': 'πού', 'ποσες': 'πόσες', 'ποσα': 'πόσα',
            'χρειαζομαι': 'χρειάζομαι', 'θελω': 'θέλω', 'μπορω': 'μπορώ',
            
            # Πρακτική άσκηση
            'πρακτικη': 'πρακτική', 'ασκηση': 'άσκηση', 'εξασκηση': 'άσκηση',
            'ξεκιναω': 'ξεκινάω', 'αρχιζω': 'ξεκινάω', 'εκκινω': 'ξεκινάω',
            
            # Έγγραφα
            'εγγραφα': 'έγγραφα', 'χαρτια': 'έγγραφα', 'δικαιολογητικα': 'έγγραφα',
            'φορμες': 'φόρμες', 'αιτηση': 'αίτηση',
            
            # Ώρες και χρόνος
            'ωρες': 'ώρες', 'χρονος': 'χρόνος', 'διαρκεια': 'διάρκεια',
            'χρονοδιαγραμμα': 'χρονοδιάγραμμα',
            
            # Επικοινωνία
            'επικοινωνια': 'επικοινωνία', 'μιλαω': 'επικοινωνώ', 'μιλησω': 'επικοινωνώ',
            'βοηθεια': 'βοήθεια', 'υποστηριξη': 'βοήθεια',
            
            # Ασφαλιστικά
            'ασφαλιστικη': 'ασφαλιστική', 'ικανοτητα': 'ικανότητα',
            'ασφαλιση': 'ασφάλιση', 'βεβαιωση': 'βεβαίωση',
            
            # Δομές
            'δομη': 'δομή', 'φορεα': 'φορέα', 'εταιρια': 'δομή',
            'γυμναστηριο': 'γυμναστήριο', 'σωματειο': 'σωματείο',
            
            # Διαδικασίες
            'συμβαση': 'σύμβαση', 'υπογραφη': 'υπογραφή',
            'σφραγιδα': 'σφραγίδα', 'διαδικασια': 'διαδικασία',
            
            # Αξιολόγηση
            'αξιολογηση': 'αξιολόγηση', 'βιβλιο': 'βιβλίο',
            'κριτηρια': 'κριτήρια', 'βαθμος': 'βαθμός',
            
            # Οικονομικά
            'κοστος': 'κόστος', 'χρηματα': 'χρήματα', 'πληρωμη': 'πληρωμή',
            'δωρεαν': 'δωρεάν', 'τιμολογηση': 'τιμολόγηση'
        }
        
        # Θεματικές κατηγορίες λέξεων
        self.topic_keywords = {
            'documents': ['έγγραφα', 'αίτηση', 'φόρμες', 'δικαιολογητικά', 'χαρτιά', 'υπεύθυνη', 'δήλωση'],
            'start': ['ξεκινάω', 'αρχή', 'εκκίνηση', 'πώς', 'βήματα', 'διαδικασία'],
            'hours': ['ώρες', '240', 'χρόνος', 'διάρκεια', 'πόσες', 'χρονοδιάγραμμα', 'deadline'],
            'insurance': ['ασφαλιστική', 'ικανότητα', 'ασφάλιση', 'βεβαίωση', 'gov.gr', 'taxisnet'],
            'contact': ['επικοινωνία', 'email', 'τηλέφωνο', 'βοήθεια', 'υπεύθυνος', 'μιλάω'],
            'evaluation': ['αξιολόγηση', 'βιβλίο', 'κριτήρια', 'βαθμός', 'απόδοση'],
            'cost': ['κόστος', 'χρήματα', 'πληρωμή', 'δωρεάν', 'οικονομικά']
        }
        
    def load_qa_data(self) -> List[Dict]:
        try:
            if os.path.exists('qa_data.json'):
                with open('qa_data.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self.get_default_qa_data()
        except Exception as e:
            st.error(f"Σφάλμα κατά τη φόρτωση δεδομένων: {e}")
            return self.get_default_qa_data()
    
    def get_default_qa_data(self) -> List[Dict]:
        return [
            {
                "id": 1,
                "category": "Γενικές Πληροφορίες",
                "question": "Πώς ξεκινάω την πρακτική μου άσκηση;",
                "answer": "Για να ξεκινήσετε την πρακτική σας άσκηση, επικοινωνήστε με τον υπεύθυνο Γεώργιο Σοφιανίδη (gsofianidis@mitropolitiko.edu.gr). Πρέπει να συμπληρώσετε 240 ώρες πρακτικής άσκησης σε δομή της επιλογής σας.",
                "keywords": ["ξεκινάω", "πρακτική", "άσκηση", "αρχή", "πώς"]
            }
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Εκτεταμένη προεπεξεργασία κειμένου"""
        # Μετατροπή σε πεζά
        text = text.lower()
        
        # Αφαίρεση ειδικών χαρακτήρων εκτός από ελληνικά
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Αντικατάσταση συνώνυμων
        words = text.split()
        processed_words = []
        for word in words:
            if word in self.synonyms:
                processed_words.append(self.synonyms[word])
            else:
                processed_words.append(word)
        
        text = ' '.join(processed_words)
        text = ' '.join(text.split())  # Αφαίρεση επιπλέον κενών
        return text
    
    def get_topic_match_score(self, question: str) -> Dict[str, float]:
        """Υπολογισμός score ανά θεματική κατηγορία"""
        processed_question = self.preprocess_text(question)
        question_words = processed_question.split()
        
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = 0
            for keyword in keywords:
                keyword_processed = self.preprocess_text(keyword)
                if keyword_processed in processed_question:
                    score += 1
                # Partial matching
                for word in question_words:
                    if (len(word) > 3 and len(keyword_processed) > 3 and 
                        (keyword_processed in word or word in keyword_processed)):
                        score += 0.5
            topic_scores[topic] = score
        
        return topic_scores
    
    def calculate_similarity(self, question: str, qa_item: Dict) -> float:
        """Βελτιωμένος υπολογισμός ομοιότητας"""
        processed_question = self.preprocess_text(question)
        processed_qa_question = self.preprocess_text(qa_item['question'])
        processed_answer = self.preprocess_text(qa_item['answer'])
        
        question_words = processed_question.split()
        qa_words = processed_qa_question.split()
        
        # 1. Άμεση ομοιότητα ερωτήσεων (40%)
        direct_similarity = difflib.SequenceMatcher(None, processed_question, processed_qa_question).ratio()
        
        # 2. Ομοιότητα λέξεων-κλειδιών (35%)
        keyword_score = 0
        if 'keywords' in qa_item:
            for keyword in qa_item['keywords']:
                keyword_processed = self.preprocess_text(keyword)
                
                # Exact match
                if keyword_processed in processed_question:
                    keyword_score += 0.4
                
                # Partial match
                for word in question_words:
                    if (len(word) > 2 and len(keyword_processed) > 2):
                        # Substring matching
                        if keyword_processed in word or word in keyword_processed:
                            keyword_score += 0.2
                        # Edit distance για παρόμοιες λέξεις
                        similarity_ratio = difflib.SequenceMatcher(None, word, keyword_processed).ratio()
                        if similarity_ratio > 0.8:
                            keyword_score += 0.3
        
        keyword_score = min(keyword_score, 1.0)
        
        # 3. Κοινές λέξεις (15%)
        common_words = set(question_words) & set(qa_words)
        word_overlap = len(common_words) / max(len(question_words), 1) if question_words else 0
        
        # 4. Θεματική ομοιότητα (10%)
        topic_scores = self.get_topic_match_score(question)
        qa_topic_scores = self.get_topic_match_score(qa_item['question'])
        
        topic_similarity = 0
        for topic in topic_scores:
            if topic_scores[topic] > 0 and qa_topic_scores[topic] > 0:
                topic_similarity += 0.3
        topic_similarity = min(topic_similarity, 1.0)
        
        # Συνολικός υπολογισμός
        total_similarity = (
            direct_similarity * 0.40 +
            keyword_score * 0.35 +
            word_overlap * 0.15 +
            topic_similarity * 0.10
        )
        
        return min(total_similarity, 1.0)
    
    def find_best_answer(self, question: str) -> Tuple[str, float, str]:
        """Βρες την καλύτερη απάντηση με βελτιωμένο matching"""
        if not self.qa_data:
            return "Λυπάμαι, δεν υπάρχουν διαθέσιμα δεδομένα.", 0.0, "Σφάλμα"
        
        # Υπολογισμός ομοιότητας για όλα τα items
        scored_items = []
        for item in self.qa_data:
            similarity = self.calculate_similarity(question, item)
            scored_items.append((item, similarity))
        
        # Ταξινόμηση κατά φθίνουσα σειρά ομοιότητας
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        best_match, similarity = scored_items[0]
        
        # Αν η ομοιότητα είναι πολύ χαμηλή, δώσε generic απάντηση
        if similarity < 0.15:
            return self.get_fallback_response(question), similarity, "Γενική Βοήθεια"
        
        return best_match['answer'], similarity, best_match.get('category', 'Γενικά')
    
    def get_fallback_response(self, question: str) -> str:
        """Γενική απάντηση όταν δεν βρίσκεται κατάλληλο match"""
        processed_question = self.preprocess_text(question)
        
        # Έλεγχος για θεματικές κατηγορίες
        topic_scores = self.get_topic_match_score(question)
        max_topic = max(topic_scores, key=topic_scores.get) if topic_scores else None
        
        if topic_scores.get(max_topic, 0) > 0:
            topic_responses = {
                'contact': "Για αυτή την ερώτηση, παρακαλώ επικοινωνήστε απευθείας με τον υπεύθυνο **Γεώργιο Σοφιανίδη** στο gsofianidis@mitropolitiko.edu.gr",
                'documents': "Σχετικά με τα έγγραφα, επικοινωνήστε με τον **Γεώργιο Σοφιανίδη** (gsofianidis@mitropolitiko.edu.gr) για λεπτομερείς οδηγίες.",
                'hours': "Για ερωτήσεις σχετικά με τις ώρες και τα χρονοδιαγράμματα, επικοινωνήστε με τον **Γεώργιο Σοφιανίδη** στο gsofianidis@mitropolitiko.edu.gr",
                'insurance': "Για θέματα ασφαλιστικής ικανότητας, δοκιμάστε πρώτα το gov.gr ή επικοινωνήστε με τον **Γεώργιο Σοφιανίδη** (gsofianidis@mitropolitiko.edu.gr)."
            }
            
            if max_topic in topic_responses:
                return topic_responses[max_topic]
        
        # Έλεγχος για λέξεις επικοινωνίας
        contact_words = ['επικοινωνία', 'τηλέφωνο', 'email', 'μαιλ', 'πού', 'ποιος', 'υπεύθυνος']
        if any(word in processed_question for word in contact_words):
            return "Ο υπεύθυνος για την πρακτική άσκηση είναι ο **Γεώργιος Σοφιανίδης**. Μπορείτε να τον επικοινωνήσετε στο gsofianidis@mitropolitiko.edu.gr"
        
        # Γενική απάντηση
        return """Δεν βρήκα συγκεκριμένη απάντηση για αυτή την ερώτηση. 
        
**Προτείνω:**
• Δοκιμάστε να αναδιατυπώσετε την ερώτηση με διαφορετικές λέξεις
• Επιλέξτε από τις συχνές ερωτήσεις στο sidebar
• Επικοινωνήστε με τον **Γεώργιο Σοφιανίδη**: gsofianidis@mitropolitiko.edu.gr"""
    
    def get_response(self, question: str) -> Dict:
        """Κύρια μέθοδος για απάντηση"""
        answer, similarity, category = self.find_best_answer(question)
        
        # Βελτίωση της απάντησης βάση confidence
        if similarity < 0.3:
            answer = f"{answer}\n\n💡 **Συμβουλή:** Δοκιμάστε να διατυπώσετε την ερώτηση διαφορετικά ή επιλέξτε από τις συχνές ερωτήσεις."
        
        response = {
            'answer': answer,
            'confidence': similarity,
            'category': category,
            'timestamp': datetime.now().strftime("%H:%M")
        }
        
        self.conversation_history.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now()
        })
        
        return response

def main():
    # Αρχικοποίηση
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PracticeTrainingChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Header με logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    
    logo_col, title_col = st.columns([1, 4])
    
    with logo_col:
        try:
            st.image("https://raw.githubusercontent.com/GiorgosBouh/chatbot.placement/main/MK_LOGO_SEO_1200x630.png", width=140)
        except:
            # Fallback αν το logo δεν φορτώνει
            st.markdown("🎓", unsafe_allow_html=True)
    
    with title_col:
        st.markdown('<h1 class="main-header">Πρακτική Άσκηση</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Μητροπολιτικό Κολλέγιο Θεσσαλονίκης • Προπονητική & Φυσική Αγωγή</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Layout με στήλες
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Quick Info Cards
        with st.container():
            quick_col1, quick_col2, quick_col3 = st.columns(3)
            
            with quick_col1:
                st.markdown("""
                <div class="info-card" style="text-align: center;">
                    <h4 style="color: #1f4e79; margin-bottom: 0.5rem;">📅 Ώρες</h4>
                    <p style="font-size: 1.2rem; font-weight: 600; color: #28a745; margin: 0;">240 ώρες</p>
                    <small style="color: #6c757d;">μέχρι 30/4</small>
                </div>
                """, unsafe_allow_html=True)
            
            with quick_col2:
                st.markdown("""
                <div class="info-card" style="text-align: center;">
                    <h4 style="color: #1f4e79; margin-bottom: 0.5rem;">📋 Σύμβαση</h4>
                    <p style="font-size: 1.2rem; font-weight: 600; color: #ffc107; margin: 0;">Moodle</p>
                    <small style="color: #6c757d;">μέχρι 15/10</small>
                </div>
                """, unsafe_allow_html=True)
            
            with quick_col3:
                st.markdown("""
                <div class="info-card" style="text-align: center;">
                    <h4 style="color: #1f4e79; margin-bottom: 0.5rem;">⏰ Ωράριο</h4>
                    <p style="font-size: 1.2rem; font-weight: 600; color: #17a2b8; margin: 0;">Δε-Σα</p>
                    <small style="color: #6c757d;">μέχρι 8ω/ημέρα</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Βελτιωμένο header styling
        st.markdown('<div style="margin-bottom: 2rem;"></div>', unsafe_allow_html=True)
        
        # Chat Interface
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="user-message">
                    <strong>Ερώτηση:</strong> {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
            else:
                confidence = message.get("confidence", 0)
                category = message.get("category", "")
                timestamp = message.get("timestamp", "")
                
                if confidence > 0.6:
                    conf_class = "confidence-high"
                    conf_icon = "🟢"
                    conf_text = "Υψηλή"
                elif confidence > 0.3:
                    conf_class = "confidence-medium"
                    conf_icon = "🟡"
                    conf_text = "Μέτρια"
                else:
                    conf_class = "confidence-low"
                    conf_icon = "🔴"
                    conf_text = "Χαμηλή"
                
                st.markdown(f'''
                <div class="bot-message {conf_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <strong style="color: #1f4e79;">💬 Απάντηση</strong>
                        <span style="font-size: 0.85rem; color: #6c757d;">
                            {conf_icon} {conf_text} • {timestamp}
                        </span>
                    </div>
                    <div style="line-height: 1.6;">
                        {message["content"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Input
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form(key='question_form', clear_on_submit=True):
            user_input = st.text_input(
                label="Γράψτε την ερώτησή σας:",
                placeholder="π.χ. Πώς ξεκινάω την πρακτική άσκηση;",
                label_visibility="collapsed"
            )
            
            col_a, col_b, col_c = st.columns([2, 1, 2])
            with col_b:
                submitted = st.form_submit_button("Αποστολή", use_container_width=True)
        
        if submitted and user_input.strip():
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            
            # Get bot response
            response = st.session_state.chatbot.get_response(user_input.strip())
            
            # Add bot message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response['answer'],
                "confidence": response['confidence'],
                "category": response['category'],
                "timestamp": response['timestamp']
            })
            
            st.rerun()
    
    # Sidebar με χρήσιμα στοιχεία
    with st.sidebar:
        st.markdown("## 📞 Επικοινωνία")
        
        st.markdown("""
        **Υπεύθυνος Πρακτικής Άσκησης**  
        **Γεώργιος Σοφιανίδης**  
        📧 gsofianidis@mitropolitiko.edu.gr
        """)
        
        st.markdown("---")
        
        # Συχνές ερωτήσεις
        st.markdown("## ❓ Συχνές Ερωτήσεις")
        
        for question in st.session_state.chatbot.frequent_questions:
            if st.button(question, key=f"faq_{question}", use_container_width=True):
                # Προσθήκη της ερώτησης στη συνομιλία
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Λήψη απάντησης
                response = st.session_state.chatbot.get_response(question)
                
                # Προσθήκη απάντησης
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response['answer'],
                    "confidence": response['confidence'],
                    "category": response['category'],
                    "timestamp": response['timestamp']
                })
                
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("## 🔗 Χρήσιμοι Σύνδεσμοι1")
        st.link_button("🏛️ Ασφαλιστική Ικανότητα", "https://www.gov.gr/ipiresies/ergasia-kai-asphalise/asphalise/asphalistike-ikanoteta")
        st.link_button("📋 ATLAS", "https://www.atlas.gov.gr/ATLAS/Pages/Home.aspx")
        st.link_button("📑 Υπεύθυνη Δήλωση", "https://www.gov.gr")
        
        st.markdown("---")
        
        if st.button("🗑️ Νέα Συνομιλία", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #6c757d; font-size: 0.9rem; padding: 2rem 0; border-top: 1px solid #e9ecef;">
            Μητροπολιτικό Κολλέγιο Θεσσαλονίκης • Τμήμα Προπονητικής & Φυσικής Αγωγής<br>
            <small>Για τεχνική υποστήριξη επικοινωνήστε με τον Γεώργιο Σοφιανίδη</small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()