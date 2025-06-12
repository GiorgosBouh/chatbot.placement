import streamlit as st
import json
import pandas as pd
from datetime import datetime
import difflib
import re
from typing import List, Dict, Tuple
import os

# Ρύθμιση σελίδας Streamlit
st.set_page_config(
    page_title="Chatbot Πρακτικής Άσκησης - Μητροπολιτικό Κολλέγιο",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS για καλύτερη εμφάνιση
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }
    
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2E86AB;
        background-color: #f0f8ff;
    }
    
    .bot-response {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class PracticeTrainingChatbot:
    def __init__(self):
        self.qa_data = self.load_qa_data()
        self.conversation_history = []
        
    def load_qa_data(self) -> List[Dict]:
        """Φόρτωση δεδομένων Q&A από αρχείο JSON"""
        try:
            if os.path.exists('qa_data.json'):
                with open('qa_data.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Προεπιλεγμένα δεδομένα αν δεν υπάρχει αρχείο
                return self.get_default_qa_data()
        except Exception as e:
            st.error(f"Σφάλμα κατά τη φόρτωση δεδομένων: {e}")
            return self.get_default_qa_data()
    
    def get_default_qa_data(self) -> List[Dict]:
        """Προεπιλεγμένα δεδομένα Q&A"""
        return [
            {
                "id": 1,
                "category": "Γενικές Πληροφορίες",
                "question": "Πώς ξεκινάω την πρακτική μου άσκηση;",
                "answer": "Για να ξεκινήσετε την πρακτική σας άσκηση, επικοινωνήστε με την υπεύθυνη Μαρία Ταμπάκη (mtampaki@mitropolitiko.edu.gr). Πρέπει να συμπληρώσετε 240 ώρες πρακτικής άσκησης σε δομή της επιλογής σας. Το κολλέγιο καλύπτει τη σύμβαση και δεν υπάρχει οικονομική υποχρέωση για τη δομή.",
                "keywords": ["ξεκινάω", "πρακτική", "άσκηση", "αρχή", "πώς", "240", "ώρες"]
            },
            {
                "id": 2,
                "category": "Έγγραφα & Διαδικασίες",
                "question": "Τι έγγραφα χρειάζομαι για την πρακτική άσκηση;",
                "answer": "Χρειάζεστε: 1) Αίτηση πραγματοποίησης πρακτικής άσκησης, 2) Στοιχεία φοιτητή (συμπληρωμένη φόρμα), 3) Υπεύθυνη δήλωση (Ν.1599/1986) ότι δεν είστε γραμμένοι στον ΟΑΕΔ, 4) Πιστοποιητικό ασφαλιστικής ικανότητας από gov.gr, 5) Στοιχεία φορέα (συμπληρωμένα από τη δομή).",
                "keywords": ["έγγραφα", "χρειάζομαι", "απαιτήσεις", "φόρμες", "δικαιολογητικά", "αίτηση"]
            },
            {
                "id": 3,
                "category": "Τοποθέτηση",
                "question": "Πού μπορώ να κάνω την πρακτική μου άσκηση;",
                "answer": "Η πρακτική άσκηση γίνεται σε δομή της αρεσκείας σας: γυμναστήρια, αθλητικά σωματεία, σχολεία, δημοτικά αθλητικά κέντρα, ιδιωτικά αθλητικά κέντρα, fitness clubs, κολυμβητικά κέντρα και άλλους αναγνωρισμένους αθλητικούς φορείς. Το ωράριο ορίζεται από τη δομή σε συμφωνία με εσάς.",
                "keywords": ["που", "πού", "τοποθέτηση", "γυμναστήρια", "σωματεία", "φορείς", "δομή"]
            }
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Προεπεξεργασία κειμένου"""
        # Μετατροπή σε πεζά
        text = text.lower()
        # Αφαίρεση ειδικών χαρακτήρων
        text = re.sub(r'[^\w\s]', ' ', text)
        # Αφαίρεση επιπλέον κενών
        text = ' '.join(text.split())
        return text
    
    def calculate_similarity(self, question: str, qa_item: Dict) -> float:
        """Υπολογισμός ομοιότητας μεταξύ ερώτησης και Q&A item"""
        processed_question = self.preprocess_text(question)
        processed_qa_question = self.preprocess_text(qa_item['question'])
        
        # Ομοιότητα με την ερώτηση
        question_similarity = difflib.SequenceMatcher(None, processed_question, processed_qa_question).ratio()
        
        # Ομοιότητα με τις λέξεις-κλειδιά
        keyword_similarity = 0
        if 'keywords' in qa_item:
            for keyword in qa_item['keywords']:
                if self.preprocess_text(keyword) in processed_question:
                    keyword_similarity += 0.2
        
        # Συνολική ομοιότητα
        return (question_similarity * 0.7) + (min(keyword_similarity, 1.0) * 0.3)
    
    def find_best_answer(self, question: str) -> Tuple[str, float, str]:
        """Εύρεση καλύτερης απάντησης"""
        if not self.qa_data:
            return "Λυπάμαι, δεν υπάρχουν διαθέσιμα δεδομένα.", 0.0, "Σφάλμα"
        
        best_match = max(self.qa_data, key=lambda x: self.calculate_similarity(question, x))
        similarity = self.calculate_similarity(question, best_match)
        
        return best_match['answer'], similarity, best_match.get('category', 'Γενικά')
    
    def get_response(self, question: str) -> Dict:
        """Κύρια μέθοδος για απάντηση"""
        answer, similarity, category = self.find_best_answer(question)
        
        response = {
            'answer': answer,
            'confidence': similarity,
            'category': category,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        # Προσθήκη στο ιστορικό
        self.conversation_history.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now()
        })
        
        return response

def main():
    # Τίτλος εφαρμογής
    st.markdown('<h1 class="main-header">🎓 Chatbot Πρακτικής Άσκησης</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Μητροπολιτικό Κολλέγιο Θεσσαλονίκης - Τμήμα Προπονητικής & Φυσικής Αγωγής</h3>', unsafe_allow_html=True)
    
    # Αρχικοποίηση chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PracticeTrainingChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar με πληροφορίες
    with st.sidebar:
        st.header("📋 Κατηγορίες Ερωτήσεων")
        categories = set(item.get('category', 'Γενικά') for item in st.session_state.chatbot.qa_data)
        for category in categories:
            st.write(f"• {category}")
        
        st.markdown("---")
        st.header("⏰ Σημαντικές Προθεσμίες")
        st.write("🎯 **240 ώρες μέχρι 30/4**")
        st.write("📝 **Σύμβαση στο moodle μέχρι 15/10**")
        st.write("⚠️ **Μόνο Δευτέρα-Σάββατο, μέχρι 8ω/ημέρα**")
        
        st.markdown("---")
        st.header("📞 Επικοινωνία")
        st.write("**Υπεύθυνη Πρακτικής Άσκησης:**")
        st.write("Μαρία Ταμπάκη, MSc, PhD(c)")
        st.write("📧 mtampaki@mitropolitiko.edu.gr")
        st.write("")
        st.write("**Programme Leader:**")
        st.write("Ιωάννης Μητρούσης, MSc, PhD(c)")
        st.write("📧 imitrousis@mitropolitiko.edu.gr")
        st.write("📞 210 4121200")
        
        st.markdown("---")
        if st.button("🗑️ Καθαρισμός Συνομιλίας"):
            st.session_state.messages = []
            st.rerun()
    
    # Κύρια περιοχή chat
    st.markdown("### 💬 Συνομιλία")
    
    # Εμφάνιση μηνυμάτων
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message"><strong>🧑‍🎓 Εσείς:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            confidence = message.get("confidence", 0)
            category = message.get("category", "")
            
            confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
            
            st.markdown(f'''
            <div class="chat-message bot-response">
                <strong>🤖 Chatbot ({category}) {confidence_color}:</strong><br>
                {message["content"]}
                <br><small>Βεβαιότητα: {confidence:.0%} | {message.get("timestamp", "")}</small>
            </div>
            ''', unsafe_allow_html=True)
    
    # Input για νέα ερώτηση
    if prompt := st.chat_input("Γράψτε την ερώτησή σας για την πρακτική άσκηση..."):
        # Προσθήκη ερώτησης στα μηνύματα
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Λήψη απάντησης από chatbot
        response = st.session_state.chatbot.get_response(prompt)
        
        # Προσθήκη απάντησης στα μηνύματα
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response['answer'],
            "confidence": response['confidence'],
            "category": response['category'],
            "timestamp": response['timestamp']
        })
        
        st.rerun()
    
    # Περιοχή πληροφοριών
    with st.expander("ℹ️ Οδηγίες Χρήσης"):
        st.markdown("""
        **Πώς να χρησιμοποιήσετε το chatbot:**
        
        1. **Γράψτε την ερώτησή σας** στο πεδίο κειμένου παρακάτω
        2. **Πατήστε Enter** ή κάντε κλικ στο εικονίδιο αποστολής
        3. **Δείτε την απάντηση** με την αντίστοιχη βεβαιότητα
        
        **Τύποι ερωτήσεων που μπορείτε να κάνετε:**
        - Διαδικασίες και έγγραφα πρακτικής άσκησης
        - Ασφαλιστική ικανότητα και ενημερότητα
        - Στοιχεία που χρειάζεται η δομή/φορέας
        - Σύμβαση και δήλωση στον ΕΡΓΑΝΗ
        - Ώρες, χρονοδιάγραμμα και προθεσμίες
        - Βιβλίο πρακτικής και αξιολόγηση
        - Επικοινωνία με υπευθύνους
        
        **Βασικά στοιχεία που πρέπει να θυμάστε:**
        - 📅 **240 ώρες** υποχρεωτικά μέχρι **30/4**
        - 📋 **Πιστοποιητικό ασφαλιστικής ικανότητας** από gov.gr
        - 📝 **Υπεύθυνη δήλωση** για ΟΑΕΔ
        - 🏢 **Η δομή δηλώνει στον ΕΡΓΑΝΗ** πριν την έναρξη
        - ✍️ **Υπογραφή + σφραγίδα** σε όλα τα έγγραφα
        
        **Σύμβολα βεβαιότητας:**
        - 🟢 Υψηλή βεβαιότητα (>70%)
        - 🟡 Μέτρια βεβαιότητα (40-70%)
        - 🔴 Χαμηλή βεβαιότητα (<40%)
        """)
    
    # Νέα ενότητα για γρήγορες συνδέσεις
    with st.expander("🔗 Χρήσιμες Συνδέσεις"):
        st.markdown("""
        **Επίσημες Ιστοσελίδες:**
        - [🏛️ Ασφαλιστική Ικανότητα (gov.gr)](https://www.gov.gr/ipiresies/ergasia-kai-asphalise/asphalise/asphalistike-ikanoteta)
        - [📋 ATLAS - Ασφαλιστική Ενημερότητα](https://www.atlas.gov.gr/ATLAS/Pages/Home.aspx)
        - [📑 Υπεύθυνη Δήλωση (gov.gr)](https://www.gov.gr)
        
        **Τι χρειάζεστε:**
        - Κωδικούς **taxisnet** (ή του γονέα σας)
        - **ΑΜΚΑ** 
        - **ΑΦΜ**
        - Στοιχεία **ταυτότητας**
        """)
    
    # Ενότητα για επείγουσες ειδοποιήσεις
    st.markdown("### ⚠️ Σημαντικές Ειδοποιήσεις")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
        <strong>🚨 ΠΡΟΣΟΧΗ - Προθεσμίες:</strong><br>
        • 240 ώρες μέχρι 30/4<br>
        • Σύμβαση στο moodle μέχρι 15/10<br>
        • ΕΡΓΑΝΗ πριν την έναρξη
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <strong>💡 Συμβουλή:</strong><br>
        Εκτυπώστε τη σύμβαση 3 φορές και φροντίστε για υπογραφή + σφραγίδα σε όλα τα αντίγραφα!
        </div>
        """, unsafe_allow_html=True)
    
    # Admin panel (για testing)
    if st.sidebar.checkbox("🔧 Admin Panel (για testing)"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Στατιστικά")
        st.sidebar.write(f"Συνολικές ερωτήσεις: {len(st.session_state.chatbot.conversation_history)}")
        st.sidebar.write(f"Διαθέσιμα Q&A: {len(st.session_state.chatbot.qa_data)}")
        
        if st.sidebar.button("📊 Εμφάνιση Ιστορικού"):
            if st.session_state.chatbot.conversation_history:
                df = pd.DataFrame([
                    {
                        'Ερώτηση': item['question'][:50] + '...',
                        'Κατηγορία': item['response']['category'],
                        'Βεβαιότητα': f"{item['response']['confidence']:.0%}",
                        'Ώρα': item['timestamp'].strftime("%H:%M:%S")
                    }
                    for item in st.session_state.chatbot.conversation_history
                ])
                st.dataframe(df)

if __name__ == "__main__":
    main()