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
    /* Βασικό styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #e8f4f8;
        padding-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat styling */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
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
    
    .confidence-high {
        border-left-color: #28a745 !important;
    }
    
    .confidence-medium {
        border-left-color: #ffc107 !important;
    }
    
    .confidence-low {
        border-left-color: #dc3545 !important;
    }
    
    /* Info cards */
    .info-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .quick-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .sub-header {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class PracticeTrainingChatbot:
    def __init__(self):
        self.qa_data = self.load_qa_data()
        self.conversation_history = []
        
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
                "answer": "Για να ξεκινήσετε την πρακτική σας άσκηση, επικοινωνήστε με την υπεύθυνη Μαρία Ταμπάκη (mtampaki@mitropolitiko.edu.gr). Πρέπει να συμπληρώσετε 240 ώρες πρακτικής άσκησης σε δομή της επιλογής σας.",
                "keywords": ["ξεκινάω", "πρακτική", "άσκηση", "αρχή", "πώς"]
            }
        ]
    
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def calculate_similarity(self, question: str, qa_item: Dict) -> float:
        processed_question = self.preprocess_text(question)
        processed_qa_question = self.preprocess_text(qa_item['question'])
        
        question_similarity = difflib.SequenceMatcher(None, processed_question, processed_qa_question).ratio()
        
        keyword_similarity = 0
        if 'keywords' in qa_item:
            for keyword in qa_item['keywords']:
                if self.preprocess_text(keyword) in processed_question:
                    keyword_similarity += 0.2
        
        return (question_similarity * 0.7) + (min(keyword_similarity, 1.0) * 0.3)
    
    def find_best_answer(self, question: str) -> Tuple[str, float, str]:
        if not self.qa_data:
            return "Λυπάμαι, δεν υπάρχουν διαθέσιμα δεδομένα.", 0.0, "Σφάλμα"
        
        best_match = max(self.qa_data, key=lambda x: self.calculate_similarity(question, x))
        similarity = self.calculate_similarity(question, best_match)
        
        return best_match['answer'], similarity, best_match.get('category', 'Γενικά')
    
    def get_response(self, question: str) -> Dict:
        answer, similarity, category = self.find_best_answer(question)
        
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
    
    # Header
    st.markdown('<h1 class="main-header">Πρακτική Άσκηση</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Μητροπολιτικό Κολλέγιο Θεσσαλονίκης • Προπονητική & Φυσική Αγωγή</p>', unsafe_allow_html=True)
    
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
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chat Interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display messages
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
                
                # Καθορισμός χρώματος βάση confidence
                if confidence > 0.7:
                    conf_class = "confidence-high"
                    conf_icon = "🟢"
                    conf_text = "Υψηλή"
                elif confidence > 0.4:
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
                            {conf_icon} {conf_text} • {category} • {timestamp}
                        </span>
                    </div>
                    <div style="line-height: 1.6;">
                        {message["content"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📞 Επικοινωνία")
        
        st.markdown("""
        **Υπεύθυνος Πρακτικής Άσκησης**  
        Γεώργιος Σοφιανίδης  
        📧 gsofianidis@mitropolitiko.edu.gr
        """)
        
        st.markdown("---")
        
        st.markdown("## 🔗 Χρήσιμοι Σύνδεσμοι")
        st.link_button("🏛️ Ασφαλιστική Ικανότητα", "https://www.gov.gr/ipiresies/ergasia-kai-asphalise/asphalise/asphalistike-ikanoteta")
        st.link_button("📋 ATLAS", "https://www.atlas.gov.gr/ATLAS/Pages/Home.aspx")
        st.link_button("📑 Υπεύθυνη Δήλωση", "https://www.gov.gr")
        
        st.markdown("---")
        
        # Categories
        st.markdown("## 📋 Κατηγορίες")
        categories = set(item.get('category', 'Γενικά') for item in st.session_state.chatbot.qa_data)
        for category in sorted(categories):
            st.markdown(f"• {category}")
        
        st.markdown("---")
        
        if st.button("🗑️ Νέα Συνομιλία", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Admin section
        if st.checkbox("🔧 Στατιστικά"):
            st.metric("Ερωτήσεις", len(st.session_state.chatbot.conversation_history))
            st.metric("Διαθέσιμα Q&A", len(st.session_state.chatbot.qa_data))
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #6c757d; font-size: 0.9rem; padding: 2rem 0; border-top: 1px solid #e9ecef;">
            Μητροπολιτικό Κολλέγιο Θεσσαλονίκης • Τμήμα Προπονητικής & Φυσικής Αγωγής<br>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()