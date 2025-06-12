import streamlit as st
import json
import pandas as pd
from datetime import datetime
import difflib
import re
from typing import List, Dict, Tuple
import os
import time

# Groq API imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Ρύθμιση σελίδας
st.set_page_config(
    page_title="Πρακτική Άσκηση - Μητροπολιτικό Κολλέγιο",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS με typing animation
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
    
    .confidence-high { border-left-color: #28a745 !important; }
    .confidence-medium { border-left-color: #ffc107 !important; }
    .confidence-low { border-left-color: #dc3545 !important; }
    
    .typing-indicator {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
    }
    
    .typing-dots {
        display: inline-block;
        margin-left: 10px;
    }
    
    .typing-dots span {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #999;
        margin: 0 2px;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    .info-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .api-status {
        background: linear-gradient(45deg, #4caf50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 1rem;
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
    
    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
        .sub-header { font-size: 1rem; }
        .logo-container img { max-width: 120px; }
    }
</style>
""", unsafe_allow_html=True)

class AdvancedPracticeChatbot:
    def __init__(self):
        self.qa_data = self.load_qa_data()
        self.conversation_history = []
        self.groq_client = self.init_groq_client()
        
        # Συχνές ερωτήσεις
        self.frequent_questions = [
            "Πώς ξεκινάω την πρακτική άσκηση;",
            "Τι έγγραφα χρειάζομαι;",
            "Πόσες ώρες πρέπει να κάνω;",
            "Πώς βγάζω ασφαλιστική ικανότητα;",
            "Με ποιον επικοινωνώ;"
        ]
        
        # System prompt για το LM
        self.system_prompt = """Είσαι ένας εξειδικευμένος σύμβουλος για θέματα πρακτικής άσκησης στο Μητροπολιτικό Κολλέγιο Θεσσαλονίκης, τμήμα Προπονητικής και Φυσικής Αγωγής.

ΚΡΙΤΙΚΕΣ ΟΔΗΓΙΕΣ:
- Χρησιμοποίησε ΜΟΝΟ ελληνικά. ΑΠΑΓΟΡΕΥΕΤΑΙ η χρήση αγγλικών ή greeklish λέξεων
- Μην προσθέτεις πληροφορίες που δεν υπάρχουν στο context (τίτλους, βαθμούς, κλπ)
- Χρησιμοποίησε ΜΟΝΟ τις πληροφορίες που σου δίνονται
- Μην εφευρίσκεις ή μην υποθέτεις στοιχεία

ΣΤΥΛ ΑΠΑΝΤΗΣΗΣ:
- Επίσημος και επαγγελματικός τόνος
- Άμεσες και συγκεκριμένες οδηγίες
- Χωρίς χαιρετισμούς ή φιλικές εκφράσεις
- Δομημένες απαντήσεις με σαφή βήματα
- Περιορισμένη χρήση emojis (μόνο για σημαντικές πληροφορίες)

ΒΑΣΙΚΕΣ ΠΛΗΡΟΦΟΡΙΕΣ (μόνο αυτές):
- Υπεύθυνος: Γεώργιος Σοφιανίδης
- Email: gsofianidis@mitropolitiko.edu.gr
- Απαιτούμενες ώρες: 240 ώρες μέχρι 30/4
- Ωράριο: Δευτέρα-Σάββατο, μέχρι 8 ώρες/ημέρα
- Σύμβαση: Ανέβασμα στο moodle μέχρι 15/10

ΟΔΗΓΙΕΣ:
- Μπες κατευθείαν στο θέμα χωρίς περιττά λόγια
- Δώσε πρακτικές και εφαρμόσιμες οδηγίες
- Χρησιμοποίησε πάντα τις πληροφορίες από το context
- Αν δεν έχεις αρκετές πληροφορίες, κατεύθυνε στον Γεώργιο Σοφιανίδη
- Μην κάνεις ερωτήσεις εκτός αν είναι απαραίτητες για διευκρίνιση
- Μην προσθέτεις τίτλους, βαθμούς ή άλλα στοιχεία που δεν αναφέρονται

Απάντησε στα ελληνικά με επαγγελματικό τόνο χρησιμοποιώντας μόνο τις δοσμένες πληροφορίες."""

    def init_groq_client(self):
        """Αρχικοποίηση Groq client"""
        try:
            if GROQ_AVAILABLE:
                # Δοκιμή για API key από streamlit secrets
                api_key = st.secrets.get("GROQ_API_KEY")
                if api_key:
                    return Groq(api_key=api_key)
        except Exception as e:
            st.sidebar.warning(f"Groq API μη διαθέσιμο: {str(e)}")
        return None

    def load_qa_data(self) -> List[Dict]:
        """Φόρτωση δεδομένων Q&A"""
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
        """Προεπιλεγμένα δεδομένα"""
        return [
            {
                "id": 1,
                "category": "Γενικές Πληροφορίες",
                "question": "Πώς ξεκινάω την πρακτική μου άσκηση;",
                "answer": "Για να ξεκινήσετε την πρακτική άσκηση, επικοινωνήστε με τον υπεύθυνο Γεώργιο Σοφιανίδη στο gsofianidis@mitropolitiko.edu.gr. Απαιτούνται 240 ώρες πρακτικής άσκησης σε δομή της επιλογής σας μέχρι 30/4.",
                "keywords": ["ξεκινάω", "πρακτική", "άσκηση", "αρχή", "πώς"]
            }
        ]

    def find_relevant_context(self, question: str, top_k: int = 3) -> str:
        """RAG: Βρες σχετικό περιεχόμενο για το LM"""
        if not self.qa_data:
            return ""

        # Υπολογισμός ομοιότητας
        scored_items = []
        for item in self.qa_data:
            score = self.calculate_similarity(question, item)
            scored_items.append((item, score))

        # Ταξινόμηση και επιλογή top_k
        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_items = scored_items[:top_k]

        # Δημιουργία context
        context = "ΣΧΕΤΙΚΕΣ ΠΛΗΡΟΦΟΡΙΕΣ:\n\n"
        for item, score in top_items:
            if score > 0.1:  # Κράτα μόνο σχετικές πληροφορίες
                context += f"Ερώτηση: {item['question']}\n"
                context += f"Απάντηση: {item['answer']}\n"
                context += f"Κατηγορία: {item.get('category', 'Γενικά')}\n\n"

        return context

    def calculate_similarity(self, question: str, qa_item: Dict) -> float:
        """Υπολογισμός ομοιότητας (απλοποιημένη έκδοση)"""
        question_lower = question.lower()
        qa_question_lower = qa_item['question'].lower()
        qa_answer_lower = qa_item['answer'].lower()

        # Βασική ομοιότητα
        similarity = difflib.SequenceMatcher(None, question_lower, qa_question_lower).ratio()

        # Keyword matching
        if 'keywords' in qa_item:
            for keyword in qa_item['keywords']:
                if keyword.lower() in question_lower:
                    similarity += 0.3

        # Answer content matching
        question_words = question_lower.split()
        for word in question_words:
            if len(word) > 3 and word in qa_answer_lower:
                similarity += 0.1

        return min(similarity, 1.0)

    def get_groq_response(self, question: str) -> Tuple[str, bool]:
        """Λήψη απάντησης από Groq LM"""
        if not self.groq_client:
            return "", False

        try:
            # Βρες σχετικό context
            context = self.find_relevant_context(question)

            # Δημιουργία του user message
            user_message = f"{context}\n\nΕΡΩΤΗΣΗ ΦΟΙΤΗΤΗ: {question}"

            # Κλήση στο Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.1-8b-instant",  # Γρήγορο και δωρεάν
                temperature=0.3,  # Χαμηλότερο για πιο επίσημες απαντήσεις
                max_tokens=800,   # Συντομότερες απαντήσεις
                top_p=1,
                stream=False
            )

            response = chat_completion.choices[0].message.content
            return response, True

        except Exception as e:
            st.error(f"Σφάλμα Groq API: {str(e)}")
            return "", False

    def get_fallback_response(self, question: str) -> str:
        """Fallback στο παλιό σύστημα"""
        if not self.qa_data:
            return "Λυπάμαι, δεν υπάρχουν διαθέσιμα δεδομένα. Επικοινωνήστε με τον Γεώργιο Σοφιανίδη: gsofianidis@mitropolitiko.edu.gr"

        # Βρες την καλύτερη απάντηση
        best_match = max(self.qa_data, key=lambda x: self.calculate_similarity(question, x))
        similarity = self.calculate_similarity(question, best_match)

        if similarity > 0.2:
            return best_match['answer']
        else:
            return f"""Δεν βρήκα συγκεκριμένη απάντηση για αυτή την ερώτηση.

**Προτείνω:**
• Δοκιμάστε να αναδιατυπώσετε την ερώτηση
• Επιλέξτε από τις συχνές ερωτήσεις στο sidebar
• Επικοινωνήστε με τον **Γεώργιο Σοφιανίδη**: gsofianidis@mitropolitiko.edu.gr"""

    def get_response(self, question: str) -> Dict:
        """Κύρια μέθοδος απάντησης"""
        start_time = time.time()

        # Δοκιμή με Groq πρώτα
        if self.groq_client:
            answer, success = self.get_groq_response(question)
            if success and answer:
                response = {
                    'answer': answer,
                    'confidence': 0.95,  # Υψηλή εμπιστοσύνη για LM
                    'source': 'AI Assistant',
                    'response_time': round(time.time() - start_time, 2),
                    'timestamp': datetime.now().strftime("%H:%M")
                }
            else:
                # Fallback
                answer = self.get_fallback_response(question)
                response = {
                    'answer': answer,
                    'confidence': 0.6,
                    'source': 'Knowledge Base',
                    'response_time': round(time.time() - start_time, 2),
                    'timestamp': datetime.now().strftime("%H:%M")
                }
        else:
            # Μόνο fallback
            answer = self.get_fallback_response(question)
            response = {
                'answer': answer,
                'confidence': 0.6,
                'source': 'Knowledge Base',
                'response_time': round(time.time() - start_time, 2),
                'timestamp': datetime.now().strftime("%H:%M")
            }

        # Αποθήκευση στο ιστορικό
        self.conversation_history.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now()
        })

        return response

def show_typing_indicator():
    """Εμφάνιση typing indicator"""
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="typing-indicator">
        <strong>🤖 Σκέφτομαι</strong>
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return typing_placeholder

def main():
    # Αρχικοποίηση
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AdvancedPracticeChatbot()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Header με logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)

    logo_col, title_col = st.columns([1, 4])

    with logo_col:
        try:
            st.image("https://raw.githubusercontent.com/GiorgosBouh/chatbot.placement/main/MK_LOGO_SEO_1200x630.png", width=140)
        except:
            st.markdown("🎓", unsafe_allow_html=True)

    with title_col:
        st.markdown('<h1 class="main-header">Πρακτική Άσκηση</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Μητροπολιτικό Κολλέγιο Θεσσαλονίκης • Προπονητική & Φυσική Αγωγή</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # API Status
    if st.session_state.chatbot.groq_client:
        st.markdown('<div class="api-status">🚀 AI Assistant Ενεργό</div>', unsafe_allow_html=True)

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
                    <strong>🎓 Φοιτητής:</strong> {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
            else:
                source = message.get("source", "Knowledge Base")
                confidence = message.get("confidence", 0)
                timestamp = message.get("timestamp", "")
                response_time = message.get("response_time", 0)

                if source == "AI Assistant":
                    # AI Response styling
                    st.markdown(f'''
                    <div class="ai-message">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                            <strong>🤖 AI Assistant</strong>
                            <span style="font-size: 0.85rem; opacity: 0.9;">
                                ⚡ {response_time}s • {timestamp}
                            </span>
                        </div>
                        <div style="line-height: 1.6;">
                            {message["content"]}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    # Fallback response styling
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
                            <strong style="color: #1f4e79;">📚 Knowledge Base</strong>
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

            # Show typing indicator για LM responses
            if st.session_state.chatbot.groq_client:
                typing_placeholder = show_typing_indicator()
                time.sleep(1)  # Simulated thinking time
                typing_placeholder.empty()

            # Get bot response
            response = st.session_state.chatbot.get_response(user_input.strip())

            # Add bot message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['answer'],
                "confidence": response['confidence'],
                "source": response['source'],
                "response_time": response['response_time'],
                "timestamp": response['timestamp']
            })

            st.rerun()

    # Sidebar
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
                    "source": response['source'],
                    "response_time": response['response_time'],
                    "timestamp": response['timestamp']
                })

                st.rerun()

        st.markdown("---")

        st.markdown("## 🔗 Χρήσιμοι Σύνδεσμοι")
        st.link_button("🏛️ Ασφαλιστική Ικανότητα", "https://www.gov.gr/ipiresies/ergasia-kai-asphalise/asphalise/asphalistike-ikanoteta")
        st.link_button("📋 ATLAS", "https://www.atlas.gov.gr/ATLAS/Pages/Home.aspx")
        st.link_button("📑 Υπεύθυνη Δήλωση", "https://www.gov.gr")

        st.markdown("---")

        # AI Status
        if st.session_state.chatbot.groq_client:
            st.success("🤖 AI Assistant Ενεργό")
            st.info("Χρησιμοποιεί Llama 3.1 8B")
        else:
            st.warning("📚 Knowledge Base Mode")
            st.info("Για AI responses, χρειάζεται Groq API key")

        st.markdown("---")

        if st.button("🗑️ Νέα Συνομιλία", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Statistics
        if st.checkbox("📊 Στατιστικά"):
            total_conversations = len(st.session_state.chatbot.conversation_history)
            ai_responses = sum(1 for conv in st.session_state.chatbot.conversation_history 
                             if conv['response'].get('source') == 'AI Assistant')
            
            st.metric("Συνολικές ερωτήσεις", total_conversations)
            st.metric("AI Απαντήσεις", ai_responses)
            if total_conversations > 0:
                ai_percentage = round((ai_responses / total_conversations) * 100, 1)
                st.metric("AI Success Rate", f"{ai_percentage}%")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; color: #6c757d; font-size: 0.9rem; padding: 2rem 0; border-top: 1px solid #e9ecef;">
            Μητροπολιτικό Κολλέγιο Θεσσαλονίκης • Τμήμα Προπονητικής & Φυσικής Αγωγής<br>
            <small>Powered by Groq AI • Για τεχνική υποστήριξη επικοινωνήστε με τον Γεώργιο Σοφιανίδη</small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()