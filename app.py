import streamlit as st
import json
import pandas as pd
from datetime import datetime
import difflib
import re
from typing import List, Dict, Tuple
import os
import time

# Groq API imports with better error handling
GROQ_AVAILABLE = False
groq_client = None

try:
    from groq import Groq
    GROQ_AVAILABLE = True
    print("✅ Groq library imported successfully")
except ImportError as e:
    GROQ_AVAILABLE = False
    print(f"❌ Groq import failed: {e}")
except Exception as e:
    GROQ_AVAILABLE = False
    print(f"❌ Groq error: {e}")

# Ρύθμιση σελίδας
st.set_page_config(
    page_title="Πρακτική Άσκηση - Μητροπολιτικό Κολλέγιο",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simplified CSS (removed complex animations that might cause issues)
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
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Κρύψιμο Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
        .sub-header { font-size: 1rem; }
        .logo-container img { max-width: 180px; }
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

    def init_groq_client(self):
        """Safer Groq client initialization"""
        if not GROQ_AVAILABLE:
            return None
            
        try:
            # Try to get API key from different sources
            api_key = None
            
            # Try Streamlit secrets first
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
            
            # Try environment variable
            elif 'GROQ_API_KEY' in os.environ:
                api_key = os.environ['GROQ_API_KEY']
                
            if api_key:
                client = Groq(api_key=api_key)
                # Test the client with a simple call
                return client
            else:
                print("⚠️ No Groq API key found")
                return None
                
        except Exception as e:
            print(f"❌ Groq client initialization failed: {e}")
            return None

    def load_qa_data(self) -> List[Dict]:
        """Load Q&A data with better error handling"""
        try:
            # Try to load from the embedded data
            qa_data_json = '''[
  {
    "id": 1,
    "category": "Γενικές Πληροφορίες",
    "question": "Πώς ξεκινάω την πρακτική μου άσκηση;",
    "answer": "**Βήμα 1:** Επικοινωνήστε με την υπεύθυνη **Γεώργιος Σοφιανίδης** στο gsofianidis@mitropolitiko.edu.gr\\n\\n**Βήμα 2:** Βρείτε δομή (γυμναστήριο, σωματείο, σχολείο) που σας ενδιαφέρει\\n\\n**Βήμα 3:** Ετοιμάστε τα απαραίτητα έγγραφα (αίτηση, ασφαλιστική ικανότητα, υπεύθυνη δήλωση)\\n\\n**Σημαντικό:** Χρειάζεστε να ολοκληρώσετε **240 ώρες μέχρι 30/4**. Το κολλέγιο καλύπτει όλα τα έξοδα της σύμβασης.",
    "keywords": ["ξεκινάω", "ξεκινω", "αρχή", "αρχίζω", "πρακτική", "άσκηση", "πώς", "πως", "βήματα"]
  },
  {
    "id": 2,
    "category": "Έγγραφα & Διαδικασίες",
    "question": "Τι έγγραφα χρειάζομαι για την πρακτική άσκηση;",
    "answer": "**Για εσάς (φοιτητή):**\\n• Αίτηση πραγματοποίησης πρακτικής άσκησης ✅\\n• Στοιχεία φοιτητή (συμπληρωμένη φόρμα) ✅\\n• **Ασφαλιστική ικανότητα** από gov.gr ⭐\\n• **Υπεύθυνη δήλωση** (δεν παίρνετε επίδομα ΟΑΕΔ) ⭐\\n\\n**Για τη δομή:**\\n• Στοιχεία φορέα (ΑΦΜ, διεύθυνση, νόμιμος εκπρόσωπος, IBAN)\\n• Ημέρες και ώρες που σας δέχεται\\n\\n**💡 Tip:** Ξεκινήστε από την ασφαλιστική ικανότητα γιατί παίρνει χρόνο!",
    "keywords": ["έγγραφα", "εγγραφα", "χρειάζομαι", "χρειαζομαι", "απαιτήσεις", "απαιτησεις", "δικαιολογητικά", "δικαιολογητικα", "αίτηση", "αιτηση"]
  }
]'''
            return json.loads(qa_data_json)
        except Exception as e:
            st.error(f"Σφάλμα κατά τη φόρτωση δεδομένων: {e}")
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

    def find_relevant_context(self, question: str, top_k: int = 3) -> str:
        """Find relevant context for the question"""
        if not self.qa_data:
            return ""

        # Calculate similarity scores
        scored_items = []
        for item in self.qa_data:
            score = self.calculate_similarity(question, item)
            scored_items.append((item, score))

        # Sort and select top_k
        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_items = scored_items[:top_k]

        # Build context
        context = "ΣΧΕΤΙΚΕΣ ΠΛΗΡΟΦΟΡΙΕΣ:\n\n"
        for item, score in top_items:
            if score > 0.1:  # Only include relevant information
                context += f"Ερώτηση: {item['question']}\n"
                context += f"Απάντηση: {item['answer']}\n"
                context += f"Κατηγορία: {item.get('category', 'Γενικά')}\n\n"

        return context

    def calculate_similarity(self, question: str, qa_item: Dict) -> float:
        """Calculate similarity score"""
        question_lower = question.lower()
        qa_question_lower = qa_item['question'].lower()
        qa_answer_lower = qa_item['answer'].lower()

        # Basic similarity
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
        """Get response from Groq with better error handling"""
        if not self.groq_client:
            return "", False

        try:
            # Find relevant context
            context = self.find_relevant_context(question)

            # Create user message
            user_message = f"{context}\n\nΕΡΩΤΗΣΗ ΦΟΙΤΗΤΗ: {question}"

            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Είσαι ένας εξειδικευμένος σύμβουλος για θέματα πρακτικής άσκησης στο Μητροπολιτικό Κολλέγιο Θεσσαλονίκης. Απάντησε στα ελληνικά με επαγγελματικό τόνο."},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=800,
                top_p=1,
                stream=False
            )

            response = chat_completion.choices[0].message.content
            return response, True

        except Exception as e:
            print(f"❌ Groq API error: {str(e)}")
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

    def get_response(self, question: str) -> Dict:
        """Main response method"""
        start_time = time.time()

        # Try Groq first
        if self.groq_client:
            answer, success = self.get_groq_response(question)
            if success and answer:
                response = {
                    'answer': answer,
                    'confidence': 0.95,
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
            # Only fallback
            answer = self.get_fallback_response(question)
            response = {
                'answer': answer,
                'confidence': 0.6,
                'source': 'Knowledge Base',
                'response_time': round(time.time() - start_time, 2),
                'timestamp': datetime.now().strftime("%H:%M")
            }

        # Save to history
        self.conversation_history.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now()
        })

        return response

def main():
    # Initialize
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AdvancedPracticeChatbot()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Header with logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)

    logo_col, title_col = st.columns([1, 4])

    with logo_col:
        # Safer image loading with fallback - ΜΕΓΑΛΥΤΕΡΟ ΛΟΓΟΤΥΠΟ
        try:
            st.image("https://raw.githubusercontent.com/GiorgosBouh/chatbot.placement/main/MK_LOGO_SEO_1200x630.png", width=220)
        except:
            st.markdown("🎓", unsafe_allow_html=True)

    with title_col:
        st.markdown('<h1 class="main-header">Πρακτική Άσκηση</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Μητροπολιτικό Κολλέγιο Θεσσαλονίκης • Προπονητική & Φυσική Αγωγή</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # API Status
    if st.session_state.chatbot.groq_client:
        st.markdown('<div class="api-status">🚀 AI Assistant Ενεργό</div>', unsafe_allow_html=True)
        
    # Επαγγελματική ενδειξη για sidebar
    st.markdown("""
    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 0.6rem; margin-bottom: 1.5rem; text-align: center; font-size: 0.9rem;">
        <strong>Πληροφορίες:</strong> Χρησιμοποιήστε το αριστερό μενού για συχνές ερωτήσεις και επικοινωνία 👈
    </div>
    """, unsafe_allow_html=True)

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
        
        **Τεχνική Υποστήριξη**  
        **Γεώργιος Μπουχουράς**  
        📧 gbouchouras@mitropolitiko.edu.gr
        """)

        st.markdown("---")

        # Συχνές ερωτήσεις
        st.markdown("## ❓ Συχνές Ερωτήσεις")

        for question in st.session_state.chatbot.frequent_questions:
            if st.button(question, key=f"faq_{question}", use_container_width=True):
                # Add question to conversation
                st.session_state.messages.append({"role": "user", "content": question})

                # Get response
                response = st.session_state.chatbot.get_response(question)

                # Add response
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