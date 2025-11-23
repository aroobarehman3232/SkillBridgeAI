
import streamlit as st
import logging
from file_utils import extract_text_from_pdf, save_as_pdf, save_as_docx
from dotenv import load_dotenv
from agent_core import (
    create_career_agent,
)
from agent_core import CareerTools
load_dotenv()

# LangChain imports
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize LLM
try:
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.5,
    )
    logging.info("OpenAI LLM initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing OpenAI LLM: {e}")
    st.error("Failed to initialize LLM. Check your API key.")


class ConversationManager:
    def __init__(self):
        self.history = InMemoryChatMessageHistory()
        self.system_message = SystemMessage(content="""
        You are CareerAgent, an AI-powered job assistant designed to help people with their job search.
        
        You can:
        1. Analyze resumes to identify key skills and experience
        2. Suggest relevant job titles based on resume content
        3. Extract keywords from resumes for job searching
        4. Search for job listings using keywords or job titles
        5. Generate tailored cover letters for specific jobs
        
        Remember to be helpful, informative, and guide the user through their job search process.
        """)
    
    def add_user_message(self, message: str):
        self.history.add_user_message(message)
    
    def add_ai_message(self, message: str):
        self.history.add_ai_message(message)
    
    def get_chat_history(self):
        return self.history.messages
    
    def format_history_as_string(self):
        messages = self.get_chat_history()
        formatted_history = ""
        
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_history += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"AI: {message.content}\n"
        
        return formatted_history



def main():
    st.set_page_config(
        page_title="SkillBridge AI",
        page_icon="👨🏻‍💻",
        layout="wide"
    )

    st.title("👨🏻‍💻 SkillBridge AI — Your Personal Job Assistant")

    # -----------------------------
    # Session Initialization
    # -----------------------------
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()

    # -----------------------------
    # MAIN TABS
    # -----------------------------
    tab_resume, tab_jobs, tab_tools = st.tabs(
        ["📄 Resume Upload", "💼 Job Finder", "⚙️ Tools"]
    )

    # ============================================================
    # 📄 TAB 1 — Resume Upload
    # ============================================================
    with tab_resume:
        st.header("📄 Upload Your Resume")

        resume_file = st.file_uploader(
            "Upload your resume (PDF only)", type=["pdf"]
        )

        if resume_file:
            with st.spinner("Extracting text from resume..."):
                try:
                    st.session_state.resume_text = extract_text_from_pdf(resume_file)
                    st.session_state.agent = create_career_agent(
                        llm, st.session_state.resume_text
                    )
                    st.success("Resume processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

        if st.session_state.resume_text:
            st.subheader("📌 Resume Extract (Preview)")
            st.text_area("", st.session_state.resume_text[:1500], height=250)

            st.success("Your resume is ready. Go to **Job Finder** tab to continue.")

    # ============================================================
    # 💼 TAB 2 — Job Finder
    # ============================================================
    with tab_jobs:
        st.header("💼 Job Search & Cover Letters")

        if st.session_state.resume_text is None:
            st.warning("Please upload your resume first in the **Resume Upload** tab.")
            st.stop()

        st.markdown("Search for jobs and generate tailored cover letters.")

        # -----------------------------
        # JOB SEARCH FORM
        # -----------------------------
        with st.form("job_search_form"):
            col1, col2 = st.columns([2, 1])

            with col1:
                job_query = st.text_input(
                    "Job Title or Keywords",
                    placeholder="e.g. Software Engineer, Data Analyst"
                )

            with col2:
                use_resume_keywords = st.checkbox(
                    "Use Resume Keywords",
                    value=True
                )

            submit_search = st.form_submit_button("🔍 Search Jobs")

        # -----------------------------
        # JOB SEARCH PROCESS
        # -----------------------------
        if submit_search:
            with st.spinner("Searching jobs..."):
                search_query = job_query

                if use_resume_keywords:
                    keywords = CareerTools.extract_keywords(llm, st.session_state.resume_text)
                    if job_query:
                        search_query = job_query + " " + " ".join(keywords[:2])
                    else:
                        search_query = " ".join(keywords)

                st.session_state.current_search_query = search_query
                result = CareerTools.search_jobs(llm, search_query)

                if result["status"] == "success":
                    st.session_state.job_search_results = result["jobs"]
                    st.session_state.job_search_message = result["message"]
                else:
                    st.session_state.job_search_results = []
                    st.session_state.job_search_message = result["message"]

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        if hasattr(st.session_state, "job_search_results"):
            if st.session_state.job_search_results:
                st.success(st.session_state.job_search_message)
                st.write(f"**Query:** {st.session_state.current_search_query}")

                # Tabs for each job
                job_tabs = st.tabs(
                    [f"{j['title']} — {j['company_name']}" for j in st.session_state.job_search_results[:10]]
                )

                for i, tab in enumerate(job_tabs):
                    job = st.session_state.job_search_results[i]

                    with tab:
                        st.subheader(job["title"])
                        st.caption(f"🏢 {job['company_name']} | 📍 {job['location']}")
                        


                        st.markdown("### 📝 Job Description")
                        st.write(job["description"])

                        st.divider()
                        st.subheader("✍️ Generate Cover Letter")

                        if st.button("Generate Cover Letter", key=f"gen_{i}"):
                            with st.spinner("Generating tailored cover letter..."):
                                job_info = (
                                    f"title: {job['title']}, "
                                    f"company: {job['company_name']}, "
                                    f"description: {job['description']}"
                                )
                                cl = CareerTools.generate_cover_letter(
                                    llm, st.session_state.resume_text, job_info
                                )
                                st.session_state[f"cover_letter_{i}"] = cl

                        if f"cover_letter_{i}" in st.session_state:
                            st.text_area(
                                "Generated Cover Letter",
                                st.session_state[f"cover_letter_{i}"],
                                height=300
                            )

                            st.markdown("### 💾 Save Cover Letter")
                            colA, colB = st.columns(2)

                            with colA:
                                if st.button("Save as PDF", key=f"pdf_{i}"):
                                    path = save_as_pdf(
                                        st.session_state[f"cover_letter_{i}"],
                                        job["title"],
                                        job["company_name"]
                                    )
                                    st.success(f"PDF saved: {path}")

                            with colB:
                                if st.button("Save as DOCX", key=f"docx_{i}"):
                                    path = save_as_docx(
                                        st.session_state[f"cover_letter_{i}"],
                                        job["title"],
                                        job["company_name"]
                                    )
                                    st.success(f"DOCX saved: {path}")
            else:
                st.warning(st.session_state.job_search_message)

    # ============================================================
    # ⚙️ TAB 3 — Quick Tools
    # ============================================================
    with tab_tools:
        st.header("⚙️ Resume Tools")

        if st.session_state.resume_text is None:
            st.warning("Upload resume in the **Resume Upload** tab.")
            st.stop()

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Suggest Suitable Job Titles"):
                with st.spinner("Generating suggestions..."):
                    out = CareerTools.suggest_job_titles(llm, st.session_state.resume_text)
                    st.subheader("Top Suggested Job Titles")
                    st.write(out)

        with col2:
            if st.button("🔍 Extract Keywords from Resume"):
                with st.spinner("Extracting keywords..."):
                    words = CareerTools.extract_keywords(llm, st.session_state.resume_text)
                    st.subheader("Resume Keywords")
                    st.write(", ".join(words))
        
        with col3:
            analysis_query = st.text_input(
                "Enter analysis query",
                placeholder="e.g. Identify strengths & weaknesses"
            )

            if st.button("📄 Analyze Resume"):
                if not analysis_query.strip():
                    st.warning("Please enter a query for analysis.")
                else:
                    with st.spinner("Analyzing resume..."):
                        out = CareerTools.analyze_resume(
                            llm,
                            st.session_state.resume_text,
                            analysis_query
                        )
                        st.subheader("Resume Analysis")
                        st.write(out)


if __name__ == "__main__":
    main()