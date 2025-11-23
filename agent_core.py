from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic import LLMChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.tools import ToolException
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents import initialize_agent, AgentType
import logging
import os
import requests
from typing import List, Dict
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_job_listings(query: str, location: str = "", limit: int = 10) -> List[Dict[str, str]]:
    """Fetches job listings from the JSearch API on RapidAPI."""
    if not query:
        raise ValueError("Empty job search query provided.")

    # API credentials & endpoint
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        raise ValueError("RAPIDAPI_KEY not set in environment.")

    api_host = "jsearch.p.rapidapi.com"
    url = "https://jsearch.p.rapidapi.com/search"

    # Define headers for RapidAPI
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": api_host
    }

    # Define query parameters
    params = {
        "query": query,
        "location": location,
        "num_pages": 1,
        "page": 1
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        job_data = data.get("data", [])

        cleaned_jobs = []
        for j in job_data[:limit]:
            cleaned_jobs.append({
                "title": j.get("job_title", "Unknown Title"),
                "company_name": j.get("employer_name", "Unknown Company"),
                "location": f"{j.get('job_city', '')}, {j.get('job_country', '')}".strip(", "),
                "description": j.get("job_description", "No description available."),
                "link": j.get("job_apply_link", "#")
            })

            

        logging.info(f"Fetched {len(cleaned_jobs)} job listings from JSearch API.")
        return cleaned_jobs

    except Exception as e:
        logging.error(f"Error fetching job listings from JSearch: {e}")
        raise RuntimeError(f"Failed to fetch job listings: {e}")


class CareerTools:
    @staticmethod
    def setup_tools(llm, resume_text: str = None):
        """Set up LangChain tools for the CareerAgent."""
        
        tools = [
            Tool(
                name="analyze_resume",
                description="Analyze a resume and extract key information like skills, experience, education, etc.",
                func=lambda query: CareerTools.analyze_resume(llm, resume_text, query),
            ),
            Tool(
                name="suggest_job_titles",
                description="Suggest job titles based on resume content",
                func=lambda _: CareerTools.suggest_job_titles(llm, resume_text),
            ),
            Tool(
                name="extract_keywords_from_resume",
                description="Extract relevant keywords from resume for job searching",
                func=lambda _: CareerTools.extract_keywords(llm, resume_text),
            ),
            Tool(
                name="search_jobs",
                description="Search for job listings by keyword or job title",
                func=lambda query: CareerTools.search_jobs(llm, query),
            ),
            Tool(
                name="generate_cover_letter",
                description="Generate a cover letter for a specific job using the resume",
                func=lambda job_info: CareerTools.generate_cover_letter(llm, resume_text, job_info),
            ),
           
        ]
        return tools
    
    @staticmethod
    def analyze_resume(llm, resume_text: str, query: str) -> str:
        """Analyze the resume based on a specific query."""
        if not resume_text:
            return "No resume uploaded. Please upload a resume first."
            
        prompt = PromptTemplate(
            template="""
            Analyze the following resume based on the query: {query}
            
            RESUME:
            {resume_text}
            
            Provide a detailed analysis addressing the query.
            """,
            input_variables=["resume_text", "query"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(resume_text=resume_text, query=query)
    
    @staticmethod
    def suggest_job_titles(llm, resume_text: str) -> str:
        """Suggest job titles based on resume content."""
        if not resume_text:
            return "No resume uploaded. Please upload a resume first."
            
        prompt = PromptTemplate(
            template="""
            Based on the following resume, suggest the top 5 job titles that would be a good fit, 
            ranked from most suitable to slightly broader:

            RESUME:
            {resume_text}

            Format the response as a numbered list with brief explanations for why each job title is suitable.
            """,
            input_variables=["resume_text"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(resume_text=resume_text)
    
    @staticmethod
    def extract_keywords(llm, resume_text: str) -> List[str]:
        """Extract relevant keywords from resume for job searching."""
        if not resume_text:
            return []
            
        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()
        
        prompt = PromptTemplate(
            template="""
            Extract the most relevant keywords for job searching from this resume. 
            Focus on skills, job titles, and technical expertise.
            Return ONLY the top 5 most relevant keywords.
            
            RESUME:
            {resume_text}
            
            {format_instructions}
            """,
            input_variables=["resume_text"],
            partial_variables={"format_instructions": format_instructions}
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(resume_text=resume_text)
        
        try:
            return output_parser.parse(result)
        except Exception as e:
            logging.error(f"Error parsing keywords: {e}")
            # Fallback parsing if output parser fails
            return [keyword.strip() for keyword in result.split(',')[:5]]
    
    @staticmethod
    def search_jobs(llm, query: str) -> Dict:
        """Search for job listings by keyword or job title."""
        try:
            jobs = fetch_job_listings(query)
            
            if not jobs:
                return {"status": "error", "message": "No jobs found matching your query. Try different keywords or broader terms.", "jobs": []}
            
            # Return the full job details in a dict for easy processing
            result = {
                "status": "success",
                "message": f"Found {len(jobs)} job(s) matching your query.",
                "jobs": jobs
            }
            
            return result
            
        except ToolException as e:
            return {"status": "error", "message": f"Error searching for jobs: {str(e)}", "jobs": []}
    

    @staticmethod
    def generate_cover_letter(llm, resume_text: str, job_info: str) -> str:
        """Generate a cover letter for a specific job using the resume."""
        if not resume_text:
            return "No resume uploaded. Please upload a resume first."
            
        try:
            # Parse job info from input - expecting format: "title: X, company: Y, description: Z"
            job_parts = job_info.split(", ")
            job_dict = {}
            
            for part in job_parts:
                if ": " in part:
                    key, value = part.split(": ", 1)
                    job_dict[key.strip()] = value.strip()
            
            job_title = job_dict.get("title", "")
            job_company = job_dict.get("company", "")
            job_description = job_dict.get("description", "")
            
            if not (job_title and job_description):
                return "Please provide both job title and description to generate a cover letter."
            
            prompt = PromptTemplate(
                template="""
                You are a professional career advisor. Write a tailored cover letter for a job application.
                
                RESUME:
                {resume_text}
                
                JOB DETAILS:
                Title: {job_title}
                Company: {job_company}
                Description: {job_description}
                
                Write a professional cover letter that highlights the candidate's relevant experience and skills
                for this specific job. Format it as a proper business letter.
                """,
                input_variables=["resume_text", "job_title", "job_company", "job_description"]
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run(
                resume_text=resume_text, 
                job_title=job_title, 
                job_company=job_company, 
                job_description=job_description
            )
            
        except Exception as e:
            logging.error(f"Error generating cover letter: {e}")
            return f"Error generating cover letter: {str(e)}"
        

def create_career_agent(llm, resume_text: str = None):
    """Create the LangChain agent for career assistance."""
    
    tools = CareerTools.setup_tools(llm, resume_text)
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )
    
    return agent