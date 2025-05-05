"""This module provides tools for the agent supervisor.

It includes:
- Web Search: For general web results using Tavily.
- Wikipedia: For encyclopedia lookups.
- ArXiv: For fetching research paper abstracts.
- YouTube: For searching YouTube videos.
- YouTube Transcript: For analyzing video content and answering questions.
- Python REPL: For executing Python code (Use with caution!).
- Wolfram Alpha: For mathematical computations and symbolic math.
"""

from typing import Annotated, List, Any, Callable, Optional, cast
import re
import importlib.util
import sys

# Core Tools & Utilities
from langchain_core.tools import tool

# Experimental Tools (Use with caution)
from langchain_experimental.utilities import PythonREPL

# Use TavilySearchResults from langchain_community like in the notebook
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from react_agent.configuration import Configuration

# Check if youtube_transcript_api is available
is_youtube_transcript_api_available = importlib.util.find_spec("youtube_transcript_api") is not None
is_pytube_available = importlib.util.find_spec("pytube") is not None

# Only import YoutubeLoader if dependencies are available
if is_youtube_transcript_api_available and is_pytube_available:
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_community.document_loaders.youtube import TranscriptFormat


# Create Tavily tool using configuration from context (more consistent approach)
def create_tavily_tool():
    """Create the Tavily search tool with configuration from context.

    Returns:
        Configured TavilySearchResults tool
    """
    configuration = Configuration.from_context()
    return TavilySearchResults(max_results=configuration.max_search_results)

def create_wikipedia_tool():
    """Create the Wikipedia search tool with configuration from context.

    Returns:
        Configured WikipediaQueryRun tool
    """
    configuration = Configuration.from_context()
    # Create the API wrapper first, then pass it to the tool
    wiki_api = WikipediaAPIWrapper(top_k_results=configuration.max_wikipedia_results)
    return WikipediaQueryRun(api_wrapper=wiki_api)

def create_arxiv_tool():
    """Create the ArXiv search tool with configuration from context.

    Returns:
        Configured ArxivQueryRun tool
    """
    configuration = Configuration.from_context()
    # Create the API wrapper first, then pass it to the tool
    arxiv_api = ArxivAPIWrapper(top_k_results=configuration.max_arxiv_results)
    return ArxivQueryRun(api_wrapper=arxiv_api)

def create_youtube_tool():
    """Create the YouTube search tool.
    
    This tool scrapes YouTube search results without using the rate-limited API.
    
    Returns:
        A YouTubeSearchTool instance
    """
    # The YouTubeSearchTool doesn't take a max_results parameter directly
    # We'll create a custom tool that uses the configuration
    configuration = Configuration.from_context()
    youtube_tool = YouTubeSearchTool()
    
    @tool
    def youtube_search(query: str) -> str:
        """Search for YouTube videos based on a query.
        
        This tool returns links to YouTube videos that match the search query.
        For specifying a specific number of results, you can append a comma
        followed by the number to the query (e.g., "python tutorial,5").
        
        The default number of results is determined by configuration.
        
        Input: A search query for YouTube videos, optionally with result count.
        Output: A list of YouTube video links.
        """
        # If the query doesn't specify a limit, add the configured limit
        if "," not in query:
            query = f"{query},{configuration.max_youtube_results}"
        return youtube_tool.run(query)
    
    return youtube_search

def create_youtube_transcript_tool():
    """Create the YouTube transcript analysis tool.
    
    This tool extracts transcripts from YouTube videos and can analyze content.
    
    Returns:
        A tool that can extract and analyze YouTube video transcripts
    """
    @tool
    def youtube_transcript(query: str) -> str:
        """Extract and analyze YouTube video transcripts to answer questions about content.
        
        This tool can be used to:
        - Extract the full transcript of a YouTube video
        - Answer questions about specific content in a video
        - Analyze what was said or shown in a video
        
        The query should include the YouTube video URL and optionally a specific question.
        Format: "URL [question]" where the question is optional.
        
        Examples:
        - "https://www.youtube.com/watch?v=abc123 What are the main points?"
        - "https://youtu.be/abc123 How many people were interviewed?"
        - "https://www.youtube.com/watch?v=abc123" (returns full transcript)
        
        Input: YouTube URL with optional question
        Output: Either the full transcript or an answer to the question
        """
        # Check if required packages are installed
        if not is_youtube_transcript_api_available or not is_pytube_available:
            missing_packages = []
            if not is_youtube_transcript_api_available:
                missing_packages.append("youtube-transcript-api")
            if not is_pytube_available:
                missing_packages.append("pytube")
                
            install_cmd = "pip install " + " ".join(missing_packages)
            return (f"Required package(s) missing: {', '.join(missing_packages)}. "
                    f"Please install them with: `{install_cmd}`")
        
        # Extract URL and question from the query
        url_match = re.search(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]+)', query)
        if not url_match:
            return "Invalid YouTube URL. Please provide a valid YouTube video URL."
        
        url = url_match.group(1)
        question = query.replace(url, "").strip()
        
        try:
            # Load the transcript
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            transcript_docs = loader.load()
            
            if not transcript_docs:
                return "Could not extract transcript from this video. It may not have subtitles or captions available."
            
            # Get the full transcript text
            full_transcript = "\n".join([doc.page_content for doc in transcript_docs])
            
            # If there's a specific question, return the full transcript for now
            # A more sophisticated implementation would use an LLM to answer the question
            if question:
                return f"Question about video: '{question}'\n\nHere is the transcript to help answer this question:\n\n{full_transcript}"
            else:
                # Return full transcript if no specific question
                return f"Full transcript of the video:\n\n{full_transcript}"
                
        except Exception as e:
            return f"Error extracting transcript: {str(e)}"
    
    return youtube_transcript

def create_wolfram_alpha_tool():
    """Create the Wolfram Alpha tool for advanced computations.
    
    This tool requires a WOLFRAM_ALPHA_APPID environment variable to be set.
    
    Returns:
        A tool that can make queries to Wolfram Alpha
    """
    configuration = Configuration.from_context()
    # WolframAlphaAPIWrapper doesn't accept timeout parameter
    wolfram = WolframAlphaAPIWrapper()
    
    @tool
    def wolfram_alpha(query: str) -> str:
        """Use Wolfram Alpha to solve mathematical problems or questions involving computation.
        
        This tool is particularly useful for:
        - Mathematical equations and symbolic math
        - Scientific calculations
        - Unit conversions
        - Data analysis
        - Solving equations that Python's standard libraries struggle with
        
        Examples of good queries:
        - "Solve x^2 + 2x + 1 = 0"
        - "Integrate x^2 * sin(x) with respect to x"
        - "Convert 100 miles per hour to meters per second"
        - "What is the half-life of uranium-235?"
        
        Input: A clear, specific question suitable for Wolfram Alpha
        Output: The result from Wolfram Alpha's computation
        """
        return wolfram.run(query)
    
    return wolfram_alpha

# Initialize tools with configuration
tavily_tool = create_tavily_tool()
wikipedia_tool = create_wikipedia_tool()
arxiv_tool = create_arxiv_tool()
youtube_tool = create_youtube_tool()
youtube_transcript_tool = create_youtube_transcript_tool()
wolfram_alpha_tool = create_wolfram_alpha_tool()


# --- Python REPL Tool ---
# WARNING: Executes arbitrary Python code locally. Be extremely careful
#          about exposing this tool, especially in production environments.
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute. Use print(...) to see output."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    # Filter out potentially sensitive REPL implementation details
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str


# --- Tool List ---

# The list of tools available to the agent supervisor.
TOOLS: List[Callable[..., Any]] = [
    tavily_tool, 
    wikipedia_tool, 
    arxiv_tool, 
    youtube_tool, 
    youtube_transcript_tool, 
    wolfram_alpha_tool, 
    python_repl_tool
]
