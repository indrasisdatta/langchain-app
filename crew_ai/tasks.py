from crewai import Task
from tools import yt_tool 
from agents import blog_researcher, blog_writer

# Research task 
research_task = Task(
    description=(
        "Identify the video {topic}.",
        "Get detailed information about the video from the channel."
    ),
    expected_output="",
    tools=[yt_tool],
    agent=blog_researcher,
)

# Writer task 
writer_task = Task(
    description=(
        "Get the info from the YouTube channel on the topic {topic}"
    ),
    expected_output="Summarize the video content and write a blog post.",
    tools=[yt_tool],
    agent=blog_writer,
    async_execution=False,
    output_file="new-blog-post.md"
)
