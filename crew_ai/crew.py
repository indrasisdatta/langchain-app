from crewai import Crew, Process, LLM 
from agents import blog_researcher, blog_writer
from tasks import research_task, writer_task

llm = LLM(
    model="groq/llama-3.2-90b-text-preview",
    temperature=0.7
)

crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, writer_task],
    process=Process.sequential,
    manager_llm=llm,
    function_calling_llm=llm,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

result = crew.kickoff(inputs= {
    "topic": "AI vs ML vs DL vs Data science"
})
print(result)