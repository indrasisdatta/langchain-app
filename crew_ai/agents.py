from crewai import Agent 

# Create a senior blog content researcher agent
blog_researcher = Agent(
    name="BlogResearcher",
    role="Blog Researcher from YouTube videos",
    goal="Get the relevant video content for the topic {topic} from YouTube channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos of Data science, Machine learning, Generative AI and providing suggestions"
    ),
    tools=[],
    allow_delegation=True
)

# Create a senior blog writer agent with YouTube tool 
blog_writer = Agent(
    name="BlogWriter",
    role="Blog Writer",
    goal="Narrate compelling tech stories about the video {topic} from Youtube channel",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, "
        "transform intricate concepts into engaging, easy-to-understand blog posts. "
    ),
    tools=[],
    allow_delegation=False
)