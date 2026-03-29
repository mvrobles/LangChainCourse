from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from dotenv import load_dotenv

import os 

load_dotenv()

def main():
    print("Hello from langchaincourse!")
    print(os.environ.get('OPENAI_API_KEY'))
    information = """
Dario Amodei (born 1983) is an American artificial intelligence (AI) researcher and entrepreneur. In 2021, he and his sister Daniela Amodei co-founded Anthropic, the company behind the large language model series Claude.[2] Prior to that, he was the vice president of research at OpenAI.[3][4]

In his capacity as Anthropic's CEO, Amodei often writes on the benefits and risks of advanced AI systems.[5] He is a proponent of an "entente" strategy in which a coalition of democratic nations use advanced AI systems in military applications to achieve a decisive advantage over adversaries while sharing the benefits with cooperating nations.[6][7][8]

Early life and education
Dario Amodei was born in San Francisco, California, in 1983.[1] His sister, Daniela, was born four years later.[1] His father, Riccardo Amodei, an Italian-American leather craftsman from Massa Marittima, Tuscany, died when Amodei was a young adult.[9][10] His mother, Elena Engel, a Jewish American born in Chicago, worked as a project manager for libraries.[1]

Dario grew up in San Francisco and graduated from Lowell High School.[11] He was a member of the USA Physics Olympiad team in 2000.[12] Amodei began college at Caltech, where he worked with Tom Tombrello as one of his Physics 11 students. He later transferred to Stanford University, where he received a bachelor of science degree in physics.[13] He also holds a PhD in biophysics from Princeton University, where he studied electrophysiology of neural circuits.[14] He was a postdoctoral scholar at the Stanford University School of Medicine.[15]

Career
From November 2014 until October 2015, he worked at Baidu.[16] After that, he worked at Google.[17] In 2016, Amodei joined OpenAI.[18]

In 2021, Dario and his sister, Daniela, founded Anthropic along with other former senior members of OpenAI.[19][20] The Amodei siblings were among those who left OpenAI due to directional differences.[21]

In November 2023, the board of directors of OpenAI approached Amodei about replacing Sam Altman and potentially merging the two startups. Amodei declined both offers.[22]

In 2025, Time magazine listed Amodei as one of the world's 100 most influential people.[23] He also was named as one of the "Architects of AI" for Time's Person of the Year.[24]

As of February 2026, Anthropic has an estimated value of $380 billion,[25] with Forbes estimating Amodei's net worth to be $7 billion.[26]
    """
    summary_template = """
    fiven the information {information} abput a person I want you to create:
    1. A short summary
    2. Two interesting facts about them"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template = summary_template # NOTE Usar input_variables ayuda a debugging
    )

    # NOTE Temperatura - Deterministico. Sirve para resumen - código
    llm = ChatOllama(temperature=0, model = 'gemma3:270m') 

    # NOTE LangChain Expression Language (LCEL) - prompt template and a llm. 
    # The pipe | creates a runnable chain by connecting output of the left component as an input to the right component
    # Output of the prompt template as input to the LLM
    chain = summary_prompt_template | llm 

    # NOTE Tne invoke method calls the chain with input "information" (Dario Amodei)
    response = chain.invoke(input = {'information': information})
    print(response.content)


if __name__ == "__main__":
    main()
