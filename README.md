# LLM-QUESTION-ANSWER

**🔍 Description:
KeywordQuest is an interactive AI-driven game where players engage in deductive reasoning to identify keywords from datasets of countries 🌍, cities 🏙️, and landmarks 🗼. Using Python and leveraging machine learning techniques, the game allows players to ask targeted questions to determine whether the keyword belongs to a country, city, or landmark. Developed with the Kaggle environments library, KeywordQuest demonstrates the application of AI in gaming through structured data interaction.**

**🔧 Technologies: Python, NumPy, pandas, JSON, Kaggle Environment**
```
import numpy as np
import pandas as pd
import json
import os
import sys
import random
from kaggle_environments import make

sys.path.append('llm_20_questions')
import keywords

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
```
## Load JSON data and create dataframes
```
my_json = json.loads(keywords.KEYWORDS_JSON)
json_country = my_json[0]
json_city = my_json[1]
json_landmark = my_json[2]
df_country = pd.json_normalize(json_country['words'])
df_city = pd.json_normalize(json_city['words'])
df_landmark = pd.json_normalize(json_landmark['words'])
```
## Combine all keywords into a single list
```
all_keywords = (
list(df_country['word']) +
list(df_city['word']) +
list(df_landmark['word'])
)
```
## Select 20 random keywords for questions
```
random_keywords = random.sample(all_keywords, 20)
```
## Define the agent function using the dataframes
```
class AgentWithQuestions:
def init(self, df_country, df_city, df_landmark, questions):
self.df_country = df_country
self.df_city = df_city
self.df_landmark = df_landmark
self.questions = questions
self.question_index = 0
```
## python Copy code
```
def __call__(self, obs, cfg):
    if obs.turnType == "ask":
        if self.question_index < len(self.questions):
            response = self.questions[self.question_index]
            print(f"Agent asks: {response}")
            self.question_index += 1
        else:
            response = "Is it a duck?"  # Fallback question if we run out
            print(f"Agent asks: {response}")
    elif obs.turnType == "guess":
        response = "country"  # Implement more complex guessing logic here
        print(f"Agent guesses: {response}")
    elif obs.turnType == "answer":
        entity = obs.entity.lower()
        if entity in self.df_country['word'].values:
            response = "country"
        elif entity in self.df_city['word'].values:
            response = "city"
        elif entity in self.df_landmark['word'].values:
            response = "landmark"
        else:
            response = "no"
        print(f"Agent answers: {response} to entity {entity}")
    return response
```
## Initialize the agent
```
agent_instance = AgentWithQuestions(df_country, df_city, df_landmark, random_keywords)
```
## Define agent function wrapper
```
def agent_fn(obs, cfg):
return agent_instance(obs, cfg)
```
## Configuration for the environment
```
debug_config = {
'episodeSteps': 61, # initial step plus 3 steps per round (ask/answer/guess)
'actTimeout': 60, # agent time per round in seconds; default is 60
'runTimeout': 1200, # max time for the episode in seconds; default is 1200
'agentTimeout': 3600 # obsolete field; default is 3600
}
```
## Initialize the environment
```
env = make(environment="llm_20_questions", configuration=debug_config, debug=True)
```
# Run the game
```
game_output = env.run(agents=[agent_fn, agent_fn, agent_fn, agent_fn])
```
## Render the game output
```
env.render(mode="ipython", width=1080, height=700)
```
