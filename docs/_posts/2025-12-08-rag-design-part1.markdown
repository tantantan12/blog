---
layout: post
title:  "Recommended Prompts and Keywords in Retrieval-Augmented Generation Systems"
date:   2025-12-08 14:34:15 -0600
categories: hci
---

## Motivation

When you use an AI system for research assistance, you may find it difficult to come up with a good prompt due to a lack of knowledge in this topic. You may not be able to evaluate the response from the AI system and have a hard time deciding whether you should terminate the conversation or continue exploring. In this article, I would like to report to you a study conducted by my co-authors ([Amit Basu][Amit] and [Jingjing Zhang][jingjing]) and me, which explored users' behavior changes when interacting with an AI system equipped with recommended keywords and prompts.

## Study Design

<!--
https://ssrnsurvey.azurewebsites.net/

This survey was saved on Github SSRNSurvey

It will need to be manually uploaded to the azure server.

You will need to push it to Azure master.

The readme.md in the folder of Study II - Prolific - Search SSRN included all the details.

-->


We recruited subjects from Prolific. The subjects are required to come from an English-speaking country and have a master's degree.
- Prompt + Keywords (56 subjects)
- Prompt (41 subjects)
- Keywords (40 subjects)
- Control (40 subjects)

Note: We removed subjects whose total duration is below 5 minutes.

### Tutorial and Comprehension Check
Subjects were first presented with a tutorial with videos demonstrating key features of ChatSSRN. The videos were rendered as GIFs on each tutorial page, and subjects were required to spend at least 15 seconds on each page. The tutorials were customized based on the assignment to the four treatment groups. The first tutorial page introduced the ChatSSRN interface. The second page demonstrated how users could click the reference to read the abstract from SSRN. The third tutorial page focused on the use of recommended prompts and keywords and was only visible to subjects who were assigned to the corresponding group. After the tutorial, subjects were given a comprehension check with two questions about ChatSSRN. Only upon passing the check were subjects able to continue with the rest of this online study.
 

### Information Tasks

| Information Task 1 | Information Task 2 |
|----------|----------|
| Imagine you are a PhD student. Your advisor asks you to identify literature on the topic:<br><br>The effectiveness of privacy legislation (e.g., COPPA, CCPA, GDPR, HIPPA)<br><br>Using ChatSSRN, find a list of 10-15 reference titles that you consider most relevant to the given topic. Provide only the titles of the selected references in the textbox below. | Imagine you are a PhD student. Now you need to prepare for a speech about:<br><br>Privacy Legislations<br><br>Please include a title of your speech and three bullet points, supported by literature from SSRN. Feel free to use the literature you identified in the first task. |


When working on both Task 1 and Task 2, subjects will be provided with the AI tool embedded in the same page. After they submit their answers to Task 1 and Task 2, they will click to enter a review page, where they can revise their answers to both Task 1 and Task 2. The AI tool is again embedded at the bottom of this revision page. 

## Results
<!--
Study I: myfolder\\Study II - Prolific - Search SSRN\\analysis\\analysis.ipynb

These four groups are for Study I:
- Prompt + Keywords
- Prompt 
- Keywords
- Control

-->
### User Behavior
From user behavior data, we find that keywords significantly reduce people's likelihood to revise their answers to the second question. Further, people tend to spend longer time in gnerating their answers when keywords are provided. This holds for both Information Tasks 1 and 2.



| Variable | revise1 | revise2 | Task 1→2 Duration | Task 2→3 Duration |
|:---------|:--------:|:--------:|:------------------:|:-------------------:|
| Keywords | 0.0477 (0.0551) | -0.1402* (0.0661) | 0.2116* (0.1023) | 0.4614* (0.2194) |
| Prompt | -0.0726 (0.0553) | 0.0500 (0.0663) | 0.0696 (0.1027) | 0.1286 (0.2201) |
| Privacy Knowledge (q1) | -0.0126 (0.0278) | 0.0795* (0.0334) | -0.0401 (0.0517) | -0.0915 (0.1108) |
| AI Usage Frequency (q12) | 0.0015 (0.0253) | -0.0028 (0.0303) | -0.0208 (0.0469) | 0.1593 (0.1006) |
| R² | 0.0159 | 0.0615 | 0.0345 | 0.0379 |
| Adj R² | -0.0070 | 0.0397 | 0.0121 | 0.0155 |
| N | 177 | 177 | 177 | 177 |

**Note:** Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10

Task 1→2 Duration and Task 2→3 Duration are log-transformed.

<!--
conversation_prompts_sep5.csv
-->

We further analzyed users' prompting behavior and found that recommended prompt will decrease the number of words in users' prompt.
 
| Variable | num_prompts | avg_prompt_words | paper_references |
|:---------|:--------:|:--------:|:--------:|
| Prompt Treatment | -0.0607 (0.7629) | -1.3798† (0.7629) | 1.7089 (1.9010) |
| Keywords Treatment | -0.0631 (0.7602) | -0.3010 (0.7603) | 0.1485 (1.8943) |
| Privacy Knowledge (q1) | -1.0391** (0.3876) | 0.0472 (0.3876) | -2.3488* (0.9659) |
| AI Usage Freq. (q12) | -0.0352 (0.3474) | 1.7154*** (0.3475) | 0.2418 (0.8657) |
| R-squared | 0.0449 | 0.1554 | 0.0425 |
| Adj R-squared | 0.0225 | 0.1355 | 0.0200 |
| N | 175 | 175 | 175 |

**Note:** Standard errors in parentheses. *** p<0.001, ** p<0.01, * p<0.05, † p<0.10

Reference category: Group 4 (Control group)
## Appendis
### Information Task Evaluation Questions

Please answer the following questions to help us better understand the two information tasks.

1. My knowledge of privacy is:
   - Far below average
   - Somewhat below average
   - Average
   - Somewhat above average
   - Far above average

2. How difficult was it for you to identify keywords to generate prompts to interact with the search tool for the first task (literature discovery)?
   - Very Easy
   - Easy
   - Neutral
   - Difficult
   - Very Difficult

3. How difficult was it for you to identify keywords to generate prompts to interact with the search tool for the second task (speech preparation)?
   - Very Easy
   - Easy
   - Neutral
   - Difficult
   - Very Difficult

4. How difficult was it for you to include keywords into appropriate prompts to interact with the search tool for the first task (literature discovery)?
   - Very Easy
   - Easy
   - Neutral
   - Difficult
   - Very Difficult

5. How difficult was it for you to include keywords into appropriate prompts to interact with the search tool for the second task (speech preparation)?
   - Very Easy
   - Easy
   - Neutral
   - Difficult
   - Very Difficult

6. How difficult was it for you to identify good search strategies for the first task (literature discovery)?
   - Very Easy
   - Easy
   - Neutral
   - Difficult
   - Very Difficult

7. How difficult was it for you to identify good search strategies for the second task (speech preparation)?
   - Very Easy
   - Easy
   - Neutral
   - Difficult
   - Very Difficult

### Search Tool Evaluation

Please answer the following questions to help us better understand the search tool.

1. Overall, what is your satisfaction for retrieved information from the search tool?
   - Very Dissatisfied
   - Dissatisfied
   - Neutral
   - Satisfied
   - Very Satisfied

2. Overall, what is your satisfaction for the interface of the search tool?
   - Very Dissatisfied
   - Dissatisfied
   - Neutral
   - Satisfied
   - Very Satisfied

3. Overall, what is your satisfaction for the search process on the search tool?
   - Very Dissatisfied
   - Dissatisfied
   - Neutral
   - Satisfied
   - Very Satisfied

4. Overall, what is your satisfaction for the received help from the search tool?
   - Very Dissatisfied
   - Dissatisfied
   - Neutral
   - Satisfied
   - Very Satisfied

5. How often do you use generative AI tools? (e.g., ChatGPT)
   - Never
   - Rarely
   - Sometimes
   - Often
   - Very Often

### Demographic Questions


- Gender 
- Age
- Country
- Education level





Links:
[Amit]:https://www.smu.edu/cox/academics/faculty/amit-basu
[Jingjing]:https://kelley.iu.edu/faculty-research/faculty-directory/profile.html?id=JJZHANG


