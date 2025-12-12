---
layout: post
title:  "Recommended Prompts and Keywords in Retrieval-Augmented Generation Systems"
date:   2025-12-08 14:34:15 -0600
categories: hci
---

## Motivation

When you use an AI system for research assistance, you may find it difficult to come up with a good prompt due to limited familiarity with the topic. You may also struggle to evaluate the AI system’s responses and to decide whether to terminate the conversation or continue exploring. In this article, I report on a study conducted by my co-authors [Amit Basu][Amit] and [Jingjing Zhang][jingjing] and me, which examines how users’ behaviors change when interacting with an AI system equipped with recommended keywords and prompts.

[Amit]: https://www.smu.edu/cox/academics/faculty/amit-basu  
[jingjing]: https://kelley.iu.edu/faculty-research/faculty-directory/profile.html?id=JJZHANG


## Study Design

<!--
https://ssrnsurvey.azurewebsites.net/

This survey was saved on Github SSRNSurvey

It will need to be manually uploaded to the azure server.

You will need to push it to Azure master.

The readme.md in the folder of Study II - Prolific - Search SSRN included all the details.

-->


We recruited subjects from Prolific. The subjects are required to come from an English-speaking country and have a master's degree.
- Prompt + Keywords (57 subjects)
- Prompt (45 subjects)
- Keywords (45 subjects)
- Control (42 subjects)

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
### Task Completion Behavior
From user behavior data, we find that keywords significantly reduce people's likelihood to revise their answers to the second question. Further, people tend to spend longer time in gnerating their answers when keywords are provided for Task 2 but not for Task 1.  


### Table: Basic Behavior

| Variable | revise1 | revise2 | Task 1→2 Duration | Task 2→3 Duration |
|:---------|:--------:|:--------:|:------------------:|:-------------------:|
| Keywords | 0.0423 (0.0524) | -0.1262* (0.0630) | 0.1267 (0.1018) | 0.4519* (0.2096) |
| Prompt | -0.0826 (0.0525) | 0.0466 (0.0630) | 0.0674 (0.1019) | 0.1365 (0.2097) |
| Privacy Knowledge (q1) | -0.0159 (0.0267) | 0.0794* (0.0320) | -0.0516 (0.0518) | -0.0598 (0.1066) |
| AI Usage Frequency (q12) | -0.0011 (0.0240) | -0.0090 (0.0288) | -0.0112 (0.0466) | 0.1806 (0.0959) |
| R² | 0.0195 | 0.0565 | 0.0187 | 0.0402 |
| Adj R² | -0.0018 | 0.0360 | -0.0026 | 0.0193 |
| N | 189 | 189 | 189 | 189 |

**Note:** Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10

Task 1→2 Duration and Task 2→3 Duration are log-transformed.

<!--
mygenAIfolder\Code\survey\conversation_prompts_sep5.csv
-->
### Prompting Behavior
We further analzyed users' prompting behavior and found that there is no significant differences among groups.
### Table: Prompting Behavior

| Variable | Log(num_prompts) | Log(avg_prompt_words) |
|:---------|:--------:|:--------:|
| Prompt Treatment | -0.0129 (0.0881) | 0.0126 (0.0684) |
| Keywords Treatment | -0.0403 (0.0880) | -0.0328 (0.0683) |
| Privacy Knowledge (q1) | -0.0796† (0.0451) | 0.0116 (0.0350) |
| AI Usage Freq. (q12) | -0.0325 (0.0401) | 0.1509*** (0.0312) |
| R-squared | 0.0288 | 0.1370 |
| Adj R-squared | 0.0075 | 0.1180 |
| N | 187 | 187 |

**Note:** Standard errors in parentheses. *** p<0.001, ** p<0.01, * p<0.05, † p<0.10

Reference category: Group 4 (Control group)
<!--
MyGenAIfolder\Responses\Alex_Information_Task1

Now in analysis.
-->

### Deligation

In this analysis, we examine whether users' prompts are highly similar to the information task. This helps us to assess to what extant users delegate the problem to AI.

 

## Table: Prompt Alignment and Vocabulary Diversity

| Variable | Max Similarity Task 1 | Max Similarity Task 2 | Avg Word Entropy Task 1 | Avg Word Entropy Task 2 |
|---|---|---|---|---|
| Prompt Treatment | -0.0172<br/>(0.0198) | -0.0580†<br/>(0.0349) | 0.0095<br/>(0.0736) | 0.1050<br/>(0.1111) |
| Keywords Treatment | -0.0361†<br/>(0.0200) | -0.0362<br/>(0.0344) | -0.0312<br/>(0.0743) | -0.1038<br/>(0.1095) |
| Privacy Knowledge | 0.0164<br/>(0.0105) | 0.0020<br/>(0.0182) | -0.0336<br/>(0.0390) | 0.0242<br/>(0.0578) |
| AI Usage Experience | 0.0150†<br/>(0.0089) | 0.0495**<br/>(0.0162) | 0.1406***<br/>(0.0331) | 0.1052*<br/>(0.0515) |
| Number of Prompts | 0.0001<br/>(0.0020) | -0.0088*<br/>(0.0034) | -0.0133†<br/>(0.0075) | -0.0071<br/>(0.0109) |
| N | 159 | 151 | 159 | 151 |
| R² | 0.0791 | 0.1324 | 0.1316 | 0.0551 |

**Note:** Threshold >=8 minutes.

**Note:** † p < 0.10, * p < 0.05, ** p < 0.01, *** p < 0.001. Values in parentheses are standard errors. Controls include privacy knowledge (q1_numeric), AI usage experience (q12_numeric), and number of prompts submitted (num_prompts). Prompt: Groups 1-2. Keywords: Groups 1,3. Control: Group 4.

### Task 1 - Information Retrieval Quality Metrics

We also find that users presented with the prompt tool have a higher chance 

### Table: Task 1 Evaluation

| Variable | Num Unique Papers | NDCG (Binary) |
|:---------|:--------:|:--------:|
| Prompt Treatment | 1.0899† (0.6294) | -0.0238 (0.0388) |
| Keywords Treatment | -0.0065 (0.6286) | -0.0374 (0.0387) |
| Privacy Knowledge (q1) | -0.0498 (0.3206) | -0.0105 (0.0198) |
| AI Usage Freq. (q12) | -0.9264** (0.2878) | -0.0454* (0.0177) |
| R-squared | 0.0774 | 0.0494 |
| N | 186 | 186 |

**Note:** Standard errors in parentheses. *** p<0.001, ** p<0.01, * p<0.05, † p<0.10

Relevance recoded as binary (2=relevant, 1,0=not relevant)


### Task 2 - Speech Quality

 

## Table: Task 2 Evaluation

| Variable | Novelty | Relevance | Salience |
|----------|---------|-----------|----------|
| Prompt Treatment | 0.2431* | 0.1778† | 0.1431 |
| | (0.1003) | (0.1076) | (0.1076) |
| Keywords Treatment | -0.0283 | -0.0051 | -0.0439 |
| | (0.1002) | (0.1075) | (0.1075) |
| Privacy Knowledge | -0.0494 | -0.0368 | -0.0593 |
| | (0.0510) | (0.0547) | (0.0547) |
| AI Experience | 0.0915* | 0.0872† | 0.0735 |
| | (0.0459) | (0.0492) | (0.0492) |
| **N** | 189 | 189 | 189 |

**Note:** Significance levels: ***p<0.001, **p<0.01, *p<0.05, †p<0.10

**Outcomes:**
- **Novelty:** How original and innovative are the ideas presented (1-5 scale)
- **Relevance:** How well does the speech address privacy legislations topic (1-5 scale)
- **Salience:** How important and timely are the issues discussed (1-5 scale)


## Appendix
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



 