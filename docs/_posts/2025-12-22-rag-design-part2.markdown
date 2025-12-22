---
layout: post
title:  "Search vs. Chatbot"
date:   2025-12-21 14:34:15 -0600
categories: AI
published: true
---
<!-- MathJax configuration -->
<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
  svg: { fontCache: 'global' }
};
</script>

<!-- MathJax CDN -->
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>

# AI-based Chatbot vs. Search Engine
## Motivation

AI is changing the landscape of information retrieval, and many say AI will replace traditional search engines. In this article, I report on a study conducted by my co-authors [Amit Basu][Amit], [Jingjing Zhang][jingjing], and me, which compares an AI chatbot with a search engine. With a controlled information retrieval setup, we ensured that the underlying knowledge base and the retriever were identical. This design attributes any differences in user behavior and task completion quality to the interface—AI chatbot versus search engine.

[Amit]: https://www.smu.edu/cox/academics/faculty/amit-basu  
[jingjing]: https://kelley.iu.edu/faculty-research/faculty-directory/profile.html?id=JJZHANG

## Research Design

We define four versions of information retrieval tools:

AI (baseline)
![alt text](../assets/images/2025-12-22-rag-design-part2/tutorial-base.png)
AI-Keywords
![alt text](../assets/images/2025-12-22-rag-design-part2/tutorial-keywords.png)
AI-Topic
![alt text](../assets/images/2025-12-22-rag-design-part2/exp-topic.png)
Search
![alt text](../assets/images/2025-12-22-rag-design-part2/exp-tutorial-search.png)
### Information Task Design

We first conducted a pilot study to gather all papers related to algorithmic trading. Specifically, we recruited ~120 student subjects, and asked each student subject to explore four topics related to algorithmic trading. For each topic, they will need to submit SSRN papers related to the topic. This gives us a set of 133 papers, most of which are related to algorithmic trading. We then hired an RA to rate each paper based on its relevance to algorithmic trading. Then, an experimentalist examined all relevant papers, and came up with four sets of questions with multiple answers. This four set of eight questions were reviewed by a finance professor and one set (two questions) were removed. 

#### Set 1
A. According to evidence from research, which of the following is NOT a consequence of AI collusion? There could be one or more answers.
- Damage competition
- Reduce market efficiency
- Decrease profit
- Increase communication cost
- I don't know

B. AI-powered algorithmic trader could lead to AI collusion. Which of the following is NOT a potential mechanism? There could be one or more answers.
- Price-trigger strategies when information-sensitive investors are prevalent and noise trading risk is high.
- Price-trigger strategies when information-insensitive investors are prevalent and noise trading risk is low.
- Over-pruning bias in learning
- Overfitting bias in learning.
- I don't know
#### Set 2
A. Which trading strategy has been found by research to perform best in terms of profit?
- Market-making algorithm
- Sniper algorithm
- Human trader
- I don't know

B. AI-powered algorithmic trader could lead to AI collusion. Which of the following is NOT a potential mechanism? There could be one or more answers.
- Price-trigger strategies when information-sensitive investors are prevalent and noise trading risk is high.
- Price-trigger strategies when information-insensitive investors are prevalent and noise trading risk is low.
- Over-pruning bias in learning
- Overfitting bias in learning.
- I don't know
#### Set 3
A. In two experimental markets (twin markets), which kind of arbitrage bot has been found to move prices into closer alignment with fundamental values?
- Arbitrage robot traders that make market liquidity
- Arbitrage robot traders that take market liquidity
- Both
- Neither
- I don't know

B. In two experimental markets (twin markets), which kind of arbitrage bot generates greater conformity to the law-of-one-price?
- Arbitrage robot traders that make market liquidity
- Arbitrage robot traders that take market liquidity
- Both
- Neither
- I don't know

### Workflow

![alt text](../assets/images/2025-12-22-rag-design-part2/exp-pre.png)

![alt text](../assets/images/2025-12-22-rag-design-part2/exp-task.png)

![alt text](../assets/images/2025-12-22-rag-design-part2/exp-post.png)
## Results

### User Satisfaction

![alt text](../assets/images/2025-12-22-rag-design-part2/satisfaction.png)

We compared user satisfaction ratings across interface groups using independent samples t-tests against the control (AI baseline) group:

| Group | Control Mean | Group Mean | Mean Difference | t-statistic | p-value |
|-------|--------------|------------|-----------------|-------------|---------|
| AI-Keywords (2) | 2.74 | 2.57 | -0.173 | -1.05 | 0.297 |
| AI-Topic (3) | 2.74 | 2.70 | -0.043 | -0.28 | 0.777 |
| Search (4) | 2.74 | 3.46 | 0.716 | 4.90 | **<0.001** |

The Search interface significantly increased user satisfaction compared to the control group (p < 0.001), with users rating it 0.72 points higher on average. In contrast, the AI-Keywords and AI-Topic interfaces showed no significant differences in satisfaction compared to the control group.

### Task Performance (Grades)

![alt text](../assets/images/2025-12-22-rag-design-part2/grades.png)

We compared task performance (grades) across interface groups using independent samples t-tests against the control (AI baseline) group:

| Group | Control Mean | Group Mean | Mean Difference | t-statistic | p-value |
|-------|--------------|------------|-----------------|-------------|---------|
| AI-Keywords (2) | 0.586 | 0.625 | 0.040 | 0.44 | 0.661 |
| AI-Topic (3) | 0.586 | 0.625 | 0.040 | 0.39 | 0.696 |
| Search (4) | 0.586 | 0.795 | 0.209 | 2.00 | **0.047** |

The Search interface showed significantly better task performance compared to the control group (p = 0.047), with users achieving 20.9 percentage points higher grades on average. The AI-Keywords and AI-Topic interfaces showed modest improvements but no statistically significant differences from the control group.

### Unknown Percentage Analysis

![alt text](../assets/images/2025-12-22-rag-design-part2/unknown_percentage.png)

We conducted independent samples t-tests comparing the unknown percentage across different interface groups against the control (AI baseline) group. The results show:

| Group | Control Mean | Group Mean | Mean Difference | t-statistic | p-value |
|-------|--------------|------------|-----------------|-------------|---------|
| AI-Keywords (2) | 10.53% | 4.17% | -6.36 | -1.91 | 0.059 |
| AI-Topic (3) | 10.53% | 6.11% | -4.42 | -1.13 | 0.260 |
| Search (4) | 10.53% | 5.02% | -5.50 | -1.52 | 0.132 |

All three alternative interfaces show lower unknown percentages compared to the control group, though none of these differences reach statistical significance at the p < 0.05 level. The AI-Keywords interface shows the largest mean difference (-6.36 percentage points) with a p-value closest to significance (p = 0.059).

### Task Difficulty Perception

We compared user perception of task difficulty across interface groups using independent samples t-tests against the control (AI baseline) group:

| Group | Control Mean | Group Mean | Mean Difference | t-statistic | p-value |
|-------|--------------|------------|-----------------|-------------|---------|
| AI-Keywords (2) | 2.78 | 2.73 | -0.049 | -0.32 | 0.749 |
| AI-Topic (3) | 2.78 | 2.83 | 0.042 | 0.28 | 0.780 |
| Search (4) | 2.78 | 3.31 | 0.525 | 3.83 | **0.0002** |

Note: The difficulty scale is inverted, with higher values indicating easier tasks (5 = very easy, 1 = very difficult).

The Search interface was perceived as significantly easier compared to the control group (p = 0.0002), with users rating it 0.53 points higher on the ease scale. The AI-Keywords and AI-Topic interfaces showed no significant differences in perceived difficulty from the control group.