---
layout: post
title:  "Search vs. Chatbot"
date:   2025-12-22 14:34:15 -0600
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


## Motivation

AI is changing the landscape of information retrieval, and many say AI will replace traditional search engines. In this article, I report on a study conducted by my co-authors [Amit Basu][Amit], [Jingjing Zhang][jingjing], and me, which compares an AI chatbot with a search engine. With a controlled information retrieval setup, we ensured that the underlying knowledge base and the retriever were identical. This design attributes any differences in user behavior and task completion quality to the interface—AI chatbot versus search engine.

[Amit]: https://www.smu.edu/cox/academics/faculty/amit-basu  
[jingjing]: https://kelley.iu.edu/faculty-research/faculty-directory/profile.html?id=JJZHANG