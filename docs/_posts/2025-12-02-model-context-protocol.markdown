---
layout: post
title:  "Model Context Protocol"
date:   2025-12-02 14:34:15 -0600
categories: AI
published: true
---
You've probably heard about Model Context Protocol (MCP) on different occasions. What is MCP? 
MCP was created to solve the problem of inregrations between LLMs and applications. With MCP, it becomes super easy to build and distribute new integrations because MCP enables the separation of integrations and app code. 

![alt text](2025-12-02-model-context-protocol/mcp.webp)
# The N $\times$ M Problem
We know that LLMs are powerful because they power other applications. If you have M applications and you want to access N different LLMs, you will need to write M $\times$ N connectors. 

# The N+M Solution 
MCP addresses this problem with the MCP server, which provides a common interface for both discovering and using integrations such as tools, prompts, and data sources. As a result, programmers no longer need to custom connectors for each model and for each integration. All they need to do is to connect to the server and get the list of integrations that the server provides and then call the functions. As an example, in Python, `use_tool()` would allow tool calling with tool name and arguments passed as parameters of the function. This architecture makes MCP a universal connector, just like USB-C.

# How MCP Enables Collaborations

App developers had to build their own tools and collectors because these were not re-usable and hard to distribute among teams. With MCP, tools, prompts, and data can be shared either indirectly through GitHub or as a running server that models can connect to directly. With unified interface for discovering and using integrations, MCP allows the sharing of integrations both globally and within organizations. 

> It is important to note that an MCP server offers three key capabilities: **actions**, **data**, and **language model prompts**.