---
layout: post
title:  "Introduction to MCP (Part I)"
date:   2025-12-02 14:34:15 -0600
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


You've probably heard about Model Context Protocol (MCP) on different occasions. What is MCP? 

# What is MCP?
MCP was created to solve the problem of inregrations between LLMs and applications. With MCP, it becomes super easy to build and distribute new integrations because MCP enables the separation of integrations and app code. 

![alt text](2025-12-02-model-context-protocol/mcp.webp)
## The N $\times$ M Problem
We know that LLMs are powerful because they power other applications. If you have M applications and you want to access N different LLMs, you will need to write M $\times$ N connectors. 

## The N+M Solution 
MCP addresses this problem with the MCP server, which provides a common interface for both discovering and using integrations such as tools, prompts, and data sources. As a result, programmers no longer need to custom connectors for each model and for each integration. All they need to do is to connect to the server and get the list of integrations that the server provides and then call the functions. As an example, in Python, `use_tool()` would allow tool calling with tool name and arguments passed as parameters of the function. This architecture makes MCP a universal connector, just like USB-C.

## How MCP Enables Collaborations

App developers had to build their own tools and collectors because these were not re-usable and hard to distribute among teams. With MCP, tools, prompts, and data can be shared either indirectly through GitHub or as a running server that models can connect to directly. With unified interface for discovering and using integrations, MCP allows the sharing of integrations both globally and within organizations. 

# MCP Capabilities
It is important to note that an MCP server offers three key capabilities: **actions**, **data**, and **language model prompts**.

## Tools
**Tools** represent actions. They are typically self-contained functions that can be called by the host application's language model. The names and descriptions are provided to the LLM via the client. If a tool is called, it will run wherever the MCP server is running, with the results being sent back to the client and application. Tools can be called by a router in an agentic system.

**Resources**
Resources represent data. They can be stored in the format of text files, blob files (PDFs), database schemes, code documentation, and others. If we consider tools to be verbs, resources are the nouns. They can be used to provide more context to the LLM; they can also be used by tools.

**Prompts**
Prompts are the language model prompts that can be used by the host application to interact with their own LLM. These server-side prompts are useful when you have a server built for a specific service or application and the builders know it well enough to have tested and built optimized prompts for interacting with said service.

## The MCP Transport

Transports in MCP implement the protocol’s communication layer. At the most basic level, they are responsible for sending and receiving messages between the client and server. MCP SDKs come with two default transports: stdio for local connections and Streamable HTTP for remote connections. While these are the most commonly used, the protocol allows for the implementation of custom transports as well, which can support use cases for which the default transports are less well-suited, as long as the underlying communication channel being used allows for bidirectional message passing. Transports may be stateless (as in the case of the stdio transport) or optionally stateful (as in the case of the Streamable HTTP transport). For stateful transports, the transport is responsible for maintaining a session between the client and server and may support things like resuming after a network interruption, or supporting multiple clients connecting to the same server. All transports must manage the entire connection lifecycle.

## The MCP Lifecycle

This lifecycle governs how each of the MCP components handle creating, using, and closing a connection:

- Initialization: The client and server exchange messages to establish which protocol versions they’re compatible with, to share and negotiate supported capabilities, and to share any implementation details that may be relevant to the connection.

- Operation: Based on the negotiated capabilities from the initialization phase, the client sends a request to the server, and the server replies with a response, which could include the results of the requested operation or an error if there was a failure.

- Shutdown: The client terminates the protocol connection, with the transport layer being responsible for communicating this termination to the server.