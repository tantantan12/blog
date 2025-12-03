---
layout: post
title:  "Virtual Environment for Python"
date:   2025-12-02 14:34:15 -0600
categories: python
---
In Python, different packages can conflict with each other due to version incompatibilities. In addition, you may need to install large packages for a single experimental project, which can clutter your global environment. You may want to delete this experimental project without affecting other projects. A virtual environment solves these problems.

A Python virtual environment is a project-specific, isolated workspace that contains its own interpreter and dependencies. It ensures that each project has exactly the packages (and versions) it needs without affecting other projects or the system-wide Python installation.

# Creating a Virtual Environment

To create a virtual environment on your computer, open the command prompt, and navigate to the folder where you want to create your project, then type this command:

```bash
C:\Users\Your Name> python -m venv myfirstproject
```



This will set up a virtual environment, and create a folder named "myfirstproject" with subfolders and files, like this:

```sql
myfirstproject
  Include
  Lib
  Scripts
  .gitignore
  pyvenv.cfg
```

For example, I used the following code to create a project `.venv`

```bash
cd C:\Users\49280549\Box\SSRN\MCP\code
python -m venv .venv
.venv\Scripts\activate
```

> Why do I use `.venv` for my virtual environment?
- Standard practice - Most Python developers use .venv by default
- Hidden by default - The leading dot makes it hidden on Unix-like systems, keeping your project directory cleaner
- Easy to gitignore - It's typically added to .gitignore so the entire virtual environment folder isn't committed to version control
- Tool recognition - Many Python tools automatically recognize and use .venv without additional configuration
# Activate Virtual Environment
To use the virtual environment, you have to activate it with this command:

```bash
C:\Users\Your Name> myfirstproject\Scripts\activate
```

After activation, your prompt will change to show that you are now working in the active environment:

```bash
(myfirstproject) C:\Users\Your Name>

```

When installing packages in your virtual environment, it will be saved in the project folder.

For example, I used the following command to activate my virtual project and execute my python file ssrn_search_tool.py.
```bash
.venv\Scripts\activate

(.venv) PS C:\Users\49280549\Box\SSRN\MCP\code> python ssrn_search_tool.py
```

# Deactivate Virtual Environment

```bash
(myfirstproject) C:\Users\Your Name> deactivate

```

Source: [source][source]. 

[source]: https://www.w3schools.com/python/python_virtualenv.asp

