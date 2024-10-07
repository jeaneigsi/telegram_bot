# Telegram Chatbot with LangChain, Gemini API, and Groq

This project demonstrates a dynamic chatbot using LangChain for memory management and integration with various APIs, including Telegram, Gemini API, Groq, and scraping for the latest marketing news.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [APIs Integrated](#apis-integrated)
- [Functions](#functions)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This chatbot uses LangChain for managing conversational memory, allowing for context-aware interactions with users. It integrates with the **Telegram API** for messaging and supports calls to external services like **Gemini API**, **Groq**, and **news scraping** for real-time updates.

## Features

- **Conversational Memory**: Powered by LangChain for dynamic memory management, allowing contextual conversation history.
- **Telegram Integration**: Handles incoming messages, responds based on the conversation context, and fetches the latest marketing and digital news.
- **APIs Support**: Integration with Gemini API and Groq to enhance chatbot functionality.
- **Real-Time News Scraping**: Fetches updates from marketing news websites.

## Setup

### Requirements

- Python 3.9+
- Redis (for storing memory state)
- Telegram API Token
- GtTS API and Groq credentials (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/chatbot-project
   cd chatbot-project
