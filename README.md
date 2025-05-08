
# Semantic Kernel Demo Agents with MCP

> **⚠️ Warning:** This project uses a prereleased version of the Semantic Kernel Vector stores, which may differ from the current released and documented version.

This repository demonstrates how to use Semantic Kernel agents with a variety of interfaces, including a command-line agent, a Chainlit app, and an MCP SSE server. It also includes utilities for parsing and storing data in a vector store.

---

## Setup

Follow these steps to set up your environment:

1. **Install [uv](https://github.com/astral-sh/uv):**
   - You can install `uv` by following the instructions in the [uv documentation](https://github.com/astral-sh/uv#installation).

2. **Install dependencies:**
   - Run the following command to install all required packages (in a new .venv folder):

     ```sh
     uv sync
     ```

---

## Load Data

To parse the contents of the directory and create a vector store:

1. Run the following command:

   ```sh
   python data/parse.py
   ```

   This will create a `chroma` directory with the parsed contents inside the `data` folder.

---

## Usage

You can interact with the agents in several ways:

### 1. Command Line Agent

- The base example is in [`agents.py`](agents.py).
- Run it to use a simple agent that answers questions about the contents of the directory:

  ```sh
  python agents.py
  ```

### 2. Chainlit App

- Start the Chainlit app to interact with the agent via a web interface:

  ```sh
  python chainlit_app.py
  ```

### 3. MCP Server

- Start the MCP SSE server to use the agent with MCP clients:

  ```sh
  python mcp_server.py
  ```

- You can then include the following configuration in your MCP clients:

  ```json
  "agent": {
      "url": "http://localhost:8000/sse"
  }
  ```

---

## Project Structure

- `agents.py` - Command-line agent example
- `chainlit_app.py` - Chainlit web app
- `mcp_server.py` - MCP SSE server
- `data/parse.py` - Script to parse and store data in the vector store
- `data/markdowns/` - Directory containing the markdown files to be parsed
- `data/chroma/` - Directory containing the persisted chroma store

---

## Notes

- This project uses prereleased features of Semantic Kernel. APIs and behaviors may change.
- For more information, see the official [Semantic Kernel documentation](https://github.com/microsoft/semantic-kernel).
