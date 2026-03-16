# Neo4j setup for Mem0 graph memory

Use this to run Neo4j locally so the Mem0 notebook can use graph storage (entities and relationships).

## 1. Install Neo4j (macOS with Homebrew)

```bash
brew install neo4j
```

This installs the Neo4j server and dependencies (including OpenJDK).

## 2. Start Neo4j

**One-off (foreground):**
```bash
neo4j console
```

**Or as a background service:**
```bash
neo4j start
# Stop later with: neo4j stop
```

**Or via Homebrew services (starts on login):**
```bash
brew services start neo4j
# Stop: brew services stop neo4j
```

## 3. Set the initial password

1. Open **Neo4j Browser**: http://localhost:7474  
2. Connect with:
   - **Username:** `neo4j`
   - **Password:** (first time) `neo4j` — you will be prompted to set a new password  
3. Choose and confirm a new password (e.g. `my-neo4j-password`).

## 4. Set environment variables

Use the **Bolt** URL (port 7687). For **local** Neo4j use `neo4j://` (no `+s`).  
For **Neo4j Aura** use `neo4j+s://<your-instance>.databases.neo4j.io`.

**In your shell (add to `~/.zshrc` to persist):**
```bash
export NEO4J_URL="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-chosen-password"
```

**Or use a `.env` file** in this project (see below) and load it before running the notebook.

## 5. Install Mem0 graph extra (if not already)

```bash
pip install "mem0ai[graph]"
```

## 6. Run the notebook

Restart the Jupyter kernel (or terminal) so it sees the new env vars, then run the notebook.  
The graph section will use Neo4j when `NEO4J_URL` and `NEO4J_PASSWORD` are set.

---

### Optional: `.env` file

Create a file named `.env` in this project (and add `.env` to `.gitignore`):

```
NEO4J_URL=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-chosen-password
```

At the top of the notebook (or in a first cell), load it with:

```python
from dotenv import load_dotenv
load_dotenv()
```

Install: `pip install python-dotenv`

---

### URL reference

| Environment | NEO4J_URL example |
|-------------|-------------------|
| Local Neo4j | `neo4j://localhost:7687` |
| Neo4j Aura  | `neo4j+s://xxxxxxxx.databases.neo4j.io` |
