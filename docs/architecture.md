# DataAgent Architecture

## System Overview

DataAgent is a multi-agent system built on LangGraph that orchestrates seven specialized AI agents to perform autonomous data analysis.

## Agent Communication Flow

```
                    ┌─────────────────────────────────────┐
                    │           Orchestrator               │
                    │         (LangGraph State)            │
                    └──────┬──────────────────────────────┘
                           │
          ┌────────────────┼────────────────────┐
          │                │                    │
          ▼                ▼                    ▼
    ┌──────────┐    ┌──────────┐        ┌──────────┐
    │  Schema  │    │ Planner  │        │  Coder   │◄─┐
    │  Agent   │    │  Agent   │        │  Agent   │  │ Retry
    └──────────┘    └──────────┘        └────┬─────┘  │ Loop
                                             │        │
                                             ▼        │
                                        ┌──────────┐  │
                                        │  Critic  │──┘
                                        │  Agent   │
                                        └────┬─────┘
                                             │
                              ┌──────────────┼──────────────┐
                              │              │              │
                              ▼              ▼              ▼
                        ┌──────────┐  ┌──────────┐  ┌──────────┐
                        │Visualizer│  │Predictor │  │  Story-  │
                        │  Agent   │  │  Agent   │  │  teller  │
                        └──────────┘  └──────────┘  └──────────┘
```

## State Management

The orchestrator maintains a shared state (`AgentState`) that flows through the graph:
- `question`: User's natural language query
- `schema`: Semantic schema from the Schema Agent
- `plan`: Analysis plan from the Planner Agent
- `sql_query`: Generated SQL from the Coder Agent
- `result_df`: Query results as a DataFrame
- `chart`: Plotly figure from the Visualizer Agent
- `validation`: Critic Agent's validation report
- `prediction`: ML results from the Predictor Agent
- `narrative`: Plain-English explanation from the Storyteller Agent
- `agent_log`: Activity log tracking each agent's execution

## Self-Correction Loop

The Critic Agent implements a debate pattern:
1. Receives output from the Coder Agent
2. Validates for empty results, unreasonable values, statistical issues
3. If confidence < 40%: REJECTS and sends back to Coder with corrections (max 3 retries)
4. If confidence 40-70%: APPROVES WITH WARNINGS
5. If confidence > 70%: VALIDATES

## RAG Pipeline

Statistical knowledge base stored in ChromaDB:
- Documents cover statistical tests, sample sizes, common errors, chart selection
- Critic Agent queries this knowledge base before validating outputs
- Ensures statistical rigor across all analyses
