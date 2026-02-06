# Persistence Backends for LangGraph

Persistence enables LangGraph applications to save and restore state across executions, supporting features like conversation history, human-in-the-loop workflows, and crash recovery.

## Overview

LangGraph supports multiple persistence backends through the `Checkpointer` interface. Checkpointers save state snapshots at each graph step, enabling:

- **Resume from interruption**: Continue execution after pauses or crashes
- **Time travel**: Rewind to previous states for debugging
- **Human-in-the-loop**: Pause for approval and resume
- **Multi-turn conversations**: Maintain context across sessions

## Checkpointer Interface

LangGraph checkpointers conform to `BaseCheckpointSaver` and expose these core methods:

- `put`: store a checkpoint with config and metadata
- `put_writes`: store pending/intermediate writes
- `get_tuple`: fetch a checkpoint tuple for a given config (`thread_id`/`checkpoint_id`)
- `list`: list checkpoints matching config/filter criteria

For async execution, async variants are used (`aput`, `aput_writes`, `aget_tuple`, `alist`).

## Memory Backend

In-memory storage for development and testing. Data is lost when the process ends.

### Python

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# Create memory checkpointer
checkpointer = InMemorySaver()

# Use with graph
graph = StateGraph(MyState)
# ... add nodes and edges ...
app = graph.compile(checkpointer=checkpointer)

# Invoke with thread ID
config = {"configurable": {"thread_id": "conversation-1"}}
result = app.invoke({"messages": [...]}, config)

# Continue same conversation
result2 = app.invoke({"messages": [...]}, config)  # Has previous context
```

### TypeScript

```typescript
import { MemorySaver } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";

// Create memory checkpointer
const checkpointer = new MemorySaver();

// Use with graph
const graph = new StateGraph({ channels: MyStateAnnotation })
  // ... add nodes and edges ...

const app = graph.compile({ checkpointer });

// Invoke with thread ID
const config = { configurable: { thread_id: "conversation-1" } };
const result = await app.invoke({ messages: [...] }, config);

// Continue same conversation
const result2 = await app.invoke({ messages: [...] }, config);
```

### When to Use

**Good for:**
- Development and testing
- Prototyping
- Short-lived sessions
- Single-process applications

**Not suitable for:**
- Production deployments
- Multi-process applications
- Long-term storage
- Distributed systems

## SQLite Backend

File-based SQL database for local persistence. Good for production single-instance deployments.

### Python

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create SQLite checkpointer
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Or with async
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")

# Use with graph
app = graph.compile(checkpointer=checkpointer)

# Same invocation pattern
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [...]}, config)
```

**Connection options:**

```python
# In-memory database (for testing)
checkpointer = SqliteSaver.from_conn_string(":memory:")

# File-based database
checkpointer = SqliteSaver.from_conn_string("./data/checkpoints.db")
```

### TypeScript

```typescript
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";

// Create SQLite checkpointer
const checkpointer = SqliteSaver.fromConnString("checkpoints.db");

const app = graph.compile({ checkpointer });
```

### When to Use

**Good for:**
- Production single-instance deployments
- Local development with persistence
- Applications with moderate traffic
- Cost-effective storage

**Not suitable for:**
- Multi-instance deployments (file locking issues)
- Very high throughput (single file bottleneck)
- Distributed systems

## PostgreSQL Backend

Production-ready relational database for multi-instance deployments.

### Python

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/mydb"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()  # Run once when initializing schema
    app = graph.compile(checkpointer=checkpointer)

# Or async
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # await checkpointer.setup()  # Run once when initializing schema
    app = graph.compile(checkpointer=checkpointer)
```

**Environment variables:**

```python
import os

DATABASE_URL = os.getenv("DATABASE_URL")
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
```

### TypeScript

```typescript
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

// Create Postgres checkpointer
const checkpointer = PostgresSaver.fromConnString(
  "postgresql://user:password@localhost:5432/mydb"
);
// await checkpointer.setup(); // Run once when initializing schema

const app = graph.compile({ checkpointer });
```

### When to Use

**Good for:**
- Production multi-instance deployments
- High-traffic applications
- Applications requiring complex queries
- Enterprise environments with existing Postgres infrastructure

**Considerations:**
- Requires Postgres server
- Higher operational complexity
- Additional infrastructure cost

## Redis Backend

Key-value store for high-performance persistence with TTL support.

### Python

```python
from langgraph.checkpoint.redis import RedisSaver

DB_URI = "redis://localhost:6379"
with RedisSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()  # Run once when initializing schema
    app = graph.compile(checkpointer=checkpointer)
```

### TypeScript

```typescript
import { RedisSaver } from "@langchain/langgraph-checkpoint-redis";

const checkpointer = RedisSaver.fromConnString("redis://localhost:6379");
// await checkpointer.setup(); // Run once when initializing schema
const app = graph.compile({ checkpointer });
```

### When to Use

**Good for:**
- High-performance requirements
- Temporary session storage
- Applications with automatic cleanup (TTL)
- Caching layer for checkpoints

**Considerations:**
- Data may be lost if Redis crashes (unless persistence configured)
- Memory-based (cost scales with data)
- Requires Redis server

## Custom Backend

Create custom backends for specialized storage systems.

### Python Implementation

```python
from langgraph.checkpoint import BaseCheckpointSaver
from typing import Optional, Iterator
import json

class CustomCheckpointer(BaseCheckpointSaver):
    """Custom checkpointer implementation."""

    def __init__(self, storage_client):
        self.client = storage_client

    def put(self, config: dict, checkpoint: dict, metadata: dict) -> None:
        """Save checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]

        # Serialize checkpoint
        data = json.dumps({
            "checkpoint": checkpoint,
            "metadata": metadata
        })

        # Save to storage
        self.client.save(
            key=f"{thread_id}/{checkpoint_id}",
            data=data
        )

    def get_tuple(self, config: dict) -> Optional[dict]:
        """Get latest checkpoint tuple."""
        thread_id = config["configurable"]["thread_id"]

        # Retrieve from storage
        latest = self.client.get_latest(thread_id)

        if latest:
            return json.loads(latest)
        return None

    def list(self, config: dict) -> Iterator[dict]:
        """List all checkpoints for thread."""
        thread_id = config["configurable"]["thread_id"]

        for checkpoint_data in self.client.list_all(thread_id):
            yield json.loads(checkpoint_data)

# Usage
storage = MyStorageClient()
checkpointer = CustomCheckpointer(storage)
app = graph.compile(checkpointer=checkpointer)
```

### TypeScript Implementation

```typescript
import { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";

class CustomCheckpointer extends BaseCheckpointSaver {
  private client: any;

  constructor(storageClient: any) {
    super();
    this.client = storageClient;
  }

  async put(config: any, checkpoint: any, metadata: any): Promise<void> {
    const threadId = config.configurable.thread_id;
    const checkpointId = checkpoint.id;

    const data = JSON.stringify({ checkpoint, metadata });

    await this.client.save(`${threadId}/${checkpointId}`, data);
  }

  async getTuple(config: any): Promise<any> {
    const threadId = config.configurable.thread_id;
    const latest = await this.client.getLatest(threadId);

    return latest ? JSON.parse(latest) : null;
  }

  async *list(config: any): AsyncIterator<any> {
    const threadId = config.configurable.thread_id;

    for await (const data of this.client.listAll(threadId)) {
      yield JSON.parse(data);
    }
  }
}
```

## Thread Management

### Thread IDs

Thread IDs identify separate conversation streams:

```python
# Different conversations
config1 = {"configurable": {"thread_id": "user-123-session-1"}}
config2 = {"configurable": {"thread_id": "user-123-session-2"}}

# Each has separate state
app.invoke({"messages": [...]}, config1)
app.invoke({"messages": [...]}, config2)
```

### Thread ID Strategies

```python
# User-based
thread_id = f"user-{user_id}"

# Session-based
thread_id = f"user-{user_id}-session-{session_id}"

# Time-based
thread_id = f"user-{user_id}-{datetime.now().strftime('%Y%m%d')}"

# Random (for anonymous users)
import uuid
thread_id = f"anon-{uuid.uuid4()}"
```

### Listing Threads

```python
# Get all checkpoints for a thread
config = {"configurable": {"thread_id": "user-123"}}
checkpoints = list(checkpointer.list(config))

for checkpoint in checkpoints:
    print(f"Checkpoint ID: {checkpoint['id']}")
    print(f"Created: {checkpoint['metadata']['created_at']}")
    print(f"Step: {checkpoint['metadata']['step']}")
```

## State Recovery

### Resume from Last Checkpoint

```python
# Graph was interrupted
config = {"configurable": {"thread_id": "user-123"}}

# Resume automatically from last checkpoint
result = app.invoke(None, config)  # Continues from where it left off
```

### Time Travel (Replay from Specific Point)

```python
# Get checkpoint history
checkpoints = list(checkpointer.list(config))

# Get specific checkpoint
specific_checkpoint = checkpoints[2]  # Third checkpoint

# Create new config pointing to that checkpoint
time_travel_config = {
    "configurable": {
        "thread_id": "user-123",
        "checkpoint_id": specific_checkpoint["id"]
    }
}

# Resume from that point
result = app.invoke(None, time_travel_config)
```

### Fork Conversation

```python
# Start from checkpoint but create new thread
original_config = {"configurable": {"thread_id": "conversation-1"}}
checkpoints = list(checkpointer.list(original_config))

# Get state at specific point
fork_point = checkpoints[5]

# Create new thread from that state
new_config = {"configurable": {"thread_id": "conversation-1-fork"}}
app.invoke(fork_point["checkpoint"]["state"], new_config)
```

## Checkpoint Metadata

Inspect metadata recorded with checkpoints:

```python
config = {"configurable": {"thread_id": "user-123"}}
checkpoints = list(checkpointer.list(config))
for cp in checkpoints:
    print(cp["metadata"])
```

## Checkpoint Cleanup

### Manual Cleanup

```python
# Backend-managed cleanup (example policy function)
def cleanup_old_checkpoints(days: int = 30):
    """Apply retention policy using backend-native tools."""
    import datetime

    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    print(f"Delete checkpoints older than: {cutoff.isoformat()}")
```

### Automatic Cleanup (Redis)

```python
# Use backend-native TTL/retention settings where supported
from langgraph.checkpoint.redis import RedisSaver

DB_URI = "redis://localhost:6379"
checkpointer = RedisSaver.from_conn_string(DB_URI)
# Apply TTL/retention using backend configuration or deployment settings
```

### Scheduled Cleanup

```python
from apscheduler.schedulers.background import BackgroundScheduler

def scheduled_cleanup():
    """Cleanup task."""
    cleanup_old_checkpoints(checkpointer, days=30)

scheduler = BackgroundScheduler()
scheduler.add_job(
    scheduled_cleanup,
    'cron',
    hour=2,  # Run at 2 AM daily
)
scheduler.start()
```

## Performance Optimization

### Batch Operations

```python
# Process multiple threads efficiently
threads = ["user-1", "user-2", "user-3"]

for thread_id in threads:
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"messages": [...]}, config)
```

### Connection Pooling

Connection pooling behavior is backend-dependent and managed by each checkpoint package internals.
Prefer documented constructors such as `from_conn_string` / `fromConnString` unless you have a backend-specific requirement.

### Caching

```python
# Cache frequently accessed checkpoints
from functools import lru_cache

class CachedCheckpointer:
    def __init__(self, base_checkpointer):
        self.base = base_checkpointer

    @lru_cache(maxsize=100)
    def get_tuple(self, thread_id: str):
        """Cached checkpoint tuple lookup."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.base.get_tuple(config)
```

## Migration Between Backends

### Export Checkpoints

```python
def export_checkpoints(source_checkpointer, thread_ids: list[str]) -> dict:
    """Export checkpoints from source."""
    exported = {}

    for thread_id in thread_ids:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = list(source_checkpointer.list(config))
        exported[thread_id] = checkpoints

    return exported
```

### Import Checkpoints

```python
def import_checkpoints(target_checkpointer, exported_data: dict):
    """Import checkpoints to target."""
    for thread_id, checkpoints in exported_data.items():
        config = {"configurable": {"thread_id": thread_id}}

        for checkpoint in checkpoints:
            target_checkpointer.put(
                config,
                checkpoint["checkpoint"],
                checkpoint["metadata"]
            )
```

### Full Migration

```python
def migrate_backend(
    source: BaseCheckpointSaver,
    target: BaseCheckpointSaver,
    thread_ids: list[str]
):
    """Migrate from one backend to another."""
    print(f"Migrating {len(thread_ids)} threads...")

    for i, thread_id in enumerate(thread_ids):
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = list(source.list(config))

        for checkpoint in checkpoints:
            target.put(
                config,
                checkpoint["checkpoint"],
                checkpoint["metadata"]
            )

        print(f"Migrated thread {i+1}/{len(thread_ids)}")

    print("Migration complete!")

# Example: SQLite to Postgres
sqlite_saver = SqliteSaver.from_conn_string("checkpoints.db")
postgres_saver = PostgresSaver.from_conn_string(DATABASE_URL)

migrate_backend(sqlite_saver, postgres_saver, ["user-1", "user-2"])
```

## Testing Persistence

### Unit Tests

```python
from langgraph.checkpoint.memory import InMemorySaver

def test_checkpoint_save_load():
    """Test checkpoint persistence."""
    checkpointer = InMemorySaver()

    # Create test state
    config = {"configurable": {"thread_id": "test-1"}}
    checkpoint = {
        "id": "checkpoint-1",
        "state": {"messages": ["Hello"]}
    }
    metadata = {"step": 1}

    # Save
    checkpointer.put(config, checkpoint, metadata)

    # Load
    loaded = checkpointer.get_tuple(config)

    assert loaded["checkpoint"]["state"] == checkpoint["state"]

def test_multiple_threads():
    """Test thread isolation."""
    checkpointer = InMemorySaver()

    config1 = {"configurable": {"thread_id": "thread-1"}}
    config2 = {"configurable": {"thread_id": "thread-2"}}

    # Save to different threads
    checkpointer.put(config1, {"id": "1", "state": {"data": "A"}}, {})
    checkpointer.put(config2, {"id": "2", "state": {"data": "B"}}, {})

    # Verify isolation
    loaded1 = checkpointer.get_tuple(config1)
    loaded2 = checkpointer.get_tuple(config2)

    assert loaded1["checkpoint"]["state"]["data"] == "A"
    assert loaded2["checkpoint"]["state"]["data"] == "B"
```

## Best Practices

1. **Choose the right backend**: Match backend to deployment architecture
2. **Use thread IDs strategically**: Design thread ID scheme for your use case
3. **Set up monitoring**: Track checkpoint size and query performance
4. **Plan for cleanup**: Implement checkpoint retention policies
5. **Test recovery**: Regularly test state restoration
6. **Secure credentials**: Use environment variables for connection strings
7. **Monitor costs**: Track storage and query costs in cloud deployments
8. **Handle failures**: Implement retry logic for backend operations

## Backend Comparison

| Feature | Memory | SQLite | PostgreSQL | Redis |
|---------|--------|--------|------------|-------|
| **Persistence** | No | Yes | Yes | Optional |
| **Multi-instance** | No | No | Yes | Yes |
| **Performance** | Fastest | Fast | Fast | Fastest |
| **Setup complexity** | None | Low | Medium | Medium |
| **Scalability** | Limited | Limited | High | High |
| **Cost** | Free | Free | $ | $$ |
| **Auto-cleanup** | N/A | Manual | Manual | TTL |
| **Best for** | Dev/test | Single instance | Production | High perf |

## Additional Resources

- LangGraph Persistence Documentation: https://docs.langchain.com/oss/python/langgraph/persistence
- Checkpointer API Reference: https://docs.langchain.com/oss/python/langgraph/persistence
- Add Memory (Persistence + Threads): https://docs.langchain.com/oss/python/langgraph/add-memory
