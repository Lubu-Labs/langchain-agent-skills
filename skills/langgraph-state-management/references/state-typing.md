# State Typing and Validation

Type-safe state management is crucial for building reliable LangGraph applications. This guide covers TypedDict, Pydantic models, validation strategies, and type safety best practices.

> Note: `StateGraph` supports `TypedDict`, dataclass, and Pydantic state schemas. In contrast, LangChain `create_agent` custom state schemas must be `TypedDict`.

## TypedDict Basics

TypedDict provides type hints for dictionary-based state without runtime validation.

### Python TypedDict

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class BasicState(TypedDict):
    """Basic state schema with type hints."""
    query: str
    results: list[dict]
    status: Literal["pending", "processing", "complete"]
    count: int
```

**Key features:**
- Provides IDE autocomplete and type checking
- No runtime validation by default
- Lightweight (no extra dependencies)
- Works with mypy and other type checkers

### Required vs Optional Fields

```python
from typing import TypedDict, NotRequired, Required

class StateWithOptional(TypedDict, total=False):
    """All fields optional by default."""
    query: str
    results: list[dict]
    metadata: dict

class MixedState(TypedDict):
    """Mix required and optional fields."""
    query: Required[str]  # Required
    results: list[dict]  # Required (default)
    metadata: NotRequired[dict]  # Optional
```

### Nested TypedDict

```python
class UserInfo(TypedDict):
    """Nested user information."""
    user_id: str
    username: str
    email: str

class RequestMetadata(TypedDict):
    """Request metadata."""
    timestamp: float
    ip_address: str

class ComplexState(TypedDict):
    """State with nested structures."""
    user: UserInfo
    metadata: RequestMetadata
    messages: list[str]
```

## Pydantic Models

Pydantic provides runtime validation, serialization, and advanced features.

### Basic Pydantic State

```python
from pydantic import BaseModel, Field
from typing import Literal

class PydanticState(BaseModel):
    """Pydantic state with validation."""
    query: str = Field(..., min_length=1, max_length=500)
    results: list[dict] = Field(default_factory=list)
    status: Literal["pending", "processing", "complete"] = "pending"
    count: int = Field(default=0, ge=0)

    class Config:
        """Pydantic config."""
        validate_assignment = True  # Validate on attribute assignment
```

**Key features:**
- Runtime validation
- Default values with `Field`
- Validators and constraints
- JSON serialization
- Data parsing and coercion

### Field Validation

```python
from pydantic import BaseModel, Field, validator, field_validator

class ValidatedState(BaseModel):
    """State with field validation."""
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., ge=0, le=150)
    score: float = Field(..., ge=0.0, le=1.0)
    tags: list[str] = Field(..., min_items=1, max_items=10)

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Custom email validation."""
        if not v or '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower()

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate and normalize tags."""
        return [tag.strip().lower() for tag in v if tag.strip()]
```

### Model Validation

```python
from pydantic import BaseModel, model_validator

class CrossFieldState(BaseModel):
    """State with cross-field validation."""
    start_date: str
    end_date: str
    duration_days: int

    @model_validator(mode='after')
    def check_dates_consistent(self) -> 'CrossFieldState':
        """Validate dates are consistent."""
        from datetime import datetime

        start = datetime.fromisoformat(self.start_date)
        end = datetime.fromisoformat(self.end_date)
        actual_duration = (end - start).days

        if actual_duration != self.duration_days:
            raise ValueError(
                f'Duration mismatch: {actual_duration} vs {self.duration_days}'
            )

        return self
```

### Pydantic with LangGraph

```python
from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

# Option 1: Pydantic model converted to dict
class PydanticModel(BaseModel):
    """Pydantic model for validation."""
    query: str
    threshold: float = Field(ge=0.0, le=1.0)

class GraphState(MessagesState):
    """MessagesState for LangGraph."""
    config: dict  # Store Pydantic model as dict

def node_with_pydantic(state: GraphState) -> dict:
    """Node using Pydantic for validation."""
    # Parse and validate
    config = PydanticModel(**state["config"])

    # Use validated config
    print(f"Query: {config.query}")
    print(f"Threshold: {config.threshold}")

    return {}

# Option 2: Hybrid approach
class ValidatedConfig(BaseModel):
    """Validated configuration."""
    max_iterations: int = Field(ge=1, le=100)
    temperature: float = Field(ge=0.0, le=2.0)

class HybridState(MessagesState):
    """MessagesState with Pydantic field."""
    config: ValidatedConfig  # Use Pydantic directly

# Usage
state = HybridState(
    messages=[],
    config=ValidatedConfig(max_iterations=10, temperature=0.7)
)
```

## TypeScript Zod Validation

Zod provides runtime validation in TypeScript.

### Basic Zod Schema with StateSchema

```typescript
import { StateSchema } from "@langchain/langgraph";
import { z } from "zod/v4";

// Define state using StateSchema with Zod types directly
const AppState = new StateSchema({
  query: z.string().min(1).max(500),
  results: z.array(z.record(z.string(), z.any())).default([]),
  status: z.enum(["pending", "processing", "complete"]).default("pending"),
  count: z.number().int().nonnegative().default(0),
});

// Extract TypeScript types
type AppStateType = typeof AppState.State;
type AppStateUpdate = typeof AppState.Update;
```

### Validation in Nodes

```typescript
import { z } from "zod/v4";

const EmailSchema = z.string().email();
const AgeSchema = z.number().int().min(0).max(150);

function validateNode(state: any): Partial<any> {
  // Validate input
  const email = EmailSchema.parse(state.email);
  const age = AgeSchema.parse(state.age);

  // Process with validated data
  console.log(`Email: ${email}, Age: ${age}`);

  return {};
}
```

### Custom Validation

```typescript
import { z } from "zod/v4";

const DateValidationSchema = z.object({
  startDate: z.string(),
  endDate: z.string(),
  durationDays: z.number(),
}).refine(
  (data) => {
    const start = new Date(data.startDate);
    const end = new Date(data.endDate);
    const actualDuration = Math.floor(
      (end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)
    );
    return actualDuration === data.durationDays;
  },
  {
    message: "Duration must match the difference between dates",
  }
);
```

## Advanced Type Patterns

### Generic State

```python
from typing import Generic, TypeVar
from langgraph.graph import MessagesState

T = TypeVar('T')

class GenericState(MessagesState, Generic[T]):
    """Generic state for reusable patterns."""
    data: T
    status: str

# Use with specific types
StringState = GenericState[str]
DictState = GenericState[dict]
```

### Union Types

```python
from typing import TypedDict, Union, Literal

class TextResult(TypedDict):
    """Text result type."""
    type: Literal["text"]
    content: str

class ImageResult(TypedDict):
    """Image result type."""
    type: Literal["image"]
    url: str
    alt_text: str

class MultiModalState(TypedDict):
    """State with union types."""
    results: list[Union[TextResult, ImageResult]]

# Type-safe access
def process_results(state: MultiModalState):
    for result in state["results"]:
        if result["type"] == "text":
            # Type checker knows this is TextResult
            print(result["content"])
        elif result["type"] == "image":
            # Type checker knows this is ImageResult
            print(result["url"], result["alt_text"])
```

### Tagged Unions

```python
from typing import TypedDict, Literal, Union

class SuccessResult(TypedDict):
    """Success result."""
    status: Literal["success"]
    data: dict

class ErrorResult(TypedDict):
    """Error result."""
    status: Literal["error"]
    error_message: str
    error_code: int

Result = Union[SuccessResult, ErrorResult]

class ResultState(TypedDict):
    """State with tagged union."""
    result: Result

def handle_result(state: ResultState):
    """Type-safe result handling."""
    result = state["result"]

    if result["status"] == "success":
        # TypedDict narrows to SuccessResult
        print(result["data"])
    else:
        # TypedDict narrows to ErrorResult
        print(f"Error {result['error_code']}: {result['error_message']}")
```

## State Validation Strategies

### Input Validation

```python
from typing import TypedDict
from pydantic import BaseModel, Field, ValidationError

class InputModel(BaseModel):
    """Validate input."""
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=10, ge=1, le=100)

class GraphState(TypedDict):
    """Graph state."""
    query: str
    max_results: int
    results: list[dict]

def validate_input(raw_input: dict) -> GraphState:
    """Validate and convert input."""
    try:
        validated = InputModel(**raw_input)
        return {
            "query": validated.query,
            "max_results": validated.max_results,
            "results": []
        }
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}")

# Usage
try:
    state = validate_input({"query": "test", "max_results": 5})
except ValueError as e:
    print(f"Validation failed: {e}")
```

### Runtime State Validation

```python
from langgraph.graph import MessagesState

class ValidatedState(MessagesState):
    """State with runtime validation."""
    iteration: int
    max_iterations: int

def validate_state(state: ValidatedState) -> None:
    """Runtime state validation."""
    if state["iteration"] > state["max_iterations"]:
        raise ValueError(
            f"Iteration {state['iteration']} exceeds max {state['max_iterations']}"
        )

    if not state["messages"]:
        raise ValueError("Messages cannot be empty")

def safe_node(state: ValidatedState) -> dict:
    """Node with state validation."""
    validate_state(state)

    # Process with validated state
    return {"iteration": state["iteration"] + 1}
```

### Conditional Validation

```python
from pydantic import BaseModel, field_validator

class ConditionalState(BaseModel):
    """State with conditional validation."""
    mode: Literal["development", "production"]
    api_key: str | None = None

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v, info):
        """Validate API key in production."""
        if info.data.get('mode') == 'production' and not v:
            raise ValueError('API key required in production mode')
        return v
```

## Type Safety Best Practices

### 1. Use Literal for Enums

```python
# ✅ Good: Type-safe with Literal
from typing import Literal

class GoodState(TypedDict):
    status: Literal["pending", "processing", "complete"]

# ❌ Bad: Any string accepted
class BadState(TypedDict):
    status: str
```

### 2. Annotate Lists and Dicts

```python
# ✅ Good: Specific types
class GoodState(TypedDict):
    results: list[dict[str, Any]]
    metadata: dict[str, str]

# ❌ Bad: Untyped
class BadState(TypedDict):
    results: list
    metadata: dict
```

### 3. Use NotRequired for Optional Fields

```python
from typing import TypedDict, NotRequired

# ✅ Good: Clear optional fields
class GoodState(TypedDict):
    query: str  # Required
    results: NotRequired[list[dict]]  # Optional

# ❌ Bad: Ambiguous
class BadState(TypedDict):
    query: str
    results: list[dict] | None  # Nullable but still required
```

### 4. Document State Fields

```python
class DocumentedState(TypedDict):
    """Well-documented state.

    Attributes:
        query: User's search query
        results: Search results from API
        status: Current processing status
        timestamp: When the query was received
    """
    query: str
    results: list[dict]
    status: Literal["pending", "processing", "complete"]
    timestamp: float
```

### 5. Validate Early

```python
def node_with_validation(state: MyState) -> dict:
    """Validate at node entry."""
    # Validate immediately
    if not state["query"]:
        raise ValueError("Query cannot be empty")

    if state["iteration"] > 100:
        raise ValueError("Too many iterations")

    # Continue with valid state
    ...
```

## Testing Type Safety

### MyPy Configuration

```ini
# mypy.ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_generics = True
check_untyped_defs = True
```

### Type Checking Tests

```python
from typing import TypedDict

class TestState(TypedDict):
    value: str
    count: int

def test_type_safety():
    """Test type safety with mypy."""
    # This should pass mypy
    state: TestState = {"value": "test", "count": 1}

    # This should fail mypy
    # state: TestState = {"value": 123, "count": "wrong"}  # Type error

def test_required_fields():
    """Test required fields."""
    # Missing required field - mypy error
    # state: TestState = {"value": "test"}  # Error: missing 'count'

    # All required fields present - OK
    state: TestState = {"value": "test", "count": 1}
    assert state["count"] == 1
```

### Runtime Validation Tests

```python
from pydantic import BaseModel, ValidationError
import pytest

class ValidatedModel(BaseModel):
    value: str
    count: int

def test_pydantic_validation():
    """Test Pydantic runtime validation."""
    # Valid input
    model = ValidatedModel(value="test", count=1)
    assert model.count == 1

    # Invalid input
    with pytest.raises(ValidationError):
        ValidatedModel(value=123, count="wrong")

    # Missing field
    with pytest.raises(ValidationError):
        ValidatedModel(value="test")
```

## Common Type Issues

### Issue: Any Type

```python
from typing import Any

# ❌ Bad: Loses type safety
class BadState(TypedDict):
    data: Any  # Could be anything

# ✅ Good: Specific type
class GoodState(TypedDict):
    data: dict[str, str] | list[int]  # Clear alternatives
```

### Issue: Mutable Defaults

```python
# ❌ Bad: Mutable default
class BadState(TypedDict):
    items: list[str] = []  # Shared between instances!

# ✅ Good: Use default_factory with Pydantic
from pydantic import BaseModel, Field

class GoodState(BaseModel):
    items: list[str] = Field(default_factory=list)
```

### Issue: Type Coercion

```python
from pydantic import BaseModel

class State(BaseModel):
    count: int

# Pydantic coerces types
state = State(count="123")  # Converts to int(123)

# To prevent coercion
class StrictState(BaseModel):
    count: int

    class Config:
        strict = True  # No coercion

# Now this raises ValidationError
# state = StrictState(count="123")
```

## Performance Considerations

### TypedDict vs Pydantic

```python
import timeit

# TypedDict (no validation)
from typing import TypedDict

class TypedDictState(TypedDict):
    value: str
    count: int

# Pydantic (with validation)
from pydantic import BaseModel

class PydanticState(BaseModel):
    value: str
    count: int

# Benchmark
typeddict_time = timeit.timeit(
    lambda: {"value": "test", "count": 1},
    number=100000
)

pydantic_time = timeit.timeit(
    lambda: PydanticState(value="test", count=1),
    number=100000
)

# TypedDict is ~10x faster (no validation overhead)
# Use TypedDict for performance-critical paths
# Use Pydantic for validation-critical paths
```

### Validation Caching

```python
from functools import lru_cache
from pydantic import BaseModel

class Config(BaseModel):
    setting1: str
    setting2: int

@lru_cache(maxsize=100)
def get_validated_config(setting1: str, setting2: int) -> Config:
    """Cache validated configs."""
    return Config(setting1=setting1, setting2=setting2)

# Repeated calls use cache
config1 = get_validated_config("test", 1)
config2 = get_validated_config("test", 1)  # From cache
assert config1 is config2
```

## Migration Path

### From Untyped to TypedDict

```python
# Before: Untyped dict
def old_node(state: dict) -> dict:
    query = state["query"]  # No type checking
    return {"results": []}

# After: TypedDict
class State(TypedDict):
    query: str
    results: list[dict]

def new_node(state: State) -> dict:
    query = state["query"]  # Type checked!
    return {"results": []}
```

### From TypedDict to Pydantic

```python
# Before: TypedDict
class OldState(TypedDict):
    email: str
    age: int

# After: Pydantic with validation
from pydantic import BaseModel, Field

class NewState(BaseModel):
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., ge=0, le=150)
```

## Additional Resources

- Pydantic Documentation: https://docs.pydantic.dev/
- Zod Documentation: https://zod.dev/
- MyPy Documentation: https://mypy.readthedocs.io/
- LangGraph State Documentation: https://docs.langchain.com/oss/python/langgraph/graph-api
