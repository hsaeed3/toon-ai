# toon-ai

Simple `litellm` client wrapper that utilizes various schema utilities from the `instructor` library to generate structured outputs with LLMs using the TOON (Token Oriented Object Notation) format.

## Installation

Install from source:

```bash
uv add git+https://github.com/hsaeed3/toon-ai.git
```

## Usage

### QuickStart

```python
from toon_ai import generate

generate(
    messages="hello! what is 45+45",
    output_type=int,
)
"""
40
"""
```

### Standard Usage (Client)

```python
from toon_ai import ToonClient


# currently only supports the `text` strategy
# but strategy 'tools' will be supported soon
client = ToonClient(
    strategy="text",
)


# generate your structured outputs!
response = client.create(
    # send messages as either strings or OpenAI Chat Completions messages
    messages = "hello! what is 45+45",

    # specify the output type, this can be any of:
    # - primitive types (int, float, bool, str)
    # - Pydantic models / TypedDicts
    output_type = int,
)
"""
4
"""
```

### Streaming Usage

> [!NOTE]
> As this implementation was designed specifically to be used within the `instructor` package it utilizes `instructor`'s `Partial` class to handle streaming responses. A `Partial` class is a Pydantic model with all fields set to optional, allowing for partial iteration of the model's fields as content comes in.

```python
from toon_ai import ToonClient
from pydantic import BaseModel


class User(BaseModel):
    name : str
    age : int


client = ToonClient()


response = client.generate(
    messages=[{"role": "user", "content": "John is 25 years old."}],
    output_type=User,
    stream=True
)

for chunk in response:
    print(chunk)
"""
name=None age=None
name='John' age=None
name='John' age=25
"""
```
