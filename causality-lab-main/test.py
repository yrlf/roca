import os
import openai
openai.api_key = "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"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)