How to add AI models

1. Create a subdirectory under `model/` for each model.
2. Implement a `predict(input_data)` function in a `model.py` file.
3. If the model requires weights or extra files, add them to the model's folder and load them in `predict` or module init.
4. If the model needs extra Python packages, list them in `model/<name>/requirements.txt` and install them.

Example:

model/my_model/

- model.py
- weights.bin
- requirements.txt

Example `model.py`:

```python
def predict(input_data):
    # parse input and run inference
    return {"status": "ok", "result": "..."}
```
