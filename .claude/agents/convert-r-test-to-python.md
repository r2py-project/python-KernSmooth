---
name: convert-r-test-to-python
description: Converts a specified R test file to its Python equivalent using provided library contexts.
---

# Convert an R Test File to Python

## Description

Translate a provided R test script into an equivalent Python test script. To ensure accurate translation, you must rely on the provided paths to the target R test file, the source R library folder, and the corresponding Python library folder. You must maintain the exact test coverage, logic, and assertions of the original R code while adapting to standard Python testing conventions (e.g., using `pytest`) if necessary.

## Execution Steps

### Step 1: Analyze Dependencies and Context

- Read the provided R library directory to understand the function signatures, data structures, and logic being tested in the R script.
- Read the provided Python library directory to identify the equivalent Python functions, classes, and modules that the new test file will need to import.

### Step 2: Resolve Datasets and Fixtures

- Scan the R test file for any required datasets or external files.
- Look for these files locally first. If they require downloading from an external source, prompt the user for the correct URL or source if it is not explicitly defined in the R script.
- Ensure the translated Python code uses appropriate libraries (e.g., `pandas` or built-in `csv`) to load these datasets correctly.

### Step 3: Translate the Test Logic

- Translate the R test file into Python syntax. Do not change any logic in the file; the workflow should remain exactly identical. Any output from the R test file should be exactly the same as the output from the Python test file, except for differences in formatting that are inherent to the language (e.g., how lists or data frames are printed).
- Map R testing framework functions (like those from `testthat`) to their exact Python equivalents (e.g., standard `assert` statements for `pytest` or `unittest.TestCase` methods), if applicable.
- Pay strict attention to handling 1-based indexing in R versus 0-based indexing in Python, as well as differences in how NA/NaN values are treated.

### Step 4: Save the Python Test File

- Check if the user specified an output folder in their prompt.
- If an output folder is specified, save the translated Python test file there.
- If no output folder is specified, create a `tests/` directory inside the provided Python library folder (if it doesn't already exist) and save the file there. The file should be named identically to the original R test file but with a `.py` extension (e.g., `test-example.R` becomes `test_example.py`).