# Boolean Retrieval Model

## Overview
This project implements a Boolean retrieval model in Python. The goal is to retrieve relevant answers to travel-related queries from a given collection of answers and associated topics. The system builds an inverted index from the answer data and processes user queries based on the titles and tags provided in provided topics JSON files.

## Requirements
- Python 3.x
- NLTK (Natural Language Toolkit)
  
### NLTK Dependencies
To use this code, ensure that the NLTK stopwords and tokenizer are downloaded:
```python
# Uncomment these lines to download required NLTK data
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
```

## File Structure
- `Answers.json`: Contains the answers with properties such as ID, Text, and Score.
- `topics_1.json` & `topics_2.json`: JSON files containing search queries with attributes like ID, Title, Body, and Tags.
- `result_binary_1.tsv`: Output file for the first set of topics.
- `result_binary_2.tsv`: Output file for the second set of topics.

## Functions

### `preprocess_text(text)`
Preprocesses the text by:
- Lowercasing
- Removing punctuation
- Tokenizing
- Removing stopwords

### `load_topics(file_path)`
Loads and returns a list of topics from a JSON file.

### `load_answers(file_path)`
Loads and returns a list of answers from a JSON file.

### `build_inverted_index(answers)`
Builds and returns an inverted index for the answers.

### `process_query(query, index)`
Processes a query and retrieves relevant answers based on the inverted index.

### `fallback_process_query(query, index)`
Fallback method for processing queries when no results are found, utilizing OR logic for both title and tags.

### `retrieve_answers(topics, index, answers)`
Retrieves and ranks answers for given topics.

### `save_results(results, file_path)`
Saves the retrieval results to a TSV file for evaluation.

## Usage
To run the model, execute the following command in your terminal:

```bash
python your_script.py <path_to_topics_1.json> <path_to_topics_2.json> <path_to_answers.json>
```

Replace `<path_to_topics_1.json>`, `<path_to_topics_2.json>`, and `<path_to_answers.json>` with the actual paths to your JSON files.

## Example
```bash
python model.py topics_1.json topics_2.json Answers.json
```

This command will generate two TSV files (`result_binary_1.tsv` and `result_binary_2.tsv`) containing the top 100 relevant answers for each query topic.

## Limitations
- The model primarily relies on the presence of exact terms in the answers.
- It may not account for variations or synonyms of keywords.