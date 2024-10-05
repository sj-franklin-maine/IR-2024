import json
import os
import argparse
from collections import defaultdict
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download

# Download stopwords if not already downloaded
# download('stopwords')
# download('punkt_tab')

# Pre-processing function to try and improve inverted index; lowercases, removes punc., tokenizes, removes stopwords
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Step 1a: Load JSON files
def load_topics(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        topics = json.load(file)

    return topics

# Step 1b: Load Answer file
def load_answers(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        answers = json.load(file)

    return answers

# Step 2: Build an inverted index for answers
def build_inverted_index(answers):
    index = defaultdict(set)
    
    for answer in answers:
        # Ensure each answer has the required fields
        if 'Id' not in answer or 'Text' not in answer:
            print(f"Warning: Answer missing 'ID' or 'Text': {answer}")
            continue
        
        answer_id = answer['Id']
        content = answer['Text']
        
        if not isinstance(content, str) or not content.strip():
            print(f"Warning: Answer ID {answer_id} has empty or invalid content.")
            continue
        
        # Preprocess content into tokens and add to index
        words = set(preprocess_text(content))
        for word in words:
            index[word].add(answer_id)
    
    # Debugging: Print the size and some content of the index
    # print(f"Inverted Index Size: {len(index)}")
    # print(f"Sample Index Content: {dict(list(index.items())[:5])}")  # Print first 5 entries
    
    return index


# Step 3a: Process Queries
def process_query(query, index):
    if 'Title' not in query or 'Tags' not in query:
        print(f"Warning: Query missing 'Title' or 'Tags': {query}")
        return []
    
    # Preprocess Title and Tags
    title_terms = preprocess_text(query['Title'])
    
    # Convert tags to a list if they are in string format and preprocess them
    tags = eval(query['Tags']) if isinstance(query['Tags'], str) else query['Tags']
    tag_terms = preprocess_text(' '.join(tags))  # Combine tags into a single string for preprocessing
    
    # Retrieve answers matching Title (AND logic)
    title_results = None
    for term in title_terms:
        if term in index:
            if title_results is None:
                title_results = index[term].copy()
            else:
                title_results = title_results.intersection(index[term])
        else:
            # If any title term is not in the index, no document can satisfy the AND condition
            title_results = set()
            break
    
    # Retrieve answers matching Tags (OR logic)
    tag_results = set()
    for term in tag_terms:
        if term in index:
            tag_results.update(index[term])
    
    # Combine Title AND results with Tag OR results
    if title_results is not None:
        combined_results = title_results.union(tag_results)
    else:
        combined_results = tag_results
    
    # Check if combined_results is empty and apply fallback if so
    if not combined_results:
        combined_results = fallback_process_query(query, index)
    
    return list(combined_results)  # Return results as a list

# Step 3b: Fallback for Processing Queries
def fallback_process_query(query, index):
    # Preprocess Title and Tags for fallback
    title_terms = preprocess_text(query['Title'])
    tags = eval(query['Tags']) if isinstance(query['Tags'], str) else query['Tags']
    tag_terms = preprocess_text(' '.join(tags))  # Combine tags into a single string for preprocessing

    # Initialize relevant_answers set
    relevant_answers = set()

    # Try to get results using OR logic for both title and tags
    for term in title_terms + tag_terms:
        if term in index:
            relevant_answers.update(index[term])
    
    return list(relevant_answers)  # Return fallback results as a list

# Step 4: Retrieve and Rank Answers
def retrieve_answers(topics, index, answers):
    q0 = "Q0" # Standard for TREC format
    results = []
    for topic in topics:
        qID = topic['Id']
        relevant_answers = process_query(topic, index)
        
        # Calculating rank and score 
        for rank, answer_id in enumerate(relevant_answers, start=1):
            if rank > 100:  # Only top 100
                break
            score = 100 - rank + 1  # Weight score based on rank (for evaluation tools)
            results.append((qID, q0, answer_id, rank, score, "boolean_run_3")) # Name the run!
            
    return results

# Step 5: Save results to TSV files
def save_results(results, file_path):
    with open(file_path, 'w') as file:
        for result in results:
            file.write("\t".join(map(str, result)) + "\n")


# Run everything here for neatness
def main(topic_file_1, topic_file_2, answers_file):
    # Load topics
    topics_1 = load_topics(topic_file_1)
    topics_2 = load_topics(topic_file_2)

    # Load answers
    answers = load_answers(answers_file)

    # Build inverted index from answers
    inverted_index = build_inverted_index(answers)
    
    # Retrieve answers for both topics
    results_1 = retrieve_answers(topics_1, inverted_index, answers)
    results_2 = retrieve_answers(topics_2, inverted_index, answers)

    # Save results to TSV files (this is what we pass to the evaluator)
    save_results(results_1, 'result_binary_1.tsv')
    save_results(results_2, 'result_binary_2.tsv')

# Main execution flow (for grabbing da arguments)
if __name__ == "__main__":
    # Allow user to specify filepath
    parser = argparse.ArgumentParser(description='Process some topic and answer files.')
    parser.add_argument('topic_file_1', type=str, help='Path to the first topic file (topics_1.json)')
    parser.add_argument('topic_file_2', type=str, help='Path to the second topic file (topics_2.json)')
    parser.add_argument('answers_file', type=str, help='Path to the answers file (Answers.json)')
    
    args = parser.parse_args()

    # Check if exactly 3 arguments were provided
    if len(vars(args)) != 3:
        print("Error: 3 arguments are required: topic_file_1, topic_file_2, and answers_file.")
        parser.print_usage()
        exit(1)

    main(args.topic_file_1, args.topic_file_2, args.answers_file)
