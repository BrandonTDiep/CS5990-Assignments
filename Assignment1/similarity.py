# -------------------------------------------------------------------------
# AUTHOR: Brandon Diep
# FILENAME: similarity.py
# SPECIFICATION: This program calculates the cosine similarity between docs and outputs the docs that are the closest
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 4 hrs
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here
docTermMatrix = []
terms = []

# get the text from each document and store the terms in an array
for row in documents:
   terms.extend(row[1].split())

# given the terms array, make sure there's no duplicates
distinctTerms = sorted(list(set(terms)))

# for each doc, check if the document's term exists (1) or doesn't exist (0)
for doc in documents:
   docRow = []

   for distinctTerm in distinctTerms:
      if distinctTerm in doc[1].split():
         docRow.append(1)
      else:
         docRow.append(0)
   docTermMatrix.append(docRow)


# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here
maxSimiliarity = -2
similarDocs = ()
for i in range(len(docTermMatrix)):
   for j in range(i+1, len(docTermMatrix)):
      similarity = cosine_similarity([docTermMatrix[i]], [docTermMatrix[j]])[0][0]
      
      if similarity > maxSimiliarity:
         maxSimiliarity = similarity
         similarDocs = (i + 1, j + 1)


# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
print(f"The most similar documents are document {similarDocs[0]} and document {similarDocs[1]} with cosine similarity = {maxSimiliarity}")
