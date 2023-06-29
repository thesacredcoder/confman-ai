from flask import Flask, request, jsonify

from flask_restful import Resource, Api
from keybert import KeyBERT, _mmr
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

app = Flask(__name__)
api = Api(app)


def extract_keywords(abstract, ngram_range=(1, 2), num_keywords=5):
    kw_model = KeyBERT("bert-base-nli-mean-tokens")
    doc_embedding = kw_model.model.embed(abstract)
    keywords = kw_model.extract_keywords(
        abstract, keyphrase_ngram_range=ngram_range, top_n=num_keywords
    )

    return keywords


def calculate_semantic_similarity(
    keywords1, keywords2, model_name="sentence-transformers/bert-base-nli-mean-tokens"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embeddings(keywords):
        if not isinstance(keywords, list):  # check if the input is a list
            keywords = [keywords]  # if not, convert it to a list
        if not all(
            isinstance(kw, str) for kw in keywords
        ):  # check if all elements are strings
            print("Invalid input:", keywords)  # print the invalid input
            return []  # return an empty list to avoid errors
        encoded_input = tokenizer(
            keywords, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = model(**encoded_input).last_hidden_state.mean(dim=1).numpy()
        return embeddings

    keywords1 = [kw[0] if isinstance(kw, tuple) else kw for kw in keywords1]
    keywords2 = [kw[0] if isinstance(kw, tuple) else kw for kw in keywords2]

    embeddings1 = get_embeddings(keywords1)
    embeddings2 = get_embeddings(keywords2)

    similarity_scores = []
    for emb1 in embeddings1:
        for emb2 in embeddings2:
            similarity_scores.append(1 - cosine(emb1, emb2))

    if not similarity_scores:
        return 0

    return sum(similarity_scores) / len(similarity_scores)


def map_reviewer_to_paper(paper, reviewers, similarity_threshold=0.0):
    extracted_keywords = extract_keywords(paper["abstract"])
    paper_keywords = [kw["word"] for kw in paper.get("keywords", [])]
    all_keywords = set(extracted_keywords).union(set(paper_keywords))

    best_match_score = -1
    best_match_reviewer = None
    for reviewer in reviewers:
        reviewer_expertises = [
            expertise["name"] for expertise in reviewer["expertises"]
        ]
        similarity_score = calculate_semantic_similarity(
            all_keywords, reviewer_expertises
        )
        print(similarity_score)
        if similarity_score >= best_match_score:
            best_match_score = similarity_score
            best_match_reviewer = reviewer

    # return best_match_reviewer

    if best_match_score >= similarity_threshold:
        return best_match_reviewer
    else:
        return None


class ReviewerMapping(Resource):
    def post(self):
        data = request.get_json()
        paper = data["paper"]
        reviewers = data["reviewers"]
        best_matching_reviewer = map_reviewer_to_paper(paper, reviewers)
        return jsonify(best_matching_reviewer)


api.add_resource(ReviewerMapping, "/reviewer-mapping")

if __name__ == "__main__":
    app.run(debug=True)
