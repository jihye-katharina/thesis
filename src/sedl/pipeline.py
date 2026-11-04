from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from data import Unit, Chunk
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
import json

@dataclass(frozen=True)
class Config:
  boundary_threshold: float = 0.80

def calculate_distances(embeddings):
  """ Calculate cosine distances between consecutive embeddings. """
  distances = []
  for i in range(len(embeddings) - 1):
    similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
    distance = 1 - similarity
    distances.append(distance)
  
  return distances

def create_chunks(units, breakpoints) -> list[Chunk]:
  """ Create initial text chunks based on identified breakpoints. """
  chunks = []
  start_idx = 0
  for bp in sorted(set(breakpoints)):
    unit_group = units[start_idx:bp+1]
    if unit_group:
      chunks.append(
          Chunk(id=f"c{len(chunks):04d}", units=tuple(unit_group))
      )
    start_idx = bp + 1

  # Add the final chunk
  unit_group = units[start_idx:]
  if unit_group:
    chunks.append(
        Chunk(id=f"c{len(chunks):04d}", units=tuple(unit_group))
    )

  return chunks

def add_context(units, window_size):
  """ Combine sentences with their neighbours for better context. """
  contextualised = []
  
  for i in range(len(units)):
    start = max(0, i - window_size)
    end = min(len(units), i + window_size + 1)
    context = ' '.join(u.text for u in units[start:end])
    contextualised.append(context)

  return contextualised

def identify_breakpoints(distances, threshold_percentile):
  """ Find natural breaking points in the text based on semantic distances. """
  threshold = np.percentile(distances, threshold_percentile)

  return [i for i, dist in enumerate(distances) if dist > threshold]

def evaluate_boundaries(chunks, units, annotation, tolerance):
  positions = {u.id: i for i, u in enumerate(units)}
  reference = sorted({
    positions[b["before_unit_id"]] for b in annotation["boundaries"]}
  )

  predicted = sorted({
    positions[ch.units[0].id] for ch in chunks
  })

  i = j = 0
  tp = 0

  while i < len(predicted) and j < len(reference):
    diff = predicted[i] - reference[j]
    if abs(diff) <= tolerance:
      tp += 1
      i += 1
      j += 1
    elif diff < 0: # Predictions is too early to match thihs reference.
      i += 1
    else: # Reference is too early to match this prediction.
      j += 1
    
  fp = len(predicted) - tp
  fn = len(reference) - tp
  """ precision = true_positives / (true_positives + false_positives)
      recall = true_positives / (true_positives + false_negatives)
      f1 = 2 * (precision * recall) / (precision + recall)
  """
  precision = tp / (tp + fp) if tp + fp else 0.0
  recall = tp / (tp + fn) if tp + fn else 0.0
  f1 = 2 * (precision * recall) / (precision + recall) if tp + fp + fn else 0.0

  return {
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "precision": precision,
    "recall": recall,
    "f1": f1
  }

def select_mmr(units, chunk_embeddings, max_num_units, lambda_weight):
  """ MMR = lambda * relevance - (1 - lambda) * redundancy
      relevance: the cosine similarity between a candidate and the reference
      redundancy: the highest cosine similarity between the candidate and the already selected
  """
  ref_chunk = chunk_embeddings.mean(axis=0) # Represent the overall meaning of the current chunk.
  relevance = cosine_similarity(chunk_embeddings, [ref_chunk])
  similarities = cosine_similarity(chunk_embeddings) # Similarity between every pair of units.

  selected = [] # Contains indices within this chunk, e.g. [1, 4].
  while len(selected) < min(max_num_units, len(units)):
    best_idx = None
    best_score = -np.inf
  
    for i in range(len(units)):
      if i in selected: continue
      
      redundancy = max(similarities[i, j] for j in selected) if selected else 0
      score = (lambda_weight * relevance[i] - (1 - lambda_weight) * redundancy)
      if score > best_score:
        best_score = score
        best_idx = i

    if best_idx is not None:
      selected.append(best_idx)

  return [units[i] for i in sorted(selected)]

def generate_bart(selected_units):
  """ Generate one summary from a chunk's MMR-selected evidence. """
  selected_text = "\n".join(u.text for u in selected_units)
  
  model_name = "facebook/bart-large-cnn"
  tokenizer = BartTokenizer.from_pretrained(model_name)
  model = BartForConditionalGeneration.from_pretrained(model_name).to("cpu")
  model.generation_config.forced_bos_token_id = tokenizer.bos_token_id

  inputs = tokenizer.encode(selected_text, return_tensors="pt", truncation=False)
  summary_ids = model.generate(inputs, min_length=0, num_beams=2, early_stopping=True)

  return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

def evaluate_summaries(chunk, summary, model, tokenizer):
  evaluation_data = {"source_text": chunk, "target_text": summary}
  """ Structure
      - system: for directives on how the model should act
      - user: for messages from the user
      - assistant: for messages from the model
  """
  messages = [
    {"role": "system", "content": "You will be given a summary generated from a lecture chunk."},
    {"role": "user", "content": "You will receive two fields:\n" +
                                "- source_text: the original lecture chunk, provided as context.\n" +
                                "- target_text: the generated summary to be evaluated.\n\n" +
                                "Assign a coherence score to target_text only." +
                                "Coherence (1–5): Assess whether the summary is logically organised, with clear connections between ideas and understandable references." +
                                "It should form a coherent explanation rather than a collection of loosely related statements." +
                                "Score: 1 = Disorganised and difficult to follow." +
                                "2 = Weak organisation with major connection problems." +
                                "3 = Generally understandable, but some connections are unclear." +
                                "4 = Well organised, with minor connection problems." +
                                "5 = Clearly structured and logically connected throughout." +
                                "Evaluation rules:" +
                                "- Focus on organisation, logical flow, transitions, and references." +
                                "- Do not reduce the score merely because source details are omitted." +
                                "- An omission affects coherence only when it makes a reference or connection within the summary unclear." +
                                "- Paraphrasing and different wording are allowed." +
                                "- Assess factual accuracy, completeness, and grammar separately; do not use them as substitutes for coherence." +
                                "- A complete single-sentence summary can be coherent." +
                                "- Copying the source does not automatically deserve a high score." +
                                "- Base your justification on specific features of target_text." +
                                "Return a score and a brief justification."                         
    }
  ]
  messages[1]["content"] += (
    "\n\nActual texts to evaluate:\n" +
    json.dumps(evaluation_data, ensure_ascii=False) +
    "\n\nEvaluate the supplied target_text. Do not invent a hypothetical summary." +
    "Return JSON only with an integer 'score' from 1 to 5 and a brief 'justification'."
  )

  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
  )
  inputs = tokenizer(text, return_tensors="pt").to(model.device) # Convert the formatted text into tensors.
  
  # Print the scores for each token generated with Greedy Search
  output = model.generate(
    **inputs, # Passes the token IDs and attention mask to the model.
    max_new_tokens=256, 
    do_sample=False, # Whether or not to use sampling. 
    num_beams=1 # Number of beams for beam search. 1 means no beam search.
  )

  response = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

  return response
