import os
import json

def calculate_avg_scores(folder_path):
    avg_score_all = 0
    i = 0
    for filename in os.listdir(folder_path):
        if filename.endswith("_scores.json"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                scores = [item['score'] for item in data if 'score' in item]
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    avg_score_all += avg_score
                    i += 1
                else:
                    avg_score = 0 
                
                print(f"File: {filename}, Average Score: {avg_score:.3f}")
            
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    avg_score_all = avg_score_all / i
    print(f'avg_score_all:{avg_score_all:.3f}')

folder_path = ""
calculate_avg_scores(folder_path)