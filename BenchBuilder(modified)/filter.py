import os
import orjson
import json
import argparse
from typing import List, Dict, Optional
import numpy as np
import wandb

def load_json(file_path: str) -> List[Dict]:
    """Load JSON file with detailed logging"""
    print(f"Loading JSON file: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
    
    with open(file_path, 'rb') as f:
        data = orjson.loads(f.read())
    
    if isinstance(data, list):
        print(f"Successfully loaded {len(data)} items from JSON array")
        return data
    else:
        print(f"Loaded single JSON object, converting to list")
        return [data]

def load_topics_file(topics_file: str) -> tuple[Dict[str, str], Dict[str, Dict]]:
    """Load topics file with multiple format support, return both names and full details"""
    cluster_number_to_name = {}
    cluster_number_to_details = {}
    
    if not topics_file or not os.path.exists(topics_file):
        print(f"Topics file not found or not specified: {topics_file}")
        return cluster_number_to_name, cluster_number_to_details
    
    try:
        # Handle CSV format
        if topics_file.endswith('.csv'):
            print(f"Loading topics from CSV: {topics_file}")
            import pandas as pd
            import ast
            
            df = pd.read_csv(topics_file, header=None)
            print(f"CSV shape: {df.shape}")
            
            for index, row in df.iterrows():
                try:
                    if len(row) >= 3:
                        cluster_id = str(row[1])
                        cluster_name = str(row[2])
                        
                        cluster_number_to_name[cluster_id] = cluster_name
                        
                        contents = []
                        for col_idx in range(3, len(row)):
                            if pd.notna(row[col_idx]) and str(row[col_idx]).strip():
                                try:
                                    content_list = ast.literal_eval(str(row[col_idx]))
                                    if isinstance(content_list, list):
                                        contents.append(content_list)
                                except (ValueError, SyntaxError):
                                    contents.append([str(row[col_idx])])
                        
                        cluster_number_to_details[cluster_id] = {
                            'name': cluster_name,
                            'contents': contents,
                            'row_id': row[0] if len(row) > 0 else index
                        }
                        
                except Exception as e:
                    print(f"Warning: Error processing row {index}: {e}")
                    continue
                    
        # Handle JSON format
        else:
            topics_map = load_json(topics_file)
            if isinstance(topics_map, list) and len(topics_map) > 0:
                topics_map = topics_map[0]
                
            if isinstance(topics_map, dict):
                if "topic_aspects" in topics_map and "OpenAI" in topics_map["topic_aspects"]:
                    for cluster_number, cluster_obj in topics_map["topic_aspects"]["OpenAI"].items():
                        cluster_number_to_name[cluster_number] = cluster_obj[0][0]
                        cluster_number_to_details[cluster_number] = {
                            'name': cluster_obj[0][0],
                            'contents': cluster_obj,
                            'row_id': cluster_number
                        }
                else:
                    cluster_number_to_name = topics_map
                    for cluster_id, name in topics_map.items():
                        cluster_number_to_details[cluster_id] = {
                            'name': name,
                            'contents': [],
                            'row_id': cluster_id
                        }
                    
        print(f"Loaded {len(cluster_number_to_name)} cluster name mappings")
        print(f"Loaded {len(cluster_number_to_details)} cluster detail mappings")
        
    except Exception as e:
        print(f"Warning: Could not load topics file {topics_file}: {e}")
        print("Will use cluster numbers as names")
    
    return cluster_number_to_name, cluster_number_to_details

def extract_content_from_conversation(conv: Dict) -> str:
    """Extract the actual content from the conversation"""
    try:
        if "post_process_conv" in conv and conv["post_process_conv"]:
            return conv["post_process_conv"]
        
        if "conversation_a" in conv and isinstance(conv["conversation_a"], list):
            for turn in conv["conversation_a"]:
                if isinstance(turn, dict) and "content" in turn:
                    if turn.get("role") == "user":
                        return turn["content"]
            if conv["conversation_a"] and "content" in conv["conversation_a"][0]:
                return conv["conversation_a"][0]["content"]
        
        if "contents" in conv:
            contents_str = conv["contents"]
            if isinstance(contents_str, str):
                try:
                    contents_data = json.loads(contents_str)
                    if isinstance(contents_data, list) and len(contents_data) > 0:
                        first_item = contents_data[0]
                        if isinstance(first_item, dict) and "content" in first_item:
                            return first_item["content"]
                except json.JSONDecodeError:
                    pass
            return str(contents_str)
        
        return ""
    
    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Error extracting content from conversation: {e}")
        return ""

def calculate_score(conversation: Dict) -> int:
    """Calculate score based on criteria_v0.1"""
    criteria = conversation.get('category_tag', {}).get('criteria_v0.1', {})
    return sum(1 for value in criteria.values() if value)

def calculate_cluster_scores(conversations: List[Dict], clusters: List) -> Dict:
    """Calculate average score for each cluster"""
    cluster_scores = {}
    for conv, cluster in zip(conversations, clusters):
        cluster = str(cluster)
        score = calculate_score(conv)
        if cluster not in cluster_scores:
            cluster_scores[cluster] = []
        cluster_scores[cluster].append(score)
    
    cluster_to_mean_score = {cluster: np.mean(scores) for cluster, scores in cluster_scores.items()}
    
    print(f"\nCluster Statistics:")
    try:
        sorted_items = sorted(cluster_to_mean_score.items(), key=lambda x: int(x[0]))
    except ValueError:
        sorted_items = sorted(cluster_to_mean_score.items(), key=lambda x: str(x[0]))
    
    for cluster, mean_score in sorted_items:
        count = len(cluster_scores[cluster])
        print(f"  Cluster {cluster}: {mean_score:.2f} avg score, {count} items")
    
    return cluster_to_mean_score

def group_conversations_by_cluster(conversations: List[Dict], clusters: List) -> Dict[str, List[tuple]]:
    """Group conversations by cluster with their scores and indices"""
    cluster_to_conversations = {}
    
    for i, (conv, cluster) in enumerate(zip(conversations, clusters)):
        cluster_str = str(cluster)
        score = calculate_score(conv)
        
        if cluster_str not in cluster_to_conversations:
            cluster_to_conversations[cluster_str] = []
        
        cluster_to_conversations[cluster_str].append((conv, score, i))
    
    return cluster_to_conversations

def filter_prompts_hierarchical(conversations: List[Dict], clusters: List, cluster_threshold: float, 
                               top_prompts_per_cluster: int, cluster_number_to_name: Dict[str, str], 
                               cluster_number_to_details: Dict[str, Dict]) -> List[Dict]:
    """Hierarchical filtering: first select clusters, then top N prompts from each cluster"""
    
    if len(conversations) != len(clusters):
        print(f"WARNING: Length mismatch!")
        print(f"  Conversations: {len(conversations)}")
        print(f"  Clusters: {len(clusters)}")
        
        if len(conversations) > len(clusters):
            print(f"  Truncating conversations to match clusters length")
            conversations = conversations[:len(clusters)]
        else:
            print(f"  Padding clusters with cluster 0 for extra conversations")
            clusters = clusters + [0] * (len(conversations) - len(clusters))
    
    # Calculate cluster scores
    cluster_scores = calculate_cluster_scores(conversations, clusters)
    
    # Select high-scoring clusters
    selected_clusters = []
    filtered_clusters = []
    
    for cluster_id, score in cluster_scores.items():
        if score >= cluster_threshold:
            selected_clusters.append(cluster_id)
        else:
            filtered_clusters.append(cluster_id)
    
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTER SELECTION")
    print("="*60)
    print(f"Cluster Threshold: {cluster_threshold}")
    print(f"Selected Clusters: {len(selected_clusters)}")
    print(f"Filtered Clusters: {len(filtered_clusters)}")
    print(f"Selection Rate: {len(selected_clusters)/(len(selected_clusters)+len(filtered_clusters))*100:.1f}%")
    print(f"Top Prompts per Cluster: {top_prompts_per_cluster}")
    
    # Print selected cluster IDs
    if selected_clusters:
        try:
            selected_cluster_ids_sorted = sorted(selected_clusters, key=lambda x: int(x))
        except ValueError:
            selected_cluster_ids_sorted = sorted(selected_clusters)
            
        print(f"\nSELECTED CLUSTER IDs ({len(selected_cluster_ids_sorted)} clusters):")
        clusters_per_line = 10
        for i in range(0, len(selected_cluster_ids_sorted), clusters_per_line):
            line_clusters = selected_cluster_ids_sorted[i:i+clusters_per_line]
            print(f"   {', '.join(f'{cluster_id:>3}' for cluster_id in line_clusters)}")
    
    # Group conversations by cluster
    cluster_to_conversations = group_conversations_by_cluster(conversations, clusters)
    
    # Select top N prompts from each selected cluster
    filtered_prompts = []
    total_selected_prompts = 0
    
    print(f"\nDETAILED CLUSTER ANALYSIS:")
    print("="*60)
    
    selected_clusters_with_scores = [(cid, cluster_scores[cid]) for cid in selected_clusters]
    selected_clusters_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (cluster_id, cluster_score) in enumerate(selected_clusters_with_scores, 1):
        cluster_name = cluster_number_to_name.get(cluster_id, f"Cluster_{cluster_id}")
        cluster_details = cluster_number_to_details.get(cluster_id, {})
        cluster_conversations = cluster_to_conversations.get(cluster_id, [])
        
        # Sort conversations in this cluster by score (descending)
        cluster_conversations.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N prompts from this cluster
        selected_from_cluster = cluster_conversations[:top_prompts_per_cluster]
        actual_selected = len(selected_from_cluster)
        total_selected_prompts += actual_selected
        
        print(f"\n{rank:2d}. Cluster {cluster_id:>3} | Score: {cluster_score:5.2f} | {cluster_name}")
        print(f"    Available: {len(cluster_conversations):3d} prompts | Selected: {actual_selected:2d} prompts")
        
        if actual_selected < top_prompts_per_cluster:
            print(f"    WARNING: Only {actual_selected} prompts available (requested {top_prompts_per_cluster})")
        
        if selected_from_cluster:
            selected_scores = [item[1] for item in selected_from_cluster]
            print(f"    Selected scores: {selected_scores[0]:2d} (best) -> {selected_scores[-1]:2d} (worst) | Avg: {np.mean(selected_scores):.1f}")
        
        # Show topic contents if available
        if cluster_details.get('contents'):
            print(f"    Topic Contents:")
            contents = cluster_details['contents']
            for content_idx, content_list in enumerate(contents[:1]):
                if isinstance(content_list, list) and content_list:
                    for item_idx, item in enumerate(content_list[:2]):
                        print(f"        - {item}")
                    if len(content_list) > 2:
                        print(f"        - ... (and {len(content_list) - 2} more items)")
        
        # Add selected conversations to filtered_prompts with metadata
        for conv, score, original_idx in selected_from_cluster:
            conv_copy = conv.copy()
            conv_copy.update({
                "prompt_score": score,
                "cluster_score": cluster_score,
                "cluster_id": cluster_id,
                "cluster_rank_in_selection": rank,
                "prompt_rank_in_cluster": selected_from_cluster.index((conv, score, original_idx)) + 1,
                "original_index": original_idx,
            })
            filtered_prompts.append(conv_copy)
        
        print("    " + "-" * 50)
    
    print("\nHIERARCHICAL FILTERING SUMMARY:")
    print("="*60)
    print(f"  Total conversations: {len(conversations):,}")
    print(f"  Total clusters: {len(cluster_scores):,}")
    print(f"  Selected clusters: {len(selected_clusters):,}")
    print(f"  Max prompts per cluster: {top_prompts_per_cluster:,}")
    print(f"  Total selected prompts: {total_selected_prompts:,}")
    print(f"  Overall retention rate: {total_selected_prompts/len(conversations)*100:.1f}%")
    print(f"  Avg prompts per selected cluster: {total_selected_prompts/len(selected_clusters):.1f}")
    
    return filtered_prompts

def to_arena_hard_questions_format(conversations: List[Dict], cluster_number_to_name: Dict[str, str]) -> List[Dict]:
    """Convert to arena-hard questions format"""
    
    arena_hard_questions = []
    
    for i, conv in enumerate(conversations):
        content = extract_content_from_conversation(conv)
        
        if not content:
            print(f"Warning: No valid content found for conversation {i}")
            continue
        
        cluster_id = conv.get('cluster_id', 0)
        cluster_name = f"Cluster_{cluster_id}"
        
        turns_list = []
        if "conversation_a" in conv and isinstance(conv["conversation_a"], list):
            for turn in conv["conversation_a"]:
                if isinstance(turn, dict) and "content" in turn:
                    turns_list.append({
                        "role": turn.get("role", "user"),
                        "content": turn["content"]
                    })
        else:
            turns_list = [{"content": content}]

        arena_hard_questions.append({
            "question_id": str(conv.get("question_id", conv.get("id", i))),
            "category": conv.get("original_category", "arena-hard-v0.1"),
            "cluster": cluster_name,
            "turns": turns_list,
            "language": conv.get("language", "English"),
            "original_data": {
                "num": conv.get("num", i),
                "id": conv.get("id", i),
                "question_id": conv.get("question_id", conv.get("id", i)),
                "tstamp": conv.get("tstamp", ""),
                "round_id": conv.get("round_id", 0),
                "roundprompt_id": conv.get("roundprompt_id", 0),
                "kind": conv.get("kind", "nlp"),
                "model_a": conv.get("model_a", ""),
                "model_b": conv.get("model_b", ""),
                "prompt_score": conv.get("prompt_score", 0),
                "cluster_score": conv.get("cluster_score", 0),
                "cluster_id": conv.get("cluster_id", 0),
                "cluster_rank_in_selection": conv.get("cluster_rank_in_selection", 0),
                "prompt_rank_in_cluster": conv.get("prompt_rank_in_cluster", 0),
                "voted_at": conv.get("voted_at", ""),
                "original_cluster": conv.get("original_cluster", ""),
                "post_process_conv": conv.get("post_process_conv", "")
            }
        })

    return arena_hard_questions

def main():
    parser = argparse.ArgumentParser(description='Filter prompts using hierarchical filtering: first select clusters, then top N prompts from each cluster.')
    parser.add_argument('--conversations_file', type=str, required=True,
                       help='Path to the JSON file containing conversations')
    parser.add_argument('--clusters_file', type=str, required=True,
                       help='Path to the JSON file containing cluster assignments')
    parser.add_argument('--cluster_threshold', type=float, default=3.0,
                       help='Minimum average score threshold for clusters')
    parser.add_argument('--top_prompts_per_cluster', type=int, default=5,
                       help='Number of top prompts to select from each qualifying cluster')
    parser.add_argument('--output_file', type=str, default='filtered_prompts.json',
                       help='Path to save the filtered prompts (JSON format)')
    parser.add_argument('--wandb_project', type=str, default='',
                       help='Wandb project name (empty to disable)')
    parser.add_argument("--topics_file", type=str, default="",
                       help="Path to the file containing topic cluster mappings (JSON or CSV)")
    parser.add_argument("--force_process_all", action='store_true',
                       help="Process all conversations even if clusters file is shorter")
    
    args = parser.parse_args()

    if args.wandb_project:
        wandb.init(project=args.wandb_project)
    
    # Load data
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    conversations = load_json(args.conversations_file)
    if not conversations:
        print("Error: No valid conversations loaded.")
        return
    
    clusters = load_json(args.clusters_file)
    
    print(f"\nLoaded data:")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Clusters: {len(clusters)}")
    
    cluster_number_to_name, cluster_number_to_details = load_topics_file(args.topics_file)
    
    if conversations:
        sample_conv = conversations[0]
        print(f"\nSample conversation:")
        print(f"  Keys: {list(sample_conv.keys())}")
        print(f"  ID: {sample_conv.get('id', 'N/A')}")
        print(f"  Score: {calculate_score(sample_conv)}")
        content_preview = extract_content_from_conversation(sample_conv)
        print(f"  Content preview: {content_preview[:100]}...")
    
    if clusters:
        print(f"\nSample cluster data:")
        print(f"  First 10 clusters: {clusters[:10]}")
        unique_clusters = set(str(c) for c in clusters)
        print(f"  Unique clusters: {len(unique_clusters)}")
    
    print("="*60)
    print("HIERARCHICAL FILTERING")
    print("="*60)
    
    filtered_prompts = filter_prompts_hierarchical(
        conversations, clusters, args.cluster_threshold, 
        args.top_prompts_per_cluster, cluster_number_to_name, cluster_number_to_details
    )
    
    if not filtered_prompts:
        print("Warning: No prompts passed the filtering criteria.")
        return
    
    print("="*60)
    print("CONVERTING TO ARENA-HARD FORMAT")
    print("="*60)
    
    arena_hard_questions = to_arena_hard_questions_format(filtered_prompts, cluster_number_to_name)

    print(f"Saving {len(arena_hard_questions)} questions to {args.output_file}")
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(arena_hard_questions, f, ensure_ascii=False, indent=2)
    
    print("="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Hierarchical Filtering Results:")
    print(f"   Original conversations: {len(conversations):,}")
    print(f"   Selected clusters: {len(set(conv.get('cluster_id') for conv in filtered_prompts)):,}")
    print(f"   Top prompts per cluster: {args.top_prompts_per_cluster}")
    print(f"   Total selected prompts: {len(filtered_prompts):,}")
    print(f"   Overall retention: {len(arena_hard_questions)/len(conversations)*100:.1f}%")
    print(f"Output saved to: {args.output_file}")

    if args.wandb_project:
        try:
            data = []
            columns = ["question_id", "content_preview", "prompt_score", "cluster_score", "cluster_rank", "prompt_rank", "cluster_name", "kind"]
            
            for conv in filtered_prompts[:100]:
                content = extract_content_from_conversation(conv)
                cluster_name = cluster_number_to_name.get(str(conv.get('cluster_id', 0)), f"Cluster_{conv.get('cluster_id', 0)}")
                data.append([
                    conv.get("id", 0),
                    content[:50] + "..." if len(content) > 50 else content,
                    conv.get("prompt_score", 0),
                    conv.get("cluster_score", 0),
                    conv.get("cluster_rank_in_selection", 0),
                    conv.get("prompt_rank_in_cluster", 0),
                    cluster_name,
                    conv.get("kind", "unknown")
                ])

            table = wandb.Table(data=data, columns=columns)
            wandb.log({"filtered_prompts_sample": table})
            print("Results logged to wandb")
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")

if __name__ == "__main__":
    main()