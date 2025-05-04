import os
import sys
import numpy as np
import argparse
import glob
from pathlib import Path
import time

def load_embeddings(video_path):
    """
    Load video and text embeddings for a specific video path.
    
    Args:
        video_path: Path to the video file (/Data/Food/food1.mp4)
    
    Returns:
        aggregated_embedding: Aggregated video embedding
        text_embedding: Text embedding
        video_name: Name of the video file without extension
    """
    # Get video filename without path and extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    category = os.path.basename(os.path.dirname(video_path))
    
    # Determine embedding directory path
    embedding_dir = os.path.join("embeddings", category)
    
    if not os.path.exists(embedding_dir):
        raise ValueError(f"Embedding directory for category {category} does not exist: {embedding_dir}")
    
    # Find the latest (most recent) aggregated and text embeddings for this video
    aggregated_pattern = os.path.join(embedding_dir, f"{video_name}_aggregated_mean_*.npy")
    text_pattern = os.path.join(embedding_dir, f"{video_name}_text_*.npy")
    
    aggregated_files = sorted(glob.glob(aggregated_pattern), reverse=True)
    text_files = sorted(glob.glob(text_pattern), reverse=True)
    
    if not aggregated_files:
        raise ValueError(f"No aggregated embedding found for video {video_name}")
    
    if not text_files:
        raise ValueError(f"No text embedding found for video {video_name}")
    
    # Load embeddings
    aggregated_embedding = np.load(aggregated_files[0])
    text_embedding = np.load(text_files[0])
    
    # Flatten to ensure 1D vectors
    aggregated_embedding = aggregated_embedding.flatten()
    text_embedding = text_embedding.flatten()
    
    return aggregated_embedding, text_embedding, video_name, category

def find_all_embeddings(base_dir="embeddings"):
    """
    Find all embeddings in the embeddings directory
    
    Returns:
        List of dictionaries containing embedding information
    """
    all_videos = []
    
    # Walk through all category folders in the embeddings directory
    for category_dir in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category_dir)
        
        if not os.path.isdir(category_path):
            continue
        
        # Get all unique video names in this category
        video_names = set()
        
        # Find all aggregated embeddings
        aggregated_files = glob.glob(os.path.join(category_path, "*_aggregated_mean_*.npy"))
        for agg_file in aggregated_files:
            # Extract video name from file name
            file_name = os.path.basename(agg_file)
            # Remove "_aggregated_mean_TIMESTAMP.npy" to get video name
            video_name = file_name.split("_aggregated_mean_")[0]
            video_names.add(video_name)
        
        # Process each unique video
        for video_name in video_names:
            # Find latest aggregated embedding
            agg_pattern = os.path.join(category_path, f"{video_name}_aggregated_mean_*.npy")
            agg_files = sorted(glob.glob(agg_pattern), reverse=True)
            
            # Find latest text embedding
            text_pattern = os.path.join(category_path, f"{video_name}_text_*.npy")
            text_files = sorted(glob.glob(text_pattern), reverse=True)
            
            # Skip videos without both embeddings
            if not agg_files or not text_files:
                print(f"Skipping {video_name} - missing embeddings")
                continue
            
            # Load latest embeddings
            agg_embedding = np.load(agg_files[0]).flatten()
            text_embedding = np.load(text_files[0]).flatten()
            
            all_videos.append({
                "video_name": video_name,
                "category": category_dir,
                "aggregated_embedding": agg_embedding,
                "text_embedding": text_embedding,
                "aggregated_path": agg_files[0],
                "text_path": text_files[0]
            })
    
    return all_videos

def calculate_similarity(source_agg_emb, source_text_emb, target_agg_emb, target_text_emb, video_weight=0.7, text_weight=0.3):
    """
    Calculate weighted similarity between source and target embeddings
    
    Args:
        source_agg_emb: Source video aggregated embedding
        source_text_emb: Source video text embedding
        target_agg_emb: Target video aggregated embedding
        target_text_emb: Target video text embedding
        video_weight: Weight for video embedding similarity (default: 0.7)
        text_weight: Weight for text embedding similarity (default: 0.3)
    
    Returns:
        similarity_score: Weighted similarity score
    """
    # Normalize embeddings for cosine similarity
    source_agg_norm = source_agg_emb / np.linalg.norm(source_agg_emb)
    source_text_norm = source_text_emb / np.linalg.norm(source_text_emb)
    target_agg_norm = target_agg_emb / np.linalg.norm(target_agg_emb)
    target_text_norm = target_text_emb / np.linalg.norm(target_text_emb)
    
    # Calculate cosine similarities
    video_similarity = np.dot(source_agg_norm, target_agg_norm)
    text_similarity = np.dot(source_text_norm, target_text_norm)
    
    # Calculate weighted similarity
    weighted_similarity = (video_weight * video_similarity) + (text_weight * text_similarity)
    
    return weighted_similarity

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Find similar videos based on embedding similarity")
    parser.add_argument("video_path", help="Path to the video file (e.g., /Data/Food/food1.mp4)")
    parser.add_argument("--k", type=int, default=5, help="Number of similar videos to find (default: 5)")
    parser.add_argument("--video-weight", type=float, default=0.7, 
                        help="Weight for video embedding similarity (default: 0.7)")
    parser.add_argument("--text-weight", type=float, default=0.3, 
                        help="Weight for text embedding similarity (default: 0.3)")
    
    args = parser.parse_args()
    
    # Check if weights sum to 1
    total_weight = args.video_weight + args.text_weight
    if abs(total_weight - 1.0) > 1e-5:
        print(f"Warning: Weights don't sum to 1.0 (sum = {total_weight}). Normalizing weights.")
        args.video_weight = args.video_weight / total_weight
        args.text_weight = args.text_weight / total_weight
    
    try:
        # Load source video embeddings
        source_agg_emb, source_text_emb, source_name, source_category = load_embeddings(args.video_path)
        
        print(f"\nProcessing: [{source_category}] {source_name}")
        print(f"Video embedding shape: {source_agg_emb.shape}")
        print(f"Text embedding shape: {source_text_emb.shape}")
        print(f"Using weights - Video: {args.video_weight:.2f}, Text: {args.text_weight:.2f}")
        
        # Find all other video embeddings
        print("\nLoading all video embeddings...")
        start_time = time.time()
        all_videos = find_all_embeddings()
        load_time = time.time() - start_time
        print(f"Loaded {len(all_videos)} videos with embeddings in {load_time:.2f} seconds")
        
        # Calculate similarity with all other videos
        print("\nCalculating similarities...")
        similarities = []
        
        for video in all_videos:
            # Skip comparing to the same video
            if video["video_name"] == source_name and video["category"] == source_category:
                continue
                
            similarity = calculate_similarity(
                source_agg_emb, 
                source_text_emb,
                video["aggregated_embedding"],
                video["text_embedding"],
                args.video_weight,
                args.text_weight
            )
            
            similarities.append({
                "video_name": video["video_name"],
                "category": video["category"],
                "similarity": similarity
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Print top k similar videos
        k = min(args.k, len(similarities))
        print(f"\nTop {k} similar videos to [{source_category}] {source_name}:")
        
        for i, item in enumerate(similarities[:k]):
            print(f"{i+1}. [{item['category']}] {item['video_name']} - Similarity: {item['similarity']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 