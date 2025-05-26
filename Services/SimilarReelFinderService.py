import os
import sys
import numpy as np
import argparse
import glob
from pathlib import Path
import time
from typing import Dict
from Services.MongoService import MongoService

class SimilarReelFinder:
    def __init__(self, video_weight=0.7, text_weight=0.3, mongo_service=None):
        self.video_weight = video_weight
        self.text_weight = text_weight
        self.mongo_service = mongo_service or MongoService()
    
    def load_embeddings_from_db(self, reel_doc):
        # Extract data from the document
        aggregated_embeddings = reel_doc.get('aggregatedEmbeddings')
        textual_embeddings = reel_doc.get('textualEmbeddings')
        path = reel_doc.get('path')
        category = reel_doc.get('category')
        reel_id = reel_doc.get('id') or str(reel_doc.get('_id'))
        
        # Convert to numpy arrays
        if isinstance(aggregated_embeddings, list):
            if isinstance(aggregated_embeddings[0], list):
                aggregated_embedding = np.array(aggregated_embeddings[0])
            else:
                aggregated_embedding = np.array(aggregated_embeddings)
        else:
            raise ValueError(f"Invalid aggregated embedding format for reel {reel_id}")
            
        if isinstance(textual_embeddings, list):
            if isinstance(textual_embeddings[0], list):
                textual_embedding = np.array(textual_embeddings[0])
            else:
                textual_embedding = np.array(textual_embeddings)
        else:
            raise ValueError(f"Invalid textual embedding format for reel {reel_id}")
        
        # Extract video name from path
        video_name = os.path.splitext(os.path.basename(path))[0]
        return aggregated_embedding, textual_embedding, video_name, category, reel_id
    
    def find_all_embeddings_from_db(self):
        all_videos = []
        all_reels = self.mongo_service.GetAllReels()
        
        for reel_doc in all_reels:
            # Extract data from the document
            aggregated_embeddings = reel_doc.get('aggregatedEmbeddings')
            textual_embeddings = reel_doc.get('textualEmbeddings')
            path = reel_doc.get('path')
            category = reel_doc.get('category')
            reel_id = reel_doc.get('id') or str(reel_doc.get('_id'))
            
            # Skip reels without embeddings
            if not aggregated_embeddings or not textual_embeddings:
                print(f"Skipping reel {reel_id} - missing embeddings")
                continue
                
            # Convert to numpy arrays
            if isinstance(aggregated_embeddings, list):
                if isinstance(aggregated_embeddings[0], list):
                    aggregated_embedding = np.array(aggregated_embeddings[0])
                else:
                    aggregated_embedding = np.array(aggregated_embeddings)
            else:
                print(f"Skipping reel {reel_id} - invalid aggregated embedding format")
                continue
                
            if isinstance(textual_embeddings, list):
                if isinstance(textual_embeddings[0], list):
                    textual_embedding = np.array(textual_embeddings[0])
                else:
                    textual_embedding = np.array(textual_embeddings)
            else:
                print(f"Skipping reel {reel_id} - invalid textual embedding format")
                continue
            
            # Extract video name from path
            video_name = os.path.splitext(os.path.basename(path))[0]
            
            all_videos.append({
                "video_name": video_name,
                "category": category,
                "aggregated_embedding": aggregated_embedding,
                "text_embedding": textual_embedding,
                "path": path,
                "reel_id": reel_id
            })
        
        return all_videos
    
    def calculate_similarity(self, source_agg_emb, source_text_emb, target_agg_emb, target_text_emb):
        # Normalize embeddings for cosine similarity
        source_agg_norm = source_agg_emb / np.linalg.norm(source_agg_emb)
        source_text_norm = source_text_emb / np.linalg.norm(source_text_emb)
        target_agg_norm = target_agg_emb / np.linalg.norm(target_agg_emb)
        target_text_norm = target_text_emb / np.linalg.norm(target_text_emb)
        
        # Calculate cosine similarities
        video_similarity = np.dot(source_agg_norm, target_agg_norm)
        text_similarity = np.dot(source_text_norm, target_text_norm)
        
        # Calculate weighted similarity
        weighted_similarity = (self.video_weight * video_similarity) + (self.text_weight * text_similarity)
        
        return weighted_similarity
    
    def find_similar_reel(self, reel_id, user_interactions: Dict[str, int] = None):
        try:
            similarities = self.mongo_service.GetSimilarities(reel_id)
            for similarity in similarities:
                if similarity.get("reel_id") not in user_interactions:
                    return similarity.get("reel_id")
            return None
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def StoreSimilarReels(self, reel_id, all_videos):
        try:
            reel = self.mongo_service.GetReel(reel_id)
            source_agg_emb, source_text_emb, source_name, source_category, source_id = self.load_embeddings_from_db(reel)
            print(f"\nProcessing: [{source_category}] {source_name}")
            print(f"Video embedding shape: {source_agg_emb.shape}")
            print(f"Text embedding shape: {source_text_emb.shape}")
            print(f"Using weights - Video: {self.video_weight:.2f}, Text: {self.text_weight:.2f}")
            print("\nCalculating similarities...")
            similarities = []
            
            for video in all_videos:
                # Skip comparing to the same video
                if (source_id and video.get("reel_id") == source_id) or (not source_id and video["video_name"] == source_name and video["category"] == source_category):
                    print(f"Skipping comparison to self: [{video['category']}] {video['video_name']}")
                    continue
                    
                similarity = self.calculate_similarity(
                    source_agg_emb, 
                    source_text_emb,
                    video["aggregated_embedding"],
                    video["text_embedding"]
                )
                
                similarities.append({
                    "similarity": similarity,
                    "reel_id": video.get("reel_id")
                })
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return []
    def StoreSimilarReelsUtil(self):
        all_reels = self.mongo_service.GetAllReels()
        mongo_service = MongoService()
        all_videos = self.find_all_embeddings_from_db()

        for reel in all_reels:
            similarities = self.StoreSimilarReels(reel.get("_id"), all_videos)
            mongo_service.StoreSimilarReels(reel.get("_id"), similarities)


def main():
    service = SimilarReelFinder()
    service.StoreSimilarReelsUtil()
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description="Find similar videos based on embedding similarity")
#     parser.add_argument("video_path", help="Path to the video file (e.g., /Data/Food/food1.mp4)")
#     parser.add_argument("--k", type=int, default=5, help="Number of similar videos to find (default: 5)")
#     parser.add_argument("--video-weight", type=float, default=0.7, 
#                         help="Weight for video embedding similarity (default: 0.7)")
#     parser.add_argument("--text-weight", type=float, default=0.3, 
#                         help="Weight for text embedding similarity (default: 0.3)")
    
#     args = parser.parse_args()
    
#     # Create SimilarReelFinder instance
#     finder = SimilarReelFinder(args.video_weight, args.text_weight)
    
#     # Find similar reels
#     finder.find_similar_reels(args.video_path, args.k)


if __name__ == "__main__":
    main() 