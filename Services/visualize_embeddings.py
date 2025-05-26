import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Services.MongoService import MongoService

def load_embeddings_from_categories(embeddings_dir="embeddings", embedding_type="frame"):
    # Identify all category folders
    category_dirs = [d for d in glob.glob(os.path.join(embeddings_dir, "*")) if os.path.isdir(d)]
    
    if not category_dirs:
        print(f"No category folders found in {embeddings_dir}")
        return None, None, None
    
    print(f"Found {len(category_dirs)} category folders")
    
    all_embeddings = []
    video_names = []
    categories = []
    
    # Pattern to match based on embedding type
    if embedding_type == "frame":
        pattern = "*_embeddings_*.npy"
    elif embedding_type == "aggregated":
        pattern = "*_aggregated_*.npy"
    elif embedding_type == "text":
        pattern = "*_text_*.npy"
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    # Load embeddings from each category
    for category_dir in category_dirs:
        category_name = os.path.basename(category_dir)
        embedding_files = glob.glob(os.path.join(category_dir, pattern))
        
        if not embedding_files:
            print(f"No {embedding_type} embedding files found in category {category_name}")
            continue
        
        print(f"Category {category_name}: Found {len(embedding_files)} {embedding_type} embedding files")
        
        # Process each embedding file in this category
        for embedding_file in embedding_files:
            # Extract video name from filename
            filename = os.path.basename(embedding_file)
            
            if embedding_type == "frame":
                video_name = filename.split("_embeddings_")[0]
            elif embedding_type == "aggregated":
                video_name = filename.split("_aggregated_")[0]
            elif embedding_type == "text":
                video_name = filename.split("_text_")[0]
            
            # Load embeddings
            embeddings = np.load(embedding_file)
            print(f" - {category_name}/{filename}: {embeddings.shape}")
            
            # Add to lists
            all_embeddings.append(embeddings)
            
            # Create labels for each embedding
            if embedding_type == "frame":
                # For frame embeddings, each frame gets labeled
                video_names.extend([video_name] * len(embeddings))
                categories.extend([category_name] * len(embeddings))
            else:
                # For aggregated or text embeddings, there's only one embedding per video
                video_names.append(video_name)
                categories.append(category_name)
    
    if not all_embeddings:
        print(f"No {embedding_type} embeddings found in any category")
        return None, None, None
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    return all_embeddings, video_names, categories

def load_embeddings_from_mongodb(embedding_type="frame"):
    """Load embeddings from MongoDB"""
    mongo_service = MongoService()
    all_reels = mongo_service.GetAllReels()
    
    if not all_reels:
        print("No reels found in MongoDB")
        return None, None, None
    
    print(f"Found {len(all_reels)} reels in MongoDB")
    
    all_embeddings = []
    video_names = []
    categories = []
    
    # Load embeddings from each reel
    for reel in all_reels:
        category = reel.get("category", "unknown")
        path = reel.get("path", "")
        video_name = os.path.splitext(os.path.basename(path))[0] if path else f"reel_{str(reel.get('_id'))}"
        
        # Get embeddings based on type
        if embedding_type == "frame":
            embeddings = reel.get("videoEmbeddings")
            if not embeddings:
                print(f"No frame embeddings found for reel {video_name}")
                continue
            
            # Convert embeddings to numpy array
            embeddings = np.array(embeddings)
            print(f" - {category}/{video_name}: {embeddings.shape}")
            
            # Add to lists
            all_embeddings.append(embeddings)
            # For frame embeddings, each frame gets labeled
            video_names.extend([video_name] * len(embeddings))
            categories.extend([category] * len(embeddings))
            
        elif embedding_type == "aggregated":
            embeddings = reel.get("aggregatedEmbeddings")
            if not embeddings:
                print(f"No aggregated embeddings found for reel {video_name}")
                continue
            
            # Handle different formats (array or array of arrays)
            if isinstance(embeddings[0], list):
                # If it's an array of arrays, take the first one
                embeddings = np.array(embeddings[0])
            else:
                embeddings = np.array(embeddings)
                
            # Reshape if needed (ensure it's 2D for consistent handling)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
                
            print(f" - {category}/{video_name}: {embeddings.shape}")
            
            # Add to lists
            all_embeddings.append(embeddings)
            video_names.append(video_name)
            categories.append(category)
            
        elif embedding_type == "text":
            embeddings = reel.get("textualEmbeddings")
            if not embeddings:
                print(f"No text embeddings found for reel {video_name}")
                continue
                
            # Handle different formats (array or array of arrays)
            if isinstance(embeddings[0], list):
                # If it's an array of arrays, take the first one
                embeddings = np.array(embeddings[0])
            else:
                embeddings = np.array(embeddings)
                
            # Reshape if needed (ensure it's 2D for consistent handling)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
                
            print(f" - {category}/{video_name}: {embeddings.shape}")
            
            # Add to lists
            all_embeddings.append(embeddings)
            video_names.append(video_name)
            categories.append(category)
    
    if not all_embeddings:
        print(f"No {embedding_type} embeddings found in MongoDB")
        return None, None, None
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    return all_embeddings, video_names, categories

def reduce_dimensions(embeddings, method="tsne", n_components=2, perplexity=None, random_state=42):
    """Reduce dimensions using t-SNE or PCA"""
    # Set default perplexity to min(30, n_samples / 5) to avoid the "perplexity must be less than n_samples" error
    if perplexity is None:
        n_samples = embeddings.shape[0]
        perplexity = min(30, max(5, n_samples // 5))
        print(f"Auto-setting perplexity to {perplexity} based on {n_samples} samples")
    
    print(f"Reducing dimensions to {n_components}D using {method}...")
    
    if method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'tsne' or 'pca'.")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def visualize_2d(reduced_embeddings, labels, categories=None, title="CLIP Embeddings Visualization"):
    """Visualize embeddings in 2D"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Get unique categories for coloring
    if categories:
        unique_categories = list(set(categories))
        # Create a color map for categories
        category_colors = {cat: color for cat, color in zip(unique_categories, plt.cm.tab10.colors)}
    
    # Get unique video names for markers
    unique_videos = list(set(labels))
    
    # Use different marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
    video_markers = {video: markers[i % len(markers)] for i, video in enumerate(unique_videos)}
    
    # Check if there are too many points to label individually
    too_many_points = len(reduced_embeddings) > 100
    
    # Plot each point
    for idx, (x, y) in enumerate(reduced_embeddings):
        video = labels[idx]
        marker = video_markers[video]
        
        if categories:
            category = categories[idx]
            color = category_colors[category]
            label = f"{category}: {video}" if labels.count(video) == 1 else None
        else:
            color = plt.cm.tab10(unique_videos.index(video) % 10)
            label = video if labels.count(video) == 1 else None
        
        ax.scatter(x, y, color=color, marker=marker, label=label, alpha=0.7)
        
        # Only add text labels if there aren't too many points
        if not too_many_points:
            ax.text(x, y, f"{idx}", fontsize=9)
    
    # Add legend (only showing each combination once)
    if categories:
        # Create legend with both category colors and video markers
        handles = []
        for cat in unique_categories:
            for video in [v for v, c in zip(labels, categories) if c == cat]:
                if f"{cat}: {video}" not in [h.get_label() for h in handles]:
                    handles.append(plt.Line2D([0], [0], marker=video_markers[video], 
                                            color=category_colors[cat], 
                                            markersize=10, 
                                            label=f"{cat}: {video}"))
    else:
        handles = [plt.Line2D([0], [0], marker=video_markers[video], 
                             color=plt.cm.tab10(unique_videos.index(video) % 10), 
                             markersize=10, 
                             label=video)
                   for video in unique_videos]
                   
    # Place legend outside the plot
    ax.legend(handles=handles, title="Videos", loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Make sure the figure adjusts to accommodate the legend
    plt.tight_layout()
    
    return fig

def visualize_3d(reduced_embeddings, labels, categories=None, title="CLIP Embeddings Visualization (3D)"):
    """Visualize embeddings in 3D"""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique categories for coloring
    if categories:
        unique_categories = list(set(categories))
        # Create a color map for categories
        category_colors = {cat: color for cat, color in zip(unique_categories, plt.cm.tab10.colors)}
    
    # Get unique video names for markers
    unique_videos = list(set(labels))
    
    # Use different marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
    video_markers = {video: markers[i % len(markers)] for i, video in enumerate(unique_videos)}
    
    # Check if there are too many points to label individually
    too_many_points = len(reduced_embeddings) > 100
    
    # Plot each point
    for idx, (x, y, z) in enumerate(reduced_embeddings):
        video = labels[idx]
        marker = video_markers[video]
        
        if categories:
            category = categories[idx]
            color = category_colors[category]
            label = f"{category}: {video}" if labels.count(video) == 1 else None
        else:
            color = plt.cm.tab10(unique_videos.index(video) % 10)
            label = video if labels.count(video) == 1 else None
        
        ax.scatter(x, y, z, color=color, marker=marker, label=label, alpha=0.7)
        
        # Only add text labels if there aren't too many points
        if not too_many_points:
            ax.text(x, y, z, f"{idx}", fontsize=9)
    
    # Add legend (only showing each combination once)
    if categories:
        # Create legend with both category colors and video markers
        handles = []
        for cat in unique_categories:
            for video in [v for v, c in zip(labels, categories) if c == cat]:
                if f"{cat}: {video}" not in [h.get_label() for h in handles]:
                    handles.append(plt.Line2D([0], [0], marker=video_markers[video], 
                                            color=category_colors[cat], 
                                            markersize=10, 
                                            label=f"{cat}: {video}"))
    else:
        handles = [plt.Line2D([0], [0], marker=video_markers[video], 
                             color=plt.cm.tab10(unique_videos.index(video) % 10), 
                             markersize=10, 
                             label=video)
                   for video in unique_videos]
                   
    # Place legend outside the plot
    ax.legend(handles=handles, title="Videos", loc='upper left', bbox_to_anchor=(1.1, 1))
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Make sure the figure adjusts to accommodate the legend
    plt.tight_layout()
    
    return fig

def process_embeddings(args, embedding_type):
    """Process and visualize a specific type of embeddings"""
    # Set the title suffix based on embedding type
    if embedding_type == "frame":
        title_suffix = "Frame Embeddings"
    elif embedding_type == "aggregated":
        title_suffix = "Aggregated Video Embeddings"
    elif embedding_type == "text":
        title_suffix = "Text Embeddings"
    
    # Load embeddings based on source
    if args.source == "files":
        embeddings, video_names, categories = load_embeddings_from_categories(args.dir, embedding_type)
    elif args.source == "mongodb":
        embeddings, video_names, categories = load_embeddings_from_mongodb(embedding_type)
    
    if embeddings is None:
        return
    
    # Sample a subset if requested
    if args.sample is not None and args.sample < len(embeddings):
        print(f"Sampling {args.sample} embeddings from {len(embeddings)} total")
        indices = np.random.choice(len(embeddings), size=args.sample, replace=False)
        embeddings = embeddings[indices]
        video_names = [video_names[i] for i in indices]
        if categories:
            categories = [categories[i] for i in indices]
    
    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(
        embeddings, 
        method=args.method, 
        n_components=args.dims,
        perplexity=args.perplexity
    )
    
    # Visualize
    if args.dims == 2:
        title = f"CLIP {title_suffix} Visualization (2D, {args.method.upper()})"
        fig = visualize_2d(reduced_embeddings, video_names, categories, title)
        output_file = f"{args.output}_{embedding_type}_{args.method}_2d.png"
    else:
        title = f"CLIP {title_suffix} Visualization (3D, {args.method.upper()})"
        fig = visualize_3d(reduced_embeddings, video_names, categories, title)
        output_file = f"{args.output}_{embedding_type}_{args.method}_3d.png"
    
    # Save figure
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize CLIP embeddings in lower-dimensional space")
    parser.add_argument("--dir", default="embeddings", help="Base directory containing category folders with embeddings")
    parser.add_argument("--method", default="tsne", choices=["tsne", "pca"], help="Dimension reduction method")
    parser.add_argument("--dims", type=int, default=2, choices=[2, 3], help="Number of dimensions for visualization")
    parser.add_argument("--perplexity", type=int, default=None, help="Perplexity for t-SNE. If not specified, will be auto-calculated based on dataset size")
    parser.add_argument("--output", default="embedding_visualization", help="Output filename prefix")
    parser.add_argument("--sample", type=int, default=None, help="Sample a subset of embeddings for visualization. Useful for large datasets")
    parser.add_argument("--type", default="all", choices=["frame", "aggregated", "text", "all"], 
                        help="Type of embeddings to visualize (frame, aggregated, text, or all)")
    parser.add_argument("--source", default="files", choices=["files", "mongodb"], 
                        help="Source of embeddings (files or MongoDB)")
    
    args = parser.parse_args()
    
    figures = []
    
    # Process embeddings based on the requested type
    if args.type == "all":
        print("\n=== Processing Frame Embeddings ===\n")
        frame_fig = process_embeddings(args, "frame")
        if frame_fig:
            figures.append(frame_fig)
        
        print("\n=== Processing Aggregated Embeddings ===\n")
        agg_fig = process_embeddings(args, "aggregated")
        if agg_fig:
            figures.append(agg_fig)
        
        print("\n=== Processing Text Embeddings ===\n")
        text_fig = process_embeddings(args, "text")
        if text_fig:
            figures.append(text_fig)
    else:
        fig = process_embeddings(args, args.type)
        if fig:
            figures.append(fig)
    
    # Show all figures
    if figures:
        plt.show()

if __name__ == "__main__":
    main() 