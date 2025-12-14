from sqlalchemy import create_engine
from project_code import MusicBrainzDatabase, CollaborationNetworkGraph, RecommendationEngine
import config
import pandas as pd

def search_artist(db):
    """Search for an artist and return a single selected artist."""
    while True:
        search_name = input("\nEnter artist name to search (or 'quit' to exit): ").strip()
        
        if search_name.lower() == 'quit':
            return None
        
        if not search_name:
            print("Please enter a valid artist name.")
            continue
        
        # Search for artist
        results = db.find_artist_by_name(search_name)
        
        if results.empty:
            print(f"No artists found matching '{search_name}'. Try again.")
            continue
        
        # Display results
        print(f"\nFound {len(results)} artist(s):")
        print("-" * 80)
        
        for idx, (_, row) in enumerate(results.head(10).iterrows(), 1):
            artist_id = row['artist_id']
            artist_name = row['artist_name']
            comment = row['comment'] if pd.notna(row['comment']) else "(no description)"
            print(f"{idx}. ID: {artist_id:7d} | {artist_name:30s} | {comment}")
        
        print("-" * 80)
        
        # Get user selection
        try:
            selection = input("\nSelect artist number (1-10): ").strip()
            
            if selection.lower() == 'back':
                continue
            
            selection_idx = int(selection) - 1
            
            # Validate selection
            if not (0 <= selection_idx < len(results.head(10))):
                print("Invalid selection. Please enter a number between 1 and 10.")
                continue
            
            # Get selected artist
            selected_row = results.head(10).iloc[selection_idx]
            selected_artist = db.find_artist_by_id(int(selected_row['artist_id']))
            
            return selected_artist
        
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 10.")
            continue

def build_graph(db, seed_artists, connections_per_seed=10):
    """Build the collaboration network graph."""
    print(f"\nBuilding collaboration network from {len(seed_artists)} seed artist(s)...")
    graph = CollaborationNetworkGraph(db)
    graph.build(seed_artists, top_k_per_seed=connections_per_seed)
    print(f"Graph built with {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges")
    return graph

def display_recommendations(df, max_rows=10):
    """Display recommendations with limited rows."""
    if df.empty:
        print("No recommendations found.")
        return
    
    # Display with limited rows
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print(df.head(max_rows))
    
    if len(df) > max_rows:
        print(f"\n... and {len(df) - max_rows} more rows")

def main():
    """Terminal-based Interactive System Music Artist Collaboration Recommendation."""
    print("\n" + "="*80)
    print("Music Artist Collaboration Recommendation System")
    print("="*80)
    
    # Setup database connection
    try:
        host = config.DB_HOST
        user = config.DB_USER
        database = config.DB_NAME
        password = config.DB_PASSWORD
        
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:5432/{database}")
        db = MusicBrainzDatabase(engine)
        print("\nConnected to MusicBrainz database")
    except Exception as e:
        print(f"\nFailed to connect to database: {e}")
        return
    
    # Collect seed artists
    seed_artists = []
    
    print("\nSelect Seed Artists")
    print("-"*80)
    print("Search for artists to inform your recommendations.")
    print("You can select multiple artists from each search result.")
    
    while True:
        print(f"\nCurrent seed artists: {len(seed_artists)}")
        if seed_artists:
            for i, artist in enumerate(seed_artists, 1):
                print(f"   {i}. {artist.name}")
        
        user_choice = input("\nWhat would you like to do?\n1. Add an artist\n2. Get recommendations\nChoice (1-2): ").strip()
        
        if user_choice == "1":
            selected = search_artist(db)
            if selected:
                seed_artists.append(selected)
                print(f"Added artist: {selected.name}")
            elif selected is None:  # User quit
                print("Exiting...")
                db.close()
                return
        
        elif user_choice == "2":
            if not seed_artists:
                print("Please add at least one artist before getting recommendations.")
                continue
            break
        
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Build graph
    print("\n" + "-"*80)
    print("Building Collaboration Network")
    print("-"*80)
    
    graph = build_graph(db, seed_artists, connections_per_seed=10)
    
    # Generate recommendations
    print("\n" + "-"*80)
    print("Generating Recommendations")
    print("-"*80)
    
    try:
        recommender = RecommendationEngine(graph, db, seed_artists)
        
        # Artist Recommendations
        print("\nPRIMARY ARTIST RECOMMENDATIONS (PageRank)")
        print("-" * 80)
        artist_primary = recommender.get_artists()
        display_recommendations(artist_primary, max_rows=10)
        
        print("\nDISCOVERY ARTIST PICKS (Betweenness and Closeness)")
        print("-" * 80)
        artist_discovery = recommender.get_discovery_artists()
        display_recommendations(artist_discovery, max_rows=5)
        
        # Song Recommendations
        print("\n" + "="*80)
        print("\nPRIMARY SONG RECOMMENDATIONS (From Neighboring Artists)")
        print("-" * 80)
        songs_shared = recommender.get_songs()
        display_recommendations(songs_shared, max_rows=10)
        
        print("\nDISCOVERY SONG PICKS (From Discovery Artists)")
        print("-" * 80)
        songs_discovery = recommender.get_discovery_songs()
        display_recommendations(songs_discovery, max_rows=5)
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
    
    print("\n" + "="*80)
    print("Recommendation process complete!")
    print("="*80 + "\n")
    
    db.close()


if __name__ == "__main__":
    main()
