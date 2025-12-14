import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Credit collaboration weights
CREDIT_WEIGHTS = {
    'vocalist': 3.0,                # High - core collaboration
    'vocal': 3.0,                   # High - core collaboration
    'performer': 3,                 # High - general performance
    'instrument': 2.0,              # Medium - instrumental contribution
    'instrumentalist': 2.0,         # Medium - instrumental contribution
    'producer': 2.0,                # Medium - production credit
    'arranger': 0.5,                # Low - arrangement credit
    'composer': 0.5,                # Low - compositional credit
    'lyricist': 0.5,                # Low - lyrical credit
    'writer': 0.5,                  # Low - writing credit
    'default': 1.0                  # Default weight for unknown credit types
}

class Artist:
    """Represents an artist with ID and name."""
    def __init__(self, id, name):
        self.id = id
        self.name = name


class MusicBrainzDatabase:
    """Handles all database operations and queries."""
    
    def __init__(self, engine):
        self.engine = engine

    def _dataframe_to_artists(self, df):
        """Convert DataFrame rows to Artist objects."""
        artists = []
        for _, row in df.iterrows():
            artists.append(Artist(row['artist_id'], row['artist_name']))
        return artists

    def find_artist_by_name(self, artist_name):
        """Find artists by name, including aliases. Shows exact matches first."""
        query = """
            SELECT DISTINCT 
                a.id AS artist_id, 
                a.name AS artist_name, 
                a.comment
            FROM musicbrainz.artist a
            LEFT JOIN musicbrainz.artist_alias aa ON aa.artist = a.id
            WHERE a.name ILIKE %s OR aa.name ILIKE %s
            GROUP BY a.id, a.name, a.comment
            ORDER BY a.name
        """
        return pd.read_sql_query(query, self.engine, params=(artist_name, artist_name))

    def find_artist_by_id(self, artist_id):
        """Find artist by ID."""
        query = """
            SELECT 
                a.id AS artist_id, 
                a.name AS artist_name, 
                a.comment
            FROM musicbrainz.artist a
            WHERE a.id = %s
        """
        df = pd.read_sql_query(query, self.engine, params=(artist_id,))
        return self._dataframe_to_artists(df)[0] if not df.empty else None

    def get_direct_collaborators(self, artist_id):
        """Get direct collaborators via artist credits on recordings."""
        query = """
            SELECT
                a2.id AS artist_id,
                a2.name AS artist_name,
                COUNT(DISTINCT COALESCE(w.id, r.id)) AS appearance_count
            FROM musicbrainz.recording r
            LEFT JOIN musicbrainz.l_recording_work lrw ON lrw.entity0 = r.id
            LEFT JOIN musicbrainz.work w ON w.id = lrw.entity1
            JOIN musicbrainz.artist_credit ac ON ac.id = r.artist_credit
            JOIN musicbrainz.artist_credit_name acn1 ON acn1.artist_credit = ac.id
            JOIN musicbrainz.artist input_artist ON input_artist.id = acn1.artist
            JOIN musicbrainz.artist_credit_name acn2 ON acn2.artist_credit = ac.id
            JOIN musicbrainz.artist a2 ON a2.id = acn2.artist
            WHERE input_artist.id = %s AND a2.id <> %s
            GROUP BY a2.id, a2.name
            ORDER BY appearance_count DESC
        """
        return pd.read_sql_query(query, self.engine, params=(artist_id, artist_id))

    def get_credit_collaborators(self, artist_id):
        """Get collaborators via recording credits (producer, vocalist, etc.)."""
        query = """
            SELECT
                a.id AS artist_id,
                a.name AS artist_name,
                lt.name AS credit_type,
                COUNT(DISTINCT COALESCE(w.id, r.id)) AS appearance_count
            FROM musicbrainz.recording r
            LEFT JOIN musicbrainz.l_recording_work lrw ON lrw.entity0 = r.id
            LEFT JOIN musicbrainz.work w ON w.id = lrw.entity1
            JOIN musicbrainz.artist_credit ac ON ac.id = r.artist_credit
            JOIN musicbrainz.artist_credit_name acn ON acn.artist_credit = ac.id
            JOIN musicbrainz.artist input_artist ON input_artist.id = acn.artist
            JOIN musicbrainz.l_artist_recording lar ON lar.entity1 = r.id
            JOIN musicbrainz.artist a ON a.id = lar.entity0
            JOIN musicbrainz.link l ON l.id = lar.link
            JOIN musicbrainz.link_type lt ON lt.id = l.link_type
            WHERE input_artist.id = %s AND a.id <> %s
            GROUP BY a.id, a.name, lt.name
            ORDER BY appearance_count DESC
        """
        return pd.read_sql_query(query, self.engine, params=(artist_id, artist_id))

    def get_featured_in_collaborations(self, artist_id):
        """Get artists where the input artist is a credit collaborator, while excluding direct collaborators."""
        query = """
            SELECT
                main_artist.id AS artist_id,
                main_artist.name AS artist_name,
                COUNT(DISTINCT COALESCE(w.id, r.id)) AS appearance_count
            FROM musicbrainz.recording r
            LEFT JOIN musicbrainz.l_recording_work lrw ON lrw.entity0 = r.id
            LEFT JOIN musicbrainz.work w ON w.id = lrw.entity1
            JOIN musicbrainz.artist_credit ac ON ac.id = r.artist_credit
            JOIN musicbrainz.artist_credit_name acn ON acn.artist_credit = ac.id
            JOIN musicbrainz.artist main_artist ON main_artist.id = acn.artist
            JOIN musicbrainz.l_artist_recording lar ON lar.entity1 = r.id
            JOIN musicbrainz.artist contributing_artist ON contributing_artist.id = lar.entity0
            WHERE contributing_artist.id = %s AND main_artist.id <> %s
            AND NOT EXISTS (
                -- Exclude if they also appear together as direct collaborators
                SELECT 1
                FROM musicbrainz.artist_credit ac2
                JOIN musicbrainz.artist_credit_name acn2_1 ON acn2_1.artist_credit = ac2.id
                JOIN musicbrainz.artist_credit_name acn2_2 ON acn2_2.artist_credit = ac2.id
                WHERE acn2_1.artist = %s AND acn2_2.artist = main_artist.id
            )
            GROUP BY main_artist.id, main_artist.name
            ORDER BY appearance_count DESC
        """
        return pd.read_sql_query(query, self.engine, params=(artist_id, artist_id, artist_id))

    def get_direct_artist_connections(self, artist_id):
        """Get artist connections like band members through l_artist_artist relationships."""
        query = """
            SELECT DISTINCT
                a.id AS artist_id,
                a.name AS artist_name
            FROM musicbrainz.artist a
            JOIN musicbrainz.l_artist_artist laa ON laa.entity1 = a.id OR laa.entity0 = a.id
            JOIN musicbrainz.link l ON laa.link = l.id
            JOIN musicbrainz.link_type lt ON l.link_type = lt.id
            WHERE (laa.entity0 = %s OR laa.entity1 = %s)
                AND a.id != %s
                AND lt.name IN ('member of band', 'collaboration', 'supporting musician')
            ORDER BY a.name
        """
        return pd.read_sql_query(query, self.engine, params=(artist_id, artist_id, artist_id))

    def get_connection_weight_between_artists(self, artist1_id, artist2_id):
        """Find the amount of shared works/recordings between two artists."""
        query = """
            SELECT COUNT(DISTINCT COALESCE(w1.id, r1.id)) AS shared_works
            FROM musicbrainz.recording r1
            LEFT JOIN musicbrainz.l_recording_work lrw1 ON lrw1.entity0 = r1.id
            LEFT JOIN musicbrainz.work w1 ON w1.id = lrw1.entity1
            JOIN musicbrainz.artist_credit ac1 ON ac1.id = r1.artist_credit
            JOIN musicbrainz.artist_credit_name acn1 ON acn1.artist_credit = ac1.id
            INNER JOIN (
                SELECT r2.id, COALESCE(w2.id, r2.id) AS work_or_recording
                FROM musicbrainz.recording r2
                LEFT JOIN musicbrainz.l_recording_work lrw2 ON lrw2.entity0 = r2.id
                LEFT JOIN musicbrainz.work w2 ON w2.id = lrw2.entity1
                JOIN musicbrainz.artist_credit ac2 ON ac2.id = r2.artist_credit
                JOIN musicbrainz.artist_credit_name acn2 ON acn2.artist_credit = ac2.id
                WHERE acn2.artist = %s
            ) artist2_recordings ON COALESCE(w1.id, r1.id) = artist2_recordings.work_or_recording
            WHERE acn1.artist = %s
        """
        result = pd.read_sql_query(query, self.engine, params=(artist2_id, artist1_id))
        return int(result.iloc[0]['shared_works']) if not result.empty else 0

    def get_shared_recordings(self, artist1_id, artist2_id):
        """Get shared recordings between two artists."""
        query = """
            SELECT DISTINCT r1.id, r1.name AS recording_name
            FROM musicbrainz.recording r1
            LEFT JOIN musicbrainz.l_recording_work lrw1 ON lrw1.entity0 = r1.id
            LEFT JOIN musicbrainz.work w1 ON w1.id = lrw1.entity1
            JOIN musicbrainz.artist_credit ac1 ON ac1.id = r1.artist_credit
            JOIN musicbrainz.artist_credit_name acn1 ON acn1.artist_credit = ac1.id
            INNER JOIN (
                SELECT r2.id, COALESCE(w2.id, r2.id) AS work_or_recording
                FROM musicbrainz.recording r2
                LEFT JOIN musicbrainz.l_recording_work lrw2 ON lrw2.entity0 = r2.id
                LEFT JOIN musicbrainz.work w2 ON w2.id = lrw2.entity1
                JOIN musicbrainz.artist_credit ac2 ON ac2.id = r2.artist_credit
                JOIN musicbrainz.artist_credit_name acn2 ON acn2.artist_credit = ac2.id
                WHERE acn2.artist = %s
            ) artist2_recordings ON COALESCE(w1.id, r1.id) = artist2_recordings.work_or_recording
            WHERE acn1.artist = %s
            ORDER BY r1.name
        """
        return pd.read_sql_query(query, self.engine, params=(artist2_id, artist1_id))

    def close(self):
        """Close database connection."""
        self.engine.dispose()


class CollaborationNetworkGraph:
    """Builds and analyzes artist collaboration graphs."""
    
    def __init__(self, database):
        """Initialize with a MusicBrainzDatabase instance."""
        self.db = database
        self.graph = nx.Graph()
    
    # ---- Graph Building ----
    def add_artist(self, artist):
        """Add a seed artist to the graph."""
        self.graph.add_node(artist.id, name=artist.name, is_seed=True)
    
    def add_connection(self, seed_artist, collab_artist, weight, connection_type):
        """Add a collaborator connection to the graph."""
        self.graph.add_node(collab_artist.id, name=collab_artist.name)
        
        if self.graph.has_edge(seed_artist.id, collab_artist.id):
            self.graph[seed_artist.id][collab_artist.id]["weight"] += weight
        else:
            self.graph.add_edge(
                seed_artist.id, collab_artist.id, 
                weight=weight, 
                connection_type=connection_type
            )
    
    def add_collaborators_from_query(self, seed_artist, query_func, connection_type, 
                                      base_weight=1.0, use_credit_weights=False):
        """Helper to add collaborators from a database query result."""
        df = query_func(seed_artist.id)
        
        for _, row in df.iterrows():
            collab_artist = Artist(row["artist_id"], row["artist_name"])
            
            # Calculate weight
            if use_credit_weights:
                credit_type = row.get("credit_type", "")
                weight = CREDIT_WEIGHTS.get(
                    credit_type.lower().strip() if credit_type else '', 
                    CREDIT_WEIGHTS['default']
                )
            else:
                weight = base_weight
            
            # Multiply by appearance count if present
            if "appearance_count" in row:
                weight *= row["appearance_count"]
            
            self.add_connection(seed_artist, collab_artist, weight, connection_type)
    
    def build_main_connections(self, artists):
        """Phase 1: Add seed artists and their collaborators from all query types."""
        print(f"\nFinding artist connections from {len(artists)} seed artists...")
        for artist in artists:
            self.add_artist(artist)
            
            # Add collaborators from three different query types
            self.add_collaborators_from_query(
                artist, self.db.get_direct_collaborators, "direct", base_weight=2.0
            )
            self.add_collaborators_from_query(
                artist, self.db.get_credit_collaborators, "credit", use_credit_weights=True
            )
            self.add_collaborators_from_query(
                artist, self.db.get_featured_in_collaborations, "featured", use_credit_weights=True
            )
    
    def filter_graph(self, top_k_per_seed=12):
        """Filter graph by keeping only the top K strongest connections per seed artist.
        
        This alsp helps to normalize the graph when seed artists have different
        numbers of collaborators, preventing popular artists from dominating.
        """
        # Get seed node IDs
        seed_ids = [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get("is_seed", False)
        ]
        
        edges_to_keep = set()
        
        # For each seed artist, keep only their top K strongest connections
        for seed_id in seed_ids:
            # Get all edges from this seed node
            seed_edges = []
            for neighbor in self.graph.neighbors(seed_id):
                edge_data = self.graph[seed_id][neighbor]
                weight = edge_data.get("weight", 0)
                seed_edges.append((seed_id, neighbor, weight))
            
            # Sort by weight descending and keep top K
            seed_edges.sort(key=lambda x: x[2], reverse=True)
            top_edges = seed_edges[:top_k_per_seed]
            
            # Mark these edges to keep
            for u, v, _ in top_edges:
                edges_to_keep.add(tuple(sorted([u, v])))
        
        # Keep all edges between non-seed nodes (interconnections)
        for u, v in self.graph.edges():
            is_u_seed = self.graph.nodes[u].get("is_seed", False)
            is_v_seed = self.graph.nodes[v].get("is_seed", False)
            if not is_u_seed and not is_v_seed:
                edges_to_keep.add(tuple(sorted([u, v])))
        
        # Remove edges not in top K
        edges_to_remove = [
            (u, v) for u, v in self.graph.edges()
            if tuple(sorted([u, v])) not in edges_to_keep
        ]
        self.graph.remove_edges_from(edges_to_remove)
        
        # Remove isolated non-seed nodes (nodes with no edges)
        nodes_to_remove = [
            node for node, data in self.graph.nodes(data=True)
            if not data.get("is_seed", False) and self.graph.degree(node) == 0
        ]
        self.graph.remove_nodes_from(nodes_to_remove)
        
        print(f"Filtered to {len(self.graph.nodes)} nodes, keeping top {top_k_per_seed} connections per seed artist")
    
    def build_interconnections(self):
        """Phase 2: Find interconnections between non-seed artists."""
        # Get all non-seed nodes
        non_seed_nodes = [
            node_id for node_id, data in self.graph.nodes(data=True) 
            if not data.get("is_seed", False)
        ]
        
        print(f"\nFinding interconnections among {len(non_seed_nodes)} collaborators...")
        
        # Check all pairs of non-seed artists
        interconnections_found = 0
        checked_pairs = set()
        for i, artist1_id in enumerate(non_seed_nodes):
            for _, artist2_id in enumerate(non_seed_nodes[i+1:], start=i+1):
                pair_key = (artist1_id, artist2_id)
                if pair_key in checked_pairs or self.graph.has_edge(artist1_id, artist2_id): continue
                
                checked_pairs.add(pair_key)
                
                # Check for direct collaborations between these two artists
                shared_works = self.db.get_connection_weight_between_artists(artist1_id, artist2_id)
                if shared_works > 0:
                    weight = shared_works * 1.5
                    self.graph.add_edge(
                        artist1_id, artist2_id,
                        weight=weight,
                        connection_type="interconnection"
                    )
                    interconnections_found += 1
        
        print(f"Added {interconnections_found} interconnections between collaborators")
    
    def build(self, artists, top_k_per_seed=15):
        """Build collaboration graph from seed Artist objects.
        
        1: Add seed artists and their collaborators from all query types.
        2: Filter to keep only top K connections per seed artist.
        3: Find interconnections between non-seed artists.
        
        Args:
            artists: List of Artist objects to use as seeds
            top_k_per_seed: Number of strongest connections to keep per seed artist (default: 15)
        """
        self.graph.clear()
        
        self.build_main_connections(artists)
        self.filter_graph(top_k_per_seed)
        self.build_interconnections()

    def clear(self):
        self.graph.clear()


class CollaborationNetworkPlot:
    """Visualizes collaboration networks with customizable colors and layout."""
    
    def __init__(self, graph):
        """Initialize plot with parameters and color scheme."""
        plt.rcParams['text.usetex'] = False
        plt.rcParams['text.parse_math'] = False
        self.colors = {
            "direct": "#B60000",            # Red
            "credit": "#B95900",            # Orange
            "featured": "#B4A200",          # Yellow
            "interconnection": "#008DCE"    # Blue
        }
        self.graph = graph
    
    def prepare(self):
        """Prepare graph for visualization."""
        self.G = nx.Graph()
        self.edge_to_type = {}  # For styling edges

        # Add nodes and edges with attributes
        for u, v, data in self.graph.edges(data=True):
            weight = data.get("weight", 0)
            connection_type = data.get("connection_type", "direct")
            
            u_name = self.graph.nodes[u].get("name", f"Artist {u}")
            v_name = self.graph.nodes[v].get("name", f"Artist {v}")
            is_seed_u = self.graph.nodes[u].get("is_seed", False)
            is_seed_v = self.graph.nodes[v].get("is_seed", False)
            
            self.G.add_node(u_name, is_seed=is_seed_u)
            self.G.add_node(v_name, is_seed=is_seed_v)
            self.G.add_edge(u_name, v_name, weight=weight, connection_type=connection_type)
            self.edge_to_type[(u_name, v_name)] = connection_type
        
        # Compute layout & styling
        self.pos = nx.spring_layout(self.G, k=0.8, weight="weight")
        self.style_edges()
        self.node_colors = [
            ("gold" if self.G.nodes[n].get("is_seed", False) else "skyblue")
            for n in self.G.nodes()
        ]
    
    def style_edges(self):
        """Style edge colors and widths."""
        edge_list = list(self.G.edges())

        self.edge_colors = []
        for u, v in edge_list:
            conn_type = self.edge_to_type.get((u, v), self.edge_to_type.get((v, u), "direct"))
            self.edge_colors.append(self.colors.get(conn_type, "#999999"))
        
        weights = [self.G[u][v]["weight"] for u, v in edge_list]
        max_w = max(weights) if weights else 1.0
        self.edge_widths = [1 + 4 * (w / max_w) for w in weights]
    
    def create_legend(self):
        """Create legend elements."""
        return [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10, label='Seed Artist'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Connected Artist'),
            Line2D([0], [0], color=self.colors["direct"], linewidth=2, label='Direct Collaboration'),
            Line2D([0], [0], color=self.colors["credit"], linewidth=2, label='Credit-based'),
            Line2D([0], [0], color=self.colors["featured"], linewidth=2, label='Featured'),
            Line2D([0], [0], color=self.colors["interconnection"], linewidth=2, label='Interconnection (artist-to-artist)')
        ]
    
    def print_statistics(self):
        """Print graph statistics."""
        
        print(f"\nVisualization graph: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        print(f"\nEdge type breakdown:")
        edge_type_counts = {}
        for conn_type in self.edge_to_type.values():
            edge_type_counts[conn_type] = edge_type_counts.get(conn_type, 0) + 1
        for edge_type, count in sorted(edge_type_counts.items()):
            print(f"  {edge_type}: {count}")
    
    def draw(self):
        """Draw the collaboration network."""
        fig, ax = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_nodes(self.G, self.pos, node_size=1200, node_color=self.node_colors, ax=ax)
        nx.draw_networkx_labels(self.G, self.pos, font_size=8, ax=ax)
        nx.draw_networkx_edges(self.G, self.pos, width=self.edge_widths, alpha=0.8, edge_color=self.edge_colors, ax=ax)
        
        ax.legend(handles=self.create_legend(), loc='upper left', fontsize=10)
        ax.axis("off")
        plt.title(f"Artist Collaboration Network")
        plt.tight_layout()
        plt.show()

    def build(self):
        self.prepare()
        self.print_statistics()
        self.draw()


class RecommendationEngine:
    """Generates artist and song recommendations from collaboration graphs."""
    
    def __init__(self, graph, database, seed_artists,
                top_n=10, top_n_primary=10, top_n_discovery=5,
                max_songs=2, max_songs_discovery=1):

        self.graph = graph
        self.db = database
        
        self.seed_artists = seed_artists
        self.seed_ids = {artist.id for artist in seed_artists}

        self.top_n = top_n
        self.top_n_primary = top_n_primary
        self.top_n_discovery = top_n_discovery
        self.max_songs = max_songs
        self.max_songs_discovery = max_songs_discovery

    def _is_valid_candidate(self, artist_id):
        """Check if artist is a valid recommendation candidate."""
        return (artist_id not in self.seed_ids and 
                not self.graph.graph.nodes[artist_id].get("is_seed", False))

    def _recommend_by_pagerank(self):
        """Recommend artists by PageRank centrality.
        
        Identifies the most "important" or influential nodes in the network.
        """
        pagerank = nx.pagerank(self.graph.graph)
        
        # Filter out seed artists and sort
        recommendations = [
            (artist_id, score) 
            for artist_id, score in pagerank.items() 
            if self._is_valid_candidate(artist_id)
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:self.top_n]

    def _recommend_by_common_neighbors(self):
        """Recommend artists by shared connections with seed artists.
        
        Recommends artists who collaborate with multiple seed artists.
        """
        seed_artist_ids = [a.id for a in self.seed_artists]
        recommendations = {}
        
        for seed_id in seed_artist_ids:
            seed_neighbors = set(self.graph.graph.neighbors(seed_id))
            for neighbor in seed_neighbors:
                if self._is_valid_candidate(neighbor):
                    recommendations[neighbor] = recommendations.get(neighbor, 0) + 1
        
        # Sort by number of shared seed connections
        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:self.top_n]

    def _recommend_by_betweenness_closeness(self, exclude_ids=None):
        """Recommend artists by betweenness * closeness centrality.
        
        Identifies bridge artists and hidden gems that connect communities.
        """
        exclude_ids = exclude_ids or set()
        
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(self.graph.graph, weight="weight")
        closeness = nx.closeness_centrality(self.graph.graph, distance="weight")
        
        # Compute combined score and filter
        recommendations = [
            (artist_id, betweenness[artist_id] * closeness[artist_id])
            for artist_id in self.graph.graph.nodes()
            if self._is_valid_candidate(artist_id) and artist_id not in exclude_ids
        ]
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:self.top_n]

    def _format_artist_results(self, artist_recs):
        """Convert artist recommendation tuples to DataFrame."""
        result = []
        for artist_id, score in artist_recs:
            artist_name = self.graph.graph.nodes[artist_id].get("name", f"Artist {artist_id}")
            result.append({
                "name": artist_name,
                "score": round(score, 4)
            })
        return pd.DataFrame(result)

    def _get_songs_for_artist(self, rec_artist_id):
        """Fetch and deduplicate songs between seed artists and a recommended artist."""
        all_songs = pd.DataFrame()
        
        # Combine songs from all seed artists
        for seed_artist in self.seed_artists:
            songs = self.db.get_shared_recordings(seed_artist.id, rec_artist_id)
            if not songs.empty:
                songs['seed_artist_name'] = seed_artist.name
                all_songs = pd.concat([all_songs, songs], ignore_index=True)
        
        if all_songs.empty:
            return []
        
        # Deduplicate by recording_name and get top songs
        unique_songs = all_songs.drop_duplicates(subset=['recording_name'])
        result = []
        
        for _, song in unique_songs.head(self.max_songs).iterrows():
            seed_artists_for_song = all_songs[
                all_songs['recording_name'] == song['recording_name']
            ]['seed_artist_name'].unique().tolist()
            
            result.append({
                "recording_name": song['recording_name'],
                "seed_artists": seed_artists_for_song
            })
        
        return result

    def _build_song_dataframe(self, artist_recs, value_column):
        """Build song recommendation DataFrame from artist recommendations."""
        result = []
        seen_songs = set()
        
        for rec_artist_id, value in artist_recs:
            rec_artist_name = self.graph.graph.nodes[rec_artist_id].get("name", f"Artist {rec_artist_id}")
            songs = self._get_songs_for_artist(rec_artist_id)
            
            for song_data in songs:
                song_key = (song_data['recording_name'], rec_artist_id)
                if song_key not in seen_songs:
                    seen_songs.add(song_key)
                    result.append({
                        "song": song_data['recording_name'],
                        "seed_artists": ", ".join(song_data['seed_artists']),
                        "recommended_artist": rec_artist_name,
                        value_column: round(value, 4) if isinstance(value, float) else value
                    })
        
        return pd.DataFrame(result)

    def get_artists(self):
        """
        Get primary artist recommendations by PageRank centrality.
        
        Returns:
            pd.DataFrame: Primary artist recommendations with name and score columns
        """
        primary_recs = self._recommend_by_pagerank()
        return self._format_artist_results(primary_recs)
    
    def get_discovery_artists(self):
        """
        Get discovery artist recommendations by betweenness * closeness centrality.
        
        Excludes artists from primary recommendations to avoid duplicates.
        
        Returns:
            pd.DataFrame: Discovery artist recommendations with name and score columns
        """
        primary_recs = self._recommend_by_pagerank()
        primary_artist_ids = {artist_id for artist_id, _ in primary_recs}
        
        discovery = self._recommend_by_betweenness_closeness(primary_artist_ids)
        return self._format_artist_results(discovery)
    
    def get_songs(self):
        """
        Get song recommendations from artists who collaborate with multiple seed artists.
        
        Returns:
            pd.DataFrame: Song recommendations with song, seed_artists, recommended_artist, and shared_with columns
        """
        shared_artists = self._recommend_by_common_neighbors()
        return self._build_song_dataframe(shared_artists, "shared_with")
    
    def get_discovery_songs(self):
        """
        Get discovery song recommendations identified by betweenness * closeness centrality,
        while excluding artists found in primary song recommendations.
        
        Returns:
            pd.DataFrame: Discovery song recommendations with song, seed_artists, recommended_artist, and score columns
        """
        shared_artists = self._recommend_by_common_neighbors()
        shared_artist_ids = {artist_id for artist_id, _ in shared_artists}
        
        discovery_artists = self._recommend_by_betweenness_closeness(shared_artist_ids)
        return self._build_song_dataframe(discovery_artists, "score")
