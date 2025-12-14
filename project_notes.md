# Data Mining Project

Create a new method of determining and recommending similar artists based on artist collaborations. The database being used for this is a trimmed version of the open source MusicBrainz postgresql database.

## Plans for a Graph Mining Approach

Build a graph of artist collaborations from input artist(s). Use the graph to determine artist connections and commmunities/clusters to help inform recommendations of new artists and recordings containing input artist and/or similar artists.

### Nodes: Artists

- Person
- Group
- Band

### Weighted Edges: Collaborations

#### Artist Credits Found in 'recording' table

- Features
- Collaborations

#### Additional Credits Found in 'l_artist_recording' and 'l_artist_work' link tables (Specific Types w/ Varying Weight)

##### Recording Credits

- Performer
  - Vocalist (High)
  - Instrumentalist (Medium)
- Arranger (Low)
- Producer (Medium)

##### Work Credits

- Writer
  - Composer (Low)
  - Lyricist (Low)
- Arranger (Low)

## Libraries Being Used

- networkx
- pandas
- sqlalchemy
- matplotlib
