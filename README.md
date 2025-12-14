# Music Artist Collaboration-based Recommendation System

## Importing the PostgreSQL Database

1. Download the [database dump file](https://drive.google.com/file/d/1Zl4OnngNfAO6alJnE7oQeNRBJLxsYtB1/view?usp=sharing)
2. Run the import commands ```
createuser musicbrainz
createdb -O musicbrainz musicbrainz
psql -d musicbrainz -c "CREATE SCHEMA musicbrainz;"
pg_restore -d musicbrainz musicbrainz_db.dump```
3. Edit config.py variables
