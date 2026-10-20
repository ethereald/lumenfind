use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use rusqlite::{Connection, OptionalExtension, params};

pub const EMBEDDING_DIM: usize = 768;
const MAX_RECENCY_BONUS: f32 = 0.02;
const RECENCY_HALF_LIFE_YEARS: f64 = 5.0;
const MILLISECONDS_PER_YEAR: f64 = 365.25 * 24.0 * 60.0 * 60.0 * 1000.0;

#[derive(Clone, Debug)]
pub struct PhotoRecord {
    pub path: PathBuf,
    pub modified_unix_ms: i64,
    pub byte_len: i64,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub path: PathBuf,
    pub score: f32,
}

pub struct PhotoDb {
    connection: Connection,
}

impl PhotoDb {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(path)?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "NORMAL")?;
        connection.execute_batch(
            "            CREATE TABLE IF NOT EXISTS photos (
               path TEXT PRIMARY KEY NOT NULL,
               modified_ms INTEGER NOT NULL,
               byte_len INTEGER NOT NULL,
               embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS photos_modified ON photos(modified_ms);",
        )?;
        Ok(Self { connection })
    }

    pub fn is_current(&self, path: &Path, modified_ms: i64, byte_len: i64) -> Result<bool> {
        let value: Option<(i64, i64)> = self
            .connection
            .query_row(
                "SELECT modified_ms, byte_len FROM photos WHERE path = ?1",
                [path.to_string_lossy().as_ref()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        Ok(value == Some((modified_ms, byte_len)))
    }

    pub fn upsert(&self, photo: &PhotoRecord) -> Result<()> {
        if photo.embedding.len() != EMBEDDING_DIM {
            bail!(
                "embedding has {} values, expected {EMBEDDING_DIM}",
                photo.embedding.len()
            );
        }
        let bytes = bytemuck::cast_slice::<f32, u8>(&photo.embedding);
        self.connection.execute(
            "INSERT INTO photos(path, modified_ms, byte_len, embedding)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(path) DO UPDATE SET
               modified_ms=excluded.modified_ms,
               byte_len=excluded.byte_len,
               embedding=excluded.embedding",
            params![
                photo.path.to_string_lossy(),
                photo.modified_unix_ms,
                photo.byte_len,
                bytes
            ],
        )?;
        Ok(())
    }

    pub fn remove_missing(&self, roots: &[PathBuf], seen: &HashSet<PathBuf>) -> Result<usize> {
        let paths = self.all_paths()?;
        let mut removed = 0;
        for path in paths {
            if roots.iter().any(|root| path.starts_with(root)) && !seen.contains(&path) {
                removed += self.connection.execute(
                    "DELETE FROM photos WHERE path = ?1",
                    [path.to_string_lossy().as_ref()],
                )?;
            }
        }
        Ok(removed)
    }

    pub fn all_paths(&self) -> Result<Vec<PathBuf>> {
        let mut statement = self.connection.prepare("SELECT path FROM photos")?;
        let values = statement.query_map([], |row| row.get::<_, String>(0))?;
        Ok(values.filter_map(Result::ok).map(PathBuf::from).collect())
    }

    pub fn count(&self) -> Result<usize> {
        Ok(self
            .connection
            .query_row("SELECT COUNT(*) FROM photos", [], |row| row.get(0))?)
    }

    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchHit>> {
        if query.len() != EMBEDDING_DIM {
            bail!("query embedding has the wrong dimension")
        }
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock predates 1970")?
            .as_millis() as i64;
        let mut statement = self
            .connection
            .prepare("SELECT path, modified_ms, embedding FROM photos")?;
        let mut rows = statement.query([])?;
        let mut hits = Vec::new();
        while let Some(row) = rows.next()? {
            let path: String = row.get(0)?;
            let modified_ms: i64 = row.get(1)?;
            let bytes: Vec<u8> = row.get(2)?;
            if bytes.len() != EMBEDDING_DIM * size_of::<f32>() {
                tracing::warn!(path, "ignoring a corrupt embedding row");
                continue;
            }
            let vector: &[f32] = bytemuck::try_cast_slice(&bytes)
                .map_err(|error| anyhow!("invalid embedding for {path}: {error}"))?;
            let score = query.iter().zip(vector).map(|(a, b)| a * b).sum();
            let rank = score + recency_bonus(modified_ms, now_ms);
            hits.push((
                SearchHit {
                    path: path.into(),
                    score,
                },
                rank,
            ));
        }
        hits.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        hits.truncate(limit);
        Ok(hits.into_iter().map(|(hit, _)| hit).collect())
    }
}

fn recency_bonus(modified_ms: i64, now_ms: i64) -> f32 {
    let age_years = now_ms.saturating_sub(modified_ms).max(0) as f64 / MILLISECONDS_PER_YEAR;
    MAX_RECENCY_BONUS * (-(age_years / RECENCY_HALF_LIFE_YEARS)).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_at(index: usize) -> Vec<f32> {
        let mut value = vec![0.0; EMBEDDING_DIM];
        value[index] = 1.0;
        value
    }

    #[test]
    fn ranks_by_cosine_dot_product() {
        let temporary = tempfile::tempdir().unwrap();
        let db = PhotoDb::open(&temporary.path().join("test.db")).unwrap();
        for (name, embedding) in [("cat.jpg", unit_at(3)), ("tree.jpg", unit_at(7))] {
            db.upsert(&PhotoRecord {
                path: name.into(),
                modified_unix_ms: 1,
                byte_len: 1,
                embedding,
            })
            .unwrap();
        }
        let hits = db.search(&unit_at(7), 1).unwrap();
        assert_eq!(hits[0].path, PathBuf::from("tree.jpg"));
        assert_eq!(hits[0].score, 1.0);
    }

    #[test]
    fn recent_photos_can_win_when_relevance_is_nearly_tied() {
        let temporary = tempfile::tempdir().unwrap();
        let db = PhotoDb::open(&temporary.path().join("test.db")).unwrap();
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let mut slightly_less_relevant = vec![0.0; EMBEDDING_DIM];
        slightly_less_relevant[0] = 0.99;
        slightly_less_relevant[1] = (1.0_f32 - 0.99_f32.powi(2)).sqrt();
        db.upsert(&PhotoRecord {
            path: "recent.jpg".into(),
            modified_unix_ms: now_ms,
            byte_len: 1,
            embedding: slightly_less_relevant,
        })
        .unwrap();
        db.upsert(&PhotoRecord {
            path: "old.jpg".into(),
            modified_unix_ms: now_ms - (10.0 * MILLISECONDS_PER_YEAR) as i64,
            byte_len: 1,
            embedding: unit_at(0),
        })
        .unwrap();

        let hits = db.search(&unit_at(0), 2).unwrap();

        assert_eq!(hits[0].path, PathBuf::from("recent.jpg"));
        assert!((hits[0].score - 0.99).abs() < f32::EPSILON);
        assert_eq!(hits[1].path, PathBuf::from("old.jpg"));
        assert_eq!(hits[1].score, 1.0);
    }

    #[test]
    fn unchanged_metadata_keeps_the_existing_embedding_current() {
        let temporary = tempfile::tempdir().unwrap();
        let db = PhotoDb::open(&temporary.path().join("test.db")).unwrap();
        let path = PathBuf::from("unchanged.jpg");
        db.upsert(&PhotoRecord {
            path: path.clone(),
            modified_unix_ms: 42,
            byte_len: 100,
            embedding: unit_at(2),
        })
        .unwrap();

        assert!(db.is_current(&path, 42, 100).unwrap());
        assert!(!db.is_current(&path, 43, 100).unwrap());
        assert!(!db.is_current(&path, 42, 101).unwrap());
    }

    #[test]
    fn recent_photos_receive_a_small_bounded_ranking_bonus() {
        let now_ms = 1_800_000_000_000;
        let recent = recency_bonus(now_ms, now_ms);
        let old = recency_bonus(now_ms - (10.0 * MILLISECONDS_PER_YEAR) as i64, now_ms);

        assert!(recent > old);
        assert!((recent - MAX_RECENCY_BONUS).abs() < f32::EPSILON);
        assert!(old > 0.0);
        assert_eq!(recency_bonus(now_ms + 1, now_ms), recent);
    }
}
