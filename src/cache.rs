//! Finished cells on disk, in the plugin's cache directory, so a view is not
//! predicted again every session (CEL-11).
//!
//! A cell is stored under a key made of everything that decides its
//! contents: the slide's identity (from the host), the model file's
//! SHA-256, the grid, the markers and their display windows, and the format
//! version. Slides without an identity are not cached. Files are written
//! atomically and the directory is pruned, oldest first, to its budget.

use crate::blend::{BlockStat, CellData, TissueCell};
use crate::geometry::{BLOCK, Grid, Idx};
use crate::markers::Marker;
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Bump whenever what a cell holds, or how it is computed, changes.
const FORMAT: u32 = 2;
const MAGIC: &[u8; 4] = b"CELC";

/// The cache key of a slide's cells.
pub fn key(
    identity: &str,
    model_sha256: &str,
    grid: &Grid,
    markers: &[Marker],
    recon: bool,
) -> String {
    let mut hasher = Sha256::new();
    let mut put = |text: String| {
        hasher.update(text.as_bytes());
        hasher.update([0]);
    };
    put(format!("format {FORMAT}"));
    put(identity.to_string());
    put(model_sha256.to_string());
    put(format!(
        "{:.9} {:.9} {} {} {} {:.6}",
        grid.sx, grid.sy, grid.window, grid.level, grid.stride, grid.downsample
    ));
    for marker in markers {
        put(format!(
            "{} {} {:.4} {:.4}",
            marker.name, marker.channel, marker.low, marker.high
        ));
    }
    put(format!("recon {recon}"));
    let digest = hasher.finalize();
    digest.iter().take(16).map(|b| format!("{b:02x}")).collect()
}

fn encode(data: &CellData, stride: usize, markers: usize, recon: bool) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT.to_le_bytes());
    match data {
        CellData::Empty => out.push(0),
        CellData::Dark => out.push(1),
        CellData::Tissue(cell) => {
            out.push(2);
            out.extend_from_slice(&(stride as u16).to_le_bytes());
            out.push(markers as u8);
            out.push(u8::from(recon));
            out.extend_from_slice(&cell.mask);
            out.extend_from_slice(&cell.planes);
            for block in &cell.blocks {
                out.extend_from_slice(&block.od.to_le_bytes());
                out.extend_from_slice(&block.coverage.to_le_bytes());
                out.extend_from_slice(&block.dark.to_le_bytes());
                for mean in &block.means {
                    out.extend_from_slice(&mean.to_le_bytes());
                }
                if recon {
                    out.extend_from_slice(&block.recon.unwrap_or(f32::NAN).to_le_bytes());
                }
            }
        }
    }
    out
}

fn decode(bytes: &[u8], stride: usize, markers: usize, recon: bool) -> Option<CellData> {
    let mut at = 0usize;
    let mut take = |n: usize| {
        let slice = bytes.get(at..at + n)?;
        at += n;
        Some(slice)
    };
    if take(4)? != MAGIC || u32::from_le_bytes(take(4)?.try_into().ok()?) != FORMAT {
        return None;
    }
    let data = match take(1)?[0] {
        0 => CellData::Empty,
        1 => CellData::Dark,
        2 => {
            let stored_stride = u16::from_le_bytes(take(2)?.try_into().ok()?) as usize;
            let stored_markers = take(1)?[0] as usize;
            let stored_recon = take(1)?[0] != 0;
            if (stored_stride, stored_markers, stored_recon) != (stride, markers, recon) {
                return None;
            }
            let area = stride * stride;
            let mask = take(area)?.to_vec();
            let planes = take(markers * area)?.to_vec();
            let per = stride / BLOCK;
            let mut float = || {
                take(4)
                    .and_then(|b| b.try_into().ok())
                    .map(f32::from_le_bytes)
            };
            let mut blocks = Vec::with_capacity(per * per);
            for _ in 0..per * per {
                let od = float()?;
                let coverage = float()?;
                let dark = float()?;
                let means = (0..markers)
                    .map(|_| float())
                    .collect::<Option<Vec<f32>>>()?;
                let recon = if recon {
                    Some(float()?).filter(|v| !v.is_nan())
                } else {
                    None
                };
                blocks.push(BlockStat {
                    od,
                    coverage,
                    dark,
                    means,
                    recon,
                });
            }
            CellData::Tissue(Box::new(TissueCell {
                planes,
                mask,
                blocks,
            }))
        }
        _ => return None,
    };
    (at == bytes.len()).then_some(data)
}

/// The cells of one slide under one key.
#[derive(Clone, Debug, PartialEq)]
pub struct Shelf {
    dir: PathBuf,
    stride: usize,
    markers: usize,
    recon: bool,
}

impl Shelf {
    pub fn new(root: &Path, key: &str, stride: usize, markers: usize, recon: bool) -> Self {
        Self {
            dir: root.join(key),
            stride,
            markers,
            recon,
        }
    }

    fn path(&self, cell: Idx) -> PathBuf {
        self.dir.join(format!("{}_{}.cell", cell.i, cell.j))
    }

    /// A stored cell; `None` when absent or unreadable.
    pub fn load(&self, cell: Idx) -> Option<CellData> {
        let bytes = fs::read(self.path(cell)).ok()?;
        decode(&bytes, self.stride, self.markers, self.recon)
    }

    /// Stores a cell, atomically.
    pub fn store(&self, cell: Idx, data: &CellData) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let path = self.path(cell);
        let temporary = path.with_extension(format!("tmp{}", std::process::id()));
        fs::write(
            &temporary,
            encode(data, self.stride, self.markers, self.recon),
        )?;
        fs::rename(&temporary, &path).inspect_err(|_| {
            let _ = fs::remove_file(&temporary);
        })
    }
}

/// Deletes the oldest files under `root` until it holds at most `budget`
/// bytes (down to 80 % of it, so pruning is not constant). Returns the bytes
/// removed.
pub fn prune(root: &Path, budget: u64) -> io::Result<u64> {
    let mut files = Vec::new();
    let Ok(shelves) = fs::read_dir(root) else {
        return Ok(0);
    };
    for shelf in shelves.filter_map(Result::ok) {
        let Ok(entries) = fs::read_dir(shelf.path()) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            if let Ok(meta) = entry.metadata()
                && meta.is_file()
            {
                let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                files.push((modified, meta.len(), entry.path()));
            }
        }
    }
    let total: u64 = files.iter().map(|f| f.1).sum();
    if total <= budget {
        return Ok(0);
    }
    files.sort_by_key(|f| f.0);
    let target = budget / 5 * 4;
    let mut remaining = total;
    let mut removed = 0;
    for (_, size, path) in files {
        if remaining <= target {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            remaining -= size;
            removed += size;
        }
    }
    // Empty shelves go too.
    if let Ok(shelves) = fs::read_dir(root) {
        for shelf in shelves.filter_map(Result::ok) {
            let _ = fs::remove_dir(shelf.path());
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Scale;
    use eov_plugin_api::protocol::FileInfo;

    fn tissue(stride: usize, markers: usize) -> CellData {
        let per = stride / BLOCK;
        CellData::Tissue(Box::new(TissueCell {
            planes: (0..markers * stride * stride).map(|i| i as u8).collect(),
            mask: vec![200; stride * stride],
            blocks: (0..per * per)
                .map(|i| BlockStat {
                    od: i as f32 / 10.0,
                    coverage: 1.0,
                    dark: 0.0,
                    means: (0..markers).map(|k| k as f32 + 0.5).collect(),
                    recon: Some(0.25),
                })
                .collect(),
        }))
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cellaris-cache-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn cells_round_trip_and_mismatches_are_misses() {
        let root = temp("round");
        let shelf = Shelf::new(&root, "k", 32, 3, true);
        for data in [CellData::Empty, CellData::Dark, tissue(32, 3)] {
            shelf.store(Idx::new(2, 5), &data).unwrap();
            assert_eq!(shelf.load(Idx::new(2, 5)), Some(data));
        }
        assert_eq!(shelf.load(Idx::new(9, 9)), None);
        // A shelf expecting two markers does not read three.
        let other = Shelf::new(&root, "k", 32, 2, true);
        assert_eq!(other.load(Idx::new(2, 5)), None);
        // A damaged file is a miss, not a panic.
        fs::write(shelf.path(Idx::new(2, 5)), b"CELC\x02\x00\x00\x00\x02\x20").unwrap();
        assert_eq!(shelf.load(Idx::new(2, 5)), None);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn keys_change_with_anything_that_changes_a_cell() {
        let info = FileInfo {
            file_id: 1,
            width: 10_000,
            height: 10_000,
            mpp_x: Some(0.25),
            mpp_y: Some(0.25),
            ..Default::default()
        };
        let grid = Grid::new(&info, Scale::of(&info, 0.325, false).unwrap(), 0.325, 256).unwrap();
        let names: Vec<String> = ["Hoechst".to_string()].to_vec();
        let markers =
            crate::markers::select(&["Hoechst".to_string(), "SMA".to_string()], &names).unwrap();
        let base = key("slide", "abc", &grid, &markers, true);
        assert_ne!(base, key("other", "abc", &grid, &markers, true));
        assert_ne!(base, key("slide", "abd", &grid, &markers, true));
        assert_ne!(base, key("slide", "abc", &grid, &markers, false));
        let mut finer = grid.clone();
        finer.sx *= 1.01;
        assert_ne!(base, key("slide", "abc", &finer, &markers, true));
        assert_eq!(base, key("slide", "abc", &grid, &markers, true));
    }

    #[test]
    fn pruning_removes_the_oldest_files_first() {
        let root = temp("prune");
        let shelf = Shelf::new(&root, "k", 32, 3, false);
        for i in 0..10 {
            shelf.store(Idx::new(i, 0), &tissue(32, 3)).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        let size = fs::metadata(shelf.path(Idx::new(0, 0))).unwrap().len();
        let removed = prune(&root, size * 5).unwrap();
        assert!(removed >= size * 6, "removed {removed}");
        assert!(shelf.load(Idx::new(0, 0)).is_none(), "the oldest survived");
        assert!(shelf.load(Idx::new(9, 0)).is_some(), "the newest went");
        assert_eq!(prune(&root, u64::MAX).unwrap(), 0);
        let _ = fs::remove_dir_all(root);
    }
}
