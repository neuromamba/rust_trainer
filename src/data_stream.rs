use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardedCursorState {
    pub shard_order: Vec<usize>,
    pub shard_pos: usize,
    pub token_pos: usize,
    pub epoch: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiWorkerCursorState {
    pub worker_states: Vec<ShardedCursorState>,
}

pub struct ShardedTokenStream {
    shards: Vec<PathBuf>,
    shuffle_shards: bool,
    rng: StdRng,
    state: ShardedCursorState,
    current_tokens: Vec<i64>,
}

#[derive(Debug)]
struct WorkerPacket {
    worker_id: usize,
    ids: Array2<i64>,
    targets: Array2<i64>,
    state: ShardedCursorState,
}

pub struct MultiWorkerShardedBatcher {
    rx: Receiver<Result<WorkerPacket, String>>,
    worker_states: Vec<Option<ShardedCursorState>>,
}

impl MultiWorkerShardedBatcher {
    pub fn from_dir<P: AsRef<Path>>(
        dir: P,
        extension_filter: &str,
        shuffle_shards: bool,
        seed: u64,
        worker_count: usize,
        prefetch_buffer: usize,
        batch: usize,
        seq_len: usize,
        packed: bool,
        restore: Option<MultiWorkerCursorState>,
    ) -> Result<Self, String> {
        let shards = list_shards(dir.as_ref(), extension_filter)?;
        if shards.is_empty() {
            return Err("no token shard files found in token dir".to_string());
        }
        let worker_count = worker_count.max(1).min(shards.len());
        let groups = partition_paths(&shards, worker_count);

        let cap = prefetch_buffer.max(1);
        let (tx, rx) = sync_channel::<Result<WorkerPacket, String>>(cap);
        for (wid, group) in groups.into_iter().enumerate() {
            let txc = tx.clone();
            let worker_seed = seed ^ ((wid as u64 + 1) * 0x9E37_79B9);
            let restore_state = restore
                .as_ref()
                .and_then(|rs| rs.worker_states.get(wid).cloned());
            thread::spawn(move || {
                let mut stream = match ShardedTokenStream::from_paths(group, shuffle_shards, worker_seed) {
                    Ok(s) => s,
                    Err(err) => {
                        let _ = txc.send(Err(err));
                        return;
                    }
                };
                if let Some(st) = restore_state {
                    if let Err(err) = stream.set_state(st) {
                        let _ = txc.send(Err(format!("restore multi-worker state failed: {err}")));
                        return;
                    }
                }
                loop {
                    let batch_out = if packed {
                        stream.next_packed_batch(batch, seq_len)
                    } else {
                        stream.next_batch(batch, seq_len)
                    };
                    match batch_out {
                        Ok((ids, targets)) => {
                            let pkt = WorkerPacket {
                                worker_id: wid,
                                ids,
                                targets,
                                state: stream.state(),
                            };
                            if txc.send(Ok(pkt)).is_err() {
                                break;
                            }
                        }
                        Err(err) => {
                            let _ = txc.send(Err(err));
                            break;
                        }
                    }
                }
            });
        }
        drop(tx);

        Ok(Self {
            rx,
            worker_states: vec![None; worker_count],
        })
    }

    pub fn next_batch(&mut self) -> Result<(Array2<i64>, Array2<i64>), String> {
        match self.rx.recv() {
            Ok(Ok(pkt)) => {
                self.worker_states[pkt.worker_id] = Some(pkt.state);
                Ok((pkt.ids, pkt.targets))
            }
            Ok(Err(err)) => Err(err),
            Err(err) => Err(format!("multi-worker channel closed: {err}")),
        }
    }

    pub fn state(&self) -> Option<MultiWorkerCursorState> {
        let mut out = Vec::with_capacity(self.worker_states.len());
        for st in &self.worker_states {
            if let Some(v) = st {
                out.push(v.clone());
            } else {
                return None;
            }
        }
        Some(MultiWorkerCursorState { worker_states: out })
    }
}

impl ShardedTokenStream {
    pub fn from_dir<P: AsRef<Path>>(
        dir: P,
        extension_filter: &str,
        shuffle_shards: bool,
        seed: u64,
    ) -> Result<Self, String> {
        let shards = list_shards(dir.as_ref(), extension_filter)?;
        Self::from_paths(shards, shuffle_shards, seed)
    }

    pub fn from_paths(shards: Vec<PathBuf>, shuffle_shards: bool, seed: u64) -> Result<Self, String> {
        if shards.is_empty() {
            return Err("empty shard list".to_string());
        }
        let mut shard_order = (0..shards.len()).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(seed);
        if shuffle_shards {
            shard_order.shuffle(&mut rng);
        }

        let mut stream = Self {
            shards,
            shuffle_shards,
            rng,
            state: ShardedCursorState {
                shard_order,
                shard_pos: 0,
                token_pos: 0,
                epoch: 0,
            },
            current_tokens: Vec::new(),
        };
        stream.load_current_shard_tokens()?;
        Ok(stream)
    }

    pub fn set_state(&mut self, state: ShardedCursorState) -> Result<(), String> {
        if state.shard_order.is_empty() || state.shard_order.len() != self.shards.len() {
            return Err("invalid shard order in restored cursor state".to_string());
        }
        if state.shard_pos >= state.shard_order.len() {
            return Err("invalid shard_pos in restored cursor state".to_string());
        }
        self.state = state;
        self.load_current_shard_tokens()?;
        if self.state.token_pos + 1 >= self.current_tokens.len() {
            self.state.token_pos = 0;
        }
        Ok(())
    }

    pub fn state(&self) -> ShardedCursorState {
        self.state.clone()
    }

    pub fn next_batch(&mut self, batch: usize, seq_len: usize) -> Result<(Array2<i64>, Array2<i64>), String> {
        let mut ids = Array2::<i64>::zeros((batch, seq_len));
        let mut targets = Array2::<i64>::zeros((batch, seq_len));

        for b in 0..batch {
            let window = self.next_window(seq_len + 1)?;
            for t in 0..seq_len {
                ids[(b, t)] = window[t];
                targets[(b, t)] = window[t + 1];
            }
        }

        Ok((ids, targets))
    }

    pub fn next_packed_batch(&mut self, batch: usize, seq_len: usize) -> Result<(Array2<i64>, Array2<i64>), String> {
        let total = batch * seq_len;
        let packed = self.next_window(total + 1)?;
        let mut ids = Array2::<i64>::zeros((batch, seq_len));
        let mut targets = Array2::<i64>::zeros((batch, seq_len));
        for b in 0..batch {
            let off = b * seq_len;
            for t in 0..seq_len {
                ids[(b, t)] = packed[off + t];
                targets[(b, t)] = packed[off + t + 1];
            }
        }
        Ok((ids, targets))
    }

    pub fn max_token_plus_one(mut self) -> Result<usize, String> {
        let mut max_tok = 0i64;
        for path in &self.shards {
            let toks = parse_token_file(path)?;
            for tok in toks {
                if tok > max_tok {
                    max_tok = tok;
                }
            }
        }
        self.current_tokens.clear();
        Ok((max_tok.saturating_add(1)).max(1) as usize)
    }

    fn next_window(&mut self, len: usize) -> Result<Vec<i64>, String> {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            if self.current_tokens.len() < 2 {
                self.advance_shard()?;
                continue;
            }
            if self.state.token_pos + 1 >= self.current_tokens.len() {
                self.advance_shard()?;
                continue;
            }

            out.push(self.current_tokens[self.state.token_pos]);
            self.state.token_pos += 1;
        }
        Ok(out)
    }

    fn advance_shard(&mut self) -> Result<(), String> {
        self.state.token_pos = 0;
        self.state.shard_pos += 1;
        if self.state.shard_pos >= self.state.shard_order.len() {
            self.state.shard_pos = 0;
            self.state.epoch += 1;
            if self.shuffle_shards {
                self.state.shard_order.shuffle(&mut self.rng);
            }
        }
        self.load_current_shard_tokens()
    }

    fn load_current_shard_tokens(&mut self) -> Result<(), String> {
        let shard_idx = self.state.shard_order[self.state.shard_pos];
        let path = &self.shards[shard_idx];
        self.current_tokens = parse_token_file(path)?;
        Ok(())
    }
}

pub fn max_token_plus_one_from_dir<P: AsRef<Path>>(dir: P, extension_filter: &str) -> Result<usize, String> {
    let shards = list_shards(dir.as_ref(), extension_filter)?;
    let mut max_tok = 0i64;
    for path in &shards {
        let toks = parse_token_file(path)?;
        for tok in toks {
            if tok > max_tok {
                max_tok = tok;
            }
        }
    }
    Ok((max_tok.saturating_add(1)).max(1) as usize)
}

fn list_shards(dir: &Path, extension_filter: &str) -> Result<Vec<PathBuf>, String> {
    let mut shards = fs::read_dir(dir)
        .map_err(|err| format!("failed reading token dir: {err}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == extension_filter)
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    shards.sort();
    if shards.is_empty() {
        return Err("no token shard files found in token dir".to_string());
    }
    Ok(shards)
}

fn partition_paths(shards: &[PathBuf], workers: usize) -> Vec<Vec<PathBuf>> {
    let mut out = vec![Vec::new(); workers];
    for (i, p) in shards.iter().enumerate() {
        out[i % workers].push(p.clone());
    }
    out
}

fn parse_token_file(path: &Path) -> Result<Vec<i64>, String> {
    let raw = fs::read_to_string(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut out = Vec::new();
    for part in raw.split_whitespace() {
        let parsed = part
            .parse::<i64>()
            .map_err(|err| format!("bad token '{part}' in {}: {err}", path.display()))?;
        out.push(parsed);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharded_stream_is_resume_stable() {
        let dir = std::env::temp_dir().join("sharded_stream_resume_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "1 2 3 4 5 6 7 8 9").unwrap();
        fs::write(dir.join("b.txt"), "10 11 12 13 14 15 16 17 18").unwrap();

        let mut s1 = ShardedTokenStream::from_dir(&dir, "txt", false, 9).unwrap();
        let (_ids1, _tgt1) = s1.next_batch(2, 4).unwrap();
        let st = s1.state();
        let (ids2a, tgt2a) = s1.next_batch(2, 4).unwrap();

        let mut s2 = ShardedTokenStream::from_dir(&dir, "txt", false, 9).unwrap();
        s2.set_state(st).unwrap();
        let (ids2b, tgt2b) = s2.next_batch(2, 4).unwrap();

        assert_eq!(ids2a, ids2b);
        assert_eq!(tgt2a, tgt2b);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn packed_batch_returns_expected_shape() {
        let dir = std::env::temp_dir().join("packed_shape_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "1 2 3 4 5 6 7 8 9 10 11 12").unwrap();

        let mut s = ShardedTokenStream::from_dir(&dir, "txt", false, 17).unwrap();
        let (ids, tgt) = s.next_packed_batch(3, 3).unwrap();
        assert_eq!(ids.dim(), (3, 3));
        assert_eq!(tgt.dim(), (3, 3));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn multiworker_batcher_produces_batches() {
        let dir = std::env::temp_dir().join("multiworker_batcher_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "1 2 3 4 5 6 7 8 9 10 11 12 13 14").unwrap();
        fs::write(dir.join("b.txt"), "15 16 17 18 19 20 21 22 23 24 25 26 27 28").unwrap();

        let mut m =
            MultiWorkerShardedBatcher::from_dir(&dir, "txt", true, 23, 2, 4, 2, 4, true, None)
                .unwrap();
        let (ids, tgt) = m.next_batch().unwrap();
        assert_eq!(ids.dim(), (2, 4));
        assert_eq!(tgt.dim(), (2, 4));
        let _ = fs::remove_dir_all(&dir);
    }
}
