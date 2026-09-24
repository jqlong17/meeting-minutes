use super::types::DiarizationOptions;

pub fn cluster_embeddings(embeddings: &[Vec<f32>], options: DiarizationOptions) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let min_k = options.min_speakers.max(1).min(n);
    let max_k = options.max_speakers.max(min_k).min(n);
    let distances = pairwise_cosine_distances(embeddings);

    let mut best_labels = vec![0usize; n];
    let mut best_score = f32::NEG_INFINITY;

    for k in min_k..=max_k {
        let labels = agglomerative_cluster(&distances, k);
        let score = silhouette_score(&distances, &labels);
        if score > best_score {
            best_score = score;
            best_labels = labels;
        }
    }

    relabel_contiguous(&best_labels)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn pairwise_cosine_distances(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut distances = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = 1.0 - cosine_similarity(&embeddings[i], &embeddings[j]);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }
    distances
}

fn agglomerative_cluster(distances: &[Vec<f32>], target_k: usize) -> Vec<usize> {
    let n = distances.len();
    if target_k >= n {
        return (0..n).collect();
    }

    let mut cluster_ids = (0..n).collect::<Vec<_>>();
    let mut cluster_members = (0..n).map(|i| vec![i]).collect::<Vec<_>>();

    while cluster_members.len() > target_k {
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut best_dist = f32::INFINITY;

        for i in 0..cluster_members.len() {
            for j in (i + 1)..cluster_members.len() {
                let dist = average_linkage(distances, &cluster_members[i], &cluster_members[j]);
                if dist < best_dist {
                    best_dist = dist;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let merged = {
            let right = cluster_members.remove(best_j);
            let left = &mut cluster_members[best_i];
            left.extend(right);
            left.clone()
        };
        cluster_members[best_i] = merged;
    }

    let mut labels = vec![0usize; n];
    for (cluster_index, members) in cluster_members.iter().enumerate() {
        for &member in members {
            labels[member] = cluster_index;
        }
    }
    labels
}

fn average_linkage(distances: &[Vec<f32>], left: &[usize], right: &[usize]) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for &i in left {
        for &j in right {
            sum += distances[i][j];
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn silhouette_score(distances: &[Vec<f32>], labels: &[usize]) -> f32 {
    let n = labels.len();
    if n <= 1 {
        return 0.0;
    }

    let mut total = 0.0f32;
    for i in 0..n {
        let same_cluster: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter_map(|(idx, label)| {
                if idx != i && *label == labels[i] {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        let a = if same_cluster.is_empty() {
            0.0
        } else {
            same_cluster.iter().map(|j| distances[i][*j]).sum::<f32>() / same_cluster.len() as f32
        };

        let mut b = f32::INFINITY;
        let unique_labels: Vec<usize> = labels
            .iter()
            .copied()
            .filter(|label| *label != labels[i])
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        for other in unique_labels {
            let members: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter_map(
                    |(idx, label)| {
                        if *label == other {
                            Some(idx)
                        } else {
                            None
                        }
                    },
                )
                .collect();
            if members.is_empty() {
                continue;
            }
            let mean = members.iter().map(|j| distances[i][*j]).sum::<f32>() / members.len() as f32;
            b = b.min(mean);
        }

        let s = if b.is_infinite() {
            0.0
        } else if a == 0.0 && b == 0.0 {
            0.0
        } else {
            (b - a) / a.max(b)
        };
        total += s;
    }

    total / n as f32
}

fn relabel_contiguous(labels: &[usize]) -> Vec<usize> {
    let mut mapping = std::collections::BTreeMap::<usize, usize>::new();
    let mut next = 0usize;
    labels
        .iter()
        .map(|label| {
            if let Some(&mapped) = mapping.get(label) {
                mapped
            } else {
                mapping.insert(*label, next);
                let value = next;
                next += 1;
                value
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_two_distinct_embeddings() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.01, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.98, 0.02],
        ];
        let labels = cluster_embeddings(
            &embeddings,
            DiarizationOptions {
                min_speakers: 2,
                max_speakers: 2,
                min_segment_ms: 500,
            },
        );
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }
}
