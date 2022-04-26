use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashSet, VecDeque},
    hash::Hash,
    marker::PhantomData,
    ops::Add,
};

trait Queue<T> {
    fn push(&mut self, value: T);
    fn pop(&mut self) -> Option<T>;
}

// First in first out. Common or garden queue.
impl<T> Queue<T> for VecDeque<T> {
    fn push(&mut self, value: T) {
        self.push_back(value)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop_front()
    }
}

// Last in first out. Actually a stack but the caller needn't care.
impl<T> Queue<T> for Vec<T> {
    fn push(&mut self, value: T) {
        self.push(value)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
}

#[derive(PartialEq, Eq)]
struct CostOrdered<T, O> {
    value: T,
    cost: O,
}

impl<T: PartialEq, O: PartialOrd> PartialOrd for CostOrdered<T, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl<T: Eq, O: Ord> Ord for CostOrdered<T, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

struct CostQueue<T, O, C> {
    heap: BinaryHeap<Reverse<CostOrdered<T, O>>>,
    cost: C,
}

// Priority queue -- pops item with lowest cost first.
impl<T: Eq, C: FnMut(&T) -> O, O: Ord> Queue<T> for CostQueue<T, O, C> {
    fn push(&mut self, value: T) {
        let cost = (self.cost)(&value);
        self.heap.push(Reverse(CostOrdered { value, cost }))
    }

    fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|v| v.0.value)
    }
}

struct Traverse<Q, S, A, N, I, H> {
    queue: Q,
    adjacent: A,
    normalise: N,
    visited: HashSet<H>,
    _phantom: PhantomData<(S, I)>,
}

impl<Q, S, A, N, I, H> Iterator for Traverse<Q, S, A, N, I, H>
where
    Q: Queue<S>,
    A: FnMut(&S) -> I,
    N: FnMut(&S) -> H,
    I: IntoIterator<Item = S>,
    H: Hash + Eq,
{
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        let mut current = self.queue.pop()?;
        while self.visited.contains(&(self.normalise)(&current)) {
            current = self.queue.pop()?;
        }
        self.visited.insert((self.normalise)(&current));
        for state in (self.adjacent)(&current) {
            if !self.visited.contains(&(self.normalise)(&state)) {
                self.queue.push(state);
            }
        }
        Some(current)
    }
}

fn traverse<Q, S, A, N, I, H>(
    mut queue: Q,
    start: S,
    adjacent: A,
    normalise: N,
) -> impl Iterator<Item = S>
where
    Q: Queue<S>,
    A: FnMut(&S) -> I,
    N: FnMut(&S) -> H,
    I: IntoIterator<Item = S>,
    H: Hash + Eq,
{
    queue.push(start);
    Traverse {
        queue,
        adjacent,
        normalise,
        visited: HashSet::new(),
        _phantom: PhantomData,
    }
}

pub fn bft<S, A, N, I, H>(start: S, adjacent: A, normalise: N) -> impl Iterator<Item = S>
where
    A: FnMut(&S) -> I,
    N: FnMut(&S) -> H,
    I: IntoIterator<Item = S>,
    H: Hash + Eq,
{
    traverse(VecDeque::new(), start, adjacent, normalise)
}

pub fn bfs<S, A, N, G, I, H>(start: S, adjacent: A, normalise: N, goal: G) -> Option<S>
where
    A: FnMut(&S) -> I,
    N: FnMut(&S) -> H,
    G: FnMut(&S) -> bool,
    I: IntoIterator<Item = S>,
    H: Hash + Eq,
{
    bft(start, adjacent, normalise).find(goal)
}

pub fn dft<S, A, N, I, H>(start: S, adjacent: A, normalise: N) -> impl Iterator<Item = S>
where
    A: FnMut(&S) -> I,
    N: FnMut(&S) -> H,
    I: IntoIterator<Item = S>,
    H: Hash + Eq,
{
    traverse(Vec::new(), start, adjacent, normalise)
}

pub fn dfs<S, A, N, G, I, H>(start: S, adjacent: A, normalise: N, goal: G) -> Option<S>
where
    A: FnMut(&S) -> I,
    N: FnMut(&S) -> H,
    G: FnMut(&S) -> bool,
    I: IntoIterator<Item = S>,
    H: Hash + Eq,
{
    dft(start, adjacent, normalise).find(goal)
}

pub fn dijkstra<S, A, N, C, G, I, H, O>(
    cost: C,
    start: S,
    adjacent: A,
    normalise: N,
    goal: G,
) -> Option<S>
where
    S: Eq,
    A: FnMut(&S) -> I,
    N: FnMut(&S) -> H,
    C: FnMut(&S) -> O,
    G: FnMut(&S) -> bool,
    I: IntoIterator<Item = S>,
    H: Hash + Eq,
    O: Ord,
{
    traverse(
        CostQueue {
            heap: BinaryHeap::new(),
            cost,
        },
        start,
        adjacent,
        normalise,
    )
    .find(goal)
}

pub fn a_star<S, A, N, C, G, I, H, O>(
    mut cost: C,
    mut heuristic: C,
    start: S,
    adjacent: A,
    normalise: N,
    goal: G,
) -> Option<S>
where
    S: Eq,
    A: FnMut(&S) -> I,
    N: FnMut(&S) -> H,
    C: FnMut(&S) -> O,
    G: FnMut(&S) -> bool,
    I: IntoIterator<Item = S>,
    H: Hash + Eq,
    O: Add,
    <O as Add>::Output: Ord,
{
    dijkstra(|s| cost(s) + heuristic(s), start, adjacent, normalise, goal)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    #[derive(PartialEq, Eq)]
    struct State {
        pos: (i32, i32),
        cost: i32,
    }

    fn bit_count(mut n: i32) -> i32 {
        let mut count = 0;
        while n > 0 {
            count += n % 2;
            n /= 2;
        }
        count
    }

    fn is_wall(n: i32, (x, y): (i32, i32)) -> bool {
        x < 0 || y < 0 || bit_count(x * x + 3 * x + 2 * x * y + y + y * y + n) % 2 == 1
    }

    #[test]
    fn year_2016_day_13() {
        let n = 1352;
        let target = (31, 39);
        let dirs = vec![(1, 0), (-1, 0), (0, 1), (0, -1)];
        assert_eq!(
            Some(90),
            super::bfs(
                State {
                    pos: (1, 1),
                    cost: 0
                },
                |s| {
                    let mut adjacent = Vec::new();
                    for (dx, dy) in &dirs {
                        let pos = (s.pos.0 + dx, s.pos.1 + dy);
                        if !is_wall(n, pos) {
                            adjacent.push(State {
                                pos,
                                cost: s.cost + 1,
                            })
                        }
                    }
                    adjacent
                },
                |s| s.pos,
                |s| s.pos == target,
            )
            .map(|s| s.cost)
        );
    }

    #[test]
    fn year_2021_day_15() {
        let cavern = vec![
            "1163751742",
            "1381373672",
            "2136511328",
            "3694931569",
            "7463417111",
            "1319128137",
            "1359912421",
            "3125421639",
            "1293138521",
            "2311944581",
        ];
        let mut risks: HashMap<(i32, i32), i32> = HashMap::new();
        for (y, line) in cavern.iter().enumerate() {
            for (x, c) in line.chars().enumerate() {
                risks.insert((x as i32, y as i32), c.to_digit(10).unwrap() as i32);
            }
        }
        let dirs = vec![(1, 0), (-1, 0), (0, 1), (0, -1)];
        let target = (9, 9);
        assert_eq!(
            Some(40),
            super::dijkstra(
                |s| s.cost,
                State {
                    pos: (0, 0),
                    cost: 0
                },
                |s| {
                    let mut adjacent = Vec::new();
                    for (dx, dy) in &dirs {
                        let pos = (s.pos.0 + dx, s.pos.1 + dy);
                        if pos.0 >= 0 && pos.0 <= 9 && pos.1 >= 0 && pos.1 <= 9 {
                            adjacent.push(State {
                                pos,
                                cost: s.cost + risks[&pos],
                            });
                        }
                    }
                    adjacent
                },
                |s| s.pos,
                |s| s.pos == target,
            )
            .map(|s| s.cost),
        )
    }
}
