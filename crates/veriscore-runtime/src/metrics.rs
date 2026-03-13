use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct StageTiming {
    pub name: &'static str,
    pub elapsed: Duration,
}

pub struct StageTimer {
    name: &'static str,
    start: Instant,
}

impl StageTimer {
    pub fn start(name: &'static str) -> Self {
        Self { name, start: Instant::now() }
    }

    pub fn finish(self) -> StageTiming {
        StageTiming { name: self.name, elapsed: self.start.elapsed() }
    }
}
